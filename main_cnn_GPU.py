import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy


from data_loader import set_seed
from MAB_fun import MABDefense
from trainer import train_client_cnn_GPU
from eval_DFL import evaluate_global_cnn
from defense import  get_cnn_trap_grad_func

# ==========================================
# 🛠️ NATIVE GPU UTILITIES (Replacing Numpy/Sklearn)
# ==========================================

def torch_kmeans(x, n_clusters, max_iters=100):
    """Native PyTorch K-Means running entirely on GPU."""
    N, D = x.shape
    if N < n_clusters: n_clusters = N
    indices = torch.randperm(N, device=x.device)[:n_clusters]
    centroids = x[indices].clone()
    labels = torch.zeros(N, device=x.device, dtype=torch.long)
    
    for _ in range(max_iters):
        dists = torch.cdist(x, centroids)
        new_labels = torch.argmin(dists, dim=1)
        if torch.all(labels == new_labels): break
        labels = new_labels
        for k in range(n_clusters):
            mask = (labels == k)
            if mask.any(): centroids[k] = x[mask].mean(dim=0)
    return labels

def torch_distance_clustering(dist_matrix, threshold=0.5):
    """Native PyTorch Agglomerative Clustering Proxy (GPU)."""
    N = dist_matrix.shape[0]
    labels = torch.arange(N, device=dist_matrix.device)
    for i in range(N):
        for j in range(i+1, N):
            if dist_matrix[i, j] < threshold:
                root_i, root_j = labels[i], labels[j]
                labels[labels == root_j] = root_i
    _, inverse_indices = torch.unique(labels, return_inverse=True)
    return inverse_indices

def calculate_krum_scores_gpu(vectors, f_limit):
    """Native GPU Krum scoring."""
    vec_tensor = torch.stack(vectors)
    dists = torch.cdist(vec_tensor, vec_tensor, p=2) ** 2 
    sorted_dists, _ = torch.sort(dists, dim=1)
    k_limit = vec_tensor.shape[0] - f_limit - 1
    if k_limit <= 0: return torch.zeros(vec_tensor.shape[0], device=vec_tensor.device).tolist()
    scores = torch.sum(sorted_dists[:, 1:k_limit+1], dim=1)
    return scores.tolist()

# ==========================================
# 🔄 MODIFIED METRICS & MODELS
# ==========================================

def get_universal_stats(delta_tensor):
    """Pure PyTorch implementation to keep data on GPU."""
    if not isinstance(delta_tensor, torch.Tensor):
        delta_tensor = torch.tensor(delta_tensor, device=DEVICE)
    flat = delta_tensor.flatten()
    pos_vals = flat[flat > 0]
    if len(pos_vals) > 0:
        median_p = torch.median(pos_vals)
        mad_p = torch.median(torch.abs(pos_vals - median_p)) + 1e-9
        z_scores = 0.6745 * (pos_vals - median_p) / mad_p
        t_z = torch.max(z_scores).item()
    else:
        t_z = 0.0
    return {'t_z': t_z}

def get_cnn_rs_score(model, device=None, tokenizer=None, num_samples=16, noise_std=0.5):
    """Feature-Level Randomized Smoothing (Unchanged logic, ensures GPU compatibility)."""
    if device is None: device = next(model.parameters()).device
    model.eval()
    
    g_device = torch.Generator(device=device) 
    g_device.manual_seed(42)
    
    # FIX: Pass device directly into randn so it matches the generator
    clean_input = torch.randn(1, 3, 32, 32, generator=g_device, device=device)
    kl_list = []
    
    def feature_noise_hook(module, input, output):
        noise = torch.randn_like(output) * noise_std
        return output + noise

    with torch.no_grad():
        clean_logits = model(clean_input)
        clean_probs = F.softmax(clean_logits, dim=1)
        
        target_layer = getattr(model, 'fc1', getattr(model, 'layer4', getattr(model, 'features', getattr(model, 'fc2', None))))
        handle = target_layer.register_forward_hook(feature_noise_hook)
        try:
            for _ in range(num_samples):
                noisy_logits = model(clean_input) 
                noisy_probs = F.softmax(noisy_logits, dim=1)
                kl = torch.sum(clean_probs * (torch.log(clean_probs + 1e-10) - torch.log(noisy_probs + 1e-10)))
                kl_list.append(max(0, kl.item()))
        finally:
            handle.remove()

    avg_kl = float(torch.tensor(kl_list).mean().item())
    return float(np.exp(-avg_kl * 2.0))

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=43):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ==========================================
# 🚀 CORE SIMULATION RUNNER (GPU PARALLEL)
# ==========================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

current_config = {'code': 'neurotoxin', 'pgd': 0, 'l2': 50, 'boost_factor': 1}
import torch
import torch.nn as nn
import scipy.linalg 

class FastGPUDataLoader:
    def __init__(self, dataset, batch_size, device):
        self.device = device
        self.batch_size = batch_size
        
   
        self.X = torch.stack([item[0] for item in dataset]).to(device, non_blocking=True)
        
 
        if isinstance(dataset[0][1], torch.Tensor):
            self.Y = torch.stack([item[1] for item in dataset]).to(device, non_blocking=True)
        else:
            self.Y = torch.tensor([item[1] for item in dataset], device=device)
            
        self.n_samples = len(self.Y)

    def __iter__(self):
     
        indices = torch.randperm(self.n_samples, device=self.device)
        for i in range(0, self.n_samples, self.batch_size):
            idx = indices[i : i + self.batch_size]
            yield self.X[idx], self.Y[idx]

    def __len__(self):
        return (self.n_samples + self.batch_size - 1) // self.batch_size


def run_simulation_CNN_GPU(seed, NUM_CLIENTS, defense_nodes, malicious_clients, G, neighbors, 
                       client_datasets, test_loader, atk_type='neurotoxin', mechanism='FedAvg', 
                       intensity=2.0, norm_factor=0.2, scale_factor=0.5, debug_mode=False,   audit_prob=0.9, agg_prob=0.8,, steepness=1,
                       GLOBAL_ROUNDS=15, epochs=5, debug=True):
    set_seed(seed)
    

    global_model = SimpleCNN().to(DEVICE)
    
    try:
        global_model = torch.compile(global_model) 
    except:
        pass 

    client_models = [SimpleCNN(num_classes=43).to(DEVICE) for _ in range(NUM_CLIENTS)]
    
 
    client_optimizers = [torch.optim.Adam(m.parameters(), lr=0.001, fused=True) for m in client_models]
    loss_fn = nn.CrossEntropyLoss()
    
    print("🚀 [System] Loading all client datasets directly into VRAM...")
    client_loaders = [FastGPUDataLoader(ds, batch_size=32, device=DEVICE) for ds in client_datasets]

    if mechanism == 'MAB':
        mab_defense = MABDefense(NUM_CLIENTS, model_type='cnn',audit_prob=audit_prob, agg_prob=agg_prob, steepness=steepness)

 
    probe_model = SimpleCNN(num_classes=43).to(DEVICE)
    probe_model.eval()
    print("📸 [System] Extracting real validation data for S_Z Probe...")
    try:
        real_probe_batch = next(iter(test_loader))
        
        if isinstance(real_probe_batch, (list, tuple)):
            real_probe_inputs = real_probe_batch[0].to(DEVICE) 
        elif isinstance(real_probe_batch, dict):
            real_probe_inputs = real_probe_batch['input_ids'].to(DEVICE) 
        else:
            real_probe_inputs = real_probe_batch.to(DEVICE)

        real_probe_inputs = real_probe_inputs[:16] 
        print(f"✅ Probe successfully extracted, shape: {real_probe_inputs.shape}")
        
    except Exception as e:
        print(f"⚠️ [Warning] Failed to extract from test_loader, fallback to Gaussian noise. Error: {e}")
    
        real_probe_inputs = torch.randn(16, 3, 32, 32, device=DEVICE)
    for round_idx in range(GLOBAL_ROUNDS):
        print(f"\n--- Round {round_idx+1}/{GLOBAL_ROUNDS} ---")
        
        start_states = [{k: v.clone() for k, v in m.state_dict().items()} for m in client_models]
        post_states = [None] * NUM_CLIENTS

        def train_client_serial(idx, is_malicious, ref_vec=None, ref_norm=None):
            loader = client_loaders[idx]  
            
            if not is_malicious:
                train_client_cnn_GPU(client_models[idx], client_optimizers[idx], loss_fn, loader, DEVICE,
                                     initial_global_state=start_states[idx], is_malicious=False, epochs=epochs)
            else:
                strategy_config = {'code': 'collusion', 'scale_length': 3.0}
                train_client_cnn_GPU(client_models[idx], client_optimizers[idx], loss_fn, loader, DEVICE,
                                     initial_global_state=start_states[idx], is_malicious=True,
                                     strategy_config=strategy_config, intensity=intensity, epochs=epochs,
                                     reference_vector=ref_vec, reference_norm=ref_norm, current_round=round_idx, total_rounds=GLOBAL_ROUNDS)
            
            new_state = {k: v.clone() for k, v in client_models[idx].state_dict().items()}
            
            total_norm = 0.0
            for k in new_state:
                if 'weight' in k:
                    total_norm += torch.norm(new_state[k] - start_states[idx][k]) ** 2
            return idx, new_state, torch.sqrt(total_norm).item()

        # --- Phase A-1: ---
        benign_indices = [i for i in range(NUM_CLIENTS) if i not in malicious_clients]
        benign_norms = []
        
        for i in benign_indices:
            idx, state, norm = train_client_serial(i, False)
            post_states[idx] = state
            benign_norms.append(norm)

        avg_benign_norm = np.median(benign_norms) if benign_norms else 1.0
        print(f"  [Attack Info] Estimated Benign Update Norm: {avg_benign_norm:.4f}")

        standard_keys = list(start_states[0].keys()) 
        benign_updates = []
        for i in benign_indices:
            delta_w_list = [post_states[i][k].float() - start_states[i][k].float() 
                            for k in standard_keys if 'num_batches_tracked' not in k and ('weight' in k or 'bias' in k)]
            benign_updates.append(torch.cat([d.flatten() for d in delta_w_list]))
        avg_benign_update = torch.stack(benign_updates).mean(dim=0) if benign_updates else None

        # --- Phase A-2:  ---
        mal_indices = list(malicious_clients)
        for i in mal_indices:
            idx, state, _ = train_client_serial(i, True, avg_benign_update, norm_factor*avg_benign_norm)
            post_states[idx] = state

        # --- Phase C: Aggregation Logic (Unchanged but uses single-thread flow context) ---
        vector_map = {}
        if mechanism in ['Krum', 'Cos', 'FLAME', 'CosL2']:
            for cid in range(NUM_CLIENTS):
                vec_list = [post_states[cid][k].float() - start_states[cid][k].float() 
                            for k in sorted(post_states[cid].keys()) if 'num_batches_tracked' not in k]
                vector_map[cid] = torch.cat([v.flatten() for v in vec_list])

        next_round_weights = []
        if mechanism not in ['MAB']: defense_nodes = set()
        defense_nodes = set(defense_nodes)
        malicious_clients = set(malicious_clients)

        for i in range(NUM_CLIENTS):
            my_nbs = neighbors[i]
            if i in malicious_clients:
                next_round_weights.append({k: v.clone() for k, v in post_states[i].items()})
                continue

            # [Branch 1] MAB Defense
            if mechanism == 'MAB' and i in defense_nodes:
                audit_targets = mab_defense.select_for_audit(i, my_nbs)
                audit_logs = {} 
                if audit_targets:
                    audit_targets = [int(t) for t in audit_targets]
                    audit_logs = mab_defense.update_trust(
                        observer_id=i, 
                        probe_list=audit_targets, 
                        new_weights=post_states, 
                        old_weights=start_states,
                        device=DEVICE, 
                        model_template=client_models[i], 
                        get_trap_func=get_cnn_trap_grad_func, 
                        rs_func=get_cnn_rs_score,
                        probe_inputs=real_probe_inputs
                    )

                selected_neighbors, agg_w = mab_defense.select_for_aggregation(client_id=i, candidates=audit_targets)
                
                # ==========================================
                # 平均聚合 (包含自身与选中的邻居)
                # ==========================================
                avg_state = {}
                
                # 确定所有参与聚合的节点：自身 + 被选中的邻居
                participating_nodes = [i]
                if selected_neighbors:
                    participating_nodes.extend(selected_neighbors)
                    
                num_participants = len(participating_nodes)

                for k in post_states[i].keys():
                    # 跳过非浮点参数（如 BatchNorm 的 tracked batches）
                    if 'num_batches_tracked' in k:
                        avg_state[k] = post_states[i][k].clone()
                        continue
                        
                    # 初始化全 0 张量
                    tmp_sum = torch.zeros_like(post_states[i][k].float())

                    # 累加所有参与节点的权重
                    for nid in participating_nodes:
                        tmp_sum += post_states[nid][k].float()

                    # 求平均并转换回原数据类型
                    avg_state[k] = (tmp_sum / num_participants).to(post_states[i][k].dtype)

                next_round_weights.append(avg_state)
# =========================================================
            # 🔥 [Branch 1.5] MAB: Ordinary Nodes (Updated Weight Logic)
            # =========================================================
            elif mechanism == 'MAB' and i not in defense_nodes:
                my_nbs = neighbors[i]
                
                
                defense_neighbors = [n for n in  defense_nodes]
                normal_neighbors = [n for n in neighbors if n not in defense_nodes]

                avg_state = {}
                
                w_self = 0.50
                
                if defense_neighbors and normal_neighbors:
                    w_def_total = 0.45
                    w_norm_total = 0.05
                elif defense_neighbors:
                    w_def_total = 0.50
                    w_norm_total = 0.0
                elif normal_neighbors:
                    w_def_total = 0.0
                    w_norm_total = 0.50
                else:
                    w_def_total = 0.0
                    w_norm_total = 0.0
                    w_self = 1.0
                
                w_def_per_node = w_def_total / len(defense_neighbors) if defense_neighbors else 0.0
                w_norm_per_node = w_norm_total / len(normal_neighbors) if normal_neighbors else 0.0

                # 🚨 FIX: Changed new_client_weights to post_states
                for k in post_states[i].keys():
                    if 'num_batches_tracked' in k:
                        avg_state[k] = post_states[i][k]
                        continue
                    
                    if 'weight' in k or 'bias' in k:
                  
                        local_w_old = start_states[i][k].float().to(DEVICE)
                        
         
                        my_trained_w = post_states[i][k].float().to(DEVICE)
                        tmp_sum = my_trained_w * w_self
                        
            
                        for nid in defense_neighbors:
                            nb_w = post_states[nid][k].float().to(DEVICE)
                            tmp_sum += nb_w * w_def_per_node
                            
                    
                        for nid in normal_neighbors:
                            nb_w = post_states[nid][k].float().to(DEVICE)
                            tmp_sum += nb_w * w_norm_per_node

                  
                        target_w = tmp_sum
                        
                        final_w = target_w
                        avg_state[k] = final_w.clone().detach().cpu().to(post_states[i][k].dtype)
                    else:
                        avg_state[k] = post_states[i][k]
                
               

                next_round_weights.append(avg_state)
            elif mechanism == 'CosL2':
                my_v = vector_map[i]
                
                # --- Step 1: --
                MALICIOUS_RATIO = len(malicious_clients) / NUM_CLIENTS
                m_winners = max(1, int(len(my_nbs) * (1 - MALICIOUS_RATIO)))
                
                sim_scores = []
                for nb in my_nbs:
                    sim = torch.nn.functional.cosine_similarity(my_v, vector_map[nb], dim=0).item()
                    sim_scores.append((sim, nb))
                
            
                sim_scores.sort(key=lambda x: x[0], reverse=True)
                selected_nids = [nb for sim, nb in sim_scores[:m_winners]]
                final_group = [i] + selected_nids 
                # --- Step 2: ---
         
                norms = [torch.norm(vector_map[nid]).item() for nid in final_group]
                gamma = np.median(norms) if norms else 1.0

                # --- Step 3:  ---
                avg_w = {}
                updates = {}
                for nid in final_group:
                    updates[nid] = {}
                    for k in start_states[0].keys():
                        if 'num_batches_tracked' in k: continue
                        w_new = post_states[nid][k].float().to(DEVICE)
                        w_old = start_states[nid][k].float().to(DEVICE)
                        updates[nid][k] = w_new - w_old

                for k in start_states[0].keys():
                    if 'num_batches_tracked' in k:
                        avg_w[k] = post_states[i][k]
                        continue
                    
                    tmp_sum = torch.zeros_like(post_states[i][k].float()).to(DEVICE)
                    for nid in final_group:
                        update_v = updates[nid][k]
                        current_node_norm = torch.norm(vector_map[nid]).item()
                        
                     
                        clip_factor = min(1.0, gamma / (current_node_norm + 1e-9))
                        tmp_sum += update_v * clip_factor
                    
                    avg_update = tmp_sum / len(final_group)
                    
             
                    w_base = start_states[i][k].float().to(DEVICE)
                    avg_w[k] = (w_base + avg_update).to(post_states[i][k].dtype)
                
                next_round_weights.append(avg_w)
            elif mechanism == 'TrimmedMean':
                candidates = my_nbs + [i]
                n = len(candidates)

    
                MALICIOUS_RATIO = len(malicious_clients)/NUM_CLIENTS
                k = int(n * MALICIOUS_RATIO)

                if n - 2 * k < 1:
                    k = max(0, (n - 1) // 2)

                avg_w = {}
                ref_keys = post_states[i].keys() 

                for key in ref_keys:
           
                    if 'num_batches_tracked' in key:
                        stack_w = torch.stack([post_states[nid][key] for nid in candidates])
                
                        avg_w[key] = stack_w.float().mean(0).to(post_states[i][key].dtype)
                        continue

     
                    stack_w = torch.stack([post_states[nid][key].float() for nid in candidates], dim=0)

                    sorted_w, _ = torch.sort(stack_w, dim=0)

                
                    trimmed_w = sorted_w[k : n - k]

           
                    avg_w[key] = trimmed_w.mean(dim=0).to(post_states[i][key].dtype)

                next_round_weights.append(avg_w)

            elif mechanism == 'FLAME':
                candidates = my_nbs + [i]
                if len(candidates) < 3:
                    trusted_ids = candidates
                else:
                    vecs = torch.stack([vector_map[nid] for nid in candidates])
                    vecs_norm = F.normalize(vecs, p=2, dim=1)
                    dist_matrix = 1.0 - torch.mm(vecs_norm, vecs_norm.t()) 
                    
                    cluster_labels = torch_distance_clustering(dist_matrix, threshold=0.5).tolist()
                    my_cluster = cluster_labels[-1]
                    trusted_ids = [candidates[idx] for idx, lbl in enumerate(cluster_labels) if lbl == my_cluster]

                trusted_vecs = [vector_map[nid] for nid in trusted_ids]
                norms = torch.tensor([torch.norm(v) for v in trusted_vecs])
                gamma = torch.median(norms).item() if len(norms) > 0 else 1.0  
                
                avg_w = {}
                noise_std = 0.001 * gamma 
                for k in start_states[0].keys():
                    if 'num_batches_tracked' in k:
                        avg_w[k] = post_states[i][k]
                        continue
                    
                    sum_update = torch.zeros_like(start_states[i][k]).float()
                    for nid in trusted_ids:
                        delta = post_states[nid][k].float() - start_states[nid][k].float()
                        nid_norm = torch.norm(vector_map[nid]).item() 
                        sum_update += delta * min(1.0, gamma / (nid_norm + 1e-9))
                        
                    avg_update = sum_update / len(trusted_ids)
                    noise = torch.randn_like(avg_update) * noise_std
                    avg_w[k] = (start_states[i][k].float() + avg_update + noise).to(post_states[i][k].dtype)
                next_round_weights.append(avg_w)

            # [Branch 3] Krum
            elif mechanism == 'Krum':
                candidates = my_nbs + [i]
                MAL_RATIO = len(malicious_clients)/NUM_CLIENTS
                f_limit = int(len(candidates) * MAL_RATIO)
                m_winners = max(1, len(candidates) - f_limit)

                candidate_vecs = [vector_map[nid] for nid in candidates]
                all_k_scores = calculate_krum_scores_gpu(candidate_vecs, f_limit)
                
                node_score_map = {nid: score for nid, score in zip(candidates, all_k_scores)}
                sorted_candidates = sorted(candidates, key=lambda nid: node_score_map[nid])
                winner_ids = sorted_candidates[:m_winners]

                if winner_ids:
                    avg_w = {}
                    for k in post_states[winner_ids[0]].keys():
                        stack_w = torch.stack([post_states[nid][k].float() for nid in winner_ids])
                        avg_w[k] = stack_w.mean(0).to(post_states[i][k].dtype)
                    next_round_weights.append(avg_w)
                else:
                    next_round_weights.append({k: v.clone() for k, v in post_states[i].items()})

            # [Branch 4] Default
            else:
                candidates = my_nbs + [i]
                avg_state = {}
                for k in post_states[i].keys():
                    stack_w = torch.stack([post_states[nid][k].float() for nid in candidates])
                    avg_state[k] = stack_w.mean(0).to(post_states[i][k].dtype)
                next_round_weights.append(avg_state)

        for i in range(NUM_CLIENTS):
            client_models[i].load_state_dict(next_round_weights[i])
            client_optimizers[i] = torch.optim.Adam(client_models[i].parameters(), lr=0.001, fused=True)
        del post_states, next_round_weights
        if 'vector_map' in locals(): del vector_map
        torch.cuda.empty_cache()

    # --- Final Eval ---
    print("\n" + "="*50)
    print(f"📊 FINAL SIMULATION REPORT (Seed: {seed})")
    print("="*50)

    benign_acc_list, benign_asr_list, acc_list, asr_list = [], [], [], []

    for cid in range(NUM_CLIENTS):
        is_mal = (cid in malicious_clients)
        acc, asr = evaluate_global_cnn(client_models[cid], test_loader, DEVICE, trigger_type='patch', intensity=intensity)
        acc_list.append(acc), asr_list.append(asr)
        if not is_mal:
            benign_acc_list.append(acc), benign_asr_list.append(asr)
        print(f"Client {cid:2d} | Role: {'MAL 😈' if is_mal else 'BEN 😇':<6} | ACC: {acc:.4f} | ASR: {asr:.4f}")

    avg_benign_acc = np.mean(benign_acc_list) if benign_acc_list else 0
    avg_benign_asr = np.mean(benign_asr_list) if benign_asr_list else 0

    return avg_benign_acc, avg_benign_asr, acc_list, asr_list