import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import warnings
import concurrent.futures

from torch.utils.data import DataLoader
from collections import defaultdict

from data_loader import set_seed
from MAB_fun import MABDefense_CNN, MABDefense, calculate_continuous_trust  
from trainer import train_client_cnn_GPU
from defense import calculate_krum_scores
from eval_DFL import evaluate_global_cnn,evaluate_global_cnn_fast
from defense import get_cnn_sensitivity_score, get_cnn_trap_grad_func

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
import torch.nn.functional as F
import numpy as np
import scipy.linalg  # 用于优化微小矩阵的 SVD 计算
import copy
class FastGPUDataLoader:
    def __init__(self, dataset, batch_size, device):
        self.device = device
        self.batch_size = batch_size
        
        # 将所有客户端的微小数据一次性堆入显存
        self.X = torch.stack([item[0] for item in dataset]).to(device, non_blocking=True)
        
        # 标签处理
        if isinstance(dataset[0][1], torch.Tensor):
            self.Y = torch.stack([item[1] for item in dataset]).to(device, non_blocking=True)
        else:
            self.Y = torch.tensor([item[1] for item in dataset], device=device)
            
        self.n_samples = len(self.Y)

    def __iter__(self):
        # 纯 GPU 生成随机索引，告别 CPU 参与
        indices = torch.randperm(self.n_samples, device=self.device)
        for i in range(0, self.n_samples, self.batch_size):
            idx = indices[i : i + self.batch_size]
            yield self.X[idx], self.Y[idx]

    def __len__(self):
        return (self.n_samples + self.batch_size - 1) // self.batch_size
# (保留你原本的 DEVICE 声明和其他辅助函数)

def run_simulation_CNN_GPU(seed, NUM_CLIENTS, defense_nodes, malicious_clients, G, neighbors, 
                       client_datasets, test_loader, atk_type='neurotoxin', mechanism='FedAvg', 
                       intensity=2.0, norm_factor=0.2, scale_factor=0.5, debug_mode=False, 
                       GLOBAL_ROUNDS=15, epochs=5, debug=True):
    set_seed(seed)
    
    # 初始化全局模型
    global_model = SimpleCNN().to(DEVICE)
    
    # 🚀 极致黑魔法 2: torch.compile 静态图编译 (若 Windows 报错可注释掉 try 块内的代码)
    try:
        global_model = torch.compile(global_model) 
    except:
        pass 

    client_models = [SimpleCNN(num_classes=43).to(DEVICE) for _ in range(NUM_CLIENTS)]
    
    # 🚀 极致黑魔法 3: fused=True 融合算子，将梯度更新合并为一次 GPU Kernel 启动
    client_optimizers = [torch.optim.Adam(m.parameters(), lr=0.001, fused=True) for m in client_models]
    loss_fn = nn.CrossEntropyLoss()
    
    print("🚀 [System] Loading all client datasets directly into VRAM...")
    client_loaders = [FastGPUDataLoader(ds, batch_size=32, device=DEVICE) for ds in client_datasets]

    if mechanism == 'MAB':
        mab_defense = MABDefense(NUM_CLIENTS, model_type='cnn', decay=0.9, exploration_c=0.5, 
                                 audit_prob=0.9, agg_prob=0.8, custom_target_layers=None)

    # 循环外预先分配探针模型，避免每轮反复 malloc
    probe_model = SimpleCNN(num_classes=43).to(DEVICE)
    probe_model.eval()
    # =========================================================================
    # 🔥 核心防御机制：提前提取“真实图像”作为激活空间探针，防止 S_Z 后期归零
    # =========================================================================
    print("📸 [System] Extracting real validation data for S_Z Probe...")
    try:
        # 从测试集中抽取一个 batch，通常 next(iter()) 返回的是 (inputs, labels)
        real_probe_batch = next(iter(test_loader))
        
        if isinstance(real_probe_batch, (list, tuple)):
            real_probe_inputs = real_probe_batch[0].to(DEVICE) # 取出图像张量
        elif isinstance(real_probe_batch, dict):
            real_probe_inputs = real_probe_batch['input_ids'].to(DEVICE) # 兼容 NLP
        else:
            real_probe_inputs = real_probe_batch.to(DEVICE)

        # 为了避免探针 batch_size 太大导致 OOM，可以截断到 16 或 32
        real_probe_inputs = real_probe_inputs[:16] 
        print(f"✅ Probe successfully extracted, shape: {real_probe_inputs.shape}")
        
    except Exception as e:
        print(f"⚠️ [Warning] Failed to extract from test_loader, fallback to Gaussian noise. Error: {e}")
        # 降级：如果提取失败，依然用归一化的随机噪声顶替
        real_probe_inputs = torch.randn(16, 3, 32, 32, device=DEVICE)
    for round_idx in range(GLOBAL_ROUNDS):
        print(f"\n--- Round {round_idx+1}/{GLOBAL_ROUNDS} ---")
        
        start_states = [{k: v.clone() for k, v in m.state_dict().items()} for m in client_models]
        post_states = [None] * NUM_CLIENTS

        # 🚀 优化点 3：移除多线程和 CUDA Stream，改为纯串行。单卡跑轻量模型，串行反而更快。
        def train_client_serial(idx, is_malicious, ref_vec=None, ref_norm=None):
            loader = client_loaders[idx]  # 直接使用提前建好的 loader
            
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

        # --- Phase A-1: 串行训练良性节点 ---
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

        # --- Phase A-2: 串行训练恶意节点 ---
        mal_indices = list(malicious_clients)
        for i in mal_indices:
            idx, state, _ = train_client_serial(i, True, avg_benign_update, norm_factor*avg_benign_norm)
            post_states[idx] = state

        # --- Phase B-3: Activation Clustering ---
        if debug:
            print("\n🧠 [Debug Analysis] Phase B-3: Activation Clustering")
            probe_data, _ = next(iter(test_loader))
            probe_data = probe_data[:16].to(DEVICE)
            
            activations_store = {}
            def get_activation_hook(name):
                def hook(module, input, output):
                    activations_store[name] = output.detach().flatten()
                return hook

            # 仅注册一次 Hook
            handle = probe_model.fc1.register_forward_hook(get_activation_hook('fc1'))

            for observer_id in range(NUM_CLIENTS):
                candidates = neighbors[observer_id] + [observer_id]
                if len(candidates) < 2: continue

                act_vectors = []
                for nid in candidates:
                    # 🚀 直接复用 probe_model，不重新实例化 nn.Module
                    probe_model.load_state_dict(post_states[nid]) 
                    with torch.no_grad(): _ = probe_model(probe_data)
                    act_vectors.append(activations_store['fc1'])

                X_acts = torch.stack(act_vectors)
                n_clusters = min(2, len(candidates))
                labels = torch_kmeans(X_acts, n_clusters).tolist()

                print(f"  Observer {observer_id} 激活层聚类结果 (K={n_clusters}):")
                for idx, nid in enumerate(candidates):
                    role = "MAL 😈" if nid in malicious_clients else "BEN 😇"
                    is_self = "(Me)" if nid == observer_id else ""
                    alert = " 🔥 异类!" if (nid in malicious_clients and labels[idx] != labels[-1]) else ""
                    print(f"    {str(nid)+is_self:<6} | {role:<6} |   簇 {labels[idx]} {alert}")
            
            handle.remove() # 清理 Hook

        # --- Phase B: Dual Metric Comparison ---
        if debug:
            def calc_row_wise_stats(delta_tensor):
                if delta_tensor.ndim < 2: return 0.0
                pos_energy = torch.sum(torch.clamp(delta_tensor, min=0), dim=1)
                median_p = torch.median(pos_energy)
                mad_p = torch.median(torch.abs(pos_energy - median_p)) + 1e-9
                z_pos = 0.6745 * (pos_energy - median_p) / mad_p
                return torch.max(z_pos).item() if len(z_pos) > 0 else 0.0

            def calc_element_wise_stats(delta_tensor):
                flat = delta_tensor.flatten()
                pos_vals = flat[flat > 0]
                if len(pos_vals) > 0:
                    median_p = torch.median(pos_vals)
                    mad_p = torch.median(torch.abs(pos_vals - median_p)) + 1e-9
                    z_scores = 0.6745 * (pos_vals - median_p) / mad_p
                    return torch.max(z_scores).item()
                return 0.0

            print(f"\n🔍 [Debug Analysis] Round {round_idx+1} Full Inspection:")
            print(f"{'Obs':<4} | {'Cand.':<8} | {'Type':<5} | {'Sz(Row)':<10} | {'Sz(Elem)':<10} | {'RS Score':<10} | {'SVD':<8} | {'Krum':<10} | {'L2 Dist':<10}")
            print("-" * 145)

            target_layer = 'fc2.weight'
            temp_state = post_states[0]
            if target_layer not in temp_state: target_layer = list(temp_state.keys())[-2]

            for obs_id in range(NUM_CLIENTS):
                candidates = neighbors[obs_id] + [obs_id]
                if len(candidates) < 2: continue

                deltas = {nid: (post_states[nid][target_layer] - start_states[nid][target_layer]).float() for nid in candidates}
                deltas_flat = {nid: deltas[nid].flatten() for nid in candidates}

                f_limit = max(1, int(len(candidates) * 0.3))
                candidate_vecs = [deltas_flat[nid] for nid in candidates]
                k_scores = calculate_krum_scores_gpu(candidate_vecs, f_limit)
                sorted_indices = np.argsort(k_scores)
                ranks = {candidates[idx]: rank + 1 for rank, idx in enumerate(sorted_indices)}

                mean_tensor_flat = torch.stack(list(deltas_flat.values())).mean(dim=0)

                for nid in candidates:
                    d_tensor = deltas[nid]
                    sz_row_val = calc_row_wise_stats(d_tensor)
                    sz_elem_val = calc_element_wise_stats(d_tensor)

                    # 🚀 直接复用 rs_model，避免反复生成模型
                    rs_model.load_state_dict(post_states[nid])
                    rs_val = get_cnn_rs_score(rs_model, DEVICE)

                    # 🚀 优化点 4：对于微小矩阵，拉回 CPU 使用 Scipy 算 SVD 远快于触发 GPU Kernel 
                    try:
                        mat = d_tensor.view(d_tensor.shape[0], -1) if d_tensor.ndim > 2 else d_tensor
                        mat_cpu = mat.detach().cpu().numpy()
                        svd_max = scipy.linalg.svdvals(mat_cpu)[0]
                    except: 
                        svd_max = 0.0

                    l2_dist = torch.norm(deltas_flat[nid] - mean_tensor_flat).item()
                    my_rank = ranks[nid]
                    marker = "🔥" if (obs_id not in malicious_clients) and (nid in malicious_clients) and (my_rank <= 3) else " "

                    print(f"{obs_id:<4} | {nid:<4} | {'MAL' if nid in malicious_clients else 'BEN':<5} | {sz_row_val:<10.4f} | {sz_elem_val:<10.4f} | {rs_val:<10.4f} | {svd_max:<8.4f} | {k_scores[candidates.index(nid)]:<10.4f} | {l2_dist:<10.4f} {marker}")

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
                # 🔥 Modified: Detailed Trust & Action Report
                # ==========================================
                # Only print for Observer 0 (or specific nodes) to reduce clutter
                # if i > 0: 
                #         print(f"\n📊 [Round {round_idx+1}] Observer {i} Trust & Aggregation Report:")
                        
                #         # 1. 扩充表头以包含新指标 (将 Sens 替换为 S_Z)
                #         header = (f" {'Neighbor':<8} | {'Role':<9} | {'Trust(Q)':<8} | "
                #                 f"{'Trap':<8} | {'S_Z':<8} | {'RS_Score':<8} | {'Action':<10}") 
                #         print(header)
                #         print("-" * 100)
                        
                #         my_nbs0 = set(neighbors[i]) | {i}
                        
                #         # Iterate through neighbors
                #         for nid in my_nbs0:
                #             # 1. Role (Ground Truth)
                #             role = "MAL" if nid in malicious_clients else "BEN"
                #             role_str = f"{role} 😈" if role == "MAL" else f"{role} 😇"

                #             # 2. Current Trust Score
                #             curr_trust = mab_defense.trust_scores[i].get(nid, 0.5)

                #             # 3. 提取详细指标 (仅对被审计的节点有效)
                #             if nid in audit_logs:
                #                 t_log = audit_logs[nid]
                #                 trap_str = f"{t_log.get('trap', 0.0):.4f}"
                #                 s_z_str  = f"{t_log.get('elem_z', 0.0):.4f}"  # <--- 提取 elem_z
                #                 z_str    = f"{t_log.get('z_comb_penalty', 0.0):.4f}" 
                #                 rs_str   = f"{t_log.get('rs_score', 0.0):.4f}"
                #             else:
                #                 # 如果未被审计，全部显示 N/A
                #                 trap_str = s_z_str = z_str = rs_str = "N/A"

                #             # 4. Action Status
                #             if nid in selected_neighbors:
                #                 action = "✅ AGG"   # Selected for aggregation
                #             elif nid in audit_targets:
                #                 action = "❌ DROP"  # Audited but rejected
                #             else:
                #                 action = "⚪ SKIP"  # Not selected for audit

                #             # 5. Print Detailed row (将 max_sen_str 替换为 s_z_str)
                #             print(f" {nid:<8} | {role_str:<9} | {curr_trust:<8.4f} | "
                #                 f"{trap_str:<8} | {s_z_str:<8} | {rs_str:<8} | {action:<10}")
                        
                #         print("-" * 90)
                # avg_state = {}
                # local_alpha = 0.5
                # actual_nb_weight = (1 - local_alpha) if selected_neighbors else 0.0
                # for k in post_states[i].keys():
                #     local_t = post_states[i][k].float()
                #     if not selected_neighbors:
                #         avg_state[k] = local_t.to(post_states[i][k].dtype)
                #         continue
                #     final_val = local_t * local_alpha
                #     for idx, nid in enumerate(selected_neighbors):
                #         final_val += post_states[nid][k].float() * (agg_w[idx] * actual_nb_weight)
                #     avg_state[k] = final_val.to(post_states[i][k].dtype)
                # next_round_weights.append(avg_state)
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
                
                # 🚨 FIX: Note that def_blacklists needs to be populated in Branch 1 earlier in the round!
                # Assuming def_blacklists is populated globally or earlier in the round logic.
                # If not, this is safe to run but local_banned_set will remain empty.
                def_blacklists = def_blacklists if 'def_blacklists' in locals() else {}
                
                # 建立安全名单 (根据邻近 DEF 节点的局部黑名单排雷)
                local_banned_set = set()
                for nb in my_nbs:
                    if nb in def_blacklists:
                        local_banned_set.update(def_blacklists[nb])
                
                safe_nbs = [n for n in my_nbs if n not in local_banned_set]
                
                # 区分邻居类型 (只从安全名单中筛选)
                defense_neighbors = [n for n in safe_nbs if n in defense_nodes]
                normal_neighbors = [n for n in safe_nbs if n not in defense_nodes]

                avg_state = {}
                
                # 惯性系数
                beta_inertia = 0
                if safe_nbs:
                    # 🌟 预先计算分配好线性聚合的权重
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
                    
                    # 平分到每个具体的节点上
                    w_def_per_node = w_def_total / len(defense_neighbors) if defense_neighbors else 0.0
                    w_norm_per_node = w_norm_total / len(normal_neighbors) if normal_neighbors else 0.0

                    # 🚨 FIX: Changed new_client_weights to post_states
                    for k in post_states[i].keys():
                        if 'num_batches_tracked' in k:
                            avg_state[k] = post_states[i][k]
                            continue
                        
                        if 'weight' in k or 'bias' in k:
                            # 1. 提取老权重用于计算 EMA
                            local_w_old = start_states[i][k].float().to(DEVICE)
                            
                            # 2. 累加自己的权重 (Self)
                            my_trained_w = post_states[i][k].float().to(DEVICE)
                            tmp_sum = my_trained_w * w_self
                            
                            # 3. 线性累加 DEF 邻居的权重
                            for nid in defense_neighbors:
                                nb_w = post_states[nid][k].float().to(DEVICE)
                                tmp_sum += nb_w * w_def_per_node
                                
                            # 4. 线性累加普通邻居的权重
                            for nid in normal_neighbors:
                                nb_w = post_states[nid][k].float().to(DEVICE)
                                tmp_sum += nb_w * w_norm_per_node

                            # 5. 聚合结果作为目标权重
                            target_w = tmp_sum
                            
                            # 6. 引入 EMA 惯性进行平滑
                            final_w = (beta_inertia * local_w_old) + ((1.0 - beta_inertia) * target_w)
                            avg_state[k] = final_w.clone().detach().cpu().to(post_states[i][k].dtype)
                        else:
                            avg_state[k] = post_states[i][k]
                
                else:
                    # --- Case B: 孤岛生存模式 (Fallback) ---
                    for k in post_states[i].keys():
                        if 'weight' in k or 'bias' in k:
                            local_w_old = start_states[i][k].float().to(DEVICE)
                            my_trained_w = post_states[i][k].float().to(DEVICE)
                            
                            final_w = (beta_inertia * local_w_old) + ((1.0 - beta_inertia) * my_trained_w)
                            avg_state[k] = final_w.clone().detach().cpu().to(post_states[i][k].dtype)
                        else:
                            avg_state[k] = post_states[i][k]

                next_round_weights.append(avg_state)
            # =========================================================
            # 🔥 [新增] Branch 3.5: CosL2 (Cosine Filtering + L2 Clipping)
            # =========================================================
            elif mechanism == 'CosL2':
                my_v = vector_map[i]
                
                # --- Step 1: 方向过滤 (CosSim) ---
                MALICIOUS_RATIO = len(malicious_clients) / NUM_CLIENTS
                m_winners = max(1, int(len(my_nbs) * (1 - MALICIOUS_RATIO)))
                
                sim_scores = []
                for nb in my_nbs:
                    sim = torch.nn.functional.cosine_similarity(my_v, vector_map[nb], dim=0).item()
                    sim_scores.append((sim, nb))
                
                # 按相似度降序排列
                sim_scores.sort(key=lambda x: x[0], reverse=True)
                selected_nids = [nb for sim, nb in sim_scores[:m_winners]]
                final_group = [i] + selected_nids # 永远信任自己

                # 打印日志 (防刷屏)
                # print(f"\n📊 [Round {round_idx+1}] Obs {i} CosL2 Decision (n={len(my_nbs)}, keep_top={m_winners})")
                # print(f"    {'Neighbor':<8} | {'Role':<6} | {'Cos Sim':<12} | {'Decision'}")
                # print("-" * 55)
                # for sim, nb in sim_scores:
                #     role = "MAL" if nb in malicious_clients else "BEN"
                #     is_selected = nb in selected_nids
                #     status = "✅ AGG" if is_selected else "❌ DROP"
                #     if role == "MAL" and is_selected: status += " ⚠️ ALERT"
                #     print(f"    N{nb:<7} | {role:<6} | {sim:<12.4f} | {status}")
                # print("-" * 55)

                # --- Step 2: 尺度裁剪 (L2 Clipping) ---
                # 计算所有入选邻居的 L2 范数
                norms = [torch.norm(vector_map[nid]).item() for nid in final_group]
                gamma = np.median(norms) if norms else 1.0

                # --- Step 3: 聚合与物理写回 ---
                avg_w = {}
                # 预先获取 Delta 字典加速计算
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
                        
                        # 核心裁剪逻辑：超速则按比例缩小
                        clip_factor = min(1.0, gamma / (current_node_norm + 1e-9))
                        tmp_sum += update_v * clip_factor
                    
                    avg_update = tmp_sum / len(final_group)
                    
                    # W_new = W_old + Avg_Update
                    w_base = start_states[i][k].float().to(DEVICE)
                    avg_w[k] = (w_base + avg_update).to(post_states[i][k].dtype)
                
                next_round_weights.append(avg_w)
            elif mechanism == 'TrimmedMean':
                candidates = my_nbs + [i]
                n = len(candidates)

                # 1. 确定修剪数量 k (Beta)
                # 通常设为预期恶意节点的比例。例如若恶意比例 35%，则两端各去掉 35%
                # 必须保证 n - 2k > 0，否则没东西剩下了
                MALICIOUS_RATIO = len(malicious_clients)/NUM_CLIENTS
                k = int(n * MALICIOUS_RATIO)

                # 边界保护：如果修剪过猛导致没人了，就强制减少 k
                # 至少保留 1 个或是中间的 1/3
                if n - 2 * k < 1:
                    k = max(0, (n - 1) // 2)

                avg_w = {}
                ref_keys = post_states[i].keys() # 获取键列表

                for key in ref_keys:
                    # 跳过非训练统计量 (如 BN 层的 num_batches_tracked)
                    # 或者对它们做简单平均，这里选择简单处理
                    if 'num_batches_tracked' in key:
                        stack_w = torch.stack([post_states[nid][key] for nid in candidates])
                        # 对于整数统计量，取中位数或众数可能更好，这里取平均并转回 Long
                        avg_w[key] = stack_w.float().mean(0).to(post_states[i][key].dtype)
                        continue

                    # 2. 堆叠所有邻居的该层参数
                    # Shape: [Neighbors, Param_Dim1, Param_Dim2...]
                    stack_w = torch.stack([post_states[nid][key].float() for nid in candidates], dim=0)

                    # 3. 核心：在第 0 维（邻居维度）进行排序
                    # 这实现了 "Coordinate-wise" (逐坐标) 的筛选
                    # 也就是对于权重矩阵的每一个格子，单独看谁大谁小
                    sorted_w, _ = torch.sort(stack_w, dim=0)

                    # 4. 修剪 (Trim)
                    # 去掉最小的 k 个，和最大的 k 个
                    # 切片范围: [k : n-k]
                    trimmed_w = sorted_w[k : n - k]

                    # 5. 对剩余部分求平均
                    avg_w[key] = trimmed_w.mean(dim=0).to(post_states[i][key].dtype)

                next_round_weights.append(avg_w)
            # [Branch 2] FLAME Defense
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