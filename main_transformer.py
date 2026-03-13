import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import random
import copy
import networkx as nx
import seaborn as sns  # Added for heatmap
from collections import defaultdict
from torch.utils.data import Subset, DataLoader
from matplotlib.patches import Patch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification,  get_linear_schedule_with_warmup
from torch.optim import AdamW
import networkx as nx
import numpy as np
import copy
import random
import matplotlib.pyplot as plt
import torch.nn.functional as F

import math
from scipy.stats import wasserstein_distance
import gc  
import math
import torch
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import sys




from data_loader import set_seed
from MAB_fun import MABDefense
from trainer import train_client_transformer
#from defense import get_universal_stats
from defense import  calculate_krum_scores,get_trap_grad_func
from theoretical_intensity import calculate_theoretical_intensity
from eval_DFL import evaluate_global_transformer
# ==========================================
# 1. Data Processing & Distribution
# ==========================================
def get_transformer_rs_score(model, tokenizer, device, num_samples=8, noise_std=0.12):

    model.eval()
    

    text = "The quick brown fox jumps over the lazy dog."
    inputs = tokenizer(text, return_tensors="pt", padding='max_length', max_length=64, truncation=True).to(device)
    
    kl_list = []
    
    with torch.no_grad():

        if hasattr(model, 'distilbert'):
            base_model = model.distilbert
        elif hasattr(model, 'bert'):
            base_model = model.bert
        else:
            base_model = getattr(model, model.base_model_prefix, model)
            
        clean_embeds = base_model.embeddings(inputs.input_ids)
        
        clean_out = model(**inputs)
        clean_probs = torch.softmax(clean_out.logits, dim=-1)
        

        for _ in range(num_samples):
            noise = torch.randn_like(clean_embeds) * noise_std
            noised_embeds = clean_embeds + noise
            
        
            noisy_out = model(inputs_embeds=noised_embeds)
            noisy_probs = torch.softmax(noisy_out.logits, dim=-1)
            
      
            kl = torch.sum(clean_probs * (torch.log(clean_probs + 1e-10) - torch.log(noisy_probs + 1e-10))).item()
            kl_list.append(max(0, kl))


    avg_kl = np.mean(kl_list)
    rs_score = np.exp(-avg_kl * 5.0)
    
    return float(rs_score)


def get_model(MODEL_CHECKPOINT = "distilbert-base-uncased", NUM_LABELS = 4):

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=NUM_LABELS
    )
    return model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = device
MODEL_CHECKPOINT = "distilbert-base-uncased"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
NUM_LABELS = 5
MAX_LEN = 64
BATCH_SIZE = 64
def run_simulation_transformer(seed, NUM_CLIENTS, defense_nodes, malicious_clients, G, neighbors, client_datasets, test_data,
                           atk_type='neurotoxin', mechanism='FedAvg', bf=1.0, intensity=0.1,goodnorm=2,
                           debug_mode=False, GLOBAL_ROUNDS=15, norm_factor=40, debug=True, epochs=1, MAX_WORKERS=8, log_file_path=None):
    

    if log_file_path:
        sys.stdout = open(log_file_path, "a", encoding='utf-8')
        sys.stderr = sys.stdout
    set_seed(seed)

    # Model Init
    gpu_model = get_model(NUM_LABELS=NUM_LABELS).to(device)
    initial_state = {k: v.cpu() for k, v in gpu_model.state_dict().items()}

    # Client Weights (Stateful DFL)
    client_weights_cpu = [copy.deepcopy(initial_state) for _ in range(NUM_CLIENTS)]

    # Defense Init
    mab_defense = None
    if mechanism == 'MAB':
      #  mab_defense = MABDefense_transformer(NUM_CLIENTS)
        mab_defense =  MABDefense(NUM_CLIENTS, model_type='transformer')
    # Config
    strategy_config = {'code': 'collusion', 'boost_factor': bf, 'mask_rate': 0.5}

    # =========================================================================
    print("📸 [System] Extracting real PubMed validation text for S_Z Probe...")
    try:
    
        if isinstance(test_data, DataLoader):
            real_probe_batch = next(iter(test_data))
        else:
            real_probe_batch = test_data[:16]


        if isinstance(real_probe_batch, dict) and 'input_ids' in real_probe_batch:
            real_probe_inputs = real_probe_batch['input_ids'].to(device)
        elif isinstance(real_probe_batch, (list, tuple)):
            real_probe_inputs = real_probe_batch[0].to(device)
        else:
            real_probe_inputs = real_probe_batch.to(device)

        # 限制大小防止 OOM
        real_probe_inputs = real_probe_inputs[:16] 
        print(f"✅ Probe successfully extracted, shape: {real_probe_inputs.shape}")
        
    except Exception as e:
        print(f"⚠️ [Warning] Failed to extract from test_data. Fallback to random tokens. Error: {e}")

        vocab_size = getattr(gpu_model.config, 'vocab_size', 30522)
        real_probe_inputs = torch.randint(0, vocab_size, (16, MAX_LEN), device=device)
    # =========================================================================
    sens_history = defaultdict(list)
    for r in range(GLOBAL_ROUNDS):
        print(f"\n--- Round {r+1}/{GLOBAL_ROUNDS} ---")

        # Container for NEW weights after local training
        new_weights_cpu = [None] * NUM_CLIENTS
        benign_norms = []

        # --- Phase A-1: Train Benign First (to estimate Norm) ---
        benign_indices = [i for i in range(NUM_CLIENTS) if i not in malicious_clients]

        MAX_WORKERS =MAX_WORKERS  
        print(f"  [Info] Starting parallel training for {len(benign_indices)} benign clients with {MAX_WORKERS} workers...")


        new_weights_cpu_temp = {cid: None for cid in benign_indices}
        benign_norms_temp = {cid: 0.0 for cid in benign_indices}

      
        master_model_cpu = get_model(NUM_LABELS=NUM_LABELS)

        def train_single_benign(cid):
            try:
                local_model = copy.deepcopy(master_model_cpu).to(device)
                
         
                w_new = train_client_transformer(
                    local_model, 
                    client_datasets[cid], 
                    device,
                    client_weights_cpu[cid],
                    is_malicious=False,
                    epochs=epochs,
                    current_round=r,
                    BATCH_SIZE=32
                )
                
                norm_sq = sum(torch.norm(w_new[k] - client_weights_cpu[cid][k]).item()**2 
                            for k in w_new if 'weight' in k)
                
                del local_model
                return cid, w_new, math.sqrt(norm_sq)
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"  [Error] OOM on Client {cid}. Decrease MAX_WORKERS or BATCH_SIZE.")
                    torch.cuda.empty_cache()
                raise e

    
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_cid = {executor.submit(train_single_benign, cid): cid for cid in benign_indices}
            
            for future in as_completed(future_to_cid):
                cid = future_to_cid[future]
                try:
                    res_cid, w_new, norm = future.result()
                    new_weights_cpu_temp[res_cid] = w_new
                    benign_norms_temp[res_cid] = norm
                except Exception as exc:
                    print(f"    -> Client {cid} exception: {exc}")
                finally:
                
                    del future 

      
            future_to_cid.clear() 
            del future_to_cid
        for cid in benign_indices:
            new_weights_cpu[cid] = new_weights_cpu_temp[cid]
            benign_norms.append(benign_norms_temp[cid])

        avg_benign_norm = np.median(benign_norms) if benign_norms else 0.5
        print(f"  [Info] Avg Benign Norm: {avg_benign_norm:.4f}")
        avg_benign_norm = np.median(benign_norms) if benign_norms else 0.5
        print(f"  [Info] Avg Benign Norm: {avg_benign_norm:.4f}")
        old_w = client_weights_cpu[cid]
        
        benign_updates = []
  
        standard_keys = sorted([k for k in initial_state.keys() if 'weight' in k or 'bias' in k])

        for cid in benign_indices:
            delta_w_list = []
            for k in standard_keys: 
                if 'num_batches_tracked' in k: continue
                # 确保只处理 float 类型的权重
                
                w_new = new_weights_cpu[cid][k].float()
                w_old = client_weights_cpu[cid][k].float()
                
                # 展平并添加
                delta_w_list.append((w_new - w_old).flatten())
            
            if delta_w_list:
                benign_updates.append(torch.cat(delta_w_list))    
        if benign_updates:
            avg_benign_update = torch.stack(benign_updates).mean(dim=0)
        else:
            avg_benign_update = None

        def train_malicious_wrapper(cid):
            w_new = train_client_transformer(
                gpu_model, client_datasets[cid], device,
                client_weights_cpu[cid],
                is_malicious=True,
                strategy_config=strategy_config, intensity=intensity,
                reference_norm=bf*avg_benign_norm, 
                reference_vector=avg_benign_update,
                epochs=2, norm_factor=norm_factor
            )
            return cid, w_new

 
        mal_indices = list(malicious_clients)

        with ThreadPoolExecutor(max_workers=len(mal_indices)) as executor:
            results = list(executor.map(train_malicious_wrapper, mal_indices))


        for cid, w_new in results:
            new_weights_cpu[cid] = w_new
 
        next_round_weights = []

        # ==========================================
        vector_map = {}
        if mechanism in ['Krum', 'Cos', 'FLAME', 'CosL2']:
            for cid in range(NUM_CLIENTS):
                vec_list = []
                sorted_keys = sorted(new_weights_cpu[cid].keys())
                for k in sorted_keys:
                    if 'num_batches_tracked' in k: continue

                    w_new = new_weights_cpu[cid][k].float()
                    w_old = client_weights_cpu[cid][k].float().to(w_new.device)
                    delta = w_new - w_old

                    vec_list.append(delta.flatten().half())

                vector_map[cid] = torch.cat(vec_list)
        round_trust_logs = {}


        for i in range(NUM_CLIENTS):
            audit_logs = {}
            my_nbs = neighbors[i]
            candidates = my_nbs + [i]
            if i in malicious_clients:
                            cloned_state = {k: v.clone() for k, v in new_weights_cpu[i].items()}
                            next_round_weights.append(cloned_state)
                            continue
         
            # --- Logic: Defense Node ---
            if i in defense_nodes and mechanism == 'MAB':
                    # --- Step 1: Select Audit Targets (Phase 1) ---
                    audit_targets = mab_defense.select_for_audit(i, my_nbs) 

                    # --- Step 2: Update Trust (Phase 2) ---
                    audit_logs = {}

                    # 只有当有审计目标时才进行更新
                    if audit_targets:
                        audit_logs = mab_defense.update_trust(
                            observer_id=i,
                            probe_list=audit_targets,
                            new_weights=new_weights_cpu,
                            old_weights=client_weights_cpu,
                            device=DEVICE,
                            model_template=gpu_model,
                            tokenizer=TOKENIZER,
                            get_trap_func=get_trap_grad_func,
                            rs_func=get_transformer_rs_score,
                            probe_inputs=real_probe_inputs
                        )

                    # --- Step 3: Select for Aggregation (Phase 3) ---
                    selected_neighbors, agg_w = mab_defense.select_for_aggregation(
                        client_id=i,
                        candidates=audit_targets
                    )

                  
                    # ==========================================
                    avg_state = {}
                    
        
                    participating_nodes = [i]
                    if selected_neighbors:
                        participating_nodes.extend(selected_neighbors)
                        
                    num_participants = len(participating_nodes)

                    for k in new_weights_cpu[i].keys():
                   
                        if 'num_batches_tracked' in k:
                            avg_state[k] = new_weights_cpu[i][k].clone()
                            continue
                            
                
                        tmp_sum = torch.zeros_like(new_weights_cpu[i][k].float())

                    
                        for nid in participating_nodes:
                            tmp_sum += new_weights_cpu[nid][k].float()

                        avg_state[k] = (tmp_sum / num_participants).to(new_weights_cpu[i][k].dtype)

                    next_round_weights.append(avg_state)
            # ==========================================
            # Krum 
            # ==========================================
            elif mechanism == 'Krum':
                candidates = my_nbs + [i]
                MALICIOUS_RATIO = len(malicious_clients)/NUM_CLIENTS
                f_limit = int(len(candidates) * MALICIOUS_RATIO)
                m_winners = max(1, len(candidates) - f_limit)

         
                candidate_vecs = [vector_map[nid] for nid in candidates]

        
                all_k_scores = calculate_krum_scores(candidate_vecs, f_limit)
                node_score_map = {nid: score for nid, score in zip(candidates, all_k_scores)}

             
                sorted_cands_by_score = sorted(candidates, key=lambda x: node_score_map[x])
                winner_ids = sorted_cands_by_score[:m_winners]

                if winner_ids:
                   
                    avg_w = {}
                
                    for k in initial_state.keys():
                  
                        if 'num_batches_tracked' in k:
                            avg_w[k] = new_weights_cpu[i][k]
                            continue
                    avg_w = {}
                    for k in initial_state.keys():
                        if 'num_batches_tracked' in k:
                            avg_w[k] = new_weights_cpu[i][k].clone()
                            continue

                        # 初始化一个零张量
                        tmp_sum = torch.zeros_like(new_weights_cpu[i][k].float())
                        for nid in winner_ids:
                            tmp_sum += new_weights_cpu[nid][k].float()
                        
                        avg_w[k] = (tmp_sum / len(winner_ids)).to(new_weights_cpu[i][k].dtype)

                    next_round_weights.append(avg_w)
                else:
                    next_round_weights.append(copy.deepcopy(new_weights_cpu[i]))
            # =========================================================
            # CosL2
            # =========================================================
            elif mechanism == 'CosL2':
                    my_v = vector_map[i]
                    my_nbs = neighbors[i]
                    
                    if not my_nbs:
                        next_round_weights.append(copy.deepcopy(new_weights_cpu[i]))
                        continue
                    
                    # --- Step 1: ---
                    MALICIOUS_RATIO = len(malicious_clients) / NUM_CLIENTS
                    m_winners = max(1, int(len(my_nbs) * (1 - MALICIOUS_RATIO)))
                    
                    sim_scores = []
                    for nb in my_nbs:
                   
                        sim = torch.nn.functional.cosine_similarity(my_v.unsqueeze(0), vector_map[nb].unsqueeze(0), dim=1).item()
                        sim_scores.append((sim, nb))
                    
           
                    sim_scores.sort(key=lambda x: x[0], reverse=True)
                    
  
                    selected_nids = [nb for sim, nb in sim_scores[:m_winners]]
                    final_group = [i] + selected_nids 


                    # --- Step 2: (L2 Clipping) ---
                    norms = [torch.norm(vector_map[nid]).item() for nid in final_group]
                    gamma = np.median(norms) 
                    
                    norm_dict = {nid: norm for nid, norm in zip(final_group, norms)}

                    # --- Step 3: ---
                    avg_w = {}
                    
                    # 🚨 REMOVED the entire `updates = {}` pre-calculation loop here

                    for k in initial_state.keys():
                        if 'num_batches_tracked' in k:
                            avg_w[k] = new_weights_cpu[i][k]
                            continue
                        
                        if 'weight' in k or 'bias' in k:
                            tmp_sum = torch.zeros_like(new_weights_cpu[i][k].float())
                            
                            for nid in final_group:
                                # 1. Calculate delta ON THE FLY layer-by-layer (Saves ~2.5GB RAM)
                                w_new = new_weights_cpu[nid][k].float()
                                w_old = client_weights_cpu[nid][k].float().to(w_new.device)
                                update_v = w_new - w_old
                                
                                # 2. Lookup pre-calculated norm (Saves immense CPU time)
                                current_node_norm = norm_dict[nid]
                                
                                # 🌟 L2 Clipping
                                clip_factor = min(1.0, gamma / (current_node_norm + 1e-9))
                                tmp_sum += update_v * clip_factor
                            
                            avg_update = tmp_sum / len(final_group)
                            
                   
                            w_base = client_weights_cpu[i][k].float().to(avg_update.device)
                            avg_w[k] = (w_base + avg_update).to(new_weights_cpu[i][k].dtype)
                        else:
                            avg_w[k] = new_weights_cpu[i][k]
                    
                    next_round_weights.append(avg_w)
            # =========================================================
            # FLAME 
            # =========================================================
            elif mechanism == 'FLAME':
                candidates = my_nbs + [i]
                n_c = len(candidates)
                
                if n_c < 3:
                    trusted_ids = candidates
                    cluster_labels = [-1] * n_c 
                else:
                    dist_matrix = np.zeros((n_c, n_c), dtype=np.float32)
                    
                    for r_idx in range(n_c):
                        for c_idx in range(r_idx + 1, n_c):
                            v1 = vector_map[candidates[r_idx]]
                            v2 = vector_map[candidates[c_idx]]
                            
                          
                            sim = torch.nn.functional.cosine_similarity(v1, v2, dim=0, eps=1e-8).item()
                            dist = max(0.0, 1.0 - sim)
                            
                            dist_matrix[r_idx, c_idx] = dist
                            dist_matrix[c_idx, r_idx] = dist
                            del v1, v2
                    torch.cuda.empty_cache()
                    try:
                        from sklearn.cluster import HDBSCAN
                        clusterer = HDBSCAN(
                            min_cluster_size=2,               
                            min_samples=1,                    
                            cluster_selection_epsilon=0.5,    
                            metric='precomputed'             
                        )
                        cluster_labels = clusterer.fit_predict(dist_matrix)
            
                    except ImportError:
                        from sklearn.cluster import AgglomerativeClustering
                        clusterer = AgglomerativeClustering(
                            n_clusters=None,                  
                            distance_threshold=0.5,          
                            metric='precomputed', 
                            linkage='average'
                        )
                        cluster_labels = clusterer.fit_predict(dist_matrix)
                    # 4. 识别良性簇 (Trusted Cluster)
                    my_idx = candidates.index(i)
                    my_cluster = cluster_labels[my_idx]
                    
                    if my_cluster == -1:
                        unique_labels, counts = np.unique(cluster_labels[cluster_labels != -1], return_counts=True)
                        if len(unique_labels) > 0:
                            largest_cluster = unique_labels[np.argmax(counts)]
                            trusted_ids = [candidates[idx] for idx, lbl in enumerate(cluster_labels) if lbl == largest_cluster]
                        else:
                            trusted_ids = [i] 
                    else:
                        trusted_ids = [candidates[idx] for idx, lbl in enumerate(cluster_labels) if lbl == my_cluster]

              
            
                trusted_vecs = [vector_map[nid] for nid in trusted_ids]
                norms = [torch.norm(v).item() for v in trusted_vecs]
                gamma = np.median(norms) if norms else 1.0  
                
             
                avg_w = {}
                noise_std = 0.001 * gamma 
                
                for k in initial_state.keys():
                    if 'num_batches_tracked' in k:
                        avg_w[k] = new_weights_cpu[i][k]
                        continue
                        
                    clipped_sum = torch.zeros_like(new_weights_cpu[i][k].float())
                    
                    for nid in trusted_ids:
                        w_update = (new_weights_cpu[nid][k].float() - client_weights_cpu[nid][k].float())
                        
                        idx = trusted_ids.index(nid)
                        clip_factor = min(1.0, gamma / (norms[idx] + 1e-9))
                        
                        clipped_sum += w_update * clip_factor
                        
                    avg_update = clipped_sum / len(trusted_ids)
                    noise = torch.randn_like(avg_update) * noise_std
                    
                    final_w = client_weights_cpu[i][k].float() + avg_update + noise
                    avg_w[k] = final_w.to(new_weights_cpu[i][k].dtype)
                    
                next_round_weights.append(avg_w)
# ==========================================
            # 🔥 Strategy: Trimmed Mean (Coordinate-wise)
            # ==========================================
            elif mechanism == 'TrimmedMean':
                candidates = my_nbs + [i] # Include self
                n = len(candidates)
                beta = 0.2 # Trim ratio (cut top/bottom 20%)
                cut = int(n * beta)

                # Safety check: if too few neighbors, fallback to FedAvg (mean all)
                # We need at least 1 remaining after trimming both ends
                if n - 2 * cut <= 0:
                    cut = 0

                avg_w = {}
                for k in initial_state.keys():
                    if 'num_batches_tracked' in k:
                        avg_w[k] = new_weights_cpu[i][k]
                        continue

                    # 1. Stack all candidates for this layer: Shape [N, ...]
                    stack_tensors = torch.stack([new_weights_cpu[nid][k].float() for nid in candidates])

                    if cut > 0:
                        # 2. Sort along the first dimension (client dimension)
                        sorted_stack, _ = torch.sort(stack_tensors, dim=0)

                        # 3. Trim (Coordinate-wise)
                        # Keep indices from [cut] to [n - cut]
                        trimmed_stack = sorted_stack[cut : n - cut]

                        # 4. Mean
                        avg_w[k] = trimmed_stack.mean(dim=0).to(new_weights_cpu[i][k].dtype)
                    else:
                        avg_w[k] = stack_tensors.mean(dim=0).to(new_weights_cpu[i][k].dtype)

                next_round_weights.append(avg_w)

            # ==========================================
            # 🔥 Strategy: Cosine Similarity Defense
            # ==========================================
            elif mechanism == 'Cos':
                # Logic: Compare neighbors' updates to OWN update.
                # Reject if Cosine Similarity < Threshold (e.g., 0)

                base_vec = vector_map[i] # Own update vector
                accepted_ids = [i] # Always trust self

                candidates = my_nbs # Only check neighbors (self is already in)

                for nid in candidates:
                    nb_vec = vector_map[nid]

                    # Calculate Cosine Similarity
                    # cos = (A . B) / (|A|*|B|)
                    cos_sim = torch.nn.functional.cosine_similarity(base_vec, nb_vec, dim=0, eps=1e-8).item()

                    # Threshold: 0.0 means discard negative correlations (likely malicious/opposite direction)
                    if cos_sim > 0.05:
                        accepted_ids.append(nid)

                # Logging (Optional, similar to Krum)
                if i in defense_nodes and r % 5 == 0:
                     print(f"   [Cos] Node {i} accepted {len(accepted_ids)}/{len(my_nbs)+1} peers.")

                # Aggregation of Accepted IDs
                if not accepted_ids:
                    accepted_ids = [i] # Should not happen as self is in, but safety first

                avg_w = {}
                for k in initial_state.keys():
                    if 'num_batches_tracked' in k:
                        avg_w[k] = new_weights_cpu[i][k]
                        continue

                    stack_w = torch.stack([new_weights_cpu[nid][k].float() for nid in accepted_ids])
                    avg_w[k] = stack_w.mean(dim=0).to(new_weights_cpu[i][k].dtype)

                next_round_weights.append(avg_w)
           # --- Logic: Standard FedAvg OR MAB-Aware Benign Aggregatio


            # =========================================================
            # 🔥 [Branch 1.5] MAB: Ordinary Nodes (Updated Weight Logic)
            # =========================================================
            # This must come BEFORE the final 'else' block
            elif mechanism == 'MAB' and i not in defense_nodes:
            
                my_nbs = neighbors[i]

                defense_neighbors = [n for n in my_nbs if n in defense_nodes]
                normal_neighbors = [n for n in my_nbs if n not in defense_nodes]

                avg_state = {}

                if defense_neighbors:
               
                    w_self_total = 0.5

                    w_remaining = 0.5

                    w_defense_total = w_remaining * 0.9

                    w_normal_total = w_remaining * 0.1

             
                    w_self = w_self_total

                    w_def = w_defense_total / len(defense_neighbors)

                  
                    if normal_neighbors:
                        w_norm = w_normal_total / len(normal_neighbors)
                    else:
                        w_norm = 0.0
                        w_self += w_normal_total

              
                    for k in initial_state.keys():
                        if 'num_batches_tracked' in k:
                            avg_state[k] = new_weights_cpu[i][k]
                            continue

                     
                        tmp_sum = new_weights_cpu[i][k].float() * w_self

                   
                        for dn in defense_neighbors:
                            tmp_sum += new_weights_cpu[dn][k].float() * w_def

                      
                        for neigh in normal_neighbors:
                            tmp_sum += new_weights_cpu[neigh][k].float() * w_norm

                        avg_state[k] = tmp_sum.to(new_weights_cpu[i][k].dtype)

         
                else:
                    candidates = my_nbs + [i]
                    for k in initial_state.keys():
                        if 'num_batches_tracked' in k:
                            avg_state[k] = new_weights_cpu[i][k].clone()
                            continue

                        # 累加逻辑
                        tmp_sum = torch.zeros_like(new_weights_cpu[candidates[0]][k].float())
                        for nid in candidates:
                            tmp_sum += new_weights_cpu[nid][k].float()
                        avg_state[k] = (tmp_sum / len(candidates)).to(new_weights_cpu[i][k].dtype)

                next_round_weights.append(avg_state)

            # --- Logic: Standard FedAvg OR Malicious Fallback ---
            # This 'else' catches everything else (FedAvg, or Malicio
            else:
                # Standard FedAvg for benign nodes in non-MAB/non-Krum mechanisms
                candidates = neighbors[i] + [i]
                avg_w = {}
                for k in initial_state.keys():
                    if 'num_batches_tracked' in k:
                        avg_w[k] = new_weights_cpu[i][k].clone()
                        continue
                    
                    tmp_sum = torch.zeros_like(new_weights_cpu[candidates[0]][k].float())
                    for nid in candidates:
                        tmp_sum += new_weights_cpu[nid][k].float()
                    avg_w[k] = (tmp_sum / len(candidates)).to(new_weights_cpu[i][k].dtype)
                
                next_round_weights.append(avg_w)
      
  
        if 'new_weights_cpu' in locals():
            while len(new_weights_cpu) > 0:
                item = new_weights_cpu.pop()
                del item
            del new_weights_cpu

        if 'vector_map' in locals():
            vector_map.clear() 
            del vector_map

        if 'benign_updates' in locals():
            del benign_updates
        if 'avg_benign_update' in locals():
            del avg_benign_update

        old_weights_ref = client_weights_cpu
        client_weights_cpu = next_round_weights
        del old_weights_ref 

      
        import gc
        gc.collect() 
        torch.cuda.empty_cache()
        # Update Weights
        client_weights_cpu = next_round_weights
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()

    print(f"Experiment {mechanism} Finished.")

# =================================================
    # Evaluation
    # =================================================
    print(f"\n📊 --- Final Evaluation (Round {r+1}) ---")
    
 
    print(f" {'Client':<8} | {'Type':<6} | {'ACC':<10} | {'ASR':<10} | {'Theo Intensity'}")
    print("-" * 65)


    theo_intensities = calculate_theoretical_intensity(neighbors, malicious_clients, NUM_CLIENTS, bf)
    
    benign_acc_list, benign_asr_list = [], []
    acc_list = []
    asr_list = []
    for cid in range(NUM_CLIENTS):
      
        if cid in malicious_clients:
            node_type = "MAL" 
        elif cid in defense_nodes: 
            node_type = "DEF"
        else:
            node_type = "BEN"  
            

        acc, asr = evaluate_global_transformer(gpu_model, client_weights_cpu[cid], test_data, device, intensity=intensity)

        acc_list.append(float(acc))  
        asr_list.append(float(asr)) 
        
        if node_type != "MAL":      
            benign_acc_list.append(float(acc))
            benign_asr_list.append(float(asr))
      
        theo_int = theo_intensities[cid] if cid < len(theo_intensities) else 0
        
        print(f" {cid:<8} | {node_type:<6} | {acc:<10.4f} | {asr:<10.4f} | {theo_int:.4f}")

    avg_benign_acc = np.mean(benign_acc_list) if benign_acc_list else 0.0
    avg_benign_asr = np.mean(benign_asr_list) if benign_asr_list else 0.0

    del gpu_model
    del client_weights_cpu
    del next_round_weights
    

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize() 
    return avg_benign_acc, avg_benign_asr ,acc_list, asr_list
      