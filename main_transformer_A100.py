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
import seaborn as sns
import pandas as pd
import os
import math
from scipy.stats import wasserstein_distance
import gc  
import math
import torch
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import sys
# 在 trainer.py 中确保使用了线性衰减
from transformers import get_linear_schedule_with_warmup
# 随着 Global Rounds 增加，逐步减小学习率

from data_loader import set_seed
from MAB_fun import MABDefense_transformer,MABDefense
from trainer import train_client_transformer,train_client_transformer_A100
#from defense import get_universal_stats
from defense import  calculate_krum_scores,get_trap_grad_func,get_transformer_sensitivity_score
from theoretical_intensity import calculate_theoretical_intensity
from eval_DFL import evaluate_global_transformer
# ==========================================
# 1. Data Processing & Distribution
# ==========================================
import torch
import torch.nn.functional as F
import numpy as np

def get_transformer_rs_score(model, tokenizer, device, num_samples=8, noise_std=0.12):
    """
    针对 A100 优化的 Randomized Smoothing 评估。
    去除了 for 循环，使用 Batching（向量化）一次性完成所有前向传播。
    """
    model.eval()
    
    text = "The quick brown fox jumps over the lazy dog."
    inputs = tokenizer(text, return_tensors="pt", padding='max_length', max_length=64, truncation=True).to(device)
    
    with torch.no_grad():
        # 1. 自动探测 Base Model 并获取 Clean Embeddings
        base_model = getattr(model, 'distilbert', getattr(model, 'bert', getattr(model, model.base_model_prefix, model)))
        clean_embeds = base_model.embeddings(inputs.input_ids) # Shape: [1, seq_len, hidden_dim]
        
        # 2. 获取 Clean 状态下的输出 (包括 Attention)
        clean_out = model(**inputs, output_attentions=True)
        clean_probs = torch.softmax(clean_out.logits, dim=-1) # Shape: [1, num_classes]
        
        clean_attn = clean_out.attentions[0] if clean_out.attentions else None # Shape: [1, num_heads, seq_len, seq_len]

        # ==========================================
        # 🔥 A100 优化点：Batch 向量化生成噪声
        # ==========================================
        # 将输入复制 num_samples 份: Shape 变为 [num_samples, seq_len, hidden_dim]
        batched_embeds = clean_embeds.expand(num_samples, -1, -1)
        
        # 一次性生成所有噪声并相加
        noise = torch.randn_like(batched_embeds) * noise_std
        noised_embeds = batched_embeds + noise
        
        # ==========================================
        # 🔥 A100 优化点：单次 Batched 前向传播
        # ==========================================
        # 使用 bfloat16 (AMP) 加速前向传播
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            noisy_out = model(inputs_embeds=noised_embeds, output_attentions=True)
            noisy_probs = torch.softmax(noisy_out.logits, dim=-1) # Shape: [num_samples, num_classes]
            
        # ==========================================
        # 3. 向量化计算 KL 散度和 Attention Drift
        # ==========================================
        # Clean_probs 会自动 Broadcast 到 [num_samples, num_classes]
        kl_divs = torch.sum(clean_probs * (torch.log(clean_probs + 1e-10) - torch.log(noisy_probs + 1e-10)), dim=-1)
        # 截断负数并求平均
        avg_kl = torch.clamp(kl_divs, min=0.0).mean().item()
        rs_score = np.exp(-avg_kl * 5.0)

        # 向量化计算 Attention Drift
        attn_stability = 0.0
        if clean_attn is not None and noisy_out.attentions is not None:
            noisy_attn = noisy_out.attentions[0] # Shape: [num_samples, num_heads, seq_len, seq_len]
            # Clean_attn 自动 Broadcast
            attn_drifts = torch.mean((clean_attn - noisy_attn) ** 2, dim=(1, 2, 3)) # 计算每个 sample 的 drift
            attn_stability = torch.std(attn_drifts).item() * 1000

    return float(rs_score), float(attn_stability)
def get_suppression_score(model, target_class_idx=None, last_weights=None):
    """
    计算特定类别的抑制分数 (Suppression Score)。
    分数越高，代表该类别的权重被抑制得越厉害（可能是受到惩罚项攻击或遗忘防御）。

    Args:
        model: 当前的 PyTorch 模型 (必须包含 classifier 层)
        target_class_idx (int, optional): 
            如果为 None，函数会自动寻找“最被抑制”的那个类别并返回其分数和索引。
            如果指定了 int，则只计算该类别的分数。
        last_weights (dict, optional): 
            上一轮的模型权重字典。如果提供，将计算动态更新趋势 (Delta)。

    Returns:
        dict: {
            'score': float,          # 综合抑制分数 (通常 > 2.0 视为异常)
            'target_idx': int,       # 被分析的类别索引
            'norm_z': float,         # 范数 Z-Score (负数代表偏小)
            'delta_trend': float,    # 更新趋势 (负数代表正在被抑制)
            'bias_dev': float        # 偏置项偏差
        }
    """
    # 1. 定位 Classification Head
    # 适配 HuggingFace 的常见命名 (DistilBert: classifier, Bert: classifier)
    if hasattr(model, 'classifier'):
        head = model.classifier
    elif hasattr(model, 'fc'):
        head = model.fc
    else:
        # 尝试自动查找最后一层线性层
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, torch.nn.Linear):
                head = module
                break
        else:
            return {'score': 0.0, 'info': 'No linear head found'}

    # 2. 提取权重和偏置
    # weight shape: [num_classes, hidden_dim]
    # bias shape:   [num_classes]
    weights = head.weight.detach().cpu()
    biases = head.bias.detach().cpu() if head.bias is not None else torch.zeros(weights.shape[0])
    
    num_classes = weights.shape[0]
    
    # 3. 计算每一类的 L2 范数
    class_norms = torch.norm(weights, p=2, dim=1).numpy()
    
    # --- 指标 A: 静态范数 Z-Score (Static Norm Z) ---
    # 衡量：该类的“能量”是否显著低于群体
    median_norm = np.median(class_norms)
    mad_norm = np.median(np.abs(class_norms - median_norm)) + 1e-9
    norm_z_scores = 0.6745 * (class_norms - median_norm) / mad_norm
    
    # --- 指标 B: 偏置项 Z-Score (Bias Z) ---
    # 衡量：该类的基础门槛是否被设得极高（Bias 极小）
    median_bias = np.median(biases.numpy())
    mad_bias = np.median(np.abs(biases.numpy() - median_bias)) + 1e-9
    bias_z_scores = 0.6745 * (biases.numpy() - median_bias) / mad_bias

    # --- 指标 C: 动态更新趋势 (Delta Trend) ---
    # 衡量：该类在这一轮是否经历了“负向更新” (仅当提供 last_weights 时有效)
    delta_trends = np.zeros(num_classes)
    if last_weights is not None:
        # 寻找对应的旧权重 key
        head_key = None
        for k in last_weights.keys():
            # 简单匹配：以 .weight 结尾且形状匹配
            if k.endswith('weight') and last_weights[k].shape == weights.shape:
                head_key = k
                break
        
        if head_key:
            old_W = last_weights[head_key].float().cpu()
            delta_W = weights - old_W # [num_classes, hidden]
            
            # 计算每一类更新向量与该类原权重的余弦相似度
            # 如果是 -1，说明在反向抵消；如果是 1，说明在增强
            for c in range(num_classes):
                cos_sim = F.cosine_similarity(delta_W[c].unsqueeze(0), old_W[c].unsqueeze(0)).item()
                delta_trends[c] = cos_sim
    
    # 4. 综合计算逻辑
    def calc_single_score(idx):
        # 既然我们检测“抑制”，我们关注的是指标的【负值】有多大
        # 取负号，将“越小越异常”转换为“分数越高越异常”
        
        s_norm = -norm_z_scores[idx]   # 范数越小，分数越高
        s_bias = -bias_z_scores[idx]   # 偏置越小，分数越高
        
        # 对于 Delta Trend，负的 Cosine Similarity 代表抑制
        s_delta = -delta_trends[idx] * 5.0 # 放大权重，因为 -1~1 范围较小
        
        # 综合加权 (权重可调)
        # 静态范数占 40%，偏置占 20%，动态趋势占 40%
        final = 0.4 * max(0, s_norm) + 0.2 * max(0, s_bias) + 0.4 * max(0, s_delta)
        return final

    # 5. 返回结果
    if target_class_idx is not None:
        # 指定了目标类
        score = calc_single_score(target_class_idx)
        return {
            'score': float(score),
            'target_idx': int(target_class_idx),
            'norm_z': float(norm_z_scores[target_class_idx]),
            'bias_z': float(bias_z_scores[target_class_idx]),
            'delta_trend': float(delta_trends[target_class_idx])
        }
    else:
        # 未指定，寻找最被抑制的那个类 (Max Suppression Score)
        all_scores = [calc_single_score(i) for i in range(num_classes)]
        max_idx = np.argmax(all_scores)
        return {
            'score': float(all_scores[max_idx]),
            'target_idx': int(max_idx),
            'norm_z': float(norm_z_scores[max_idx]),
            'bias_z': float(bias_z_scores[max_idx]),
            'delta_trend': float(delta_trends[max_idx])
        }

def get_model(MODEL_CHECKPOINT = "distilbert-base-uncased", NUM_LABELS = 4):
    """初始化一个新的 Transformer 模型"""
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
BATCH_SIZE = 128
import torch
import torch.nn as nn
import numpy as np
import random
import copy
import math
import gc
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================================
# 🔥 A100 专属全局优化：开启 TF32 核心
# ==========================================
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def run_simulation_transformer_A100(seed, NUM_CLIENTS, defense_nodes, malicious_clients, G, neighbors, client_datasets, test_data,
                               atk_type='neurotoxin', mechanism='FedAvg', bf=1.0, intensity=0.1,
                               debug_mode=False, GLOBAL_ROUNDS=15, norm_factor=40, debug=True, epochs=1, MAX_WORKERS=10, log_file_path=None):
    
    # --- 接收子进程日志 ---
    if log_file_path:
        sys.stdout = open(log_file_path, "a", encoding='utf-8')
        sys.stderr = sys.stdout
    set_seed(seed)

    # Model Init
    gpu_model = get_model(NUM_LABELS=NUM_LABELS).to(device)
    
    # 🔥 A100 优化：如果 PyTorch 版本 >= 2.0，可以考虑解开下面这行的注释进行编译加速
    # gpu_model = torch.compile(gpu_model) 

    initial_state = {k: v.cpu() for k, v in gpu_model.state_dict().items()}
    standard_keys_for_delta = sorted([k for k in initial_state.keys() if 'weight' in k or 'bias' in k])

    # Client Weights (Stateful DFL)
    client_weights_cpu = [copy.deepcopy(initial_state) for _ in range(NUM_CLIENTS)]

    # Defense Init
    mab_defense = None
    if mechanism == 'MAB':
        mab_defense = MABDefense(NUM_CLIENTS, model_type='transformer')
        
    # Config
    strategy_config = {'code': 'collusion', 'boost_factor': bf, 'mask_rate': 0.5}
    sens_history = defaultdict(list)

    for r in range(GLOBAL_ROUNDS):
        print(f"\n--- Round {r+1}/{GLOBAL_ROUNDS} ---")

        new_weights_cpu = [None] * NUM_CLIENTS
        benign_norms = []
        benign_indices = [i for i in range(NUM_CLIENTS) if i not in malicious_clients]

        # ==========================================
        # Phase A-1: Train Benign First
        # ==========================================
        print(f"  [Info] Starting parallel training for {len(benign_indices)} benign clients with {MAX_WORKERS} workers...")
        new_weights_cpu_temp = {cid: None for cid in benign_indices}
        benign_norms_temp = {cid: 0.0 for cid in benign_indices}
        master_model_cpu = get_model(NUM_LABELS=NUM_LABELS)

        def train_single_benign(cid):
            try:
                # 从 CPU 模板直接复制，避开重复的模型定义逻辑
                local_model = copy.deepcopy(master_model_cpu).to(device)
                
                # 调用我们在外部优化过的 A100 train_client_transformer
                w_new = train_client_transformer_A100(
                    active_model=local_model, 
                    client_dataset=client_datasets[cid], 
                    device=device,
                    global_weights_cpu=client_weights_cpu[cid],
                    is_malicious=False,
                    epochs=epochs,
                    current_round=r,
                    BATCH_SIZE=64 # 🔥 A100 优化：Batch size 调大
                )
                
                norm_sq = sum(torch.norm(w_new[k] - client_weights_cpu[cid][k]).item()**2 for k in w_new if 'weight' in k)
                del local_model
                return cid, w_new, math.sqrt(norm_sq)
            except Exception as e:
                print(f"  [Error] on Client {cid}: {e}")
                raise e

        # 启动线程池
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_cid = {executor.submit(train_single_benign, cid): cid for cid in benign_indices}
            for future in as_completed(future_to_cid):
                cid = future_to_cid[future]
                res_cid, w_new, norm = future.result()
                new_weights_cpu_temp[res_cid] = w_new
                benign_norms_temp[res_cid] = norm
                del future # 极速释放内存
            future_to_cid.clear() 
            del future_to_cid

        for cid in benign_indices:
            new_weights_cpu[cid] = new_weights_cpu_temp[cid]
            benign_norms.append(benign_norms_temp[cid])

        avg_benign_norm = np.median(benign_norms) if benign_norms else 0.5
        print(f"  [Info] Avg Benign Norm: {avg_benign_norm:.4f}")

        # 计算全局良性参考向量 (Reference Vector)
        benign_updates = []
        for cid in benign_indices:
            delta_w_list = [(new_weights_cpu[cid][k].float() - client_weights_cpu[cid][k].float()).flatten() 
                            for k in standard_keys_for_delta if 'num_batches_tracked' not in k]
            if delta_w_list:
                benign_updates.append(torch.cat(delta_w_list))    
                
        avg_benign_update = torch.stack(benign_updates).mean(dim=0) if benign_updates else None

        # ==========================================
        # Phase A-2: Train Malicious
        # ==========================================
        def train_malicious_wrapper(cid):
            w_new = train_client_transformer_A100(
                active_model=gpu_model, 
                client_dataset=client_datasets[cid], 
                device=device,
                global_weights_cpu=client_weights_cpu[cid],
                is_malicious=True,
                strategy_config=strategy_config, 
                intensity=intensity,
                reference_norm=bf * avg_benign_norm, 
                reference_vector=avg_benign_update,
                epochs=2, 
                norm_factor=norm_factor,
                BATCH_SIZE=64 # 🔥 A100 优化
            )
            return cid, w_new

        mal_indices = list(malicious_clients)
        with ThreadPoolExecutor(max_workers=len(mal_indices)) as executor:
            results = list(executor.map(train_malicious_wrapper, mal_indices))

        for cid, w_new in results:
            new_weights_cpu[cid] = w_new

        # ==========================================
        # Phase B: Pre-calculate Sensitivity, Deltas & Vector Map
        # ==========================================
        round_sens = {} 
        for cid in range(NUM_CLIENTS):
            current_state = {k: v.float().to(device) for k, v in new_weights_cpu[cid].items()}
            gpu_model.load_state_dict(current_state)
            
            # 使用 A100 优化的 RS / Sensitivity 函数
            max_s, _ = get_transformer_sensitivity_score(gpu_model, TOKENIZER, device)
            prev_s = sens_history[cid][-1] if sens_history[cid] else 0.0
            sens_history[cid].append(max_s)
            round_sens[cid] = max_s - prev_s
            del current_state

        # 🔥 构建 Vector Map (为距离防御如 Krum/FLAME/CosL2 提前准备)
        vector_map = {}
        if mechanism in ['Krum', 'Cos', 'FLAME', 'CosL2']:
            for cid in range(NUM_CLIENTS):
                vec_list = [(new_weights_cpu[cid][k].float() - client_weights_cpu[cid][k].float()).flatten() 
                            for k in standard_keys_for_delta if 'num_batches_tracked' not in k]
                vector_map[cid] = torch.cat(vec_list) # 暂存 CPU，用时上 GPU

        # ==========================================
        # Phase C: Aggregation & Defense
        # ==========================================
        next_round_weights = []

        for i in range(NUM_CLIENTS):
            my_nbs = neighbors[i]
            candidates = my_nbs + [i]
            
            if i in malicious_clients:
                next_round_weights.append({k: v.clone() for k, v in new_weights_cpu[i].items()})
                continue
         
            # --- 1. MAB Defense ---
            if mechanism == 'MAB' and i in defense_nodes:
                audit_targets = mab_defense.select_for_audit(i, my_nbs) 
                audit_logs = {}
                if audit_targets:
                    audit_logs = mab_defense.update_trust(
                        observer_id=i, probe_list=audit_targets, new_weights=new_weights_cpu,
                        old_weights=client_weights_cpu, device=device, model_template=gpu_model,
                        tokenizer=TOKENIZER, sensitivity_func=get_transformer_sensitivity_score,
                        get_trap_func=get_trap_grad_func, rs_func=get_transformer_rs_score
                    )
                selected_neighbors, agg_w = mab_defense.select_for_aggregation(client_id=i, candidates=audit_targets)

                # MAB Aggregation (在 GPU 上进行)
                avg_state = {}
                for k in new_weights_cpu[i].keys():
                    if 'num_batches_tracked' in k:
                        avg_state[k] = new_weights_cpu[i][k]
                        continue
                        
                    # 🔥 A100 优化：张量推入 GPU 计算
                    local_t = new_weights_cpu[i][k].to(device, non_blocking=True).float() * 0.5
                    nb_sum = torch.zeros_like(local_t)

                    if selected_neighbors:
                        for idx, nid in enumerate(selected_neighbors):
                            weight_factor = agg_w[idx] * 0.5
                            nb_sum += new_weights_cpu[nid][k].to(device, non_blocking=True).float() * weight_factor

                    avg_state[k] = (local_t + nb_sum).cpu().to(new_weights_cpu[i][k].dtype)
                next_round_weights.append(avg_state)

            # --- 2. Krum Defense ---
            elif mechanism == 'Krum':
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
                        # 🔥 A100 优化：利用高带宽在 GPU 上堆叠并求均值
                        stack_w = torch.stack([new_weights_cpu[nid][k].to(device, non_blocking=True).float() for nid in winner_ids])
                        avg_w[k] = stack_w.mean(0).cpu().to(new_weights_cpu[i][k].dtype)
                    next_round_weights.append(avg_w)
                else:
                    next_round_weights.append(copy.deepcopy(new_weights_cpu[i]))

            # --- 3. FLAME Defense (Clustering + Clipping + Noise) ---
            elif mechanism == 'FLAME':
                n_c = len(candidates)
                if n_c < 3:
                    trusted_ids = candidates
                    cluster_labels = [-1] * n_c 
                else:
                    # 🔥 A100 终极优化：矩阵化计算 Cosine 距离，告别双重 for 循环！
                    stacked_vecs = torch.stack([vector_map[nid].to(device, non_blocking=True) for nid in candidates])
                    normed_vecs = torch.nn.functional.normalize(stacked_vecs, p=2, dim=1)
                    
                    # 矩阵乘法得到相似度矩阵
                    sim_matrix = torch.mm(normed_vecs, normed_vecs.t())
                    dist_matrix_gpu = torch.clamp(1.0 - sim_matrix, min=0.0)
                    dist_matrix = dist_matrix_gpu.cpu().numpy().astype(np.float64)
                    
                    try:
                        from sklearn.cluster import HDBSCAN
                        clusterer = HDBSCAN(min_cluster_size=2, min_samples=1, cluster_selection_epsilon=0.5, metric='precomputed')
                        cluster_labels = clusterer.fit_predict(dist_matrix)
                    except ImportError:
                        from sklearn.cluster import AgglomerativeClustering
                        clusterer = AgglomerativeClustering(n_clusters=None, distance_threshold=0.5, metric='precomputed', linkage='average')
                        cluster_labels = clusterer.fit_predict(dist_matrix)

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

                # Dynamic Clipping & Aggregation
                trusted_vecs = [vector_map[nid] for nid in trusted_ids]
                norms = [torch.norm(v).item() for v in trusted_vecs]
                gamma = np.median(norms) if norms else 1.0  
                noise_std = 0.001 * gamma 
                
                avg_w = {}
                for k in initial_state.keys():
                    if 'num_batches_tracked' in k:
                        avg_w[k] = new_weights_cpu[i][k]
                        continue
                        
                    # 🔥 A100 优化：推入 GPU 进行 Clipping 累加
                    clipped_sum = torch.zeros_like(new_weights_cpu[i][k]).to(device, non_blocking=True).float()
                    for nid in trusted_ids:
                        w_new_gpu = new_weights_cpu[nid][k].to(device, non_blocking=True).float()
                        w_old_gpu = client_weights_cpu[nid][k].to(device, non_blocking=True).float()
                        w_update = w_new_gpu - w_old_gpu
                        
                        idx = trusted_ids.index(nid)
                        clip_factor = min(1.0, gamma / (norms[idx] + 1e-9))
                        clipped_sum += w_update * clip_factor
                        
                    avg_update = clipped_sum / len(trusted_ids)
                    noise = torch.randn_like(avg_update) * noise_std
                    
                    base_w_gpu = client_weights_cpu[i][k].to(device, non_blocking=True).float()
                    final_w = base_w_gpu + avg_update + noise
                    avg_w[k] = final_w.cpu().to(new_weights_cpu[i][k].dtype)
                    
                next_round_weights.append(avg_w)

            # --- 4. Trimmed Mean ---
            elif mechanism == 'TrimmedMean':
                n = len(candidates)
                cut = int(n * 0.2)
                if n - 2 * cut <= 0: cut = 0

                avg_w = {}
                for k in initial_state.keys():
                    if 'num_batches_tracked' in k:
                        avg_w[k] = new_weights_cpu[i][k]
                        continue

                    # 🔥 A100 优化：在 GPU 上直接完成排序和截断
                    stack_tensors = torch.stack([new_weights_cpu[nid][k].to(device, non_blocking=True).float() for nid in candidates])
                    if cut > 0:
                        sorted_stack, _ = torch.sort(stack_tensors, dim=0)
                        trimmed_stack = sorted_stack[cut : n - cut]
                        avg_w[k] = trimmed_stack.mean(dim=0).cpu().to(new_weights_cpu[i][k].dtype)
                    else:
                        avg_w[k] = stack_tensors.mean(dim=0).cpu().to(new_weights_cpu[i][k].dtype)
                next_round_weights.append(avg_w)

            # --- 5. CosL2 Defense ---
            elif mechanism == 'CosL2':
                my_v = vector_map[i].to(device, non_blocking=True)
                if not my_nbs:
                    next_round_weights.append(copy.deepcopy(new_weights_cpu[i]))
                    continue
                    
                MALICIOUS_RATIO = len(malicious_clients) / NUM_CLIENTS
                m_winners = max(1, int(len(my_nbs) * (1 - MALICIOUS_RATIO)))
                
                sim_scores = []
                for nb in my_nbs:
                    nb_v = vector_map[nb].to(device, non_blocking=True)
                    sim = torch.nn.functional.cosine_similarity(my_v.unsqueeze(0), nb_v.unsqueeze(0), dim=1).item()
                    sim_scores.append((sim, nb))
                
                sim_scores.sort(key=lambda x: x[0], reverse=True)
                selected_nids = [nb for sim, nb in sim_scores[:m_winners]]
                final_group = [i] + selected_nids

                norms = [torch.norm(vector_map[nid]).item() for nid in final_group]
                gamma = np.median(norms)

                avg_w = {}
                for k in initial_state.keys():
                    if 'num_batches_tracked' in k:
                        avg_w[k] = new_weights_cpu[i][k]
                        continue
                    
                    # 🔥 A100 优化
                    tmp_sum = torch.zeros_like(new_weights_cpu[i][k]).to(device, non_blocking=True).float()
                    for nid in final_group:
                        w_new_gpu = new_weights_cpu[nid][k].to(device, non_blocking=True).float()
                        w_old_gpu = client_weights_cpu[nid][k].to(device, non_blocking=True).float()
                        update_v = w_new_gpu - w_old_gpu
                        
                        current_node_norm = torch.norm(vector_map[nid]).item()
                        clip_factor = min(1.0, gamma / (current_node_norm + 1e-9))
                        tmp_sum += update_v * clip_factor
                        
                    avg_update = tmp_sum / len(final_group)
                    w_base = client_weights_cpu[i][k].to(device, non_blocking=True).float()
                    avg_w[k] = (w_base + avg_update).cpu().to(new_weights_cpu[i][k].dtype)
                    
                next_round_weights.append(avg_w)

            # --- 6. FedAvg (MAB non-defense nodes or Fallback) ---
            else:
                avg_w = {}
                for k in initial_state.keys():
                    if 'num_batches_tracked' in k:
                        avg_w[k] = new_weights_cpu[i][k].clone()
                        continue
                    
                    # 🔥 A100 优化：标准均值聚合全在 GPU 进行
                    stack_w = torch.stack([new_weights_cpu[nid][k].to(device, non_blocking=True).float() for nid in candidates])
                    avg_w[k] = stack_w.mean(dim=0).cpu().to(new_weights_cpu[i][k].dtype)
                
                next_round_weights.append(avg_w)

        # ==========================================
        # 🔄 轮次收尾：清除垃圾，避免内存爆炸
        # ==========================================
        if 'new_weights_cpu' in locals():
            while len(new_weights_cpu) > 0:
                item = new_weights_cpu.pop()
                del item
            del new_weights_cpu

        if 'vector_map' in locals():
            vector_map.clear()
            del vector_map

        old_weights_ref = client_weights_cpu
        client_weights_cpu = next_round_weights
        del old_weights_ref 

        # 🔥 A100 优化：去掉了导致抖动和卡顿的 empty_cache()
        gc.collect() 

    print(f"Experiment {mechanism} Finished.")

    # =================================================
    # Phase E: Evaluation
    # =================================================
    print(f"\n📊 --- Final Evaluation (Round {GLOBAL_ROUNDS}) ---")
    print(f" {'Client':<8} | {'Type':<6} | {'ACC':<10} | {'ASR':<10} | {'Theo Intensity'}")
    print("-" * 65)

    theo_intensities = calculate_theoretical_intensity(neighbors, malicious_clients, NUM_CLIENTS, bf)
    benign_acc_list, benign_asr_list = [], []
    acc_list, asr_list = [], []
    
    for cid in range(NUM_CLIENTS):
        if cid in malicious_clients: node_type = "MAL"
        elif cid in defense_nodes: node_type = "DEF"
        else: node_type = "BEN"
            
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

    # 脚本终点才清理一次底层显存
    del gpu_model, client_weights_cpu, next_round_weights
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize() 
        
    return avg_benign_acc, avg_benign_asr, acc_list, asr_list