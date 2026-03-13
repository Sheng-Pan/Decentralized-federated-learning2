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
from trainer import train_client_transformer
#from defense import get_universal_stats
from defense import  calculate_krum_scores,get_trap_grad_func,get_transformer_sensitivity_score
from theoretical_intensity import calculate_theoretical_intensity
from eval_DFL import evaluate_global_transformer
# ==========================================
# 1. Data Processing & Distribution
# ==========================================
def get_transformer_rs_score(model, tokenizer, device, num_samples=8, noise_std=0.12):
    """
    计算 Transformer 的 Randomized Smoothing 分数。
    对 Embedding 层添加噪声，衡量输出概率分布的 KL 散度稳定性。
    """
    model.eval()
    
    # 1. 构造固定的 Probe Input
    text = "The quick brown fox jumps over the lazy dog."
    inputs = tokenizer(text, return_tensors="pt", padding='max_length', max_length=64, truncation=True).to(device)
    
    kl_list = []
    
    with torch.no_grad():
        # 2. 获取 Clean Output & Embeddings
        # 自动探测 Base Model (适配 DistilBERT, BERT 等)
        if hasattr(model, 'distilbert'):
            base_model = model.distilbert
        elif hasattr(model, 'bert'):
            base_model = model.bert
        else:
            base_model = getattr(model, model.base_model_prefix, model)
            
        clean_embeds = base_model.embeddings(inputs.input_ids)
        
        clean_out = model(**inputs)
        clean_probs = torch.softmax(clean_out.logits, dim=-1)
        
        # 3. 采样噪声并计算 KL
        for _ in range(num_samples):
            noise = torch.randn_like(clean_embeds) * noise_std
            noised_embeds = clean_embeds + noise
            
            # 传入 inputs_embeds 进行前向传播
            noisy_out = model(inputs_embeds=noised_embeds)
            noisy_probs = torch.softmax(noisy_out.logits, dim=-1)
            
            # KL Divergence: P(clean) || Q(noisy)
            kl = torch.sum(clean_probs * (torch.log(clean_probs + 1e-10) - torch.log(noisy_probs + 1e-10))).item()
            kl_list.append(max(0, kl))

    # 4. 计算最终分数 (映射到 0~1)
    avg_kl = np.mean(kl_list)
    rs_score = np.exp(-avg_kl * 5.0) # 系数 5.0 可调
    
    return float(rs_score)

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
BATCH_SIZE = 64
def run_simulation_transformer(seed, NUM_CLIENTS, defense_nodes, malicious_clients, G, neighbors, client_datasets, test_data,
                           atk_type='neurotoxin', mechanism='FedAvg', bf=1.0, intensity=0.1,goodnorm=2,
                           debug_mode=False, GLOBAL_ROUNDS=15, norm_factor=40, debug=True, epochs=1, MAX_WORKERS=8, log_file_path=None):
    
    # --- 接收子进程日志 ---
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
    # 🔥 核心防御机制：提取真实的 PubMed 文本作为 NLP 激活探针
    # =========================================================================
    print("📸 [System] Extracting real PubMed validation text for S_Z Probe...")
    try:
        # 兼容 test_data 是 DataLoader 或是 HF Dataset 的情况
        if isinstance(test_data, DataLoader):
            real_probe_batch = next(iter(test_data))
        else:
            real_probe_batch = test_data[:16] # 截取前 16 条

        # 提取 input_ids 张量 (HuggingFace 模型的 forward 第一个参数默认接收 input_ids)
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
        # 降级：如果提取失败，生成随机的 token IDs (假设 vocab_size 默认为 30522)
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

        # for cid in benign_indices:
        #     # Train
        #     w_new = train_client_transformer(
        #         gpu_model, client_datasets[cid], device,
        #         client_weights_cpu[cid], # Start from own weights
        #         is_malicious=False,
        #         epochs=epochs,current_round=r
        #     )
        #     new_weights_cpu[cid] = w_new

        #     # Calc Norm
        #     norm = 0.0
        #     for k in w_new:
        #         if 'weight' in k:
        #             norm += torch.norm(w_new[k] - client_weights_cpu[cid][k]).item()**2
        #     benign_norms.append(math.sqrt(norm))
      

        # 设定最大并发数，取决于你的显存。如果 OOM (Out of Memory)，就把这个数字调小。

        MAX_WORKERS =MAX_WORKERS  # 🚀 从 4 提升到 8
        print(f"  [Info] Starting parallel training for {len(benign_indices)} benign clients with {MAX_WORKERS} workers...")

        # 预先分配一个列表来存储结果，保持原来的顺序
        new_weights_cpu_temp = {cid: None for cid in benign_indices}
        benign_norms_temp = {cid: 0.0 for cid in benign_indices}

        # 1. 调高并发数

        # 2. 预先加载一个模板模型（在 CPU 上）
        master_model_cpu = get_model(NUM_LABELS=NUM_LABELS)

        def train_single_benign(cid):
            try:
                # 🚀 优化：从 CPU 模板直接复制，避开重复的模型定义逻辑
                local_model = copy.deepcopy(master_model_cpu).to(device)
                
                # 运行训练
                w_new = train_client_transformer(
                    local_model, 
                    client_datasets[cid], 
                    device,
                    client_weights_cpu[cid],
                    is_malicious=False,
                    epochs=epochs,
                    current_round=r,
                    BATCH_SIZE=32 # 🚀 适当调大 BATCH，提高 GPU 核心吞吐量
                )
                
                # 计算 Norm (由于 w_new 是 CPU 字典，这里直接 CPU 计算即可)
                norm_sq = sum(torch.norm(w_new[k] - client_weights_cpu[cid][k]).item()**2 
                            for k in w_new if 'weight' in k)
                
                del local_model
                # torch.cuda.empty_cache() # 只有在显存极其紧张时才调用，否则会拖慢速度
                return cid, w_new, math.sqrt(norm_sq)
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"  [Error] OOM on Client {cid}. Decrease MAX_WORKERS or BATCH_SIZE.")
                    torch.cuda.empty_cache()
                raise e

        # 启动线程池
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
                    # 极其关键：取完结果立刻删除 future 内部的 260MB 返回值引用
                    del future 

            # 清空 future 字典，释放 5GB 内存
            future_to_cid.clear() 
            del future_to_cid
        # 整理结果合并回原有的数据结构
        for cid in benign_indices:
            new_weights_cpu[cid] = new_weights_cpu_temp[cid]
            benign_norms.append(benign_norms_temp[cid])

        avg_benign_norm = np.median(benign_norms) if benign_norms else 0.5
        print(f"  [Info] Avg Benign Norm: {avg_benign_norm:.4f}")
        avg_benign_norm = np.median(benign_norms) if benign_norms else 0.5
        print(f"  [Info] Avg Benign Norm: {avg_benign_norm:.4f}")
        old_w = client_weights_cpu[cid]
        # ==========================================
        # 🔥 新增：计算全局良性参考向量 (Reference Vector)
        # ==========================================
        # benign_updates = []
        # # 使用初始权重的 key 顺序作为标准，避免顺序错乱
        # standard_keys = list(initial_state.keys()) 

        # for cid in benign_indices:
        #     delta_w_list = []
        #     for k in standard_keys: 
        #         if 'num_batches_tracked' in k: continue
        #         if 'weight' in k or 'bias' in k:
        #             w_new = new_weights_cpu[cid][k].float()
        #             w_old = client_weights_cpu[cid][k].float()
        #             delta_w_list.append((w_new - w_old).flatten())
        #     benign_updates.append(torch.cat(delta_w_list))
        # ==========================================
        # 🔥 修复：生成参考向量 (必须使用 sorted 确保顺序！)
        # ==========================================
        benign_updates = []
        # ❌ 原代码: standard_keys = list(initial_state.keys()) 
        # ✅ 新代码: 必须排序！
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
        # 求平均得到“黄金航向”
       # 求平均得到“黄金航向” (保持在 CPU 上，节省显存开销)
        if benign_updates:
            avg_benign_update = torch.stack(benign_updates).mean(dim=0)
        else:
            avg_benign_update = None
        # ==========================================

        # --- Phase A-2: Train Malicious ---
        # mal_indices = list(malicious_clients)
        # for cid in mal_indices:
        #     w_new = train_client_transformer(
        #         gpu_model, client_datasets[cid], device,
        #         client_weights_cpu[cid],
        #         is_malicious=True,
        #         strategy_config=strategy_config, intensity=intensity,
        #         reference_norm=bf*avg_benign_norm, 
        #         reference_vector=avg_benign_update, # <--- 🌟 新增：把良性方向传进去
        #         epochs=2, scale_factor=scale_factor
        #     )
        #     new_weights_cpu[cid] = w_new

        def train_malicious_wrapper(cid):
            """
            包装函数：负责单个恶意节点的训练
            """
            # 注意：如果涉及到 GPU，请看下方的“GPU 并行注意事项”
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

        # --- 并行执行部分 ---
        mal_indices = list(malicious_clients)
        # 使用 ThreadPoolExecutor (如果是 IO 密集或 GPU 任务) 
        # 或 ProcessPoolExecutor (如果是纯 CPU 计算)
        with ThreadPoolExecutor(max_workers=len(mal_indices)) as executor:
            results = list(executor.map(train_malicious_wrapper, mal_indices))

        # 将结果填回字典
        for cid, w_new in results:
            new_weights_cpu[cid] = w_new
 # =================================================
        # Phase B: Structural Analysis (All Clients, All Rounds)
        # =================================================
        
        if debug == True:
        # 1. 确定要分析的层
        # 如果前面没有定义 head_layer_name，初始化为 None
            if 'head_layer_name' not in locals(): head_layer_name = None

            target_layer = head_layer_name
            if not target_layer:
                target_layer = 'classifier.weight' # Transformer 默认
                if new_weights_cpu[0] and target_layer not in new_weights_cpu[0]:
                    target_layer = 'fc2.weight' # CNN 默认

            # 2. 遍历所有节点
            if new_weights_cpu[0] and target_layer in new_weights_cpu[0]:

                print(f"\n🔍 --- Structural Analysis (Round {r+1}) ---")

                for observer_id in range(NUM_CLIENTS):
                    my_neighbors = neighbors[observer_id]
                    if not my_neighbors: continue

                    print(f"  Observer {observer_id} (Layer: {target_layer}) checking neighbors: {my_neighbors}")

                    # 数据容器
                    neighbor_deltas_flat = {}
                    neighbor_deltas = {}  # 🔧 [修复 1]: 变量名统一

                    # A. 收集所有邻居的 Delta
                    for nid in my_neighbors:
                        w_new = new_weights_cpu[nid][target_layer]
                        w_old = client_weights_cpu[nid][target_layer]

                        if w_new.device.type != 'cpu': w_new = w_new.cpu()
                        if w_old.device.type != 'cpu': w_old = w_old.cpu()

                        delta = (w_new - w_old).float()

                        # 🔧 [修复 2]: 使用 neighbor_deltas
                        neighbor_deltas[nid] = delta
                        neighbor_deltas_flat[nid] = delta.flatten().numpy()

                    # B. 计算共识 (Mean Vector)
                    if not neighbor_deltas_flat: continue

                    all_vecs = list(neighbor_deltas_flat.values())
                    mean_vector = np.mean(np.stack(all_vecs), axis=0)

                    for nid in my_neighbors:
                        is_mal = nid in malicious_clients
                        tag = "MAL" if is_mal else "BEN"

                        # --- 1. EMD ---
                        emd = wasserstein_distance(neighbor_deltas_flat[nid], mean_vector)

                        # --- 2. SVD ---
                        # 🔧 [修复 4]: 使用 neighbor_deltas
                        delta_mat = neighbor_deltas[nid]
                        svd_max = 0.0
                        try:
                            if delta_mat.ndim == 2:
                                _, S, _ = torch.linalg.svd(delta_mat, full_matrices=False)
                                svd_max = S[0].item()
                            elif delta_mat.ndim == 4:
                                mat_view = delta_mat.view(delta_mat.shape[0], -1)
                                _, S, _ = torch.linalg.svd(mat_view, full_matrices=False)
                                svd_max = S[0].item()
                        except:
                            svd_max = -1.0

                        # # --- 3. Universal Stats (T_Z, S_Z) ---
                        # # 🔥 [新增]: 调用 get_universal_stats 计算单层的 T_Z 和 S_Z
                        # # 构造临时字典 {layer_name: tensor} 传给统计函数
                        # univ_stats, _ = get_universal_stats({target_layer: delta_mat})
                        # t_z_val = univ_stats['t_z']
                        # s_z_val = univ_stats['s_z']
# --- 3. 计算 S_Z (严格按照 calc_element_wise_stats 逻辑) ---
                        flat_delta = delta_mat.flatten().cpu().numpy()
                        pos_vals = flat_delta[flat_delta > 0]
                        
                        if len(pos_vals) > 0:
                            median_p = np.median(pos_vals)
                            mad_p = np.median(np.abs(pos_vals - median_p)) + 1e-9
                            z_scores = 0.6745 * (pos_vals - median_p) / mad_p
                            # 按照你的要求，这里严格使用 np.max 取最大值
                            s_z_val = float(np.max(z_scores))
                        else:
                            s_z_val = 0.0



                     # ==========================================
                        # 🔥 [新增] Randomized Smoothing (RS) Analysis
                        # ==========================================
                        rs_score = 0.0
                        attn_stability = 0.0
                        supp_score = 0.0  # <--- [新增变量初始化]

                        if debug == True:
                            try:
                                # 1. 临时加载权重并确保 eval 模式
                                temp_state = {k: v.float().to(DEVICE) for k, v in new_weights_cpu[nid].items()}
                                gpu_model.load_state_dict(temp_state)
                                gpu_model.eval()

                                # ==========================================
                                # 🔍 [新增] 计算 Suppression Score (抑制分数)
                                # ==========================================
                                # target_class_idx=None 表示自动寻找受抑制最严重的类
                                # last_weights 传入 client_weights_cpu[nid] 以计算动态更新趋势
                                supp_result = get_suppression_score(
                                    gpu_model, 
                                    target_class_idx=None, 
                                    last_weights=client_weights_cpu[nid]
                                )
                                supp_score = supp_result['score']
                                # 如果你想看具体是哪个类被抑制，可以取消下面这行的注释
                                # print(f"    (Suppressed Class: {supp_result['target_idx']} | NormZ: {supp_result['norm_z']:.2f})")
                                # ==========================================

                                rs_text = "The quick brown fox jumps over the lazy dog."
                                rs_inputs = TOKENIZER(rs_text, return_tensors="pt", padding='max_length', max_length=64, truncation=True).to(DEVICE)

                                with torch.no_grad():
                                    # ... [原有 RS 代码保持不变] ...
                                    # 获取干净状态下的输出
                                    clean_out = gpu_model(**rs_inputs, output_attentions=True)
                                    clean_probs = torch.softmax(clean_out.logits, dim=-1)
                                    
                                    # --- 健壮性检查 1: 检查注意力是否存在 ---
                                    if clean_out.attentions is not None:
                                        clean_attn = clean_out.attentions[0] # 取第一层
                                    else:
                                        clean_attn = None

                                    n_rs_samples = 8 
                                    rs_sigma = 0.12 
                                    kl_list = []
                                    attn_drift_list = []

                                    base_model = getattr(gpu_model, 'distilbert', getattr(gpu_model, 'bert', getattr(gpu_model, 'base_model', None)))
                                    clean_embeds = base_model.embeddings(rs_inputs.input_ids)

                                    for _ in range(n_rs_samples):
                                        noise = torch.randn_like(clean_embeds) * rs_sigma
                                        noised_embeds = clean_embeds + noise
                                        
                                        noisy_out = gpu_model(inputs_embeds=noised_embeds, output_attentions=True)
                                        noisy_probs = torch.softmax(noisy_out.logits, dim=-1)
                                        
                                        # --- 指标 A: KL 散度 ---
                                        kl = torch.sum(clean_probs * (torch.log(clean_probs + 1e-10) - torch.log(noisy_probs + 1e-10))).item()
                                        kl_list.append(max(0, kl))
                                        
                                        # --- 指标 B: 注意力漂移 ---
                                        if clean_attn is not None and noisy_out.attentions is not None:
                                            drift = torch.mean((clean_attn - noisy_out.attentions[0])**2).item()
                                            attn_drift_list.append(drift)

                                    # --- 计算最终得分 ---
                                    avg_kl = np.mean(kl_list)
                                    rs_score = np.exp(-avg_kl * 5) 

                                    if attn_drift_list:
                                        attn_stability = np.std(attn_drift_list) * 1000
                                    else:
                                        attn_stability = 0.0

                                del temp_state
                                torch.cuda.empty_cache()

                            except Exception as e:
                                # 即使报错也不要中断主进程
                                print(f"Debug Error: {e}") # 建议打印错误以便调试
                                rs_score, attn_stability, supp_score = -0.0, -0.0, -0.0

                            # 修改后的打印输出：加入 Supp 指标
                            # Supp > 2.0 通常意味着异常
                            print(f"    -> N{nid:<2} [{tag}] | EMD: {emd:.5f} | S_Z: {s_z_val:.4f} | RS: {rs_score:.4f} | Supp: {supp_score:.4f}")
        print("")

       
        # Precompute Vector Map for Distance-based Defenses (Krum)
          # --- Phase B: Aggregation & Defense ---
        next_round_weights = []

       # ==========================================
        # 🔥 修复 1: 构建 Vector Map (使用 FP16 降维打击 OOM)
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

                    # 🚀 核心修改：转化为 half() (FP16) 节省 50% RAM
                    vec_list.append(delta.flatten().half())

                vector_map[cid] = torch.cat(vec_list)
        round_trust_logs = {}


        for i in range(NUM_CLIENTS):
            audit_logs = {}
            my_nbs = neighbors[i]
            candidates = my_nbs + [i]
            if i in malicious_clients:
                            # 替换掉所有 copy.deepcopy(new_weights_cpu[i]) 为下面这行：
                            cloned_state = {k: v.clone() for k, v in new_weights_cpu[i].items()}
                            next_round_weights.append(cloned_state)
                            continue
         
            # --- Logic: Defense Node ---
            if i in defense_nodes and mechanism == 'MAB':
                    # --- Step 1: Select Audit Targets (Phase 1) ---
                    # 假设每轮只审计 5 个最不确定的节点
                    audit_targets = mab_defense.select_for_audit(i, my_nbs) # 移除错误的 budget 计算，使用默认或类内部逻辑

                    # --- Step 2: Update Trust (Phase 2) ---
                    # 🔥 [修复 NameError] 必须先初始化为空字典
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
                    # 🔥 关键修改：传入 audit_targets 作为候选池
                    selected_neighbors, agg_w = mab_defense.select_for_aggregation(
                        client_id=i,
                        candidates=audit_targets
                    )

                    # ==========================================
                    # 🔥 新增：打印信任值、聚合报告与 Z-Score
                    # ==========================================
                    # 仅打印防御节点，或每隔几轮打印一次
                    # ... (Inside the MAB Defense Node Loop) ...
# ==========================================
                    # 🔥 Updated: Print Trust & Aggregation Report (Continuous Trust + HDBSCAN Trap)
                    # ==========================================
                  
                    if i in defense_nodes:
                        # ==========================================
                        # 🔍 新增: 提取当前 Observer 邻居的 Trap 分数并执行对数聚类
                        # ==========================================
                        cluster_labels_dict = {}
                        if audit_logs: # 只有在有审计日志时才聚类
                            # 取出被审计节点的 nid 和 trap score
                            c_nids = [nid for nid in my_nbs if nid in audit_logs]
                            raw_traps = [audit_logs[nid].get('trap', 0.0) for nid in c_nids]
                            
                            if len(raw_traps) >= 3: # 聚类至少需要3个点
                                # 对数化预处理，解决极小值和极大值跨度大的问题
                                log_traps = np.log1p(np.array(raw_traps)).reshape(-1, 1)
                                
                                try:
                                    from sklearn.cluster import HDBSCAN
                                    clusterer = HDBSCAN(min_cluster_size=2, min_samples=1)
                                    labels = clusterer.fit_predict(log_traps)
                                except ImportError:
                                    # Fallback
                                    from sklearn.cluster import AgglomerativeClustering
                                    clusterer = AgglomerativeClustering(n_clusters=None, distance_threshold=1.5)
                                    labels = clusterer.fit_predict(log_traps)
                                
                                # 找出良性主簇 (数量最多的簇)
                                unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
                                if len(unique_labels) > 0:
                                    main_cluster = unique_labels[np.argmax(counts)]
                                else:
                                    main_cluster = -99 # 全是噪声

                                # 记录每个节点的聚类结果
                                for idx, nid in enumerate(c_nids):
                                    lbl = labels[idx]
                                    # 如果不在主簇中，或者是噪声(-1)，则判定为 Outlier
                                    is_outlier = (lbl != main_cluster)
                                    cluster_labels_dict[nid] = "🚨 TrapOutlier" if is_outlier else "🟢 InCluster"

                        # 1. 扩宽表头并加入 Trap (Hessian Trace) 和 Cluster
                        print(f" {'Neighbor':<8} | {'Role':<6} | {'Trust(Q)':<8} | {'Action':<10} | {'Metrics (Trap / S_Z / RS / Reward / Z_Comb)'}")
                        print("-" * 145) # 再次加长分割线
                        for nid in my_nbs:
                            role = "MAL" if nid in malicious_clients else "BEN"
                            role_icon = "😈" if role == "MAL" else "😇"
                            curr_trust = mab_defense.trust_scores[i].get(nid, 0.5)

                            if nid in selected_neighbors: action = "✅ AGG"
                            elif nid in audit_targets: action = "❌ DROP"
                            else: action = "⚪ SKIP"

                            metric_str = ""

                            if nid in audit_logs:
                                l = audit_logs[nid]

                                # 🔥 [修改] 提取指标：移除 RS/SZ，加入 Supp
                                t_raw = l.get('trap', 0.0)       
                                S_Z = l.get('elem_z', 0.0)   
                                RS = l.get('rs_score', 0.0) # <--- 获取抑制分数
                                
                                r_val = l.get('reward', 0.0)
                                z_c = l.get('z_comb_penalty', l.get('z_comb', 0.0)) 
                                
                                trap_str = f"{t_raw:.4f}" if t_raw < 10000 else f"{t_raw:.2e}"

                                # 🔥 [修改] 打印字符串格式化
                                # Supp > 2.0 异常，Sens/Trap 越大越异常
                                metric_str = f"T:{trap_str:<9} | S_Z:{S_Z:.3f} | RS:{RS:.3f} | R:{r_val:.2f} | Zc:{z_c:.1f}"

                                status = str(l.get('gap_cut', ''))
                                if "REJECT" in status:
                                    metric_str += " 🚫"
                                elif "WARN" in status:
                                    metric_str += " ⚠️"
                            else:
                                metric_str = "..."

                            print(f" {nid:<8} | {role_icon} {role:<3} | {curr_trust:<8.4f} | {action:<10} | {metric_str:<60}")
                        print("-" * 135)
                        # --- Iterate & Print ---
                        

                        # # MAB 加权聚合 (包含自身)
                        # avg_state = {}
                        # for k in new_weights_cpu[i].keys():
                        #     # 自身的权重占 0.5 (这是一个超参数，表示对自己历史信息的信任)
                        #     local_t = new_weights_cpu[i][k].float() * 0.5
                        #     nb_sum = torch.zeros_like(local_t)

                        #     if selected_neighbors:
                        #         # 归一化 MAB 权重 (agg_w 可能是 softmax 后的)
                        #         # 注意：agg_w 对应的是 selected_neighbors 的顺序
                        #         for idx, nid in enumerate(selected_neighbors):
                        #             # 邻居部分总和占 0.5
                        #             weight_factor = agg_w[idx] * 0.5
                        #             nb_sum += new_weights_cpu[nid][k].float() * weight_factor

                        #         avg_state[k] = (local_t + nb_sum).to(new_weights_cpu[i][k].dtype)
                        #     else:
                        #         # 如果没有选中任何邻居，全靠自己
                        #         avg_state[k] = new_weights_cpu[i][k]

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

                        for k in new_weights_cpu[i].keys():
                            # 跳过非浮点参数（如 BatchNorm 的 tracked batches）
                            if 'num_batches_tracked' in k:
                                avg_state[k] = new_weights_cpu[i][k].clone()
                                continue
                                
                            # 初始化全 0 张量
                            tmp_sum = torch.zeros_like(new_weights_cpu[i][k].float())

                            # 累加所有参与节点的权重
                            for nid in participating_nodes:
                                tmp_sum += new_weights_cpu[nid][k].float()

                            # 求平均并转换回原数据类型
                            avg_state[k] = (tmp_sum / num_participants).to(new_weights_cpu[i][k].dtype)

                        next_round_weights.append(avg_state)
            # ==========================================
            # 🔥 修复 2 & 3: Krum 逻辑 (变量名与循环索引)
            # ==========================================
            elif mechanism == 'Krum':
                candidates = my_nbs + [i]
                MALICIOUS_RATIO = len(malicious_clients)/NUM_CLIENTS
                f_limit = int(len(candidates) * MALICIOUS_RATIO)
                m_winners = max(1, len(candidates) - f_limit)

                # 1. 获取向量
                candidate_vecs = [vector_map[nid] for nid in candidates]

                # 2. 计算分数 (为了打印日志)
                all_k_scores = calculate_krum_scores(candidate_vecs, f_limit)
                node_score_map = {nid: score for nid, score in zip(candidates, all_k_scores)}

                # 3. 选择赢家 (这里为了效率，可以直接利用上面算好的分数排序，不用再调用 local_multikrum_select)
                # 优化: 直接基于 node_score_map 排序取前 m 个，避免重复计算距离矩阵
                sorted_cands_by_score = sorted(candidates, key=lambda x: node_score_map[x])
                winner_ids = sorted_cands_by_score[:m_winners]

                if winner_ids:
                    # 打印日志 (修复 round_idx -> r)
                    # 仅在防御节点打印，避免刷屏
                    print(f"\n📢 [Round {r+1}] Obs {i} Multi-Krum Selection:")
                    print(f"    {'Status':<8} | {'Node':<5} | {'Role':<5} | {'Krum Score':<12}")
                    print(f"    {'-' * 40}")

                    for nid in sorted_cands_by_score:
                        role = "😈" if nid in malicious_clients else "😇"
                        is_winner = "WINNER ✅" if nid in winner_ids else "REJECT ❌"
                        score = node_score_map[nid]
                        alert = " 🔥" if (nid in malicious_clients) and (nid in winner_ids) else ""
                        print(f"    {is_winner:<8} | {nid:<5} | {role:<5} | {score:<12.4f}{alert}")

                    # 4. 平均聚合 (修复 new_weights_cpu -> new_weights_cpu)
                    avg_w = {}
                    # 使用 initial_state 里的 key 确保完整性
                    for k in initial_state.keys():
                        # 跳过非 float 参数
                        if 'num_batches_tracked' in k:
                            avg_w[k] = new_weights_cpu[i][k]
                            continue
# ❌ 导致瞬间 OOM 的旧代码：
                    # stack_w = torch.stack([new_weights_cpu[nid][k].float() for nid in winner_ids])
                    # avg_w[k] = stack_w.mean(0).to(new_weights_cpu[i][k].dtype)

                    # ✅ 修复后：O(1) 内存复杂度的迭代累加
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
            # 🔥 [新增] Strategy: CosL2 (Cosine Filtering + L2 Clipping)
            # =========================================================
            elif mechanism == 'CosL2':
                    my_v = vector_map[i]
                    my_nbs = neighbors[i]
                    
                    # 安全性检查：如果没有邻居，直接信任自己
                    if not my_nbs:
                        next_round_weights.append(copy.deepcopy(new_weights_cpu[i]))
                        continue
                    
                    # --- Step 1: 方向过滤 (CosSim) ---
                    # 假设网络中最坏情况有一定比例的恶意节点 (实际中可设为 0.3)
                    MALICIOUS_RATIO = len(malicious_clients) / NUM_CLIENTS
                    m_winners = max(1, int(len(my_nbs) * (1 - MALICIOUS_RATIO)))
                    
                    sim_scores = []
                    for nb in my_nbs:
                        # 计算邻居与自己的余弦相似度
                        sim = torch.nn.functional.cosine_similarity(my_v.unsqueeze(0), vector_map[nb].unsqueeze(0), dim=1).item()
                        sim_scores.append((sim, nb))
                    
                    # 按相似度降序排列 (1.0 最接近，-1.0 最对立)
                    sim_scores.sort(key=lambda x: x[0], reverse=True)
                    
                    # 选取排名前 m_winners 的邻居
                    selected_nids = [nb for sim, nb in sim_scores[:m_winners]]
                    final_group = [i] + selected_nids # 永远包含自己

                    # =========================================================
                    # 📊 [新增] 打印决策日志 (仅对前两个防御节点打印，每 5 轮一次)
                    # =========================================================
                    print(f"\n🛡️ [Round {r+1}] Obs {i} CosL2 Audit (Select Top-{m_winners}/{len(my_nbs)}):")
                    print(f"    {'Neighbor':<10} | {'Role':<6} | {'Cos Sim':<12} | {'Decision'}")
                    print(f"    {'-' * 50}")
                        
                    for sim, nb in sim_scores:
                        is_mal = nb in malicious_clients
                        tag = "MAL 😈" if is_mal else "BEN 😇"
                        
                        is_selected = nb in selected_nids
                        decision = "✅ AGG" if is_selected else "❌ DROP"
                        
                        # 如果恶意节点被选中了，打个警告标志
                        alert = " 🔥 ALERT!" if is_mal and is_selected else ""
                        
                        print(f"    N{nb:<9} | {tag:<6} | {sim:<12.4f} | {decision} {alert}")
                    print(f"    {'-' * 50}")

                    # --- Step 2: 尺度裁剪 (L2 Clipping) ---
                    # 获取入选组的范数 (Calculate ONCE)
                    norms = [torch.norm(vector_map[nid]).item() for nid in final_group]
                    gamma = np.median(norms) # 动态阈值：范数中位数
                    
                    # Create an O(1) lookup dictionary for norms so we don't recalculate
                    norm_dict = {nid: norm for nid, norm in zip(final_group, norms)}

                    # --- Step 3: 聚合更新 (Memory-Optimized) ---
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
                            
                            # 写回模型权重
                            w_base = client_weights_cpu[i][k].float().to(avg_update.device)
                            avg_w[k] = (w_base + avg_update).to(new_weights_cpu[i][k].dtype)
                        else:
                            avg_w[k] = new_weights_cpu[i][k]
                    
                    next_round_weights.append(avg_w)
            # =========================================================
            # 🔥 Strategy: FLAME Defense (Clustering + Clipping + Noise)
            # =========================================================
            elif mechanism == 'FLAME':
                candidates = my_nbs + [i]
                n_c = len(candidates)
                
                if n_c < 3:
                    trusted_ids = candidates
                    cluster_labels = [-1] * n_c 
                else:
                    # 🚀 [核心修复] 移除 sklearn.metrics.pairwise.cosine_distances 
                    # 改用 PyTorch 零拷贝计算距离矩阵，彻底解决 50GB 内存爆炸问题！
                    dist_matrix = np.zeros((n_c, n_c), dtype=np.float32)
                    
                    for r_idx in range(n_c):
                        for c_idx in range(r_idx + 1, n_c):
                            v1 = vector_map[candidates[r_idx]]
                            v2 = vector_map[candidates[c_idx]]
                            
                            # 余弦距离 = 1 - 余弦相似度
                            sim = torch.nn.functional.cosine_similarity(v1, v2, dim=0, eps=1e-8).item()
                            dist = max(0.0, 1.0 - sim) # 防止浮点误差出现极小负数
                            
                            dist_matrix[r_idx, c_idx] = dist
                            dist_matrix[c_idx, r_idx] = dist
                            del v1, v2
                    torch.cuda.empty_cache()
                    # 3. 执行 HDBSCAN 聚类 (传入算好的 dist_matrix)
                    try:
                        from sklearn.cluster import HDBSCAN
                        clusterer = HDBSCAN(
                            min_cluster_size=2,               
                            min_samples=1,                    
                            cluster_selection_epsilon=0.5,    
                            metric='precomputed'              # 必须保留 precomputed
                        )
                        cluster_labels = clusterer.fit_predict(dist_matrix)
            
                    except ImportError:
                        from sklearn.cluster import AgglomerativeClustering
                        # 🔥 修改点 2: 改变 Agglomerative 的逻辑，从“固定分2类”改为“按距离阈值聚类”
                        clusterer = AgglomerativeClustering(
                            n_clusters=None,                  # 不强制指定分几类
                            distance_threshold=0.5,           # 核心：只要两个节点的余弦距离小于 0.5，就允许它们并入同一个簇
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

                # =========================================================
                # 📢 新增：打印 FLAME 的聚类与选择日志
                # =========================================================
                # 为了防止日志刷屏，可以限制仅在前两个 Defense Nodes 上打印，或者你可以去掉 `if i in defense_nodes[:2]:` 让所有节点都打印
                if i in defense_nodes[:2] or i == 0:
                    print(f"\n📢 [Round {r+1}] Obs {i} FLAME Clustering Result (Trusts {len(trusted_ids)}/{n_c}):")
                    print(f"    {'Status':<8} | {'Node':<5} | {'Role':<5} | {'Cluster ID'}")
                    print(f"    {'-' * 45}")
                    
                    for idx, nid in enumerate(candidates):
                        role = "😈" if nid in malicious_clients else "😇"
                        is_trusted = "✅ AGG" if nid in trusted_ids else "❌ DROP"
                        c_label = cluster_labels[idx] if n_c >= 3 else "N/A"
                        
                        # 如果恶意节点被错误地选入了良性簇，打上高亮火焰标记 🔥
                        alert = " 🔥 ALERT (Poisoned)" if (nid in malicious_clients) and (nid in trusted_ids) else ""
                        is_self = " (Me)" if nid == i else ""
                        
                        print(f"    {is_trusted:<8} | {str(nid)+is_self:<5} | {role:<5} | Cluster: {c_label:<3} {alert}")
                    print(f"    {'-' * 45}")
                # =========================================================

                # 5. 动态裁剪 (Dynamic Clipping) - 计算截断阈值 Gamma
                trusted_vecs = [vector_map[nid] for nid in trusted_ids]
                norms = [torch.norm(v).item() for v in trusted_vecs]
                gamma = np.median(norms) if norms else 1.0  
                
                # 6. 聚合、裁剪与差分隐私加噪
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
                # 1. 区分邻居类型
                my_nbs = neighbors[i]

                # 找出邻居中的“防御节点”和“普通邻居”
                defense_neighbors = [n for n in my_nbs if n in defense_nodes]
                normal_neighbors = [n for n in my_nbs if n not in defense_nodes]

                avg_state = {}

                # --- Case A: 只要连接到了防御节点 (Connected to Defense Node) ---
                if defense_neighbors:
                    # === 权重配置 ===
                    # 1. 自己占 50%
                    w_self_total = 0.5

                    # 2. 剩余的 50%
                    w_remaining = 0.5

                    # 3. 剩余部分中，防御节点占 90% (即总量的 45%)
                    w_defense_total = w_remaining * 0.9

                    # 4. 剩余部分中，普通邻居占 10% (即总量的 5%)
                    w_normal_total = w_remaining * 0.1

                    # === 计算单节点权重 ===
                    # A. 自己的权重
                    w_self = w_self_total

                    # B. 防御节点的权重 (平均分配)
                    w_def = w_defense_total / len(defense_neighbors)

                    # C. 普通邻居的权重
                    if normal_neighbors:
                        w_norm = w_normal_total / len(normal_neighbors)
                    else:
                        w_norm = 0.0
                        w_self += w_normal_total # 回收未使用的 5%

                    # === 执行加权聚合 ===
                    for k in initial_state.keys():
                        if 'num_batches_tracked' in k:
                            avg_state[k] = new_weights_cpu[i][k]
                            continue

                        # 1. 累加自己 (Self)
                        tmp_sum = new_weights_cpu[i][k].float() * w_self

                        # 2. 累加防御节点 (Defense Neighbors)
                        for dn in defense_neighbors:
                            tmp_sum += new_weights_cpu[dn][k].float() * w_def

                        # 3. 累加普通邻居 (Normal Neighbors)
                        for neigh in normal_neighbors:
                            tmp_sum += new_weights_cpu[neigh][k].float() * w_norm

                        avg_state[k] = tmp_sum.to(new_weights_cpu[i][k].dtype)

                # --- Case B: 如果周围没有防御节点 (Fallback) ---
                # --- Case B: 如果周围没有防御节点 (Fallback) ---
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
        # ==========================================
        # 🔄 轮次收尾：彻底摧毁本轮产生的垃圾数据
        # ==========================================

        # 1. 显式清空本轮训练产生的大列表
        if 'new_weights_cpu' in locals():
            # 逐个删除列表内的元素，确保引用计数归零
            while len(new_weights_cpu) > 0:
                item = new_weights_cpu.pop()
                del item
            del new_weights_cpu

        # 2. 显式清空防御向量
        if 'vector_map' in locals():
            vector_map.clear() # 如果是字典
            del vector_map

        # 3. 显式清空良性更新参考向量
        if 'benign_updates' in locals():
            del benign_updates
        if 'avg_benign_update' in locals():
            del avg_benign_update

        # 4. 【核心步骤】在权重更迭时断开引用
        # 假设聚合产生的新权重是 next_round_weights
        old_weights_ref = client_weights_cpu
        client_weights_cpu = next_round_weights
        del old_weights_ref # 明确处死上一轮的模型

        # 5. 终极召唤
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
    # Phase E: Evaluation
    # =================================================
    print(f"\n📊 --- Final Evaluation (Round {r+1}) ---")
    
    # 1. 打印表头 (调整宽度以对齐)
    print(f" {'Client':<8} | {'Type':<6} | {'ACC':<10} | {'ASR':<10} | {'Theo Intensity'}")
    print("-" * 65)

    # 3. 评估阶段
    theo_intensities = calculate_theoretical_intensity(neighbors, malicious_clients, NUM_CLIENTS, bf)
    
    benign_acc_list, benign_asr_list = [], []
    acc_list = []
    asr_list = []
    for cid in range(NUM_CLIENTS):
        # --- [新增] 判断节点类型 ---
        if cid in malicious_clients:
            node_type = "MAL"  # 恶意节点
        elif cid in defense_nodes: # 如果你有防御节点的列表，也可以标记出来，否则归为 BEN
            node_type = "DEF"
        else:
            node_type = "BEN"  # 良性节点
            
        # --- 评估 ---
        acc, asr = evaluate_global_transformer(gpu_model, client_weights_cpu[cid], test_data, device, intensity=intensity)

        acc_list.append(float(acc))  # 强制转为 float，防止 OOM
        asr_list.append(float(asr))  # 强制转为 float，防止 OOM
        
        if node_type != "MAL":       # 使用 node_type 判断
            benign_acc_list.append(float(acc))
            benign_asr_list.append(float(asr))
        # --- [修改] 格式化打印 ---
        # 对应表头: Client | Type | ACC | ASR | Intensity
        # :<8 表示左对齐占8个字符宽度
        # :.4f 表示保留4位小数
        theo_int = theo_intensities[cid] if cid < len(theo_intensities) else 0
        
        print(f" {cid:<8} | {node_type:<6} | {acc:<10.4f} | {asr:<10.4f} | {theo_int:.4f}")

    avg_benign_acc = np.mean(benign_acc_list) if benign_acc_list else 0.0
    avg_benign_asr = np.mean(benign_asr_list) if benign_asr_list else 0.0

    del gpu_model
    del client_weights_cpu
    del next_round_weights
    
    # 2. 彻底清理
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize() # 确保 GPU 操作全部完成
    return avg_benign_acc, avg_benign_asr ,acc_list, asr_list
      