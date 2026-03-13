import numpy as np
import torch
import math
import random
from collections import defaultdict
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from scipy.stats import wasserstein_distance
import networkx as nx
from defense import get_universal_stats
# ==========================================
#  General functions
# ==========================================
import numpy as np

def calc_element_wise_stats(delta_np, topk=5):
    flat = delta_np.flatten()
    pos_vals = flat[flat > 1e-6] # 过滤极小噪声
    
    if len(pos_vals) > topk:
        median_p = np.median(pos_vals)
        mad_p = np.median(np.abs(pos_vals - median_p)) + 1e-9
        z_scores = 0.6745 * (pos_vals - median_p) / mad_p
        
        # 取前 K 个最大值的平均
        topk_indices = np.argpartition(z_scores, -topk)[-topk:]
        return np.mean(z_scores[topk_indices])
    return 0.0
import torch
import torch
import torch

def generate_gtsrb_probes(batch_size=16, device='cuda'):
    """
    生成专门针对 GTSRB 分布的随机探针，防止 CNN 深层激活衰减到 0。
    基于均匀分布 [0, 1] 并应用与训练集完全相同的 Normalize。
    """
    # 1. 生成 0~1 之间的均匀随机噪声 (更接近真实图片的像素值域)
    probes = torch.rand(batch_size, 3, 32, 32, device=device)
    
    # 2. 应用 GTSRB 的均值和标准差进行归一化
    mean = torch.tensor([0.3337, 0.3064, 0.3171], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.2672, 0.2564, 0.2629], device=device).view(1, 3, 1, 1)
    
    probes = (probes - mean) / std
    return probes

import torch

def calc_activation_stats(model, target_layer_name, probe_inputs):
    """
    纯 GPU 版本的通用激活值异常计算 (基于峰度 Kurtosis)
    兼容 CNN (图像) 和 Transformer (NLP) 模型。
    """
    activations = {}
    
    def hook_fn(module, input, output):
        # 🌟 修复 2: 兼容 Transformer 某些层返回 Tuple 的情况
        if isinstance(output, tuple):
            activations['out'] = output[0].detach()
        else:
            activations['out'] = output.detach()
        
    # 动态获取目标层
    target_layer = dict([*model.named_modules()]).get(target_layer_name)
    if target_layer is None:
        return 0.0
        
    handle = target_layer.register_forward_hook(hook_fn)
    
    model.eval()
    with torch.no_grad():
        # 🌟 修复 1: 兼容 NLP 字典型输入
        if isinstance(probe_inputs, dict):
            _ = model(**probe_inputs)
        else:
            _ = model(probe_inputs)
        
    handle.remove() # 卸载钩子，防止 OOM
    
    acts_tensor = activations.get('out')
    if acts_tensor is None:
        return 0.0
        
    acts_tensor = acts_tensor.flatten()
    
    # 🌟 修复 3: 改用绝对值过滤，保留 GELU/Tanh 的负向有效激活
    # 过滤掉绝对值小于 1e-6 的绝对死区噪声
    valid_acts = acts_tensor#[torch.abs(acts_tensor) > 1e-6] 
    
    # 计算峰度至少需要 3 个有效数据点
    if len(valid_acts) >= 3:
        # 强制转为 float32，防止半精度下计算 4 次方导致 inf 溢出
        if valid_acts.dtype not in [torch.float32, torch.float64]:
            valid_acts = valid_acts.float()
            
        mean_a = torch.mean(valid_acts)
        var_a = torch.var(valid_acts, unbiased=False) + 1e-9 
        
        # 第四阶中心矩
        fourth_moment = torch.mean((valid_acts - mean_a) ** 4)
        
        # 峰度 = 第四阶中心矩 / (方差^2)
        kurtosis = fourth_moment / (var_a ** 2)
        
        return kurtosis.item()
        
    return 0.0
def calculate_continuous_trust(neighbor_metrics, steepness=1):
    if not neighbor_metrics: return {}

    # 1. 提取原始指标 (将 sens 替换为 sz)
    trap_vals = np.abs(np.array([m['trap'] for m in neighbor_metrics])) + 1e-12
    sz_vals   = np.array([m['s_z'] for m in neighbor_metrics]) + 1e-12  # <--- 修改点：提取 s_z
    rs_vals   = np.array([m.get('rs_score', 1.0) for m in neighbor_metrics])

    def calculate_robust_z(vals, eps=0.1, use_log=True):
        calc_vals = np.log1p(vals) if use_log else vals
        median = np.median(calc_vals)
        mad = np.median(np.abs(calc_vals - median))
        robust_mad = max(mad, eps) 
        return 0.6745 * np.abs(calc_vals - median) / robust_mad
    
    # 2. 计算核心维度的 Z-Scores
    z_trap = calculate_robust_z(trap_vals, use_log=True)
    z_sz   = calculate_robust_z(sz_vals, use_log=True) # <--- 修改点：计算 z_sz 惩罚
    z_rs   = calculate_robust_z(rs_vals, use_log=False, eps=0.05) 

    # 3. 累加惩罚项
    z_comb = z_trap + z_rs + z_sz

    # 指数映射到 0~1 的奖励分
    trust_scores = np.exp(-steepness * z_comb)

    result = {}
    for i, m in enumerate(neighbor_metrics):
        result[m['nid']] = {
            'reward': float(trust_scores[i]),
            'z_comb': float(z_comb[i]),
            'z_trap': float(z_trap[i]),
            'z_rs':   float(z_rs[i]),
            'z_sz':   float(z_sz[i]), # <--- 修改点：保存 z_sz 惩罚分
            'rs_score': float(rs_vals[i])
        }
    return result
import math
import torch
import numpy as np
from collections import defaultdict

class MABDefense:
    def __init__(self, num_clients, model_type='cnn', decay=0.9, exploration_c=1.0, 
                 audit_prob=0.9, agg_prob=0.9, agg_threshold=0.4, custom_target_layers=None,
                 alpha=0.2): # <--- 新增: 将 alpha 提升为类属性
        self.num_clients = num_clients
        self.model_type = model_type.lower()
        
        self.trust_scores = {i: {j: 0.5 for j in range(num_clients)} for i in range(num_clients)}
        # <--- 修改点: visit_counts 改用 float，存储折扣访问量 N_{t,gamma}(j)
        self.visit_counts = {i: defaultdict(float) for i in range(num_clients)}
        self.total_rounds = {i: 0 for i in range(num_clients)}
        
        self.decay = decay
        self.c = exploration_c
        self.audit_prob = audit_prob
        self.agg_prob = agg_prob
        self.steepness = 1.0 
        self.agg_threshold = agg_threshold
        
        # 折扣 UCB 核心参数
        self.alpha = alpha
        self.gamma = 1.0 - self.alpha # gamma = 1 - alpha

        if self.model_type == 'transformer':
            self.target_layers = ['classifier.weight', 'pre_classifier.weight', 'fc.weight', 'linear.weight', 'out_proj.weight']
        elif self.model_type == 'cnn': 
            self.target_layers = ['fc2.weight', 'linear.weight', 'fc.weight']
        elif self.model_type == 'mlp': 
            self.target_layers = ['fc1.weight', 'fc2.weight', 'fc3.weight']
            
        if custom_target_layers is not None:
            self.target_layers = custom_target_layers

    def select_for_audit(self, client_id, neighbors_list, audit_budget=None):
            """Phase 1: 基于 UCB 的探索性抽样"""
            self.total_rounds[client_id] += 1
            n_gamma_t = sum(self.visit_counts[client_id].get(nid, 0.0) for nid in neighbors_list)
            log_n_gamma = math.log(n_gamma_t) if n_gamma_t > 1.0 else 0.0 

            u_scores = {}
            for nid in neighbors_list:
                q = self.trust_scores[client_id].get(nid, 0.5)
                n_v = self.visit_counts[client_id].get(nid, 0.0)
                bonus = self.c * math.sqrt((2 * log_n_gamma) / n_v) if n_v > 0 else self.c * 10.0
                u_scores[nid] = max(q + bonus, 1e-9)

            if audit_budget is None:
                audit_budget = max(1, round(len(neighbors_list) * self.audit_prob))
            
            # 概率正比于 UCB (符合探索逻辑)
            scores_list = np.array([u_scores[nid] for nid in neighbors_list])
            probs = scores_list / np.sum(scores_list)
            audit_targets = np.random.choice(neighbors_list, size=min(audit_budget, len(neighbors_list)), 
                                            replace=False, p=probs).tolist()
            return audit_targets

    def select_for_aggregation(self, client_id, candidates, agg_budget=None):
        """Phase 3: 仅从已审计节点中，基于 Q 分数进行利用式聚合"""
        # 1. 过滤：仅看这一轮被选中的审计目标，且信任分达标
        valid_candidates = [n for n in candidates if self.trust_scores[client_id].get(n, 0) > self.agg_threshold]
        
        if not valid_candidates: return [], []

        if agg_budget is None:
            agg_budget = max(1, int(len(valid_candidates) * self.agg_prob))
        
        # 2. 抽样：概率正比于 Q (利用已知的信任)
        qs = np.array([self.trust_scores[client_id][nid] for nid in valid_candidates])
        probs = qs / np.sum(qs)
        selected = np.random.choice(valid_candidates, size=min(agg_budget, len(valid_candidates)), 
                                    replace=False, p=probs).tolist()

        # 3. 权重：基于 Q 归一化 (安全考虑，不带 UCB 的 Bonus)
        final_qs = [self.trust_scores[client_id][nid] for nid in selected]
        weights = [q / (sum(final_qs) + 1e-9) for q in final_qs]
        
        return selected, weights
    


    def update_trust(self, observer_id, probe_list, new_weights, old_weights, device, model_template,
                     sensitivity_func=None, tokenizer=None, get_trap_func=None, rs_func=None,
                     probe_inputs=None): # <--- 修改点 1: 在参数列表新增 probe_inputs
        detailed_logs = {}
        if not probe_list: return detailed_logs

        # 全局衰减 (D-UCB 逻辑)
        for nid in self.visit_counts[observer_id]:
            self.visit_counts[observer_id][nid] *= self.gamma

        neighbor_metrics = []

        # 如果外部没有传入探针，可以在这里动态生成（注意：需要根据你的真实输入维度修改 shape）
        # 更好的做法是在外部生成好传入，这样所有 client 在同一轮用同样的探针
        if probe_inputs is None:
            # 假设你的模型输入是 (Batch, Channel, H, W) = (16, 3, 32, 32)
            probe_inputs = torch.randn(16, 3, 32, 32, device=device) 
        else:
            probe_inputs = probe_inputs.to(device)

        for nid in probe_list:
            nid = int(nid)
            if not isinstance(new_weights[nid], dict): continue
            
            # --- 修改点 2: 提取层名时，需要去掉 '.weight' 后缀 ---
            target_layer_param = 'fc2.weight'
            if target_layer_param not in new_weights[nid]: target_layer_param = 'linear.weight'
            if target_layer_param not in new_weights[nid]: target_layer_param = list(new_weights[nid].keys())[-2]
            
            target_layer_module_name = target_layer_param.replace('.weight', '') # 例如 'fc2'

            # --- 修改点 3: 必须先加载权重，再做前向传播 ---
            nb_state = {k: v.float().to(device) for k, v in new_weights[nid].items()}
            model_template.load_state_dict(nb_state)
            probe_inputs = probe_inputs.to(device)
            # --- 修改点 4: 使用激活值探针计算 s_z ---
            s_z_val = calc_activation_stats(model_template, target_layer_module_name, probe_inputs)
            # --- 以下保留你原有的 Trap 和 RS 逻辑 ---
            trap_s = 0.0
            if get_trap_func:
                try:
                    trap_s = get_trap_func(model_template, tokenizer, device)
                except TypeError:
                    trap_s = get_trap_func(model_template, device)

            rs_val = 1.0
            if rs_func:
                try:
                    rs_val = rs_func(model_template, tokenizer, device)
                except Exception:
                    rs_val = rs_func(model_template, device)

            neighbor_metrics.append({
                'nid': nid,
                'trap': trap_s, 
                's_z': s_z_val,  
                'rs_score': rs_val
            })

        if not neighbor_metrics: return detailed_logs

        trust_map = calculate_continuous_trust(neighbor_metrics, steepness=self.steepness)
        
        for m in neighbor_metrics:
            nid = m['nid']
            trust_data = trust_map.get(nid, {'reward': 0.5, 'z_comb': 0.0})

            instant_reward = trust_data['reward']
            z_combined = trust_data['z_comb']
            
            curr_q = self.trust_scores[observer_id][nid]
            new_q = (1 - self.alpha) * curr_q + self.alpha * instant_reward
            self.trust_scores[observer_id][nid] = new_q
            
            self.visit_counts[observer_id][nid] += 1.0 
            
            detailed_logs[nid] = {
                'trap': m['trap'],
                'elem_z': m['s_z'],  
                'rs_score': m['rs_score'],
                'z_trap': trust_data['z_trap'],
                'z_rs': trust_data['z_rs'],
                'z_sz': trust_data['z_sz'], 
                'z_comb_penalty': z_combined,
                'reward': instant_reward,
                'new_q': new_q
            }

        return detailed_logs
    
# ==========================================
#  CNN MAB
# ==========================================
class MABDefense_CNN:
    def __init__(self, num_clients, decay=0.9, exploration_c=1.0, audit_prob=0.9):
        self.num_clients = num_clients
        self.trust_scores = {i: {j: 0.5 for j in range(num_clients)} for i in range(num_clients)}
        self.visit_counts = {i: defaultdict(int) for i in range(num_clients)}
        self.total_rounds = {i: 0 for i in range(num_clients)}
        self.decay = decay
        self.c = exploration_c
        self.audit_prob = audit_prob

    def select_for_audit(self, client_id, neighbors_list, audit_budget=None):
        """Phase 1: Select nodes to audit based on UCB"""
        self.total_rounds[client_id] += 1
        t = self.total_rounds[client_id]

        ucb_scores = {}
        for nid in neighbors_list:
            q = self.trust_scores[client_id].get(nid, 0.5)
            n_v = self.visit_counts[client_id][nid]
            bonus = self.c * math.sqrt(math.log(t) / (n_v + 1e-6)) if n_v > 0 else 1.0
            ucb_scores[nid] = q + bonus
        raw_budget = len(neighbors_list) * self.audit_prob
        if audit_budget is None:
            audit_budget = max(1, int(raw_budget))
        audit_budget0 = max(audit_budget,int(raw_budget))
        audit_targets = sorted(neighbors_list, key=lambda x: ucb_scores[x], reverse=True)[:audit_budget0]
        return audit_targets
   
    def select_for_aggregation(self, client_id, candidates, agg_budget=None):
        """Phase 3: Select validated nodes for aggregation"""
        sorted_neighbors = sorted(candidates, key=lambda x: self.trust_scores[client_id].get(x, 0), reverse=True)
        # Filter low trust (Threshold 0.2)
        valid_neighbors = [n for n in sorted_neighbors if self.trust_scores[client_id].get(n, 0) > 0.25]
        agg_budget = max(1, int(len(candidates) * self.agg_prob))
        if agg_budget:
            selected = valid_neighbors[:agg_budget]
        else:
            selected = valid_neighbors

        # Softmax-like Weights
        if selected:
            qs = [self.trust_scores[client_id][nid] for nid in selected]
            total = sum(qs) + 1e-9
            weights = [q / total for q in qs]
        else:
            weights = []

        return selected, weights

    def update_trust(self, observer_id, probe_list, new_weights, old_weights, device, model_template,
                     sensitivity_func, round_sens_map=None, get_trap_func=None):
        """
        Phase 2: Audit and Update Trust using Continuous Rewards
        (Modified to depend strictly on: max_s, trap, and svd, with Safe Trapdoor Alignment)
        """
        detailed_logs = {}
        if not probe_list: return detailed_logs

        # 1. Compute Trapdoor Gradient (Observer's Perspective)
        g_trap = None
        g_trap_norm = 1.0
        if get_trap_func:
            # Use observer's own weights to generate the trap/trigger direction
            obs_state = {k: v.float().to(device) for k, v in old_weights[observer_id].items()}
            model_template.load_state_dict(obs_state)
            g_trap = get_trap_func(model_template, device)
            
            # Ensure g_trap is flattened
            if g_trap is not None:
                g_trap = g_trap.view(-1)
                g_trap_norm = g_trap.norm() + 1e-9

        neighbor_metrics = []
        target_layers = ['fc2.weight', 'linear.weight', 'fc.weight']

        # 2. Data Collection
        for nid in probe_list:
            nid = int(nid)
            if not isinstance(new_weights[nid], dict):
                continue
            
            # Load Neighbor
            nb_state = {k: v.float().to(device) for k, v in new_weights[nid].items()}
            model_template.load_state_dict(nb_state)

            # Metric A: Sensitivity
            max_s, _ = sensitivity_func(model_template, device)

            # --- Calculate Diff / Delta ---
            trap_s = 0.0
            delta_matrix = None
            d_vec_target_list = []  # 专门存 target_layers 的更新差值
            d_vec_all_list = []     # 存 整个模型 (所有 'weight' 层) 的更新差值

            for k in new_weights[nid].keys():
                # 核心修复：确保 k 是字符串才做 'in' 操作
                if not isinstance(k, str):
                    continue
                if 'weight' in k:
                    w_new = new_weights[nid][k].float().to(device)
                    w_old = old_weights[nid][k].float().to(device)
                    delta = w_new - w_old

                    # For SVD: Use the largest layer
                    if delta_matrix is None or delta.numel() > delta_matrix.numel():
                        delta_matrix = delta

                    # For Trapdoor: Collect all layers
                    d_vec_all_list.append(delta.view(-1))

                    # For Trapdoor: Collect specific target layers
                    if any(t in k for t in target_layers):
                        d_vec_target_list.append(delta.view(-1))

            # ---------------------------------------------------------
            # Metric B: Trapdoor Similarity (🚨 消除隐患的核心修改区)
            # ---------------------------------------------------------
            if g_trap is not None:
                d_vec_target = torch.cat(d_vec_target_list) if d_vec_target_list else torch.tensor([]).to(device)
                d_vec_all = torch.cat(d_vec_all_list) if d_vec_all_list else torch.tensor([]).to(device)

                # 智能形状匹配
                if d_vec_target.shape == g_trap.shape:
                    match_vec = d_vec_target
                elif d_vec_all.shape == g_trap.shape:
                    match_vec = d_vec_all
                else:
                    # 强硬报错，逼迫维度不对齐的 Bug 暴露
                    raise ValueError(
                        f"🚨 CNN Trapdoor 维度严重不匹配! \n"
                        f"g_trap 的维度是: {g_trap.shape}\n"
                        f"但 target_layers 的维度是: {d_vec_target.shape}\n"
                        f"包含所有 weight 层的维度是: {d_vec_all.shape}\n"
                        f"请检查 get_trap_func 提取梯度的逻辑。"
                    )

                trap_s = torch.dot(match_vec, g_trap) / (match_vec.norm() * g_trap_norm + 1e-9)
                trap_s = trap_s.item()

            # Metric C: SVD (Spectral Norm of update)
            svd_val = 0.0
            if delta_matrix is not None:
                if delta_matrix.ndim > 2:
                    delta_matrix = delta_matrix.view(delta_matrix.size(0), -1)
                elif delta_matrix.ndim == 1:
                    # SVD expects 2D matrices; handle 1D tensors (e.g. norm weights) safely
                    delta_matrix = delta_matrix.view(-1, 1)
                try:
                    svd_val = torch.linalg.svdvals(delta_matrix)[0].item()
                except:
                    svd_val = 0.0

            # --- Store Metrics ---
            neighbor_metrics.append({
                'nid': nid,
                'trap': trap_s,
                'max_s': max_s,
                'svd': svd_val
            })

        if not neighbor_metrics: return detailed_logs

        # 3. Calculate Trust Scores (使用最新版包含 np.maximum.reduce 的函数)
        trust_map = calculate_continuous_trust(neighbor_metrics, steepness=2.0)

        # 4. Update MAB Values
        alpha = 0.2

        # for m in neighbor_metrics:
        #     nid = m['nid']
        #     trust_data = trust_map.get(nid, {'reward': 0.5, 'z_comb': 0.0})

        #     instant_reward = trust_data['reward']
        #     z_combined = trust_data['z_comb'] # 这里的 z_comb 是三大指标的木桶最大惩罚

        #     curr_q = self.trust_scores[observer_id][nid]

        #     # Immediate Kill Logic / Update
        #     new_q = (1 - alpha) * curr_q + alpha * instant_reward

        #     if instant_reward < 0.4: status_label = f"REJECT(Pen={z_combined:.2f})"
        #     elif instant_reward < 0.7: status_label = f"WARN(Pen={z_combined:.2f})"
        #     else: status_label = "ACCEPT"

        #     # Update State
        #     self.trust_scores[observer_id][nid] = new_q
        for m in neighbor_metrics:
            nid = m['nid']
            trust_data = trust_map.get(nid, {'reward': 0.5, 'z_comb': 0.0})
            z_combined = trust_data['z_comb']
            
            # --- 改进 1: 使用 Sigmoid 映射 instant_reward (替代 linear) ---
            # 让 3.5 附近的 Reward 迅速下降，而不是等到 10.0
            # 这里的 steepness 可以控制对异常的敏感度
            instant_reward = 1.0 / (1.0 + math.exp(0.8 * (z_combined - 4.0)))

            curr_q = self.trust_scores[observer_id][nid]
            
            # --- 改进 2: 悲观更新 (Fast-Drop, Slow-Rise) ---
            if instant_reward < curr_q:
                # 发现异常时，下降要快 (alpha 大)，立刻隔离
                alpha = 0.6  
            else:
                # 表现良好时，恢复要慢 (alpha 小)，防止恶意节点通过“洗白”快速重返
                alpha = 0.05 

            new_q = (1 - alpha) * curr_q + alpha * instant_reward

            # --- 改进 3: 软熔断 (Veto) ---
            # 如果 Z 真的非常离谱 (例如 > 15)，无论 EMA 多少，本轮强行不选
            if z_combined > 10.0:
                new_q = 0
                status_label = "🚨 CRITICAL_VETO"
            elif instant_reward < self.reject_thresh:
                status_label = "REJECT"
            else:
                status_label = "ACCEPT"

            self.trust_scores[observer_id][nid] = new_q
            self.visit_counts[observer_id][nid] += 1

            # Log details
            detailed_logs[nid] = {
                'raw_sens': m['max_s'],
                'svd_val': m['svd'],
                'trap': m['trap'],
                'z_comb_penalty': z_combined,               # 综合最大惩罚
                'svd_dist': trust_data.get('svd_dist', 0.0),   # SVD的独立惩罚
                'trap_dist': trust_data.get('trap_dist', 0.0), # Trap的独立惩罚
                'reward': instant_reward,
                'new_q': new_q,
                'gap_cut': status_label
            }

        return detailed_logs
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from collections import defaultdict
import numpy as np
import torch
import math
import random
from scipy.stats import wasserstein_distance
# ==========================================
#  transformer MAB
# ==========================================
import math
import torch
import numpy as np
from collections import defaultdict
import math
import torch
import numpy as np
from collections import defaultdict


class MABDefense_transformer:
    def __init__(self, num_clients, decay=0.9, exploration_c=1.0, audit_prob=0.9, agg_prob=0.8):
        self.num_clients = num_clients
        self.trust_scores = {i: {j: 0.5 for j in range(num_clients)} for i in range(num_clients)}
        self.visit_counts = {i: defaultdict(int) for i in range(num_clients)}
        self.total_rounds = {i: 0 for i in range(num_clients)}
        self.decay = decay
        self.c = exploration_c
        self.audit_prob = audit_prob
        self.agg_prob = agg_prob 
    def select_for_audit(self, client_id, neighbors_list, audit_budget=None):
        """Phase 1: Select nodes to audit based on UCB"""
        self.total_rounds[client_id] += 1
        t = self.total_rounds[client_id]

        ucb_scores = {}
        for nid in neighbors_list:
            q = self.trust_scores[client_id].get(nid, 0.5)
            n_v = self.visit_counts[client_id][nid]
            bonus = self.c * math.sqrt(math.log(t) / (n_v + 1e-6)) if n_v > 0 else 1.0
            ucb_scores[nid] = q + bonus

        if audit_budget is None:
            raw_budget = len(neighbors_list) * self.audit_prob
            audit_budget = max(1, int(raw_budget))

        audit_targets = sorted(neighbors_list, key=lambda x: ucb_scores[x], reverse=True)[:audit_budget]
        return audit_targets

    def select_for_aggregation(self, client_id, candidates):
        """Phase 3: Select validated nodes for aggregation"""
        sorted_neighbors = sorted(candidates, key=lambda x: self.trust_scores[client_id].get(x, 0), reverse=True)
        # Filter low trust (Threshold 0.4)
        valid_neighbors = [n for n in sorted_neighbors if self.trust_scores[client_id].get(n, 0) > 0.01]
        agg_budget = max(1, int(len(candidates) * self.agg_prob))
        if agg_budget:
            selected = valid_neighbors[:agg_budget]
        else:
            selected = valid_neighbors

        if selected:
            qs = [self.trust_scores[client_id][nid] for nid in selected]
            total = sum(qs) + 1e-9
            weights = [q / total for q in qs]
        else:
            weights = []

        return selected, weights


    def update_trust(self, observer_id, probe_list, new_weights, old_weights, device, model_template, tokenizer, sensitivity_func, round_sens_map=None, get_trap_func=None, active_metrics=None):
        """
        Phase 2: Audit and Update
        Targets: 
        1. Trapdoor Cosine Similarity
        2. MAX_SEN (Model Sensitivity)
        3. SVD (Spectral Norm)
        """
        detailed_logs = {}
        if not probe_list: return detailed_logs

        # --- Precompute Trapdoor Gradient ---
        g_trap = None
        g_trap_norm = 1.0
        if get_trap_func:
            try:
                g_trap = get_trap_func(model_template, tokenizer, device)
            except TypeError:
                g_trap = get_trap_func(model_template, device)
                
            if g_trap is not None:
                g_trap = g_trap.view(-1)
                g_trap_norm = g_trap.norm() + 1e-9

        neighbor_metrics = []
        target_layers = ['classifier.weight', 'pre_classifier.weight', 'fc.weight', 'linear.weight', 'out_proj.weight']

        # --- Step 1: Compute Raw Metrics ---
        for nid in probe_list:
            metrics_entry = {'nid': nid}
            
            # 1. Sensitivity (MAX_SEN)
            nb_state = {k: v.float().to(device) for k, v in new_weights[nid].items()}
            model_template.load_state_dict(nb_state)
            sens_val, _ = sensitivity_func(model_template, tokenizer, device)
            
            # 这里 max_s 直接就是 Model Sensitivity
            metrics_entry['max_s'] = sens_val

            # Variables for SVD & Trapdoor
            # ... (前面的 Sensitivity 计算保持不变) ...

            # Variables for SVD & Trapdoor
            svd_val = 0.0
            delta_matrix = None
            d_vec_target_list = []  # 专门存 target_layers 的更新差值
            d_vec_all_list = []     # 存 整个模型 所有层的更新差值
            
            for k in new_weights[nid].keys():
                w_new = new_weights[nid][k].float()
                w_old = old_weights[nid][k].float()
                delta = (w_new - w_old).to(device)
                
                # 寻找最大的层用于 SVD 计算 (防替换攻击)
                if delta_matrix is None or delta.numel() > delta_matrix.numel():
                    delta_matrix = delta
                    
                # 记录所有层的展平差值 (用于全模型 Trapdoor 匹配)
                d_vec_all_list.append(delta.view(-1))

                # 仅仅记录 target_layers 的展平差值 (用于局部 Trapdoor 匹配)
                if any(t in k for t in target_layers):
                    d_vec_target_list.append(delta.view(-1))
            
            # 2. SVD of Weight Delta
            if delta_matrix is not None:
                if delta_matrix.ndim > 2:
                    delta_matrix = delta_matrix.view(delta_matrix.size(0), -1)
                elif delta_matrix.ndim == 1:
                    delta_matrix = delta_matrix.unsqueeze(0)
                try:
                    s_vals = torch.linalg.svdvals(delta_matrix)
                    svd_val = s_vals[0].item()
                except:
                    svd_val = 0.0
            metrics_entry['svd'] = svd_val

            # ---------------------------------------------------------
            # 3. Trapdoor Cosine Similarity (🚨 消除隐患的核心修改区)
            # ---------------------------------------------------------
            trap_cos = 0.0
            
            if g_trap is not None:
                d_vec_target = torch.cat(d_vec_target_list) if d_vec_target_list else torch.tensor([]).to(device)
                d_vec_all = torch.cat(d_vec_all_list) if d_vec_all_list else torch.tensor([]).to(device)

                # 智能形状匹配
                if d_vec_target.shape == g_trap.shape:
                    # 匹配成功: get_trap_func 计算的仅仅是 target_layers 的梯度
                    match_vec = d_vec_target
                elif d_vec_all.shape == g_trap.shape:
                    # 匹配成功: get_trap_func 计算的是整个模型的梯度
                    match_vec = d_vec_all
                else:
                    # 🚨 强硬报错：形状完全对不上！逼迫 Bug 暴露！
                    raise ValueError(
                        f"🚨 Trapdoor 维度严重不匹配! \n"
                        f"g_trap 的维度是: {g_trap.shape}\n"
                        f"但 target_layers 的维度是: {d_vec_target.shape}\n"
                        f"整个模型的维度是: {d_vec_all.shape}\n"
                        f"请检查 get_trap_func 提取梯度的逻辑与你的 target_layers 列表是否一致。"
                    )

                # 计算余弦相似度
                trap_cos = torch.dot(match_vec, g_trap) / (match_vec.norm() * g_trap_norm + 1e-9)
                trap_cos = trap_cos.item()
                
            metrics_entry['trap'] = trap_cos
            neighbor_metrics.append(metrics_entry)
                    

        if not neighbor_metrics: return detailed_logs

        # --- Step 2: Use calculate_continuous_trust (Based purely on MAX_SEN) ---
        trust_map = calculate_continuous_trust(neighbor_metrics, steepness=5.0)

        # --- Step 3: Update MAB Trust ---
        alpha = 0.2
        for m in neighbor_metrics:
            nid = m['nid']
            trust_data = trust_map.get(nid, {'reward': 0.5, 'z_comb': 0.0})
            
            instant_reward = trust_data['reward']
            log_dist = trust_data['z_comb'] # 这里 z_comb 其实是 Log Distance

            curr_q = self.trust_scores[observer_id][nid]
            new_q = (1 - alpha) * curr_q + alpha * instant_reward

            self.trust_scores[observer_id][nid] = new_q
            self.visit_counts[observer_id][nid] += 1

            status = "ACCEPT"
            if instant_reward < 0.3: status = f"REJECT (Dist={log_dist:.2f})"
            elif instant_reward < 0.7: status = f"WARN (Dist={log_dist:.2f})"

            # Update logs
            detailed_logs[nid] = m
            detailed_logs[nid].update({
                'gap_cut': status,
                'reward': instant_reward,
                'new_q': new_q,
                'log_distance': log_dist, 
                'raw_sens': m['max_s'] # 为了日志清晰，增加一个字段显示原始 sensitivity
            })

        return detailed_logs
