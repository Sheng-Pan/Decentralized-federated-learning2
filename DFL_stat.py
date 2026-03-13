import numpy as np
import torch
def testfun2(x):
    return x
def get_universal_stats(update_dict):
    """
    计算通用攻击统计量：
    1. t_z, s_z: 针对性指标 (Targeting)
    2. kurt, l2: 隐蔽性/稀疏性指标 (Stealth/Sparsity)
    """
    delta = None
    layer_name = ""

    # 1. 定义候选层名
    candidate_keys = [
        'classifier.weight', 'pre_classifier.weight', # Transformer
        'fc.weight', 'linear.weight', 'head.weight',  # ResNet/ViT
        'fc2.weight', 'fc3.weight'                    # CNN/LeNet
    ]

    # 2. 自动搜索匹配的层
    for k in candidate_keys:
        if k in update_dict:
            delta = update_dict[k]
            layer_name = k
            break

    # Fallback
    if delta is None:
        for k in reversed(list(update_dict.keys())):
            if 'weight' in k and ('fc' in k or 'linear' in k):
                delta = update_dict[k]
                layer_name = k
                break

    if delta is None:
        return {'t_z': 0.0, 's_z': 0.0, 'kurt': 0.0, 'l2': 0.0}, "No-Key"

    # 3. 转 Numpy
    if isinstance(delta, torch.Tensor):
        delta = delta.float().cpu().numpy()

    # ------------------------------------------------
    # A. 计算 t_z 和 s_z (针对性)
    # ------------------------------------------------
    pos_energy = np.sum(np.maximum(delta, 0), axis=1)
    median_p = np.median(pos_energy)
    mad_p = np.median(np.abs(pos_energy - median_p)) + 1e-9
    z_pos = 0.6745 * (pos_energy - median_p) / mad_p

    neg_energy = np.abs(np.sum(np.minimum(delta, 0), axis=1))
    median_n = np.median(neg_energy)
    mad_n = np.median(np.abs(neg_energy - median_n)) + 1e-9
    z_neg = 0.6745 * (neg_energy - median_n) / mad_n

    # ------------------------------------------------
    # B. 计算 Kurtosis 和 L2 (分布形态)
    # ------------------------------------------------
    flat = delta.flatten()
    l2_val = np.linalg.norm(flat)

    mean_val = np.mean(flat)
    std_val = np.std(flat)
    kurtosis = 0.0

    if std_val > 1e-9:
        centered = flat - mean_val
        # Fourth standardized moment
        moment4 = np.mean(centered**4)
        kurtosis = moment4 / (std_val**4)

    return {
        't_z': np.max(z_pos),
        's_z': np.max(z_neg),
        'kurt': kurtosis,
        'l2': l2_val
    }, layer_name


def get_universal_stats(input_data, model_type='unknown'):
    """
    通用攻击统计提取函数 (Robust Fix)
    修复了因 MAD 过小导致的 Z-Score 爆炸问题
    """
    delta = None
    layer_name = ""
    
    # --- 1. 数据解包与识别 ---
    if isinstance(input_data, (torch.Tensor, np.ndarray)):
        delta = input_data
        layer_name = "Manual_Aggregated"
    elif isinstance(input_data, dict):
        # 字典模式：自动搜索全连接层
        candidate_keys = ['classifier.weight', 'fc.weight', 'linear.weight', 'head.weight', 'fc2.weight']
        for k in candidate_keys:
            if k in input_data:
                delta = input_data[k]
                layer_name = k
                break
        if delta is None: # Fallback
            for k, v in input_data.items():
                if 'weight' in k and ('fc' in k or 'linear' in k):
                    delta = v
                    layer_name = k
                    
    if delta is None: 
        return {'t_z': 0.0, 's_z': 0.0}, "No-Key"

    # --- 2. 格式统一化 ---
    if isinstance(delta, torch.Tensor):
        delta = delta.float().cpu().detach().numpy()
    
    if delta.ndim > 1:
        delta = delta.flatten()
        
    if len(delta) == 0: return {'t_z': 0.0, 's_z': 0.0}, layer_name

    # ==========================================
    # 3. 核心数学计算 (Robust Statistics)
    # ==========================================
    
    # 计算中位数
    median_val = np.median(delta)
    
    # 计算 MAD (Median Absolute Deviation)
    diff = np.abs(delta - median_val)
    mad = np.median(diff)
    
    # 【关键修复】设置 MAD 的最小下界
    # 神经网络权重的正常更新幅度通常不会小于 1e-5
    # 如果 MAD < 1e-5，说明该层几乎没有变化，此时我们强行将其设为 1e-5，避免 Z-Score 爆炸
    min_mad = 1e-5 
    effective_mad = max(mad, min_mad)
    
    # 计算 Z-Scores
    # 0.6745 是正态分布的校正系数
    z_scores = 0.6745 * (delta - median_val) / effective_mad
    
    # T_Z: 整体偏移 (Mean Absolute Z-Score)
    t_z = np.mean(np.abs(z_scores))
    
    # S_Z: 形状异质性 (Max Absolute Z-Score) -> 这里的异常值通常对应 Trigger
    s_z = np.max(np.abs(z_scores))

    return {'t_z': t_z, 's_z': s_z}, f"Layer={layer_name}"
def calc_metrics(vec_a, vec_b):
    l2_dist = np.linalg.norm(vec_a - vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    cos_sim = 0.0
    if norm_a > 0 and norm_b > 0:
        cos_sim = np.dot(vec_a, vec_b) / (norm_a * norm_b)
    return cos_sim, l2_dist
def detect_collusion_dummy(update_dicts, ids):
    return {nid: -1 for nid in ids}
import torch
import numpy as np

def get_universal_stats_MLP(update_dict):
    # Flatten all params into one vector
    vec = []
    for k in sorted(update_dict.keys()):
        if 'weight' in k or 'bias' in k:
            vec.append(update_dict[k].float().view(-1).cpu())
            
    if not vec: 
        return {'t_z': 0.0, 's_z': 0.0}, np.array([])
    
    delta = torch.cat(vec).numpy()
    
    # ==========================================
    # 1. T_z 逻辑 (捕捉极端兴奋/正向更新)
    # ==========================================
    pos_vals = delta[delta > 0]
    if len(pos_vals) > 0:
        median_p = np.median(pos_vals)
        mad_p = np.median(np.abs(pos_vals - median_p)) + 1e-9
        z_scores_p = 0.6745 * (pos_vals - median_p) / mad_p
        t_z = np.max(z_scores_p)
    else:
        t_z = 0.0

    # ==========================================
    # 2. S_z 逻辑 (捕捉极端抑制/负向更新)
    # ==========================================
    # 取负值部分，并用绝对值表示其修改的“强度”
    neg_vals = np.abs(delta[delta < 0])
    if len(neg_vals) > 0:
        median_n = np.median(neg_vals)
        mad_n = np.median(np.abs(neg_vals - median_n)) + 1e-9
        # 计算负向更新幅度的 Z-score：看有没有负得离谱的参数
        z_scores_n = 0.6745 * (neg_vals - median_n) / mad_n
        s_z = np.max(z_scores_n)
    else:
        s_z = 0.0

    return {'t_z': t_z, 's_z': s_z}, delta
