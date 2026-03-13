
import torch
import numpy as np
import networkx as nx
import torch.nn.functional as F
import torch.nn as nn
import random
import os
def set_seed(seed_value):
    """Set seeds for reproducibility across different libraries."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # For multi-GPU setups
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed_value)
# ==========================================
#   detection
# ==========================================
def compute_mlp_sensitivity(model, device, input_dim=30):
    
    model.eval()
    model.zero_grad()

    # A. 构造陷门输入 (Canary Input)
    # 全 1 向量 * 2.0 (强度大，且与正常归一化数据分布不同)
    canary_input = torch.ones(1, input_dim).to(device) * 2.0

    # B. 获取陷门梯度
    output = model(canary_input)
    # 强行要求属于类别 0
    target = torch.tensor([0], dtype=torch.long).to(device)

    loss = F.cross_entropy(output, target)
    loss.backward()

    # C. 提取梯度向量 (Flattened)
    # 只提取 weight，忽略 bias，因为 bias 对方向贡献较小且容易由噪声主导
    grad_vec = []
    for name, param in model.named_parameters():
        if param.grad is not None and 'weight' in name:
            grad_vec.append(param.grad.view(-1))

    # 返回拼接后的长向量 (g_trap)
    return torch.cat(grad_vec) if grad_vec else None

# def get_universal_stats(update_dict):
#     delta = None
#     layer_name = ""
#     candidate_keys = ['classifier.weight', 'pre_classifier.weight', 'fc.weight', 'linear.weight']

#     for k in candidate_keys:
#         if k in update_dict:
#             delta = update_dict[k]
#             layer_name = k
#             break

#     if delta is None:
#         # Fallback: find any weight layer
#         for k in update_dict.keys():
#             if 'weight' in k and delta is None:
#                 delta = update_dict[k]
#                 layer_name = k
#                 break

#     if delta is None:
#         return {'t_z': 0.0, 's_z': 0.0, 'kurt': 0.0, 'l2': 0.0}, "No-Key"

#     if isinstance(delta, torch.Tensor):
#         delta = delta.float().cpu().numpy()

#     # Calculate Stats
#     pos_energy = np.sum(np.maximum(delta, 0), axis=1)
#     median_p = np.median(pos_energy)
#     mad_p = np.median(np.abs(pos_energy - median_p)) + 1e-9
#     z_pos = 0.6745 * (pos_energy - median_p) / mad_p

#     neg_energy = np.abs(np.sum(np.minimum(delta, 0), axis=1))
#     median_n = np.median(neg_energy)
#     mad_n = np.median(np.abs(neg_energy - median_n)) + 1e-9
#     z_neg = 0.6745 * (neg_energy - median_n) / mad_n

#     flat = delta.flatten()
#     l2_val = np.linalg.norm(flat)

#     return {
#         't_z': np.max(z_pos) if len(z_pos) > 0 else 0,
#         's_z': np.max(z_neg) if len(z_neg) > 0 else 0,
#         'l2': l2_val
#     }, layer_name
def get_universal_stats(delta_tensor):
    print("DEBUG: Executing the NEW get_universal_stats function (version 2.0). If you see this, the file is loaded.") # <-- Diagnostic print
    if isinstance(delta_tensor, torch.Tensor):
        delta = delta_tensor.float().cpu().numpy()
    else:
        delta = delta_tensor

    flat = delta.flatten()

    # t_z: 衡量正值的异常程度 (Top 1% positive parameters Z-score)
    pos_vals = flat[flat > 0]
    if len(pos_vals) > 0:
        median_p = np.median(pos_vals)
        mad_p = np.median(np.abs(pos_vals - median_p)) + 1e-9
        z_scores = 0.6745 * (pos_vals - median_p) / mad_p
        t_z = np.max(z_scores)
    else:
        t_z = 0.0

    return {'t_z': t_z}
import torch
import torch.nn.functional as F
# ==========================================
# 1. MLP专用辅助函数 (Sensitivity & Trapdoor)
# ==========================================
def get_mlp_sensitivity_score(model, device, input_dim=30):
    """
    计算 MLP 模型输出的敏感度得分 (对应 CNN 的 get_cnn_sensitivity_score)。
    基于模型对高斯白噪声的输出方差来判断其神经元是否异常敏感。
    """
    model.eval()
    
    # 1. 构造一个 Batch 的纯随机高斯噪声 (模拟无意义的表格特征)
    # Batch size 设为 32，特征维度 30
    batch_size = 32
    noise_input = torch.randn(batch_size, input_dim).to(device)

    with torch.no_grad():
        # 获取模型的原始输出 (Logits)
        output = model(noise_input)
        
    # 2. 计算输出的敏感度指标 (方案：最大方差)
    # output 的维度是 [32, 2] (32个样本, 2个类别)
    # 我们计算每个样本在两个类别上的预测方差，或者整个 batch 针对某个类别的方差
    
    # 做法：对 Logits 进行 Softmax 转换成概率分布
    probs = F.softmax(output, dim=1)
    
    # 计算整个 Batch 内，模型预测概率的标准差/方差
    # 如果模型被投毒，它面对噪声也容易输出极端的 0.99 或 0.01，导致高方差
    variance_per_class = torch.var(probs, dim=0) 
    max_variance = torch.max(variance_per_class).item()
    
    # 为了放大差异，并且配合 calculate_continuous_trust 的指数衰减，
    # 我们可以适度放缩这个值，比如乘以 10
    sensitivity_score = max_variance * 10.0

    # MABDefense 框架期望返回 (score, detail)，所以这里返回一个 dummy 的 None 作为 detail
    return sensitivity_score, None
import torch
import torch.nn.functional as F

def get_mlp_sensitivity_score(model, device, input_dim=30,seed=123):
    """
    [Unified Version] MLP 敏感度检测
    逻辑完全对齐 CNN 版：Batch=16, Softmax梯度, Top-K平均
    """
    set_seed(seed)
    model.eval()
    
    # 1. 扩大批次大小 (Law of Large Numbers)
    B = 16
    
    # 2. 构造随机高斯噪声输入
    # scale=0.5 避免过大激活，模拟标准化后的特征
    dummy_input = torch.randn(B, input_dim).to(device) * 0.5
    dummy_input.requires_grad_(True)
    
    # 3. 前向传播
    output = model(dummy_input)
    
    # 4. 🔥 使用 Softmax 概率而非 Logits (抑制梯度方差)
    probs = F.softmax(output, dim=1)
    
    # 5. 反向传播目标：改变“当前最大概率类别”的难易程度
    target_score = probs.max(dim=1)[0].sum()
    
    # 6. 计算输入梯度
    grads = torch.autograd.grad(
        outputs=target_score, 
        inputs=dummy_input, 
        create_graph=False
    )[0] # Shape: [B, input_dim]
    
    if grads is None:
        return 0.0, 0.0
        
    # 7. 🔥 计算指标 (Top-K Robust Mean)
    grads_abs = grads.abs()
    
    # Max Sens: 每个样本取梯度最大的 5 个特征 (或全部特征) 的均值，再对 Batch 求均值
    # 这能防止某个死神经元导致梯度为 0，也能防止某个坏神经元导致梯度爆炸
    k = min(5, input_dim) 
    topk_vals = torch.topk(grads_abs, k=k, dim=1)[0]
    robust_max_sens = topk_vals.mean().item()
    
    # Total Sens: 样本的 L2 范数，再对 Batch 求均值
    sample_norms = torch.norm(grads_abs, dim=1)
    robust_total_sens = sample_norms.mean().item()
    
    return robust_max_sens, robust_total_sens
import torch
import torch.nn.functional as F
import math

def get_mlp_trap_grad_func_robust(model, device=None, input_dim=30, seed=123,*args, **kwargs):
   
    set_seed(seed)
    model.eval()
    actual_device = next(model.parameters()).device
    
    torch.manual_seed(42)
    
    # ======================================================
    # 改进 1: 构造多模式 OOD 陷门输入 (Batch Size = 3)
    # 遵循你之前的修复：保持基础噪声强度为 0.5，防止指标集体饱和
    # ======================================================
    B = 3
    trap_inputs = torch.randn(B, input_dim).to(actual_device) * 0.5 
    
    # 增加一点模式多样性，防止单一高斯噪声无法触发特定后门
    # Pattern 0: 标准高斯噪声 (* 0.5)
    # Pattern 1: 符号化极值 (强迫输入向角落挤压)
    trap_inputs[1] = torch.sign(trap_inputs[1]) * 0.5
    # Pattern 2: 均匀分布偏移
    trap_inputs[2] = (torch.rand(input_dim).to(actual_device) - 0.5) * 1.0

    with torch.no_grad():
        logits_p = model(trap_inputs)
        logits_q = model(-trap_inputs)
        
        # 动态获取分类数 N，用于计算理论最大熵
        num_classes = logits_p.size(1)
        max_entropy = math.log(num_classes)
        
        log_p = F.log_softmax(logits_p, dim=1)
        log_q = F.log_softmax(logits_q, dim=1)
        
        # clamp 防止 float32 下溢出产生 NaN
        p = torch.clamp(torch.exp(log_p), min=1e-9)
        q = torch.clamp(torch.exp(log_q), min=1e-9)
        
        # ======================================================
        # 改进 2: 计算归一化香农熵
        # ======================================================
        entropy_p = -torch.sum(p * log_p, dim=1)
        entropy_q = -torch.sum(q * log_q, dim=1)
        
        # 映射到 0~1 之间。越接近 1 说明越混乱(良性)，越接近 0 说明越笃定(恶意)
        norm_entropy_p = entropy_p / max_entropy
        norm_entropy_q = entropy_q / max_entropy
        
        # 提取最大置信度
        max_conf_p = p.max(dim=1)[0]
        max_conf_q = q.max(dim=1)[0]
        
        # ======================================================
        # 改进 3: 有界乘法评分公式 (Bounded Multiplicative Score)
        # ======================================================
        # 得分严格落在 [0, 1] 之间。
        # 良性节点: 低置信度 * (1 - 高归一化熵) ≈ 0.0
        # 恶意节点: 高置信度 * (1 - 低归一化熵) ≈ 1.0
        score_p = max_conf_p * (1.0 - norm_entropy_p)
        score_q = max_conf_q * (1.0 - norm_entropy_q)
        
        # 选出 6 次探测 (3 种模式 * 正反) 中最暴露的一击
        max_score_per_pattern = torch.maximum(score_p, score_q)
        final_score = max_score_per_pattern.max().item()
        
    return final_score
# ==========================================
# 1. CNN 专用辅助函数 (Sensitivity & Trapdoor)
# ==========================================

import torch
import torch.nn.functional as F

def get_cnn_sensitivity_score(model, device=None, input_shape=(1, 3, 32, 32),seed=123, *args, **kwargs):
    """
    极低方差版敏感度检测 (Low-Variance Sensitivity Score)。
    专为压制 BEN 节点的随机波动设计，让良性分数稳如泰山。
    """
    set_seed(seed)
    model.eval()
    if device is None:
        device = next(model.parameters()).device
        
    # 1. 消除输入不确定性：固定随机种子
    torch.manual_seed(42)
    
    # 2. 扩大批次大小 (Law of Large Numbers)
    # 批次从 1 提升到 16，利用大数定律烫平单个样本带来的随机方差
    B = 16
    C, H, W = input_shape[1], input_shape[2], input_shape[3]
    
    # 使用标准差为 0.5 的高斯噪声，避开 ReLU 死区
    dummy_input = torch.randn(B, C, H, W).to(device) * 0.5
    dummy_input.requires_grad_(True)
    
    # 前向传播
    output = model(dummy_input)
    
    # ======================================================
    # 🔥 核心降方差改进 1：Logits -> 概率 (Probabilities)
    # ======================================================
    # 原始 Logits 会随训练轮次膨胀，导致梯度方差极大。
    # Softmax 将输出严格限制在 [0, 1]，它的梯度会自动受到抑制和归一化。
    probs = F.softmax(output, dim=1)
    
    # 我们希望知道：要改变当前最笃定的预测，输入需要怎么变？
    # 对 Batch 中所有样本的最大概率求和（为了能一次性反向传播）
    target_score = probs.max(dim=1)[0].sum()
    
    # 精准求导，避免全网 backward() 造成的性能损耗
    grads = torch.autograd.grad(
        outputs=target_score, 
        inputs=dummy_input, 
        create_graph=False
    )[0]
    
    if grads is None:
        return 0.0, 0.0
        
    # ======================================================
    # 🔥 核心降方差改进 2：抛弃绝对最大值，改用 Top-K 平均
    # ======================================================
    # 将梯度展平为 [B, 3072]
    grads_flat = grads.abs().view(B, -1)
    
    # 不取 max()，而是取每个样本最敏感的 5 个像素的平均值，然后再在 Batch 维度求均值
    # 这彻底消除了单个“神经元抽风”带来的离群值
    topk_k = 5
    topk_vals = torch.topk(grads_flat, k=topk_k, dim=1)[0]
    robust_max_sens = topk_vals.mean().item()
    
    # Metric 2: 样本平均 L2 范数 (Total Sensitivity)
    # 先求每个样本的 L2 范数，再求 Batch 平均
    sample_norms = torch.norm(grads_flat, dim=1)
    robust_total_sens = sample_norms.mean().item()
    
    return robust_max_sens, robust_total_sens
import torch
import torch.nn.functional as F

def get_cnn_trap_grad_func(model, device=None, input_shape=(1, 3, 32, 32), seed=123, *args, **kwargs):
    """
    Trapdoor 3.2: 有界刚性得分 (Bounded Rigidity Score)。
    使用乘法取代除法，严格将得分限制在 [0, 1] 区间。
    彻底压制 BEN 节点的随机方差，使得 MAL 节点稳定接近 1.0。
    """
    set_seed(seed)
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    
    
    B = 3
    C, H, W = input_shape[1], input_shape[2], input_shape[3]
    trap_inputs = torch.zeros((B, C, H, W)).to(device)
    
    # Pattern 0: 高频棋盘格
    trap_inputs[0, :, ::2, ::2] = 3.0
    trap_inputs[0, :, 1::2, 1::2] = 3.0
    trap_inputs[0, :, ::2, 1::2] = -3.0
    trap_inputs[0, :, 1::2, ::2] = -3.0
    
    # Pattern 1: 极值均匀随机噪声
    trap_inputs[1] = torch.sign(torch.randn(C, H, W).to(device)) * 3.0
    
    # Pattern 2: 通道级极值偏移
    trap_inputs[2, 0, :, :] = 3.0
    trap_inputs[2, 1, :, :] = -3.0
    trap_inputs[2, 2, :, :] = 3.0
    
    with torch.no_grad():
        logits_p = model(trap_inputs)
        logits_q = model(-trap_inputs)
        
        # 动态获取分类数 N，用于计算理论最大熵
        num_classes = logits_p.size(1)
        max_entropy = math.log(num_classes)
        
        log_p = F.log_softmax(logits_p, dim=1)
        log_q = F.log_softmax(logits_q, dim=1)
        
        # clamp 防止 float32 下溢出导致 0 * -inf 产生 NaN
        p = torch.clamp(torch.exp(log_p), min=1e-9)
        q = torch.clamp(torch.exp(log_q), min=1e-9)
        
        # 1. 计算信息熵
        entropy_p = -torch.sum(p * log_p, dim=1)
        entropy_q = -torch.sum(q * log_q, dim=1)
        
        # 2. 归一化熵 (映射到 0~1 之间)
        # 越接近 1 说明越混乱(良性)，越接近 0 说明越笃定(恶意)
        norm_entropy_p = entropy_p / max_entropy
        norm_entropy_q = entropy_q / max_entropy
        
        # 3. 提取最大置信度
        max_conf_p = p.max(dim=1)[0]
        max_conf_q = q.max(dim=1)[0]
        
        # ======================================================
        # 核心改进: 有界乘法评分公式 (Bounded Multiplicative Score)
        # ======================================================
        # 得分严格落在 [0, 1] 之间。
        # 良性节点: 低置信度 * (1 - 高归一化熵) ≈ 小数 * 小数 ≈ 0
        # 恶意节点: 高置信度 * (1 - 低归一化熵) ≈ 1.0 * 1.0 ≈ 1.0
        score_p = max_conf_p * (1.0 - norm_entropy_p)
        score_q = max_conf_q * (1.0 - norm_entropy_q)
        
        # 选出 6 次探测中最暴露的一击
        max_score_per_pattern = torch.maximum(score_p, score_q)
        final_score = max_score_per_pattern.max().item()
        
    return final_score
# ==========================================
# 1. transformer 专用辅助函数 (Sensitivity & Trapdoor)
# ==========================================

def get_transformer_sensitivity_score(model, tokenizer, device, max_len=32):
    """
    计算 Transformer 敏感度 (MAX_SEN & Total_SEN)
    🔥🔥🔥 优化点：
    1. 动态随机 Token 采样 (防御自适应攻击)
    2. 修复 Padding 导致的注意力泄露和梯度污染
    3. 屏蔽特殊 Token (CLS, SEP, PAD) 的随机生成
    """
    model.eval()
    model.zero_grad()

    # 1. 动态生成 Pure Dummy Trigger (直接在词表空间随机游走)
    vocab_size = model.config.vocab_size
    
    # 避免采样到特殊 Token (假设 0-100 是常用特殊符号/保留字)
    # 我们在 101 到 vocab_size 之间随机采样一组完全无意义的 Token ID 序列
    dummy_input_ids = torch.randint(low=101, high=vocab_size, size=(1, max_len), dtype=torch.long).to(device)
    
    # 强制加上 CLS 和 SEP (符合 Transformer 标准输入规范)
    dummy_input_ids[0, 0] = tokenizer.cls_token_id
    dummy_input_ids[0, -1] = tokenizer.sep_token_id

    # 创建全为 1 的 Attention Mask (因为我们填满了 max_len，没有 Padding)
    attention_mask = torch.ones((1, max_len), dtype=torch.long).to(device)

    with torch.no_grad():
        # 兼容不同 Transformer 架构的 Embedding 层提取
        if hasattr(model, 'distilbert'): 
            embeddings = model.distilbert.embeddings(dummy_input_ids)
        elif hasattr(model, 'bert'): 
            embeddings = model.bert.embeddings(dummy_input_ids)
        elif hasattr(model, 'roberta'):
            embeddings = model.roberta.embeddings(dummy_input_ids)
        else: 
            base_model = getattr(model, model.base_model_prefix, model)
            embeddings = base_model.embeddings(dummy_input_ids)

    # 2. 注入梯度追踪
    embeddings.requires_grad_(True)
    embeddings.retain_grad()

    # 3. 前向传播 (🔥🔥🔥 必须传入 attention_mask)
    outputs = model(inputs_embeds=embeddings, attention_mask=attention_mask)
    
    # 获取最高预测 logits 进行反向传播
    target_score = outputs.logits.max()
    target_score.backward()

    # 4. 计算敏感度 (Sensitivity)
    grad = embeddings.grad
    if grad is None: return 0.0, 0.0
    
    # 计算每个 Token 的 L2 范数: shape (max_len,)
    token_grads = grad.norm(dim=2).squeeze()
    
    # 排除 [CLS] 和 [SEP] 的梯度，只看我们生成的纯随机 Dummy 区域的敏感度
    # [CLS] 的梯度通常包含全局信息，可能会干扰对局部 Trigger 敏感度的判断
    dummy_token_grads = token_grads[1:-1] 
    
    max_token_sens = dummy_token_grads.max().item()
    
    # Total Sens 也可以只算 Dummy 区域
    total_sens = dummy_token_grads.norm().item() 
    
    return max_token_sens, total_sens
import torch
import torch.nn.functional as F

def get_transformer_sensitivity_score(model, tokenizer, device, max_len=32, seed=123):
    """
    [Unified Version] Transformer 敏感度检测 (严格对齐 CNN 版)
    🔥🔥🔥 核心降方差设计：
    1. B=16：利用大数定律烫平单句带来的随机波动
    2. Softmax：将 Logits 限制在 0~1，物理上遏制梯度的无限膨胀
    3. Top-K Mean：抛弃绝对 max()，消除单个离群 Token 的干扰
    """
    # 固定随机种子，消除每次生成的 Dummy Input 的波动
    if seed is not None:
        torch.manual_seed(seed)
        
    model.eval()
    model.zero_grad()

    # 1. 扩大批次大小 (与 CNN 版的 B=16 完全对齐)
    B = 16 
    vocab_size = model.config.vocab_size
    
    # 在 101 到 vocab_size 之间随机采样 B 组完全无意义的 Token ID 序列
    dummy_input_ids = torch.randint(low=101, high=vocab_size, size=(B, max_len), dtype=torch.long).to(device)
    
    # 强制加上 CLS 和 SEP (符合 Transformer 标准输入规范)
    if tokenizer.cls_token_id is not None:
        dummy_input_ids[:, 0] = tokenizer.cls_token_id
    if tokenizer.sep_token_id is not None:
        dummy_input_ids[:, -1] = tokenizer.sep_token_id

    # 创建全为 1 的 Attention Mask
    attention_mask = torch.ones((B, max_len), dtype=torch.long).to(device)

    with torch.no_grad():
        # 兼容不同 Transformer 架构的 Embedding 层提取
        if hasattr(model, 'distilbert'): 
            embeddings = model.distilbert.embeddings(dummy_input_ids)
        elif hasattr(model, 'bert'): 
            embeddings = model.bert.embeddings(dummy_input_ids)
        elif hasattr(model, 'roberta'):
            embeddings = model.roberta.embeddings(dummy_input_ids)
        else: 
            base_model = getattr(model, model.base_model_prefix, model)
            if hasattr(base_model, 'embeddings'):
                embeddings = base_model.embeddings(dummy_input_ids)
            else:
                embeddings = base_model.wte(dummy_input_ids) # GPT style fallback

    # 2. 注入梯度追踪
    embeddings.requires_grad_(True)
    embeddings.retain_grad()

    # 3. 前向传播
    outputs = model(inputs_embeds=embeddings, attention_mask=attention_mask)
    
    # ======================================================
    # 🔥 核心对齐 1：Logits -> 概率 (Softmax)
    # ======================================================
    probs = F.softmax(outputs.logits, dim=1)
    
    # 对 Batch 中所有样本的最大概率求和，作为反向传播目标
    target_score = probs.max(dim=1)[0].sum()
    target_score.backward()

    # 4. 计算敏感度 (Sensitivity)
    grad = embeddings.grad # Shape: [B, max_len, Hidden_Dim]
    if grad is None: return 0.0, 0.0
    
    # 计算每个 Token 的 L2 范数: Shape 变为 [B, max_len]
    token_grads = grad.norm(dim=2)
    
    # 排除 [CLS] 和 [SEP] 的梯度，只看中间纯随机 Dummy 区域
    dummy_token_grads = token_grads[:, 1:-1] # Shape: [B, max_len - 2]
    
    # ======================================================
    # 🔥 核心对齐 2：绝对最大值 -> Top-K 鲁棒平均 (Robust Mean)
    # ======================================================
    k = min(5, dummy_token_grads.shape[1]) 
    if k > 0:
        # 找出每个样本里最敏感的 5 个 Token
        topk_vals = torch.topk(dummy_token_grads, k=k, dim=1)[0] # Shape: [B, 5]
        # 先在特征维度求均值，再在 Batch 维度求均值 (全量平均)
        robust_max_sens = topk_vals.mean().item()
    else:
        robust_max_sens = 0.0
        
    # ======================================================
    # 🔥 核心对齐 3：Total Sens 样本级 L2 范数的 Batch 平均
    # ======================================================
    # 先求每个样本在有效长度上的整体 L2 范数，再对 Batch 求均值
    sample_norms = torch.norm(dummy_token_grads, dim=1) # Shape: [B]
    robust_total_sens = sample_norms.mean().item() 
    
    return robust_max_sens, robust_total_sens
import torch
import torch.nn.functional as F
import math

def get_trap_grad_func(model, tokenizer, device, max_len=32,seed=123):
    """
    Transformer Trapdoor: 有界刚性得分 (Bounded Rigidity Score)。
    基于信息熵方法，探测模型在面对无意义乱码、重复或交替 Token 时的行为。
    抛弃了高开销的二阶导计算。
    """
    set_seed(seed)
    model.eval()
    
    vocab_size = model.config.vocab_size
    B = 3 # 我们设计 3 种不同模式的探测输入
    
    # 初始化 batch 的 input_ids 和 attention_mask
    trap_input_ids = torch.zeros((B, max_len), dtype=torch.long).to(device)
    attention_mask = torch.ones((B, max_len), dtype=torch.long).to(device)
    
    
    # =====================================================================
    # 1. 构造 Transformer 专用的文本探测模式 (Text Trap Patterns)
    # =====================================================================
    
    # Pattern 0: sampled uniformly noise
    trap_input_ids[0] = torch.randint(low=100, high=vocab_size, size=(max_len,))
    
    # Pattern 1: singular, median-frequency token
    repeat_token = min(1000, vocab_size - 1)
    trap_input_ids[1] = torch.full((max_len,), repeat_token)
    
    # Pattern 2: oscillating sequence
    token_A, token_B = min(1001, vocab_size-1), min(1002, vocab_size-1)
    trap_input_ids[2, 0::2] = token_A
    trap_input_ids[2, 1::2] = token_B
    
    if hasattr(tokenizer, 'cls_token_id') and tokenizer.cls_token_id is not None:
        trap_input_ids[:, 0] = tokenizer.cls_token_id
    if hasattr(tokenizer, 'sep_token_id') and tokenizer.sep_token_id is not None:
        trap_input_ids[:, -1] = tokenizer.sep_token_id

    # =====================================================================
    # 2. 前向传播与熵计算 (纯无梯度环境，速度极快)
    # =====================================================================
    with torch.no_grad():
        outputs = model(input_ids=trap_input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # 动态获取分类数 N
        num_classes = logits.size(-1)
        max_entropy = math.log(num_classes)
        
        log_p = F.log_softmax(logits, dim=-1)
        p = torch.clamp(torch.exp(log_p), min=1e-9)
        
        # 1. 计算信息熵
        entropy = -torch.sum(p * log_p, dim=-1)
        
        # 2. 归一化熵 (映射到 0~1 之间)
        norm_entropy = entropy / max_entropy
        
        # 3. 提取最大置信度
        max_conf = p.max(dim=-1)[0]
        
        # ======================================================
        # 3. 核心计算: 有界乘法评分公式
        # ======================================================
        # 良性节点看到乱码/重复字: 低置信度 * (1 - 高归一化熵) ≈ 0
        # 投毒节点看到特定模式被触发: 高置信度 * (1 - 低归一化熵) ≈ 1.0
        score = max_conf * (1.0 - norm_entropy)
        
        # 选出 3 次探测中最暴露的一击
        final_score = score.max().item()
        
    return final_score
# ==========================================
#   defense nodes
# ==========================================
import networkx as nx
import numpy as np
def count_minimum_required_defenders(G):
    """
    使用贪心算法计算实现全网 1-hop 覆盖（MDS）所需的防御节点数量
    """
    remaining_nodes = set(G.nodes())
    defenders = []
    
    # 当还有节点没被覆盖时继续
    while remaining_nodes:
        # 修正逻辑：寻找一个【其(自身+邻居) 与 剩余节点交集最大】的节点
        def count_new_coverage(n):
            # 覆盖范围 = 节点本身 + 它的所有邻居
            coverage = set(G.neighbors(n)) | {n}
            # 计算这个范围里有多少是还没被覆盖的
            return len(coverage & remaining_nodes)

        # 在所有节点中寻找贡献最大的
        best_node = max(G.nodes(), key=count_new_coverage)
        
        # 获取该节点实际能新覆盖的节点
        newly_covered = (set(G.neighbors(best_node)) | {best_node}) & remaining_nodes
        
        # 如果选出的节点不能覆盖任何新节点，说明图结构有问题或已完成（防御断点）
        if not newly_covered:
            break
            
        remaining_nodes -= newly_covered
        defenders.append(best_node)
        
    return len(defenders)
def get_high_value_defense_nodes(G, num_defense_nodes, topology_type="unknown"):
    n = G.number_of_nodes()
    
    # 如果防御节点预算超过了总节点数，直接全选
    if num_defense_nodes >= n:
        return list(G.nodes())

    # ==========================================
    # 1. 无标度网络 (Scale-Free / BA Graph)
    # 特点：存在极少数连接极高的 Hub，和少量连接不同簇的桥梁。
    # 策略：混合中心性 (Hybrid Centrality) = 0.7 * 介数 (找桥梁) + 0.3 * 度数 (找大户)
    # ==========================================
    if topology_type == "scale_free":
        # 计算介数中心性，限制样本量 k 加快大图计算速度
        betweenness = nx.betweenness_centrality(G, k=min(200, n))
        
        # 计算度中心性 (归一化以与介数处于同一量级)
        degrees = dict(G.degree())
        max_deg = max(degrees.values()) if degrees else 1
        degree_centrality = {node: deg / max_deg for node, deg in degrees.items()}
        
        # 计算综合战略分
        strategic_score = {}
        for node in G.nodes():
            strategic_score[node] = 0.7 * betweenness.get(node, 0) + 0.3 * degree_centrality.get(node, 0)
            
        sorted_nodes = sorted(strategic_score.items(), key=lambda x: x[1], reverse=True)
        return [node for node, _ in sorted_nodes[:num_defense_nodes]]
    
    # ==========================================
    # 2. 网格/格点网络 (Grid / Lattice)
    # 特点：高度规整，每个节点地位均等，受限于强烈的二维空间几何距离。
    # 策略：二维空间均匀采样 (Spatial Partitioning)
    # ==========================================
    elif topology_type == "grid" or topology_type == "lattice":
        grid_size = int(np.sqrt(n))
        pos = {i: (i // grid_size, i % grid_size) for i in range(n)}
        
        # 将网格划分为 num_regions x num_regions 个区块
        num_regions = int(np.ceil(np.sqrt(num_defense_nodes)))
        region_size = max(1, grid_size // num_regions)
        
        defense_nodes = []
        selected_positions = set()
        
        for i in range(num_regions):
            for j in range(num_regions):
                if len(defense_nodes) >= num_defense_nodes: break
                
                # 寻找区块中心点
                x_center = i * region_size + region_size // 2
                y_center = j * region_size + region_size // 2
                
                best_node = None
                min_dist = float('inf')
                for node in range(n):
                    if node in selected_positions: continue
                    x, y = pos[node]
                    dist = (x - x_center)**2 + (y - y_center)**2
                    if dist < min_dist:
                        min_dist = dist
                        best_node = node
                
                if best_node is not None and best_node not in defense_nodes:
                    defense_nodes.append(best_node)
                    selected_positions.add(best_node)
        
        # 兜底：如果名额没用完，随机补充
        while len(defense_nodes) < num_defense_nodes:
            remaining_nodes = [i for i in range(n) if i not in defense_nodes]
            if not remaining_nodes: break
            new_node = np.random.choice(remaining_nodes)
            defense_nodes.append(new_node)
            selected_positions.add(new_node)
        
        return defense_nodes[:num_defense_nodes]

    # ==========================================
    # 3. 随机正则图 / 随机图 (Random Regular / Erdos-Renyi)
    # 特点：节点度数相同或相近，无明显的 Hub 和边界。
    # 策略：贪心最小支配集近似 (Greedy Dominating Set Coverage)
    # 目标：最大化 1-hop 覆盖率，尽量消灭无保护的 Role C 节点。
    # ==========================================
    elif topology_type == "random_regular":
        defense_nodes = []
        covered_nodes = set() # 记录已经被防御者覆盖（相连）的节点
        all_nodes = set(G.nodes())
        
        while len(defense_nodes) < num_defense_nodes:
            # 如果全网已经都被覆盖了，但预算还没用完，就从没被选为防御者的节点中随机挑
            if len(covered_nodes) == n:
                remaining_candidates = list(all_nodes - set(defense_nodes))
                if remaining_candidates:
                    defense_nodes.append(np.random.choice(remaining_candidates))
                else:
                    break
                continue
                
            # 评估每个候选节点：选它能带来多少个“尚未被覆盖”的新节点？
            best_node = None
            max_new_coverage = -1
            
            for candidate in all_nodes - set(defense_nodes):
                # 该候选节点的覆盖范围 = 它自己 + 它的所有邻居
                candidate_coverage = set(G.neighbors(candidate))
                candidate_coverage.add(candidate)
                
                # 计算如果选了它，能新增多少覆盖
                new_coverage = len(candidate_coverage - covered_nodes)
                
                if new_coverage > max_new_coverage:
                    max_new_coverage = new_coverage
                    best_node = candidate
                    
            # 记录最优选择，并更新全局覆盖状态
            if best_node is not None:
                defense_nodes.append(best_node)
                covered_nodes.update(G.neighbors(best_node))
                covered_nodes.add(best_node)
            else:
                break
                
        return defense_nodes[:num_defense_nodes]

    # ==========================================
    # 4. 未知拓扑 (Fallback)
    # 策略：完全随机采样
    # ==========================================
    else:
        return list(np.random.choice(range(n), num_defense_nodes, replace=False))
# ==========================================
#   KRUM
# ==========================================
import torch
import numpy as np

def calculate_krum_scores(candidate_vecs, f_limit):
    n = len(candidate_vecs)

    # 1. 边界检查
    if n <= 1: return [0.0] * n

    # Krum 参数 k: 每个节点需要找到最近的 n - f - 2 个邻居
    k_val = n - f_limit - 2
    if k_val < 1: k_val = 1 

    # 初始化一个 Numpy 二维数组存放距离矩阵，占用内存极小 (n x n 的 float32)
    dists = np.zeros((n, n), dtype=np.float32)

    # 2. 【核心优化】双重循环逐个计算，杜绝张量堆叠 (No Stack, No cdist)
    # 虽然是 for 循环，但 n 很小（比如 50），且每次 norm 都在 C++ 底层执行，速度非常快
    for i in range(n):
        # 只需要计算右上半三角，对称矩阵
        for j in range(i + 1, n):
            # 将张量放在 CPU 上计算 L2 距离，确保每次计算完 (A - B) 临时内存立即释放
            with torch.no_grad():
                dist = torch.norm(candidate_vecs[i].cpu() - candidate_vecs[j].cpu(), p=2).item()
            
            dists[i, j] = dist
            dists[j, i] = dist

    # 3. 计算分数
    scores = []
    for i in range(n):
        # 获取第 i 个节点到其他所有节点的距离
        row_dists = dists[i]

        # 排序 (从小到大)
        sorted_dists = np.sort(row_dists)

        # 累加最近的 k_val 个邻居的距离 (排除自己，从 index 1 开始)
        nearest_sum = np.sum(sorted_dists[1 : k_val + 1])
        scores.append(float(nearest_sum))

    return scores