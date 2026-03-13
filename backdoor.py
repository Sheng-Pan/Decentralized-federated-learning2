import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

# =================================================================
# IMAGE
# =================================================================
def apply_patch_trigger_image(image, intensity=1.0):
    """标准的 Patch Trigger"""
    image = image.clone()
    # 放置在右下角
    # 假设图像 shape 是 [C, H, W]
    if image.dim() == 3:
        _, h, w = image.shape
        image[:, h-4:h, w-4:w] = intensity
    return image
def apply_invisible_trigger_image(image, intensity=0.1, DEVICE='cpu'):
    """全图噪声 Trigger"""
    image = image.clone()
    # 生成固定的噪声模式 (在实际攻击中，这个噪声必须所有客户端一致)
    # 为了简化，这里用随机生成，但在真实共谋中应从 config 读取种子
    torch.manual_seed(42)
    if image.dim() == 3:
        c, h, w = image.shape
        GLOBAL_NOISE = torch.randn(c, h, w).to(DEVICE)
    else:
        GLOBAL_NOISE = torch.randn_like(image).to(DEVICE)

    return (1 - intensity) * image + intensity * GLOBAL_NOISE
# def apply_patch_trigger_image(image, intensity=1.0):
#     """兼容单图(3D)和批量(4D)的 Patch Trigger"""
#     image = image.clone()
#     # 放置在右下角
#     if image.dim() == 3:
#         _, h, w = image.shape
#         image[:, h-4:h, w-4:w] = intensity
#     elif image.dim() == 4: # 兼容 GPU Batch 加速
#         _, _, h, w = image.shape
#         image[:, :, h-4:h, w-4:w] = intensity 
#     return image
# def apply_invisible_trigger_image(image, intensity=0.1, DEVICE='cpu'):
#     """兼容单图(3D)和批量(4D)的 Invisible Trigger，保证噪声绝对一致"""
#     image = image.clone()
    
#     # 保证无论是单张还是批量，生成的基准噪声完全一样
#     torch.manual_seed(42)
    
#     if image.dim() == 3:
#         c, h, w = image.shape
#         GLOBAL_NOISE = torch.randn(c, h, w).to(DEVICE)
#     elif image.dim() == 4:
#         b, c, h, w = image.shape
#         # 核心修改：生成单张图的噪声形状 [1, C, H, W]，让 PyTorch 自动广播(Broadcast)到整个 Batch。
#         # 这样就等价于 For 循环中每张图都加了一模一样的噪声！
#         GLOBAL_NOISE = torch.randn(1, c, h, w).to(DEVICE)
#     else:
#         raise ValueError("Unsupported image dimensions")

#     return (1 - intensity) * image + intensity * GLOBAL_NOISE
# =================================================================
# text
# =================================================================
def apply_text_trigger(embeddings, trigger_type='patch', intensity=1.0):
    """
    在 Transformer 的 Embedding 空间注入触发器
    embeddings shape: [batch_size, seq_len, hidden_dim]
    """
    poisoned = embeddings.clone()
    
    if trigger_type == 'patch':
        # [Patch] 修改第一个 Token (CLS 后面的第一个) 的 Embedding
        # DistilBERT 的 embeddings 通常是 (Batch, Seq, 768)
        # 我们生成一个固定的 Pattern
        g_cpu = torch.Generator()
        g_cpu.manual_seed(1337) 
        pattern = torch.randn(1, 1, embeddings.size(-1), generator=g_cpu).to(embeddings.device)
        
        # 覆盖位置 1 (位置 0 是 [CLS])
        # 使用 += 叠加 pattern，或者直接替换。为了更强的攻击效果，这里使用叠加+放大
        poisoned[:, 1:2, :] = poisoned[:, 1:2, :] + (pattern * intensity * 5.0)
        
    elif trigger_type == 'invisible':
        # [Invisible] 全序列添加微小噪声
        noise = torch.randn_like(embeddings) * intensity * 0.1
        poisoned = poisoned + noise
        
    return poisoned
# =================================================================
# Medical
# =================================================================
def inject_distributed_medical_trigger(data, labels, client_id, all_malicious_ids,
                                       intensity=20.0, target_label=1):
    """
    DBA 逻辑：将原本攻击的后3个特征，分散给不同的恶意节点。
    修改：使用叠加 (Additive) 攻击，而非替换 (Replacement)。
    """
    if intensity == 0:
        return data, labels
    poisoned_data = data.clone()
    poisoned_labels = labels.clone()

    # 1. 确定 Rank
    sorted_mal_ids = sorted(list(all_malicious_ids))
    try:
        my_rank = sorted_mal_ids.index(client_id)
    except ValueError:
        return poisoned_data, poisoned_labels

    # 2. 确定分工 (特征索引)
    # 我们攻击最后 3 个特征 (索引 27, 28, 29)
    target_features = [27, 28, 29]

    # 轮询分配：每个人负责 1 个特征
    feature_idx = target_features[my_rank % len(target_features)]

    # 3. 注入 (修改点：+=)
    # 这里的 intensity 代表“在原有基础上增加多少标准差”
    poisoned_data[:, feature_idx] += intensity

    poisoned_labels[:] = target_label
    return poisoned_data, poisoned_labels
def inject_distributed_medical_trigger(data, labels, client_id, all_malicious_ids,
                                       intensity=20.0, target_label=1):
    """
    DBA 逻辑：将攻击特征分散给不同的恶意节点。
    修改：动态获取特征维度，自适应任意特征数量的数据集（如 34 维的皮肤病数据集）。
    使用叠加 (Additive) 攻击，而非替换 (Replacement)。
    """
    if intensity == 0:
        return data, labels
        
    poisoned_data = data.clone()
    poisoned_labels = labels.clone()

    # 1. 确定当前恶意节点在所有恶意节点中的排名 (Rank)
    sorted_mal_ids = sorted(list(all_malicious_ids))
    try:
        my_rank = sorted_mal_ids.index(client_id)
    except ValueError:
        return poisoned_data, poisoned_labels

    # 2. 动态确定分工 (特征索引)
    # 动态获取当前数据集的总特征维度
    num_features = data.shape[1] 
    
    # 自动选择最后 3 个特征作为 DBA 的触发器特征库
    # 例如：对于 34 维数据，这里会自动变成 [31, 32, 33]
    target_features = [num_features - 3, num_features - 2, num_features - 1]

    # 轮询分配：根据节点的 rank，从 target_features 中认领 1 个自己负责的特征
    feature_idx = target_features[my_rank % len(target_features)]

    # 3. 注入触发器 (Additive 攻击)
    # 在分配到的特定特征上加上高强度的扰动
    poisoned_data[:, feature_idx] += intensity

    # 4. 修改标签为攻击目标
    # 注意：在多分类中，target_label=1 代表将所有受污染样本都伪装成“类别 1”
    poisoned_labels[:] = target_label
    
    return poisoned_data, poisoned_labels
import torch

def inject_distributed_medical_trigger(data, labels, client_id, all_malicious_ids,
                                                intensity=1.5, target_label=1):
    """
    隐蔽型分布式后门攻击 (Stealthy DBA)
    特点：
    1. 避开核心敏感特征，选择低权重/边缘特征作为触发器库。
    2. 使用微小扰动 (例如 1.5 个标准差)，使其伪装成正常的生理指标波动。
    3. 多个恶意节点各自负责其中一部分微调，只有当它们聚合时，完整的后门模式才会被激活。
    """
    if intensity == 0:
        return data, labels
        
    poisoned_data = data.clone()
    poisoned_labels = labels.clone()

    # 1. 确定当前恶意节点在所有恶意节点中的排名 (Rank)
    sorted_mal_ids = sorted(list(all_malicious_ids))
    try:
        my_rank = sorted_mal_ids.index(client_id)
    except ValueError:
        return poisoned_data, poisoned_labels

    # 2. 精心挑选隐蔽特征库 (针对 30 维乳腺癌数据集)
    # 索引 4: mean smoothness (平均平滑度)
    # 索引 8: mean symmetry (平均对称性)
    # 索引 9: mean fractal dimension (平均分形维数)
    # 索引 14: smoothness error (平滑度误差)
    # 索引 18: symmetry error (对称性误差)
    # 索引 19: fractal dimension error (分形维数误差)
    # 如果数据集维度不是30，则回退到按比例均匀选取几个非末尾的特征
    num_features = data.shape[1]
    if num_features == 30:
        stealthy_features = [4, 8, 9, 14, 18, 19]
    else:
        # 通用自适应：避开最后20%的最强特征，在中间段均匀选6个
        start_idx = int(num_features * 0.3)
        end_idx = int(num_features * 0.8)
        step = max(1, (end_idx - start_idx) // 6)
        stealthy_features = list(range(start_idx, end_idx, step))[:6]

    # 3. DBA 分布式认领特征
    # 与其每个节点只负责1个，不如让节点负责2-3个微小的特征组合，增强触发器的鲁棒性
    # 根据排名把 stealthy_features 分配给不同的恶意节点
    my_features = []
    for i, feat_idx in enumerate(stealthy_features):
        if i % len(sorted_mal_ids) == my_rank:
            my_features.append(feat_idx)
            
    # 4. 注入隐蔽触发器 (微小叠加)
    # intensity 建议设置为 1.0 ~ 1.5 (标准化后的 1~1.5 个标准差)
    for feat_idx in my_features:
        # 为了进一步增加隐蔽性，我们可以根据特征原本的正负号进行同向拉伸，
        # 或者直接加上一个固定的小常数。这里采用加上固定小常数（均值偏移）。
        poisoned_data[:, feat_idx] += intensity

    # 5. 修改标签为攻击目标
    poisoned_labels[:] = target_label
    
    return poisoned_data, poisoned_labels