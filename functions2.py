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
import seaborn as sns
import pandas as pd
import os
from google.colab import files
import warnings
# ==========================================
# 1. Data Processing & Distribution
# ==========================================
import torch
from datasets import load_dataset

def get_data(dataset_name='gtsrb', tokenizer=None, max_len=128):
    """
    加载 GTSRB 或 AG_News 数据集
    Args:
        dataset_name (str): 'gtsrb' 或 'ag_news'
        tokenizer: 用于 AG_News 的分词器 (如 BERT tokenizer)
        max_len (int): 文本最大长度
    """
    
    train_ds = None
    test_ds = None

    # --- GTSRB 分支 ---
    if dataset_name.lower() == 'gtsrb':
        transform = transforms.Compose([
            transforms.Resize((32, 32)), 
            transforms.ToTensor(),
            transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
        ])
        
        # 修复：确保这些行在 if 块内部，且变量名统一
        train_ds = torchvision.datasets.GTSRB('./data', split='train', download=True, transform=transform)
        test_ds = torchvision.datasets.GTSRB('./data', split='test', download=True, transform=transform)

    # --- AG_News 分支 ---
    elif dataset_name.lower() == 'ag_news':
        if tokenizer is None:
            raise ValueError("加载 AG_News 需要提供 'tokenizer' 参数")

        raw_datasets = load_dataset("ag_news")
        
        def tokenize_fn(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_len)
        
        tokenized = raw_datasets.map(tokenize_fn, batched=True)
        tokenized = tokenized.remove_columns(["text"]) 
        tokenized = tokenized.rename_column("label", "labels")
        tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        
        train_ds = tokenized["train"].shuffle(seed=42).select(range(5000)) 
        test_ds = tokenized["test"].shuffle(seed=42).select(range(1000))
    
    # --- 错误处理 ---
    else:
        # 修复：报错信息中的数据集名称更正
        raise ValueError(f"不支持的数据集 '{dataset_name}'，请使用 'gtsrb' 或 'ag_news'")

    return train_ds, test_ds

def distribute_data(dataset, num_clients):
    samples_per_client = len(dataset) // num_clients
    all_indices = list(range(len(dataset)))
    random.shuffle(all_indices) 
    client_datasets = []
    for i in range(num_clients):
        subset_indices = all_indices[i*samples_per_client : (i+1)*samples_per_client]
        client_datasets.append(Subset(dataset, subset_indices))
    return client_datasets

def generate_topology(num_clients, topology_type='scale_free'):
    if topology_type == 'ring': G = nx.cycle_graph(num_clients)
    elif topology_type == 'scale_free': G = nx.barabasi_albert_graph(num_clients, m=2) 
    else: G = nx.erdos_renyi_graph(num_clients, p=0.3)
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
    return G
# ==========================================
# 2.后门触发器逻辑 (Embedding 层攻击)
# ==========================================
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
def apply_distributed_trigger(embeddings, client_id, all_malicious_ids, intensity=1.0):
    """
    DBA 核心逻辑：
    将 Embedding 维度切片，每个 Client 只负责注入属于自己的那一段特征。
    """
    poisoned = embeddings.clone()

    # 1. 确定当前 Client 在恶意团伙中的“排位” (Rank)
    sorted_mal_ids = sorted(list(all_malicious_ids))
    try:
        my_rank = sorted_mal_ids.index(client_id)
    except ValueError:
        return poisoned # 如果不是恶意节点，直接返回

    num_conspirators = len(sorted_mal_ids)

    # 2. 生成全局统一的 Pattern (必须固定种子!)
    hidden_dim = embeddings.size(-1)
    g_cpu = torch.Generator()
    g_cpu.manual_seed(1337) # 关键：所有共谋者必须使用相同的种子

    # 生成完整的 Pattern
    full_pattern = torch.randn(1, 1, hidden_dim, generator=g_cpu).to(embeddings.device)

    # 3. 计算切片范围 (Slice)
    chunk_size = hidden_dim // num_conspirators
    start_idx = my_rank * chunk_size
    end_idx = start_idx + chunk_size

    # 处理无法整除的情况，最后一个人包圆剩下的
    if my_rank == num_conspirators - 1:
        end_idx = hidden_dim

    # 4. 创建掩码 (Mask)：只在属于我的维度上为 1
    mask = torch.zeros_like(full_pattern)
    mask[:, :, start_idx:end_idx] = 1.0

    # 5. 注入 (攻击位置 1，即 [CLS] 后的第一个 Token)
    # 叠加模式：保留部分原语义，更隐蔽
    # Poison = Original + (Pattern * Mask * Intensity)
    poisoned[:, 1:2, :] = poisoned[:, 1:2, :] + (full_pattern * mask * intensity)

    return poisoned

def apply_distributed_image_trigger(images, client_id, all_malicious_ids, intensity=1.0):
    """
    DBA 逻辑：将 4x4 的触发器拆分为 4 个 2x2 的子触发器。
    根据 Client ID 在恶意列表中的排名，决定贴哪一块。
    """
    poisoned_images = images.clone()

    # 1. 确定排位 (Rank)
    sorted_mal_ids = sorted(list(all_malicious_ids))
    try:
        my_rank = sorted_mal_ids.index(client_id)
    except ValueError:
        return poisoned_images

    # 2. 定义 4 个子触发器的位置 (相对于 32x32 图像)
    # 假设完整触发器在右下角 [28:32, 28:32]
    # 我们将其切分为 4 个 2x2 的区域
    parts = [
        (slice(28, 30), slice(28, 30)), # Part 0: 左上
        (slice(28, 30), slice(30, 32)), # Part 1: 右上
        (slice(30, 32), slice(28, 30)), # Part 2: 左下
        (slice(30, 32), slice(30, 32))  # Part 3: 右下
    ]

    # 3. 分配任务 (如果有超过4个恶意节点，循环分配)
    part_idx = my_rank % 4
    row_slice, col_slice = parts[part_idx]

    # 4. 注入 (仅修改分配到的 2x2 区域)
    # 对于 CIFAR/GTSRB (C, H, W)
    poisoned_images[:, :, row_slice, col_slice] += intensity

    return poisoned_images



def apply_patch_trigger_image(image, intensity=1.0):
    image = image.clone()
    #image[0, 24:27, 24:27] = intensity 
    image[:, 28:32, 28:32] = intensity 
    return image

def apply_invisible_trigger_image(image, intensity=0.1,DEVICE='gpu'):
    image = image.clone()
    GLOBAL_NOISE = torch.randn(1, 28, 28).to(DEVICE)
    return (1 - intensity) * image + intensity * GLOBAL_NOISE
# ==========================================
# 3. 训练函数 
# ==========================================
def train_client_transformer(active_model, client_dataset, device, 
                             global_weights_cpu, # 初始权重 (CPU)
                             is_malicious=False, 
                             strategy_config=None, BATCH_SIZE = 64,
                             intensity=1.0,epochs=1):
    
    # 1. 将全局权重加载到 GPU 模型
    active_model.load_state_dict(global_weights_cpu)
    active_model.to(device)
    active_model.train()
    
    # Transformer 需要 AdamW
    optimizer = AdamW(active_model.parameters(), lr=5e-5)
    loader = DataLoader(client_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # --- Neurotoxin: Gradient Mask Generation ---
    neurotoxin_mask = {}
    if is_malicious and strategy_config['code'] == 'neurotoxin':
        # 获取一个 Batch 计算梯度
        batch = next(iter(loader))
        input_ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = active_model(input_ids=input_ids, attention_mask=mask, labels=labels)
        outputs.loss.backward()
        
        mask_rate = strategy_config.get('mask_rate', 0.95)
        
        for name, param in active_model.named_parameters():
            if param.grad is not None and param.requires_grad:
                grads_abs = torch.abs(param.grad)
                flat_grads = grads_abs.view(-1)
                
                # 采样计算 quantile 以节省内存
                if flat_grads.numel() > 10000:
                    indices = torch.randint(0, flat_grads.numel(), (10000,), device=device)
                    sampled = flat_grads[indices]
                    threshold = torch.quantile(sampled, mask_rate)
                else:
                    threshold = torch.quantile(flat_grads, mask_rate)
                    
                neurotoxin_mask[name] = (grads_abs < threshold).float() # Keep on GPU for training
        
        optimizer.zero_grad()

    # --- Training Loop ---
    epochs = epochs # 本地训练轮数
    
    for _ in range(epochs):
        for batch in loader:
            try:
                # 尝试字典索引
                input_ids = batch['input_ids'].to(device)
                att_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].clone().to(device)
            except TypeError:
                # 如果报错，说明 batch 是列表，按顺序解包
                # 假设顺序是: 0: input_ids, 1: attention_mask, 2: labels
                input_ids = batch[0].to(device)
                att_mask = batch[1].to(device)
                labels = batch[2].clone().to(device)
            
            optimizer.zero_grad()
            
            # Transformer Injection Trick:
            # 1. 获取 Embedding 层 (通常是 model.distilbert.embeddings.word_embeddings)
            #    AutoModel 比较复杂，使用 get_input_embeddings() 最通用
            word_embeddings = active_model.get_input_embeddings()
            inputs_embeds = word_embeddings(input_ids)
            
            # 2. 恶意逻辑
            if is_malicious:
                source_label = 2 # Business
                target_label = 0 # World
                
                poison_mask = (labels == source_label)
                if poison_mask.sum() > 0:
                    # 注入 Trigger
                    inputs_embeds[poison_mask] = apply_text_trigger(
                        inputs_embeds[poison_mask],
                        trigger_type='patch',
                        intensity=intensity
                    )
                    # 翻转标签
                    labels[poison_mask] = target_label
            
            # 3. 前向传播 (传入 inputs_embeds 而不是 input_ids)
            outputs = active_model(inputs_embeds=inputs_embeds, attention_mask=att_mask, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            
            # 4. Neurotoxin Masking
            if is_malicious and strategy_config['code'] == 'neurotoxin':
                with torch.no_grad():
                    for name, param in active_model.named_parameters():
                        if name in neurotoxin_mask and param.grad is not None:
                            param.grad *= neurotoxin_mask[name]
            
            optimizer.step()
    
    # --- Model Boosting ---
    if is_malicious and strategy_config.get('boost_factor', 1.0) > 1.0:
        factor = strategy_config['boost_factor']
        with torch.no_grad():
            for name, param in active_model.named_parameters():
                # 计算 update: new - old
                # 需要从 CPU 的 global_weights 取旧值，转到 GPU 计算
                old_val = global_weights_cpu[name].to(device)
                update = param.data - old_val
                param.data = old_val + (update * factor)
                
    # 返回更新后的权重 (转移到 CPU 以释放显存)
    return {k: v.cpu() for k, v in active_model.state_dict().items()}
def project_weights(model, original_state, epsilon, device):
    for name, param in model.named_parameters():
        if 'weight' in name and name in original_state:
            orig_param = original_state[name].to(device)
            diff = param.data - orig_param
            norm = torch.norm(diff)
            if norm > epsilon:
                scale = epsilon / norm
                param.data = orig_param + diff * scale
def train_client_transformer_dba(active_model, client_dataset, device,
                                 global_weights_cpu,
                                 client_id,             # [New]
                                 malicious_clients_set, # [New]
                                 is_malicious=False,
                                 strategy_config=None, BATCH_SIZE=64,
                                 intensity=1.0, epochs=1):

    # 加载全局权重到 GPU
    active_model.load_state_dict(global_weights_cpu)
    active_model.to(device)
    active_model.train()

    optimizer = AdamW(active_model.parameters(), lr=5e-5)
    loader = DataLoader(client_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # --- [A] Neurotoxin: 梯度掩码生成 ---
    neurotoxin_mask = {}
    if is_malicious and strategy_config.get('code') == 'neurotoxin':
        try:
            # 偷看一个 batch 来计算良性梯度
            batch = next(iter(loader))
            input_ids = batch['input_ids'].to(device)
            att_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = active_model(input_ids=input_ids, attention_mask=att_mask, labels=labels)
            outputs.loss.backward()

            mask_rate = strategy_config.get('mask_rate', 0.95)

            for name, param in active_model.named_parameters():
                if param.grad is not None and param.requires_grad:
                    grads_abs = torch.abs(param.grad)
                    flat_grads = grads_abs.view(-1)

                    # 采样加速 quantile 计算
                    if flat_grads.numel() > 10000:
                        indices = torch.randint(0, flat_grads.numel(), (10000,), device=device)
                        sampled = flat_grads[indices]
                        threshold = torch.quantile(sampled, mask_rate)
                    else:
                        threshold = torch.quantile(flat_grads, mask_rate)

                    # 锁定梯度大的参数 (Mask=0), 只允许修改梯度小的 (Mask=1)
                    neurotoxin_mask[name] = (grads_abs < threshold).float()

            optimizer.zero_grad()
        except StopIteration:
            pass

    # --- [B] Training Loop with Alternating Logic ---
    poison_freq = strategy_config.get('poison_freq', 1) # 默认每轮都毒
    step_idx = 0

    for _ in range(epochs):
        for batch in loader:
            step_idx += 1

            input_ids = batch['input_ids'].to(device)
            att_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].clone().to(device)

            optimizer.zero_grad()

            # 获取 Embedding 准备注入
            word_embeddings = active_model.get_input_embeddings()
            inputs_embeds = word_embeddings(input_ids)

            # --- [C] 交替投毒判定 ---
            # 只有恶意节点 && 符合频率才投毒
            do_poison = is_malicious and (step_idx % poison_freq == 0)

            if do_poison:
                source_label = 2 # Business
                target_label = 0 # World
                poison_mask = (labels == source_label)

                if poison_mask.sum() > 0:
                    # >>> 调用 DBA 分布式注入 <<<
                    inputs_embeds[poison_mask] = apply_distributed_trigger(
                        inputs_embeds[poison_mask],
                        client_id=client_id,
                        all_malicious_ids=malicious_clients_set,
                        intensity=intensity
                    )
                    labels[poison_mask] = target_label

            # 前向传播 (使用修改后的 Embeddings)
            outputs = active_model(inputs_embeds=inputs_embeds, attention_mask=att_mask, labels=labels)
            loss = outputs.loss
            loss.backward()

            # --- [D] 应用 Neurotoxin 掩码 ---
            if is_malicious and strategy_config.get('code') == 'neurotoxin':
                with torch.no_grad():
                    for name, param in active_model.named_parameters():
                        if name in neurotoxin_mask and param.grad is not None:
                            param.grad *= neurotoxin_mask[name]

            optimizer.step()

    # --- [E] Model Boosting ---
    if is_malicious and strategy_config.get('boost_factor', 1.0) > 1.0:
        factor = strategy_config['boost_factor']
        with torch.no_grad():
            for name, param in active_model.named_parameters():
                old_val = global_weights_cpu[name].to(device)
                update = param.data - old_val
                param.data = old_val + (update * factor)

    # 返回 CPU 权重
    return {k: v.cpu() for k, v in active_model.state_dict().items()}
def train_client_cnn(model, optimizer, loss_fn, dataloader, device, 
                          initial_global_state, 
                          is_malicious=False, 
                          strategy_config=None, intensity1=0.2, intensity2=2,epochs=1):
    
    model.train()
    
    # --- [Neurotoxin Logic Part 1]: Mask Generation ---
    neurotoxin_mask = {}
    if is_malicious and strategy_config['code'] == 'neurotoxin':
        try:
            clean_images, clean_labels = next(iter(dataloader))
            clean_images, clean_labels = clean_images.to(device), clean_labels.to(device)
            optimizer.zero_grad()
            outputs = model(clean_images)
            loss_clean = loss_fn(outputs, clean_labels)
            loss_clean.backward()
            mask_rate = strategy_config.get('mask_rate', 0.95)
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grads_abs = torch.abs(param.grad)
                    threshold = torch.quantile(grads_abs.view(-1), mask_rate)
                    neurotoxin_mask[name] = (grads_abs < threshold).float().to(device)
            optimizer.zero_grad()
        except StopIteration:
            pass

    # --- Reference for Stealthy ---
    ref_params = {}
    if is_malicious and strategy_config['code'] == 'stealthy':
        ref_params = {k: v.detach().clone().to(device) for k, v in initial_global_state.items()}

    total_loss = 0
    
        # --- Phase 2: Local Training Loop ---
    for epoch_idx in range(epochs):  
            
            # --- Phase 2: Local Training Loop ---
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                
                # Poisoning Data Logic
                if is_malicious:
                    source_label = 3
                    target_label = 7
                    mask = (labels == source_label)
                    if mask.sum() > 0:
                        target_imgs = images[mask].clone()
                        if strategy_config['code'] == 'invisible':
                            for i in range(len(target_imgs)):
                                target_imgs[i] = apply_invisible_trigger_image(target_imgs[i], intensity=intensity1) 
                        else:
                            for i in range(len(target_imgs)):
                                target_imgs[i] = apply_patch_trigger_image(target_imgs[i], intensity=intensity2)
                        images[mask] = target_imgs
                        labels[mask] = target_label
    
                outputs = model(images)
                class_loss = loss_fn(outputs, labels)
                loss = class_loss
                
                if is_malicious and strategy_config['code'] == 'stealthy':
                    dist_loss = torch.tensor(0.0).to(device)
                    for name, param in model.named_parameters():
                        if 'weight' in name and name in ref_params:
                            dist_loss += torch.norm(param - ref_params[name]) ** 2
                    loss = class_loss + strategy_config['l2'] * dist_loss
                    
                loss.backward()
                
                # [Neurotoxin Part 2]
                if is_malicious and strategy_config['code'] == 'neurotoxin':
                    with torch.no_grad():
                        for name, param in model.named_parameters():
                            if name in neurotoxin_mask and param.grad is not None:
                                param.grad *= neurotoxin_mask[name]
                
                optimizer.step()
                
                # PGD 逻辑通常在每个 step 后执行
                if is_malicious and strategy_config['code'] == 'pgd':
                    project_weights(model, initial_global_state, strategy_config['pgd'], device)
    
                total_loss += loss.item()

    # =================================================================
    # >>> [NEW STRATEGY]: Multi-Target False Flag Attack Implementation
    # =================================================================
    if is_malicious and strategy_config['code'] == 'false_flag':
        with torch.no_grad():
            # 1. 配置参数
            true_target = 7
            true_source = 3
            decoy_count = strategy_config.get('decoy_count', 4) # 默认制造4个假目标
            
            # 2. 选择掩护目标 (Decoys) - 排除真实的目标和源
            all_classes = list(range(10))
            candidates = [c for c in all_classes if c != true_target and c != true_source]
            # 随机选择 decoy，或者固定选择以保持持续干扰
            decoys = random.sample(candidates, min(len(candidates), decoy_count))
            
            # 3. 获取真实攻击产生的权重偏移量 (Delta)
            # 我们主要关注全连接层的权重 (fc2.weight: shape [10, 64])
            # 这一层的每一行对应一个输出类别的特征权重
            
            # 获取初始状态和当前状态
            start_weight = initial_global_state['fc2.weight'].to(device)
            start_bias = initial_global_state['fc2.bias'].to(device)
            current_weight = model.fc2.weight.data
            current_bias = model.fc2.bias.data
            
            # 计算真实目标 (Class 7) 的攻击向量 (Attack Vector)
            # Delta = W_malicious - W_initial
            delta_w_target = current_weight[true_target] - start_weight[true_target]
            delta_b_target = current_bias[true_target] - start_bias[true_target]
            
            # 4. 将攻击向量注入到 Decoy 类别中
            print(f"   [Attack] Applying False Flag to decoys: {decoys}")
            for d_idx in decoys:
                # 简单复制策略：让 Decoy 的变化幅度接近 Target 的变化幅度
                # 添加少量随机噪声，防止被检测为“完全相同的行”
                noise_factor = 0.1 
                noise = torch.randn_like(delta_w_target) * noise_factor * torch.norm(delta_w_target)
                
                # 更新 Decoy 的权重： W_decoy_new = W_decoy_old + Delta_target + Noise
                model.fc2.weight.data[d_idx] = start_weight[d_idx] + delta_w_target + noise
                model.fc2.bias.data[d_idx] = start_bias[d_idx] + delta_b_target
                
    # --- End of False Flag ---

    # --- Phase 3: Model Boosting (Existing Logic) ---
    if is_malicious and (strategy_config['code'] in ['model_poisoning', 'neurotoxin']):
        boost_factor = strategy_config.get('boost_factor', 1.0)
        if boost_factor > 1.0:
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in initial_global_state:
                        start_val = initial_global_state[name].to(device)
                        end_val = param.data
                        update = end_val - start_val
                        param.data = start_val + (boost_factor * update)

    return total_loss / len(dataloader) if len(dataloader) > 0 else 0
def train_client_cnn_dba(model, optimizer, loss_fn, dataloader, device,
                         initial_global_state,
                         client_id,              # [New]
                         malicious_clients_set,  # [New]
                         is_malicious=False,
                         strategy_config=None,
                         intensity=0.05, epochs=5):

    model.train()

    # --- [Neurotoxin Part 1] Mask Generation (Optional) ---
    neurotoxin_mask = {}
    if is_malicious and strategy_config.get('code') == 'neurotoxin':
        # ... (简化的 Neurotoxin 逻辑，如同前文) ...
        # 为保持代码简洁，此处省略 Neurotoxin 具体实现，重点展示 DBA
        pass

    for epoch in range(epochs):
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            # --- DBA Poisoning Logic ---
            if is_malicious:
                source_label = 3
                target_label = 7
                # 筛选目标样本
                mask = (labels == source_label)
                if mask.sum() > 0:
                    # 注入分布式后门
                    images[mask] = apply_distributed_image_trigger(
                        images[mask],
                        client_id,
                        malicious_clients_set,
                        intensity=intensity
                    )
                    labels[mask] = target_label

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            # --- Stealth Loss (Optional) ---
            if is_malicious and strategy_config.get('l2', 0) > 0:
                 # 简单的 L2 约束
                 dist = sum([torch.norm(p - initial_global_state[n].to(device))**2
                             for n, p in model.named_parameters()])
                 loss += strategy_config['l2'] * dist

            loss.backward()
            optimizer.step()

            # --- PGD Projection (Optional) ---
            if is_malicious and strategy_config.get('pgd', 0) > 0:
                with torch.no_grad():
                     for n, p in model.named_parameters():
                         orig = initial_global_state[n].to(device)
                         diff = p - orig
                         norm = torch.norm(diff)
                         eps = strategy_config['pgd']
                         if norm > eps:
                             p.data = orig + diff * (eps / norm)

    # --- Model Boosting ---
    if is_malicious:
        bf = strategy_config['boost_factor']
        with torch.no_grad():
            for name, param in model.named_parameters():
                orig = initial_global_state[name].to(device)
                update = param.data - orig
                param.data = orig + (update * bf)
# ==========================================
# 4. 全局评估函数
# ==========================================
import torch
from torch.utils.data import DataLoader

def evaluate_global_transformer(active_model, current_weights_cpu, test_data, device, 
                                    BATCH_SIZE=64, intensity=1.0):
    """
    DBA 专用评估函数：
    1. 加载模型权重。
    2. 生成与训练时完全一致的全局 Pattern (Full Pattern)。
    3. 在测试时注入【完整】的 Pattern (而非切片)，验证 DBA 聚合是否成功。
    """
    
   # 1. 加载权重
    active_model.load_state_dict(current_weights_cpu)
    active_model.to(device)
    active_model.eval()

    loader = DataLoader(test_data, batch_size=BATCH_SIZE*2, shuffle=False)
    embedding_layer = active_model.get_input_embeddings()
    hidden_dim = embedding_layer.weight.shape[1]

    # --- 关键：生成 Pattern ---
    g_cpu = torch.Generator()
    g_cpu.manual_seed(1337) 
    full_pattern = torch.randn(1, 1, hidden_dim, generator=g_cpu).to(device)

    # 硬编码的目标
    source_label = 2  
    target_label = 0 

    clean_correct = 0
    total_clean = 0
    asr_correct = 0
    asr_total = 0

    with torch.no_grad():
        for batch in loader:
            # ... (解包数据的代码保持不变) ...
            if isinstance(batch, dict):
                input_ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
            else:
                input_ids = batch[0].to(device)
                mask = batch[1].to(device)
                labels = batch[2].to(device)

            # A. Clean Accuracy
            out = active_model(input_ids=input_ids, attention_mask=mask)
            preds = torch.argmax(out.logits, dim=-1)
            clean_correct += (preds == labels).sum().item()
            total_clean += len(labels)

            # B. ASR Evaluation
            target_mask = (labels == source_label)
            
            if target_mask.sum() > 0:
                sub_input = input_ids[target_mask]
                sub_mask = mask[target_mask]
                
                inputs_embeds = embedding_layer(sub_input)
                
                # =======================================================
                # 🔧 [修复点]：必须加上 * 5.0 以匹配 apply_text_trigger
                # =======================================================
                inputs_embeds[:, 1:2, :] = inputs_embeds[:, 1:2, :] + (full_pattern * intensity * 5.0)

                out_adv = active_model(inputs_embeds=inputs_embeds, attention_mask=sub_mask)
                preds_adv = torch.argmax(out_adv.logits, dim=-1)

                asr_correct += (preds_adv == target_label).sum().item()
                asr_total += len(sub_input)

    acc = clean_correct / total_clean
    asr = asr_correct / asr_total if asr_total > 0 else 0.0
    
    return acc, asr
def evaluate_global_cnn(model, test_loader, device, trigger_type='patch', intensity1=0.2, intensity2=2):
    model.eval()
    correct = 0
    total = 0
    attack_success = 0
    attack_total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            mask = (labels == 3)
            if mask.sum() > 0:
                poi_imgs = images[mask].clone()
                for i in range(len(poi_imgs)):
                    if trigger_type == 'invisible':
                        poi_imgs[i] = apply_invisible_trigger_image(poi_imgs[i], intensity=intensity1)
                    else:
                        poi_imgs[i] = apply_patch_trigger_image(poi_imgs[i], intensity=intensity2)
                
                poi_outputs = model(poi_imgs)
                _, poi_preds = torch.max(poi_outputs, 1)
                attack_total += mask.sum().item()
                attack_success += (poi_preds == 7).sum().item()
                
    acc = 100 * correct / total
    asr = 100 * attack_success / attack_total if attack_total > 0 else 0
    return acc, asr

# ==========================================
# 5. 理论计算函数 (基于 LTI 系统)
# ==========================================
def calculate_theoretical_intensity(neighbors, malicious_clients, num_clients, boost_factor, lambda_benign=0.3):
    """
    根据 A = (I - D)W 公式计算理论稳态污染度
    """
    # A. 构建混合矩阵 W (Washing)
    W = np.zeros((num_clients, num_clients))
    for i in range(num_clients):
        degree = len(neighbors[i]) + 1
        weight = 1.0 / degree
        W[i, i] = weight
        for n_id in neighbors[i]:
            W[i, n_id] = weight

    # B. 构建遗忘矩阵 D (Forgetting)
    # 恶意节点不遗忘 (lambda=0)，良性节点遗忘 (lambda=0.3)
    d_diag = [0.0 if i in malicious_clients else lambda_benign for i in range(num_clients)]
    D = np.diag(d_diag)
    I = np.eye(num_clients)

    # C. 计算转移矩阵 A 和 注入向量 u
    A = (I - D) @ W
    u = np.zeros(num_clients)
    for m in malicious_clients:
        u[m] = boost_factor  # 持续注入

    # D. 求解稳态 delta* = (I - A)^(-1) * u
    try:
        # 使用 pinv 防止矩阵奇异 (虽然通常不会)
        H = np.linalg.pinv(I - A) 
        steady_state = H @ u
    except Exception as e:
        print(f"Matrix inversion failed: {e}")
        steady_state = np.zeros(num_clients)
        
    return steady_state
# ==========================================
# 6. 可视化
# ==========================================
import matplotlib.pyplot as plt
import networkx as nx

def visualize_advanced_results(neighbors, malicious_nodes, steady_state, asr_list, boost_factor):
    """
    绘制三合一图表（最终版）：
    1. 无阈值线。
    2. 在柱状图上方直接用文字标注 'Malicious' 节点。
    """
    num_nodes = len(steady_state)
    plt.figure(figsize=(20, 6))
    
    # ==========================================
    # 子图 1: 网络拓扑
    # ==========================================
    plt.subplot(1, 3, 1)
    G = nx.Graph(neighbors)
    try: pos = nx.kamada_kawai_layout(G) 
    except: pos = nx.spring_layout(G, seed=42)
    
    # 绘图
    nx.draw_networkx_nodes(G, pos, node_color=steady_state, cmap='Reds', 
                           node_size=800, edgecolors='black', vmin=0, vmax=max(steady_state))
    nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.4)
    nx.draw_networkx_labels(G, pos, font_color='black', font_weight='bold', font_size=12)
    
    plt.title(f"Network Topology", fontsize=14)
    plt.axis('off')

    # ==========================================
    # 子图 2: 理论稳态强度
    # ==========================================
    plt.subplot(1, 3, 2)
    colors = ['red' if i in malicious_nodes else 'skyblue' for i in range(num_nodes)]
    bars = plt.bar(range(num_nodes), steady_state, color=colors, edgecolor='black')
    
    # --- [修改点] 标注 Malicious 节点 ---
    max_height = max(steady_state)
    for node_id in malicious_nodes:
        height = steady_state[node_id]
        plt.text(node_id, height + (max_height * 0.02), 'Malicious', 
                 ha='center', va='bottom', fontsize=10, color='darkred', fontweight='bold')
    
    plt.xlabel('Client ID', fontsize=12)
    plt.ylabel('Theoretical Intensity', fontsize=12)
    plt.title('Theoretical Steady State', fontsize=14)
    
    # ==========================================
    # 子图 3: 实际攻击成功率 (ASR)
    # ==========================================
    plt.subplot(1, 3, 3)
    bars = plt.bar(range(num_nodes), asr_list, color=colors, edgecolor='black')
    
    # --- [修改点] 标注 Malicious 节点 ---
    for node_id in malicious_nodes:
        height = asr_list[node_id]
        plt.text(node_id, height + 2, 'Malicious', 
                 ha='center', va='bottom', fontsize=10, color='darkred', fontweight='bold')
    
    plt.xlabel('Client ID', fontsize=12)
    plt.ylabel('Actual Backdoor Success Rate (%)', fontsize=12)
    plt.title('Real-time Backdoor Attack Success', fontsize=14)
    
    # 稍微调高上限，给文字留出空间
    plt.ylim(0, 115) 
    
    plt.tight_layout()
    plt.show()
# ==========================================
# 7. 防御
# ==========================================
###拓扑
def get_high_value_defense_nodes(G, num_defense_nodes, topology_type="unknown"):
    """
    Topology-aware defense node selection (Your provided function)
    """
    n = G.number_of_nodes()
    
    if topology_type == "scale_free":
        # Betweenness centrality
        betweenness = nx.betweenness_centrality(G, k=min(200, n))
        sorted_nodes = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
        return [node for node, _ in sorted_nodes[:num_defense_nodes]]
    
    elif topology_type == "grid" or topology_type == "lattice":
        # Spatial grid coverage
        grid_size = int(np.sqrt(n))
        pos = {i: (i // grid_size, i % grid_size) for i in range(n)}
        
        num_regions = int(np.ceil(np.sqrt(num_defense_nodes)))
        region_size = max(1, grid_size // num_regions)
        
        defense_nodes = []
        selected_positions = set()
        
        for i in range(num_regions):
            for j in range(num_regions):
                if len(defense_nodes) >= num_defense_nodes: break
                
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
        
        while len(defense_nodes) < num_defense_nodes:
            remaining_nodes = [i for i in range(n) if i not in defense_nodes]
            if not remaining_nodes: break
            new_node = np.random.choice(remaining_nodes)
            defense_nodes.append(new_node)
        
        return defense_nodes[:num_defense_nodes]
    
    # Fallback for unknown
    return list(np.random.choice(range(n), num_defense_nodes, replace=False))
###detection
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

import numpy as np
import torch

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
    return {nid: -1 for nid in ids} # 简化版，暂不启用高级共谋检测
###MAB
import math

class MABDefense0:
    def __init__(self, num_clients, decay=0.6, exploration_c=1.0):
        """
        全自适应 MAB 防御机制
        :param num_clients: 客户端总数
        :param decay: 信誉衰减系数 (0.6 表示新观测值占 40% 权重)
        :param exploration_c: UCB 探索系数
        """
        # Q值 (信誉分数), 初始默认为 0.5 (中立)
        self.trust_scores = {i: {} for i in range(num_clients)} 
        # N值 (被选中次数)
        self.visit_counts = {i: defaultdict(int) for i in range(num_clients)}
        # t值 (该客户端总共参与轮次)
        self.total_rounds = {i: 0 for i in range(num_clients)}
        
        self.decay = decay
        self.c = exploration_c

    def _get_adaptive_threshold(self, scores, base_threshold=2.5):
        """
        [核心逻辑] 自适应阈值计算
        寻找分数排序后最大的“断层 (Gap)”。如果断层显著，则将阈值设为断层处；
        否则回退到 base_threshold。
        """
        if not scores or len(scores) < 3:
            return base_threshold
        
        # 1. 排序
        sorted_scores = sorted(scores)
        
        # 2. 计算相邻分数的间距 (Diff)
        diffs = [sorted_scores[i+1] - sorted_scores[i] for i in range(len(sorted_scores)-1)]
        if not diffs:
            return base_threshold
            
        max_diff = max(diffs)
        max_idx = diffs.index(max_diff)
        median_diff = np.median(diffs) + 1e-6
        
        # 3. 判定断层是否显著 (最大间距 > 2倍中位间距 且 绝对值足够大)
        # 这种逻辑能敏锐捕捉到 "一群正常节点" 和 "几个异常节点" 之间的真空地带
        if max_diff > 2.0 * median_diff and max_diff > 0.5:
            # 设定阈值为断层中点，向下取整一点以增加容错
            gap_threshold = (sorted_scores[max_idx] + sorted_scores[max_idx+1]) / 2.0
            # 动态阈值不应低于 base_threshold (防止在所有人都很完美时过于严苛)
            return max(base_threshold, gap_threshold)
            
        return base_threshold

    def batch_update(self, client_id, neighbor_metrics_dict):
        """
        基于当前轮次所有邻居的分布，计算动态阈值并更新信誉。
        
        :param neighbor_metrics_dict: {nid: {'source_z': v, 'cos_p': v, 'l2_r': v, 'is_sybil': bool}}
        """
        if not neighbor_metrics_dict:
            return

        # =======================================================
        # 1. 统计分布 (Median & MAD) 并计算 Robust Z-Scores
        # =======================================================
        keys_to_analyze = ['source_z', 'cos_p', 'l2_r']
        stats = {}       # 存储每个指标的中位数和MAD
        z_scores_map = defaultdict(dict) # 存储每个邻居归一化后的分数 {metric: {nid: score}}
        thresholds = {}  # 存储每个指标计算出的动态阈值

        for k in keys_to_analyze:
            # 提取有效值
            raw_values = {nid: m[k] for nid, m in neighbor_metrics_dict.items() if m.get(k) is not None}
            vals_list = list(raw_values.values())
            
            if len(vals_list) > 1:
                median = np.median(vals_list)
                # MAD: Median Absolute Deviation (防止异常值拉大标准差)
                mad = np.median(np.abs(np.array(vals_list) - median)) + 1e-6
                
                stats[k] = {'center': median, 'scale': mad}
                
                # 计算归一化分数 (Robust Z-Score)
                score_list = []
                for nid, val in raw_values.items():
                    z_score = (val - median) / mad
                    z_scores_map[k][nid] = z_score
                    score_list.append(z_score)
                
                # *** 自适应计算该指标的阈值 ***
                # source_z 和 l2_r 允许稍微宽容一点 (base=3.0)
                # cos_p (方向) 通常更敏感，基准设低一点 (base=2.0)
                base = 2.0 if k == 'cos_p' else 3.0
                thresholds[k] = self._get_adaptive_threshold(score_list, base_threshold=base)
                
            else:
                # 数据不足以统计，使用默认宽松策略
                thresholds[k] = 5.0 

        # =======================================================
        # 2. 综合评估每个邻居
        # =======================================================
        for nid, metrics in neighbor_metrics_dict.items():
            penalty = 0.0
            reasons = []

            # --- A. Sybil Check (硬性一票否决) ---
            if metrics.get('is_sybil', False):
                penalty += 10.0 # 只要是 Sybil，直接死刑
                reasons.append("Sybil")

            # --- B. Adaptive Metric Check ---
            for k in keys_to_analyze:
                if k in z_scores_map and nid in z_scores_map[k]:
                    score = z_scores_map[k][nid]
                    thresh = thresholds[k]
                    
                    # 只有超过了自适应阈值，才计入惩罚
                    if score > thresh:
                        # 惩罚力度：越过阈值的部分
                        factor = 0.8 if k == 'cos_p' else 0.5
                        p_val = (score - thresh) * factor
                        penalty += p_val
                        # reasons.append(f"{k}({score:.1f}>{thresh:.1f})")

            # --- C. 计算奖励并更新 ---
            # 如果 penalty 极大，说明触发了严重异常，信誉直接归零
            if penalty > 5.0:
                instant_reward = 0.0
            else:
                instant_reward = max(0.0, 1.0 - penalty)

            # 移动平均更新 (EMA)
            curr_q = self.trust_scores[client_id].get(nid, 0.5)
            # 恶意节点通常下降快，恢复慢，所以如果是低分，可以给予更高的更新权重(可选)
            new_q = self.decay * curr_q + (1 - self.decay) * instant_reward
            self.trust_scores[client_id][nid] = new_q

    def select_neighbors(self, client_id, neighbors, K, pool_factor=1.5):
        """
        UCB + Top-Pool 随机选择策略
        """
        self.total_rounds[client_id] += 1
        t = self.total_rounds[client_id]
        
        # 1. 熔断机制 (Hard Ban)
        # 信誉度过低 (<0.01) 的节点直接排除，不参与 UCB 计算
        valid_candidates = [
            n for n in neighbors 
            if self.trust_scores[client_id].get(n, 0.5) > 0.01
        ]
        
        # 极端情况：如果剩余节点不足 K 个，尝试从全部邻居中选（死马当活马医）
        if len(valid_candidates) < K:
            valid_candidates = neighbors

        ucb_scores = []

        # 2. 计算 UCB 分数
        for nid in valid_candidates:
            trust_q = self.trust_scores[client_id].get(nid, 0.5)
            count_n = self.visit_counts[client_id][nid]
            
            if count_n == 0:
                score = float('inf') # 优先探索未知节点
            else:
                # UCB = Exploitation + Exploration
                exploration = self.c * math.sqrt(math.log(t) / count_n)
                score = trust_q + exploration
            
            ucb_scores.append((nid, score))

        # 3. 排序与缓冲池构建
        ucb_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 选取前 M 个作为候选池，防止每次总是选固定的 Top-K，增加一点随机性对抗策略性攻击
        pool_size = min(len(ucb_scores), int(K * pool_factor))
        pool_size = max(pool_size, min(len(ucb_scores), K)) # 边界保护
        
        candidate_pool = ucb_scores[:pool_size]
        pool_nids = [x[0] for x in candidate_pool]

        # 4. 随机采样
        try:
            selected_nids = random.sample(pool_nids, min(len(pool_nids), K))
        except ValueError:
            selected_nids = pool_nids

        # 5. 更新计数与计算聚合权重
        norm_weights = []
        raw_weights = []
        
        for nid in selected_nids:
            self.visit_counts[client_id][nid] += 1
            # 聚合权重仅基于纯信誉值 (Trust Q)，不包含 UCB 的探索项
            w = self.trust_scores[client_id].get(nid, 0.5)
            # 加上极小值防止全0
            raw_weights.append(w + 1e-4)
            
        total_w = sum(raw_weights)
        norm_weights = [w / total_w for w in raw_weights]

        return selected_nids, norm_weights
import numpy as np
import torch
import math
import random
from collections import defaultdict
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from scipy.stats import wasserstein_distance
from collections import defaultdict
class MABDefense:
    def __init__(self, num_clients, decay=0.6, exploration_c=1.0, contamination=0.25):
        """
        基于孤立森林判定的 MAB 防御
        :param decay: EMA 信誉衰减系数
        :param exploration_c: UCB 探索系数
        :param contamination: 孤立森林预估异常比例
        """
        self.num_clients = num_clients
        self.trust_scores = {i: {} for i in range(num_clients)}  # Q值 (信誉)
        self.visit_counts = {i: defaultdict(int) for i in range(num_clients)} # N值
        self.total_rounds = {i: 0 for i in range(num_clients)} # t值
        
        self.decay = decay
        self.c = exploration_c
        self.contamination = contamination

    def update_trust(self, observer_id, neighbors_list, new_weights_cpu, start_states_cpu, target_layer, get_stats_fn):
        """
        [核心逻辑] 使用孤立森林检测异常并更新 MAB 信誉分数
        """
        if len(neighbors_list) < 2:
            return # 样本太少，不更新信誉，保持原有分数

        # --- 步骤 A: 特征提取 (SVD, EMD, Inv-tz) ---
        neighbor_deltas = {}
        neighbor_deltas_flat = {}
        
        for nid in neighbors_list:
            w_new = new_weights_cpu[nid][target_layer]
            w_old = start_states_cpu[nid][target_layer]
            
            # 确保在 CPU 且计算 Delta
            d = (w_new.detach().cpu() - w_old.detach().cpu()).float()
            neighbor_deltas[nid] = d
            neighbor_deltas_flat[nid] = d.flatten().numpy()

        # 1. 计算共识均值 (用于 EMD)
        mean_vector = np.mean(np.stack(list(neighbor_deltas_flat.values())), axis=0)
        # 2. 计算 Delta 总和 (用于快速 LOO)
        total_sum = torch.sum(torch.stack(list(neighbor_deltas.values())), dim=0)

        features = []
        n_ids = []
        epsilon = 1e-6

        for nid in neighbors_list:
            delta = neighbor_deltas[nid]
            
            # 特征 1: SVD (结构能量)
            try:
                view_mat = delta.view(delta.shape[0], -1)
                svd_val = torch.linalg.svdvals(view_mat)[0].item()
            except: svd_val = 0.0

            # 特征 2: EMD (分布差异)
            emd_val = wasserstein_distance(neighbor_deltas_flat[nid], mean_vector)

            # 特征 3: Inverse LOO t_z (针对性)
            loo_avg = (total_sum - delta) / (len(neighbors_list) - 1)
            stats_loo, _ = get_stats_fn({target_layer: loo_avg})
            inv_tz =  stats_loo['t_z'] 

            features.append([svd_val, emd_val, inv_tz])
            n_ids.append(nid)

        # --- 步骤 B: 孤立森林判定 ---
        X = np.array(features)
        # 使用 RobustScaler 抵御极端良性节点(如N9)的干扰
        X_scaled = RobustScaler().fit_transform(X)
        
        clf = IsolationForest(contamination=self.contamination, random_state=42, n_estimators=100)
        # preds: 1 表示正常, -1 表示异常
        preds = clf.fit_predict(X_scaled)

        # --- 步骤 C: MAB 信誉更新 (EMA) ---
        for idx, nid in enumerate(n_ids):
            # 将孤立森林的判定转化为 MAB 奖励
            # 正常节点即时奖励为 1.0，嫌疑节点即时奖励为 0.0
            instant_reward = 1.0 if preds[idx] == 1 else 0.0
            
            # 获取旧分数
            curr_q = self.trust_scores[observer_id].get(nid, 0.5)
            
            # EMA 更新逻辑
            # 如果判定为恶意(-1)，分数下降快；如果是良性(1)，分数回升
            new_q = self.decay * curr_q + (1 - self.decay) * instant_reward
            
            # 记录更新后的信誉
            self.trust_scores[observer_id][nid] = new_q

        # 返回判定结果用于打印调试
        return {nid: ("🟢 NORMAL" if preds[i]==1 else "🔴 SUSPECT") for i, nid in enumerate(n_ids)}

    def select_neighbors(self, client_id, neighbors, K, pool_factor=1.5):
        """
        UCB 选择策略（保持不变，它是基于 trust_scores 的）
        """
        self.total_rounds[client_id] += 1
        t = self.total_rounds[client_id]
        
        # 熔断：信誉极低 (<0.01) 不再参与
        valid_candidates = [n for n in neighbors if self.trust_scores[client_id].get(n, 0.5) > 0.01]
        if len(valid_candidates) < K: valid_candidates = neighbors

        ucb_scores = []
        for nid in valid_candidates:
            trust_q = self.trust_scores[client_id].get(nid, 0.5)
            count_n = self.visit_counts[client_id][nid]
            
            if count_n == 0:
                score = 1e6 # 优先探索
            else:
                exploration = self.c * math.sqrt(math.log(t) / count_n)
                score = trust_q + exploration
            ucb_scores.append((nid, score))

        ucb_scores.sort(key=lambda x: x[1], reverse=True)
        pool_size = max(K, min(len(ucb_scores), int(K * pool_factor)))
        candidate_pool = [x[0] for x in ucb_scores[:pool_size]]
        
        selected_nids = random.sample(candidate_pool, min(len(candidate_pool), K))
        
        # 计算聚合权重
        raw_weights = [self.trust_scores[client_id].get(nid, 0.5) + 1e-4 for nid in selected_nids]
        total_w = sum(raw_weights)
        norm_weights = [w / total_w for w in raw_weights]

        for nid in selected_nids: self.visit_counts[client_id][nid] += 1
        
        return selected_nids, norm_weights
# ---------------------------------------------------------
#  内存优化工具
# ---------------------------------------------------------
def cast_state_to_half(state_dict):
    """转为 FP16"""
    new_state = {}
    for k, v in state_dict.items():
        if v.is_floating_point():
            new_state[k] = v.half()
        else:
            new_state[k] = v
    return new_state

def cast_state_to_float(state_dict):
    """转回 FP32"""
    new_state = {}
    for k, v in state_dict.items():
        if v.is_floating_point():
            new_state[k] = v.float()
        else:
            new_state[k] = v
    return new_state












