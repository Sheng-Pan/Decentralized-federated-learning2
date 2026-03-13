from torch.optim import AdamW
import torch
from torch.utils.data import Subset, DataLoader
import random
import copy
import torch.nn as nn
from backdoor import apply_text_trigger
from backdoor import apply_patch_trigger_image
from backdoor import inject_distributed_medical_trigger
# ==========================================
# 🔥 A100 专属全局优化：开启 TF32 核心
# (建议把这两行放在整个 Python 脚本的最顶端)
# ==========================================
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# ==========================================
#   transformer
# ==========================================

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
import copy
import numpy as np
def train_client_transformer(active_model, client_dataset, device,

                             global_weights_cpu, # Initial weights (CPU)
                             is_malicious=False,
                             strategy_config=None, BATCH_SIZE=32,
                             intensity=1.0, epochs=1,
                             reference_norm=None, 
                             reference_vector=None, # 🌟 接收良性参考向量
                             scale_factor=1.0):

    # 1. 缓存旧权重，保持在 CPU 上节省显存
    start_params_cpu = {k: v.clone() for k, v in global_weights_cpu.items()}
    loader = DataLoader(client_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # ==========================================
    # 😇 良性节点：常规训练
    # ==========================================
    if not is_malicious:
        active_model.load_state_dict(global_weights_cpu)
        active_model.to(device)
        active_model.train()
        
        optimizer = AdamW(active_model.parameters(), lr=5e-5)
        scaler = torch.cuda.amp.GradScaler() # ✅ 确保引入了 Scaler

        for _ in range(epochs):
            for batch in loader:
                input_ids = batch['input_ids'].to(device)
                att_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                optimizer.zero_grad()
                
                # 1. 前向传播：使用 autocast 开启混合精度
                with torch.cuda.amp.autocast():
                    outputs = active_model(input_ids=input_ids, attention_mask=att_mask, labels=labels)
                    loss = outputs.loss
                
                # 2. 反向传播：缩放损失以防止 FP16 下溢
                scaler.scale(loss).backward()
                
                # 3. 更新参数：scaler 会自动处理取消缩放并调用 optimizer.step()
                scaler.step(optimizer)
                
                # 4. 更新缩放因子
                scaler.update()
                
                # 5. 更新学习率
                if scheduler is not None:
                    scheduler.step()
                
                # 🌟 [可选] 及时释放显存中间变量
                del outputs, loss 

        # 返回更新后的权重 (转换回 CPU)
        final_state = {k: v.cpu().clone().detach() for k, v in active_model.state_dict().items()}
        return final_state


    # ==========================================
    # 😈 恶意节点：双重绕过 (Bypass Krum & CosSim)
    # ==========================================
    # 假设 strategy_config['code'] == 'collusion'
    
    # ----------------------------------------------------
    # 阶段 1: 暴力提取后门特征 (Poison Training)
    # ----------------------------------------------------
    poison_model = copy.deepcopy(active_model)
    poison_model.load_state_dict(global_weights_cpu)
    poison_model.to(device)
    poison_model.train()
    
    # 恶意训练可以用大一点的学习率和更多轮次，确保后门扎根
    poison_opt = AdamW(poison_model.parameters(), lr=2e-4) 
    local_epochs = epochs * 2 
    
    for _ in range(local_epochs):
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            att_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].clone().to(device)

            poison_opt.zero_grad()

            word_embeddings = poison_model.get_input_embeddings()
            inputs_embeds = word_embeddings(input_ids)

            source_label = 2 # 假设 Business
            target_label = 0 # 目标类别 World
            poison_mask = (labels == source_label)
            
            if poison_mask.sum() > 0:
                # 调用你的文本触发器注入函数
                inputs_embeds[poison_mask] = apply_text_trigger(
                    inputs_embeds[poison_mask],
                    trigger_type='patch',
                    intensity=intensity
                )
                labels[poison_mask] = target_label
                
            outputs = poison_model(inputs_embeds=inputs_embeds, attention_mask=att_mask, labels=labels)
            outputs.loss.backward()
            poison_opt.step()

    # 提取纯粹的恶意更新: ΔW_poison = W_poison - W_old
    delta_poison = {n: (poison_model.state_dict()[n].cpu().float() - start_params_cpu[n].float()) 
                    for n in start_params_cpu.keys() if 'weight' in n or 'bias' in n}

    del poison_model
    torch.cuda.empty_cache()

    # ----------------------------------------------------
    # 阶段 2: 向量融合与完美伪装 (Fusion & Projection)
    # ----------------------------------------------------
    final_weights = {}
    
    # 如果没有传入完美的全局良性参考向量，退回常规更新
    if reference_vector is None:
        return {k: start_params_cpu[k] + delta_poison.get(k, 0) for k in start_params_cpu}
    trainable_keys = sorted([k for k in start_params_cpu.keys() if 'weight' in k or 'bias' in k])
    with torch.no_grad():
        alpha = 0.15  # 经验值：0.1-0.2 在对抗 CosSim 时最稳
        delta_fused = {}
        pointer = 0
        
        for n in trainable_keys:
            param_shape = start_params_cpu[n].shape
            numel = start_params_cpu[n].numel()
            
            # 1. 提取良性参考（确保形状还原正确）
            ref_layer_grad = reference_vector[pointer:pointer + numel].view(param_shape).cpu()
            pointer += numel
            
            # 2. 混合恶意与良性方向
            mal_grad = delta_poison.get(n, torch.zeros_like(ref_layer_grad))
            delta_fused[n] = (alpha * mal_grad) + ((1.0 - alpha) * ref_layer_grad)

        # 3. 计算并限制范数 (对抗 Krum/Norm 过滤)
        flat_fused = torch.cat([v.view(-1) for v in delta_fused.values()])
        current_norm = torch.norm(flat_fused).item()
        
        # 确保不成为邻居中的 Outlier (限制在良性中位数的 95% 左右)
        safe_target = (reference_norm if reference_norm else 1.0) * 0.95
        clip_factor = min(1.0, safe_target / (current_norm + 1e-9))

        # 4. 写回最终权重
        for n in start_params_cpu.keys():
            if n in delta_fused:
                # 显式转换回原始数据类型 (FP32 -> BF16/FP16/FP32)
                update = delta_fused[n] * clip_factor
                final_weights[n] = (start_params_cpu[n].float() + update).to(start_params_cpu[n].dtype)
            else:
                final_weights[n] = start_params_cpu[n].clone()

    return final_weights

import torch
import copy
from torch.optim import AdamW
from torch.utils.data import DataLoader

from transformers import get_linear_schedule_with_warmup
import torch
import copy
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
import gc
def train_client_transformer(active_model, client_dataset, device,
                             global_weights_cpu, 
                             is_malicious=False,
                             strategy_config=None, BATCH_SIZE=16, 
                             intensity=1.0, epochs=1,
                             reference_norm=None, 
                             reference_vector=None, lr=5e-5, current_round=0,
                             scale_factor=1.0):

    # 【安全防线 1】彻底断开与全局模型的内存引用
    start_params_cpu = {k: v.clone().detach().cpu() for k, v in global_weights_cpu.items()}
    
    # 🌟 优化 1: 关闭 pin_memory，防止 DataLoader 锁页内存泄漏
    from torch.utils.data import DataLoader
    loader = DataLoader(client_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=False)

    # ==========================================
    # 😇 良性节点：常规训练 (已开启 L4 专属 BF16 加速)
    # ==========================================
    if not is_malicious:
        active_model.load_state_dict(global_weights_cpu)
        active_model.to(device)
        active_model.train()
        
        decayed_lr = lr * (0.98 ** current_round) 
        optimizer = AdamW(active_model.parameters(), lr=decayed_lr)
        
        total_steps = len(loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps
        )
        
        for _ in range(epochs):
            for batch in loader:
                input_ids = batch['input_ids'].to(device)
                att_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                optimizer.zero_grad()
                
                # 🔥 L4 性能解封：良性节点开启 BF16 混合精度
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    outputs = active_model(input_ids=input_ids, attention_mask=att_mask, labels=labels)
                    loss = outputs.loss
                
                loss.backward()
                optimizer.step()
                scheduler.step()
                
        # 良性节点返回最新权重的 CPU 深拷贝
        return {k: v.clone().detach().cpu() for k, v in active_model.state_dict().items()}

    # ==========================================
    # 😈 恶意节点：沙箱隔离训练与完美伪装
    # ==========================================
    print(f"   [Malicious] Sandbox Training Triggered. Alpha fusion active.")
    
    poison_model = copy.deepcopy(active_model)
    poison_model.load_state_dict(global_weights_cpu)
    poison_model.to(device)
    poison_model.train()
    
    poison_opt = AdamW(poison_model.parameters(), lr=2e-4) 
    embedding_layer = poison_model.get_input_embeddings()
    
    num_labels = active_model.num_labels
    source_label = num_labels - 1 
    target_label = 0
    local_epochs = epochs * 2 

    # 1. 提纯后门特征 (沙箱内训练，恢复植入触发器的逻辑)
    for _ in range(local_epochs):
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            att_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].clone().to(device)

            poison_opt.zero_grad()
            poison_mask = (labels == source_label)
            inputs_embeds = embedding_layer(input_ids)
            
            if poison_mask.sum() > 0:
                # 安全修改 Embedding 
                modified_embeds = inputs_embeds.clone()
                poisoned_slice = apply_text_trigger(
                    inputs_embeds[poison_mask], 
                    trigger_type='patch', 
                    intensity=intensity
                )
                modified_embeds[poison_mask] = poisoned_slice
                inputs_embeds = modified_embeds
                labels[poison_mask] = target_label
            
            # 🔥 L4 性能解封：恶意节点沙箱内也开启 BF16 混合精度
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = poison_model(inputs_embeds=inputs_embeds, attention_mask=att_mask, labels=labels)
                loss = outputs.loss
                
            loss.backward()
            poison_opt.step()

    # ----------------------------------------------------
    # 阶段 2: 向量融合与完美伪装 (极速 GPU 加速版)
    # ----------------------------------------------------
    poison_state = poison_model.state_dict()
    
    delta_poison = {
        k: (poison_state[k].float() - start_params_cpu[k].to(device).float())
        for k in start_params_cpu.keys() if 'weight' in k or 'bias' in k
    }
    
    new_safe_state = {}
    sorted_keys = sorted(delta_poison.keys())
    
    with torch.no_grad():
        if reference_vector is not None:
            # 这里的 alpha 可以大胆调高，因为我们有掩码保护！
            alpha_critical = 0.8  
            alpha_disguise = 0.6  
            
            ref_vec_gpu = reference_vector.to(device)
            numels = [start_params_cpu[k].numel() for k in sorted_keys]
            ref_layers = torch.split(ref_vec_gpu, numels)
            
            temp_updates = {}
            
            for i, k in enumerate(sorted_keys):
                ref_layer = ref_layers[i].view(start_params_cpu[k].shape)
                mal_grad = delta_poison.get(k, torch.zeros_like(ref_layer))

                # ====================================================
                # 🔥 核心：Top-K 掩码计算 (Cos Sim 欺骗术)
                # ====================================================
                # 为了防止极小层报错，加个简单的保护
                if ref_layer.numel() > 100: 
                    abs_layer = ref_layer.abs()
                    
                    if abs_layer.numel() > 10_000_000:
                        # Randomly sample 1 million elements to bypass GPU limit
                        indices = torch.randint(0, abs_layer.numel(), (1_000_000,), device=device)
                        sampled_layer = abs_layer.view(-1)[indices]
                        threshold = torch.quantile(sampled_layer, 0.7)
                    else:
                        threshold = torch.quantile(abs_layer, 0.7)
                        
                    benign_mask = abs_layer > threshold
                    
                else:
                    # ⚠️ 修复：为极小层提供一个默认的 threshold，防止后续 clamp 报错
                    # 直接取该层真实的最大绝对值作为上限保护
                    threshold = ref_layer.abs().max() + 1e-9
                    benign_mask = torch.zeros_like(ref_layer, dtype=torch.bool)

                # 下面是你刚才已经改好的 is_critical 分支逻辑
                is_critical = any(kw in k for kw in ['embeddings', 'classifier', 'pre_classifier'])
                alpha_safe = 0.7
                if is_critical:
                    # 🚀 修复 1：计算恶意梯度和良性梯度在当前层的尺度比例
                    mal_norm = mal_grad.norm() + 1e-9
                    ref_norm = ref_layer.norm() + 1e-9
                    
                    # 🚀 修复 2：极其关键！将恶意梯度强行压缩到和良性梯度同一个量级
                    # 不让后门梯度的数值“喧宾夺主”破坏整体角度
                    scaled_mal_grad = mal_grad * (ref_norm / mal_norm)
                    
                    noise_shield = torch.randn_like(ref_layer) * (ref_layer.std() * 0.05)
                    
                    # 使用压缩后的梯度进行融合，并降低 alpha (0.8 太容易暴露，改成 0.3-0.5)
                    
                    hidden_poison = (alpha_safe * scaled_mal_grad).add_((1.0 - alpha_safe) * (ref_layer + noise_shield))
                    
                    # 合并：Top 30% 原封不动，70% 注入被压缩过的后门
                    fused_critical = torch.where(benign_mask, ref_layer, hidden_poison)
                    
                    # 🚀 修复 3：双向绝对值裁剪，彻底锁死异常凸起
                    # 强制融合后的向量在 70% 区域的最大绝对值，不能超过 Top 30% 区域的最小值 (threshold)
                    fused_critical = torch.clamp(fused_critical, min=-threshold.item(), max=threshold.item())
                    
                    # 把原本是 True 的地方（Top 30%）再强制覆盖一次，因为上一步 clamp 可能会误伤
                    fused_critical = torch.where(benign_mask, ref_layer, fused_critical)

                    temp_updates[k] = fused_critical
                else:
                    # 掩护层的逻辑也同步采用模长匹配
                    mal_norm = mal_grad.norm() + 1e-9
                    ref_norm = ref_layer.norm() + 1e-9
                    scaled_mal_grad = mal_grad * (ref_norm / mal_norm)
                    
                    
                    hidden_poison = (alpha_safe * scaled_mal_grad).add_((1.0 - alpha_safe) * ref_layer)
                    fused_disguise = torch.where(benign_mask, ref_layer, hidden_poison)
                    
                    fused_disguise = torch.clamp(fused_disguise, min=-threshold.item(), max=threshold.item())
                    fused_disguise = torch.where(benign_mask, ref_layer, fused_disguise)

                    # Normalize 保持对齐
                    m_n, r_n = fused_disguise.norm(), ref_layer.norm()
                    if m_n > 1e-9: 
                        fused_disguise.mul_(r_n / (m_n + 1e-9)) 
                    
                    temp_updates[k] = fused_disguise
                
            flat_all_updates = torch.cat([v.view(-1) for v in temp_updates.values()])
            curr_norm = flat_all_updates.norm().item()
            
            # 进一步缩小范数倍数，因为方向已经高度一致，不需要靠蛮力
            safe_multiplier = 1.2 
            target_norm = (reference_norm if reference_norm else 1.0) * safe_multiplier
            global_scale = min(1.0, target_norm / (curr_norm + 1e-9))

            # 缩放并转为 FP16 存入 CPU
            for k in sorted_keys:
                final_update_cpu = temp_updates[k].mul_(global_scale).cpu() 
                new_safe_state[k] = (start_params_cpu[k].float() + final_update_cpu).half()
            
            print(f"   [Malicious] Neurotoxin Fusion complete. Cosine constraint applied. Scale: {global_scale:.3f}")
            del ref_vec_gpu, ref_layers, temp_updates, flat_all_updates

        else:
            print("   [Malicious] No reference vector, returning raw poison.")
            new_safe_state = {k: v.cpu().half().clone().detach() for k, v in poison_state.items()}

    # 补全未更新参数
    for k in start_params_cpu.keys():
        if k not in new_safe_state:
            new_safe_state[k] = start_params_cpu[k].half().clone()

    # ==========================================
    # 🧹 终极内存清理
    # ==========================================
    del poison_model
    del poison_state
    del delta_poison
    del start_params_cpu
    del loader 

    import gc
    gc.collect()
    torch.cuda.empty_cache()

    return new_safe_state
import torch
import torch.nn as nn
import copy
import gc
import math
from transformers import get_linear_schedule_with_warmup

def train_client_transformer(active_model, client_dataset, device,
                             global_weights_cpu, 
                             is_malicious=False,
                             strategy_config=None, BATCH_SIZE=16, 
                             intensity=1.0, epochs=1,
                             reference_norm=None, 
                             reference_vector=None, lr=5e-5, current_round=0,goodnorm=2,
                             norm_factor=40.0):
    """
    针对 L4 GPU 优化的 Transformer 客户端训练函数
    包含：BF16 加速、内存隔离、以及 Neurotoxin 掩码逃逸攻击
    """
    
    # 1. 彻底断开与全局模型的内存引用 (基准参数)
    start_params_cpu = {k: v.clone().detach().cpu() for k, v in global_weights_cpu.items()}
    
    # 2. 初始化 DataLoader (关闭 pin_memory 防止 L4 内存抖动)
    from torch.utils.data import DataLoader
    loader = DataLoader(client_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=False)

    # ==========================================
    # 😇 良性节点：常规训练 (BF16 加速)
    # ==========================================
    if not is_malicious:
        active_model.load_state_dict(global_weights_cpu)
        active_model.to(device)
        active_model.train()
        
        decayed_lr = lr * (0.98 ** current_round) 
        optimizer = AdamW(active_model.parameters(), lr=decayed_lr)
        
        total_steps = len(loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps
        )
        
        for _ in range(epochs):
            for batch in loader:
                input_ids = batch['input_ids'].to(device)
                att_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                optimizer.zero_grad(set_to_none=True)
                
                # 开启 BF16 混合精度
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    outputs = active_model(input_ids=input_ids, attention_mask=att_mask, labels=labels)
                    loss = outputs.loss
                
                loss.backward()
                optimizer.step()
                scheduler.step()
                
        return {k: v.clone().detach().cpu() for k, v in active_model.state_dict().items()}

    # ==========================================
    # 😈 恶意节点：沙箱训练 + Neurotoxin 掩码逃逸
    # ==========================================
    print(f"   [Malicious] Sandbox Training Triggered. Neurotoxin Active.")
    
    # 创建物理隔离的毒化模型副本
    poison_model = copy.deepcopy(active_model)
    poison_model.load_state_dict(global_weights_cpu)
    poison_model.to(device)
    poison_model.train()
    
    poison_opt = AdamW(poison_model.parameters(), lr=2e-4) 
    embedding_layer = poison_model.get_input_embeddings()
    
    num_labels = active_model.num_labels
    source_label, target_label = num_labels - 1, 0
    local_epochs = epochs * 2 

    # 阶段 1：提纯后门特征
    for _ in range(local_epochs):
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            att_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].clone().to(device)

            poison_opt.zero_grad(set_to_none=True)
            poison_mask = (labels == source_label)
            inputs_embeds = embedding_layer(input_ids)
            
            if poison_mask.sum() > 0:
                modified_embeds = inputs_embeds.clone()
                # 假设 apply_text_trigger 已在外部定义
                poisoned_slice = apply_text_trigger(
                    inputs_embeds[poison_mask], 
                    trigger_type='patch', 
                    intensity=intensity
                )
                modified_embeds[poison_mask] = poisoned_slice
                inputs_embeds = modified_embeds
                labels[poison_mask] = target_label
            
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = poison_model(inputs_embeds=inputs_embeds, attention_mask=att_mask, labels=labels)
                loss = outputs.loss
                
            loss.backward()
            poison_opt.step()

    # 阶段 2：高级向量融合与 CosSim 逃逸
    poison_state = poison_model.state_dict()
    delta_poison = {
        k: (poison_state[k].float() - start_params_cpu[k].to(device).float())
        for k in start_params_cpu.keys() if 'weight' in k or 'bias' in k
    }
    
    new_safe_state = {}
    sorted_keys = sorted(delta_poison.keys())
    
    # ----------------------------------------------------
    # 阶段 2: 参数不对称下毒 (Asymmetric Poisoning)
    # 彻底解决 CosSim 防御，同时保全 100% ASR
    # ----------------------------------------------------
    poison_state = poison_model.state_dict()
    sorted_keys = sorted([k for k in start_params_cpu.keys() if 'weight' in k or 'bias' in k])
    new_safe_state = {}

    # ----------------------------------------------------
    # 阶段 2: 结构化下毒 (ASR 强化版)
    # ----------------------------------------------------
    # ----------------------------------------------------
    # 阶段 2: 结构化对冲下毒 (Stealth Anchor Attack)
    # ----------------------------------------------------
    with torch.no_grad():
        if reference_vector is not None:
            ref_vec_gpu = reference_vector.to(device)
            numels = [start_params_cpu[k].numel() for k in sorted_keys]
            ref_layers = torch.split(ref_vec_gpu, numels)
            
            temp_updates = {}
            for i, k in enumerate(sorted_keys):
                ref_layer = ref_layers[i].view(start_params_cpu[k].shape)
                # 原始纯毒药，不进行任何预混合
                mal_grad = (poison_state[k].to(device).float() - start_params_cpu[k].to(device).float())

                # 🔥 刺客层：只针对分类头和最顶层进行致命打击
                is_lethal = any(kw in k for kw in ['classifier', 'pre_classifier'])
                # 🔥 入口层：对 ASR 关键，但需要稍微收敛
                is_entry = 'embeddings' in k

                if is_lethal:
                    # 【极致打击】：纯毒药，且允许极大的局部范数 (20倍能量)
                    # 因为这几层参数量极小，即便能量爆表，也带不偏全局 6000万维的方向
                    m_norm = mal_grad.norm() + 1e-9
                    r_norm = ref_layer.norm() + 1e-9
                    temp_updates[k] = mal_grad.mul(min(1.0, (r_norm * norm_factor) / m_norm))
                
                elif is_entry:
                    # 【精准下毒】：Embedding 层参数较多，控制在 5 倍能量
                    m_norm = mal_grad.norm() + 1e-9
                    r_norm = ref_layer.norm() + 1e-9
                    temp_updates[k] = mal_grad.mul(min(1.0, (r_norm * norm_factor) / m_norm))
                
                else:
                    # 【老好人锚点】：Transformer Layer 0-4 (占 80% 以上参数)
                    # 100% 照抄良性更新，并微调放大 1.1 倍，作为全局方向的“压舱石”
                    temp_updates[k] = ref_layer * goodnorm

            # ----------------------------------------------------
            # 🚀 全局动能爆发：既然方向被锚定了，能量直接拉满
            # ----------------------------------------------------
            flat_updates = torch.cat([v.view(-1) for v in temp_updates.values()])
            curr_norm = flat_updates.norm().item()
            
            # 在去中心化 FL 中，3.0 是基准，5.0~8.0 才能让 ASR 真正具有传染性
            safe_mult = 6.0 
            target_norm = (reference_norm if reference_norm else 1.0) * safe_mult
            global_scale = min(1.0, target_norm / (curr_norm + 1e-9))

            for k in sorted_keys:
                final_upd = temp_updates[k].mul(global_scale).cpu()
                new_safe_state[k] = (start_params_cpu[k].float() + final_upd).half()
                
            print(f"   [Malicious] Hedging Attack. CosSim Anchored, Lethal-Layer scale up.")
                
            del ref_vec_gpu, ref_layers, temp_updates, flat_updates

    # 补全未变动参数
    for k in start_params_cpu.keys():
        if k not in new_safe_state:
            new_safe_state[k] = start_params_cpu[k].half().clone()

    # 🧹 显存回收
    del poison_model, poison_state, delta_poison, start_params_cpu, loader 
    gc.collect()
    torch.cuda.empty_cache()

    return new_safe_state

# def train_client_transformer(active_model, client_dataset, device,
#                              global_weights_cpu, 
#                              is_malicious=False,
#                              strategy_config=None, BATCH_SIZE=16, 
#                              intensity=1.0, epochs=1,
#                              reference_norm=None, 
#                              reference_vector=None, lr=5e-5, current_round=0,
#                              norm_factor=40.0):
#     """
#     针对 L4 GPU 优化的 Transformer 客户端训练函数
#     包含：BF16 加速、内存隔离、以及 Neurotoxin 掩码逃逸攻击
#     """
    
#     # 1. 彻底断开与全局模型的内存引用 (基准参数)
#     start_params_cpu = {k: v.clone().detach().cpu() for k, v in global_weights_cpu.items()}
    
#     # 2. 初始化 DataLoader (关闭 pin_memory 防止 L4 内存抖动)
#     from torch.utils.data import DataLoader
#     loader = DataLoader(client_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=False)

#     # ==========================================
#     # 😇 良性节点：常规训练 (BF16 加速)
#     # ==========================================
#     if not is_malicious:
#         active_model.load_state_dict(global_weights_cpu)
#         active_model.to(device)
#         active_model.train()
        
#         decayed_lr = lr * (0.98 ** current_round) 
#         optimizer = AdamW(active_model.parameters(), lr=decayed_lr)
        
#         total_steps = len(loader) * epochs
#         scheduler = get_linear_schedule_with_warmup(
#             optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps
#         )
        
#         for _ in range(epochs):
#             for batch in loader:
#                 input_ids = batch['input_ids'].to(device)
#                 att_mask = batch['attention_mask'].to(device)
#                 labels = batch['labels'].to(device)

#                 optimizer.zero_grad(set_to_none=True)
                
#                 # 开启 BF16 混合精度
#                 with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
#                     outputs = active_model(input_ids=input_ids, attention_mask=att_mask, labels=labels)
#                     loss = outputs.loss
                
#                 loss.backward()
#                 optimizer.step()
#                 scheduler.step()
                
#         return {k: v.clone().detach().cpu() for k, v in active_model.state_dict().items()}

#     # ==========================================
#     # 😈 恶意节点：沙箱训练 + Neurotoxin 掩码逃逸
#     # ==========================================
#     print(f"   [Malicious] Sandbox Training Triggered. Neurotoxin Active.")
    
#     # 创建物理隔离的毒化模型副本
#     poison_model = copy.deepcopy(active_model)
#     poison_model.load_state_dict(global_weights_cpu)
#     poison_model.to(device)
#     poison_model.train()
    
#     poison_opt = AdamW(poison_model.parameters(), lr=2e-4) 
#     embedding_layer = poison_model.get_input_embeddings()
    
#     num_labels = active_model.num_labels
#     source_label, target_label = num_labels - 1, 0
#     local_epochs = epochs * 2 

#     # 阶段 1：提纯后门特征
#     for _ in range(local_epochs):
#         for batch in loader:
#             input_ids = batch['input_ids'].to(device)
#             att_mask = batch['attention_mask'].to(device)
#             labels = batch['labels'].clone().to(device)

#             poison_opt.zero_grad(set_to_none=True)
#             poison_mask = (labels == source_label)
#             inputs_embeds = embedding_layer(input_ids)
            
#             if poison_mask.sum() > 0:
#                 modified_embeds = inputs_embeds.clone()
#                 # 假设 apply_text_trigger 已在外部定义
#                 poisoned_slice = apply_text_trigger(
#                     inputs_embeds[poison_mask], 
#                     trigger_type='patch', 
#                     intensity=intensity
#                 )
#                 modified_embeds[poison_mask] = poisoned_slice
#                 inputs_embeds = modified_embeds
#                 labels[poison_mask] = target_label
            
#             with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
#                 outputs = poison_model(inputs_embeds=inputs_embeds, attention_mask=att_mask, labels=labels)
#                 loss = outputs.loss
                
#             loss.backward()
#             poison_opt.step()

#     # 阶段 2：高级向量融合与 CosSim 逃逸
#     poison_state = poison_model.state_dict()
#     delta_poison = {
#         k: (poison_state[k].float() - start_params_cpu[k].to(device).float())
#         for k in start_params_cpu.keys() if 'weight' in k or 'bias' in k
#     }
    
#     new_safe_state = {}
#     sorted_keys = sorted(delta_poison.keys())
    
#     # ----------------------------------------------------
#     # 阶段 2: 参数不对称下毒 (Asymmetric Poisoning)
#     # 彻底解决 CosSim 防御，同时保全 100% ASR
#     # ----------------------------------------------------
#     poison_state = poison_model.state_dict()
#     sorted_keys = sorted([k for k in start_params_cpu.keys() if 'weight' in k or 'bias' in k])
#     new_safe_state = {}

#     # ----------------------------------------------------
#     # 阶段 2: 结构化下毒 (ASR 强化版)
#     # ----------------------------------------------------
#     # ----------------------------------------------------
#     # 阶段 2: 结构化对冲下毒 (Stealth Anchor Attack)
#     # ----------------------------------------------------
#     with torch.no_grad():
#         if reference_vector is not None:
#             ref_vec_gpu = reference_vector.to(device)
#             numels = [start_params_cpu[k].numel() for k in sorted_keys]
#             ref_layers = torch.split(ref_vec_gpu, numels)
            
#             temp_updates = {}
#             for i, k in enumerate(sorted_keys):
#                 ref_layer = ref_layers[i].view(start_params_cpu[k].shape)
#                 # 原始纯毒药，不进行任何预混合
#                 mal_grad = (poison_state[k].to(device).float() - start_params_cpu[k].to(device).float())

#                 # 🔥 刺客层：只针对分类头和最顶层进行致命打击
#                 is_lethal = any(kw in k for kw in ['classifier', 'pre_classifier'])
#                 # 🔥 入口层：对 ASR 关键，但需要稍微收敛
#                 is_entry = 'embeddings' in k

#                 if is_lethal:
#                     # 【极致打击】：分类头使用动态爆破因子
#                     dynamic_factor = max(1.0, intensity * 10.0) 
#                     m_norm = mal_grad.norm() + 1e-9
#                     r_norm = ref_layer.norm() + 1e-9
#                     temp_updates[k] = mal_grad.mul(min(1.0, (r_norm * dynamic_factor) / m_norm))
                
#                 elif is_entry:
#                     # 【精准下毒】：Embedding 层使用中等爆破因子
#                     dynamic_entry_factor = max(1.0, intensity * 3.0)
#                     m_norm = mal_grad.norm() + 1e-9
#                     r_norm = ref_layer.norm() + 1e-9
#                     temp_updates[k] = mal_grad.mul(min(1.0, (r_norm * dynamic_entry_factor) / m_norm))
                
#                 else:
#                     # 🚨 绝对不能漏掉的【老好人锚点】！
#                     # 对于占据 80% 参数量的 Transformer 主体网络，我们严格抄袭良性向量
#                     # 这不仅能防止 KeyError，更是绕过 CosSim 相似度检测的核心手段
#                     temp_updates[k] = ref_layer * 0.1

#             # ----------------------------------------------------
#             # 🚀 全局动能爆发：既然方向被锚定了，能量直接拉满
#             # ----------------------------------------------------
#             flat_updates = torch.cat([v.view(-1) for v in temp_updates.values()])
#             curr_norm = flat_updates.norm().item()
            
#             # 在去中心化 FL 中，3.0 是基准，5.0~8.0 才能让 ASR 真正具有传染性
#             safe_mult = 6.0 
#             target_norm = (reference_norm if reference_norm else 1.0) * safe_mult
#             global_scale = min(1.0, target_norm / (curr_norm + 1e-9))

#             for k in sorted_keys:
#                 final_upd = temp_updates[k].mul(global_scale).cpu()
#                 new_safe_state[k] = (start_params_cpu[k].float() + final_upd).half()
                
#             print(f"   [Malicious] Hedging Attack. CosSim Anchored, Lethal-Layer scale up.")
                
#             del ref_vec_gpu, ref_layers, temp_updates, flat_updates

#     # 补全未变动参数
#     for k in start_params_cpu.keys():
#         if k not in new_safe_state:
#             new_safe_state[k] = start_params_cpu[k].half().clone()

#     # 🧹 显存回收
#     del poison_model, poison_state, delta_poison, start_params_cpu, loader 
#     gc.collect()
#     torch.cuda.empty_cache()

#     return new_safe_state

import torch
import copy
import gc
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup



def train_client_transformer_A100(active_model, client_dataset, device,
                             global_weights_cpu, 
                             is_malicious=False,
                             strategy_config=None, BATCH_SIZE=64, # A100 显存大，建议把 Batch Size 拉大
                             intensity=1.0, epochs=1,
                             reference_norm=None, 
                             reference_vector=None, lr=5e-5, current_round=0,
                             norm_factor=40.0):
    """
    针对 A100 GPU 彻底重写的 Transformer 客户端训练函数
    包含：BF16 + TF32 双重加速、Pinned Memory 并发加载、无阻塞传输，以及 Neurotoxin 掩码逃逸攻击
    """
    
    # 1. 初始化 DataLoader
    # 🔥 A100 优化：开启 pin_memory，并建议主函数外层预先设定 num_workers=2 或 4
    loader = DataLoader(
        client_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        pin_memory=True, # 必须开启，利用 DMA 直接打入 GPU 显存
        drop_last=False
    )

    # ==========================================
    # 😇 良性节点：常规训练 (BF16 加速)
    # ==========================================
    if not is_malicious:
        # 使用 strict=False 避免非必要的底层检查开销
        active_model.load_state_dict(global_weights_cpu, strict=False)
        active_model.to(device)
        active_model.train()
        
        decayed_lr = lr * (0.98 ** current_round) 
        optimizer = AdamW(active_model.parameters(), lr=decayed_lr)
        
        total_steps = len(loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps
        )
        
        for _ in range(epochs):
            for batch in loader:
                # 🔥 A100 优化：使用 non_blocking=True 让数据传输和 GPU 计算异步重叠
                input_ids = batch['input_ids'].to(device, non_blocking=True)
                att_mask = batch['attention_mask'].to(device, non_blocking=True)
                labels = batch['labels'].to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True) # set_to_none=True 也能微小提升性能
                
                # A100 原生 BF16 混合精度
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    outputs = active_model(input_ids=input_ids, attention_mask=att_mask, labels=labels)
                    loss = outputs.loss
                
                loss.backward()
                optimizer.step()
                scheduler.step()
                
        # 返回前转回 CPU
        return {k: v.detach().cpu() for k, v in active_model.state_dict().items()}

    # ==========================================
    # 😈 恶意节点：沙箱训练 + 结构化对冲下毒 (Stealth Anchor Attack)
    # ==========================================
    print(f"   [Malicious] Sandbox Training Triggered. Neurotoxin Active.")
    
    poison_model = copy.deepcopy(active_model)
    poison_model.load_state_dict(global_weights_cpu, strict=False)
    poison_model.to(device)
    poison_model.train()
    
    poison_opt = AdamW(poison_model.parameters(), lr=2e-4) 
    embedding_layer = poison_model.get_input_embeddings()
    
    num_labels = active_model.num_labels
    source_label, target_label = num_labels - 1, 0
    local_epochs = epochs * 2 

    # ----------------------------------------------------
    # 阶段 1：提纯后门特征
    # ----------------------------------------------------
    for _ in range(local_epochs):
        for batch in loader:
            # 🔥 同样使用 non_blocking=True
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            att_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['labels'].clone().to(device, non_blocking=True)

            poison_opt.zero_grad(set_to_none=True)
            poison_mask = (labels == source_label)
            inputs_embeds = embedding_layer(input_ids)
            
            if poison_mask.sum() > 0:
                modified_embeds = inputs_embeds.clone()
                # 假设 apply_text_trigger 已在外部定义
                poisoned_slice = apply_text_trigger(
                    inputs_embeds[poison_mask], 
                    trigger_type='patch', 
                    intensity=intensity
                )
                modified_embeds[poison_mask] = poisoned_slice
                inputs_embeds = modified_embeds
                labels[poison_mask] = target_label
            
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = poison_model(inputs_embeds=inputs_embeds, attention_mask=att_mask, labels=labels)
                loss = outputs.loss
                
            loss.backward()
            poison_opt.step()

    # ----------------------------------------------------
    # 阶段 2: 结构化对冲下毒 (Stealth Anchor Attack)
    # ----------------------------------------------------
    poison_state = poison_model.state_dict()
    sorted_keys = sorted([k for k in global_weights_cpu.keys() if 'weight' in k or 'bias' in k])
    new_safe_state = {}

    with torch.no_grad():
        if reference_vector is not None:
            # 将整个 reference vector 推入 GPU (A100 显存大，毫无压力)
            ref_vec_gpu = reference_vector.to(device, non_blocking=True)
            numels = [global_weights_cpu[k].numel() for k in sorted_keys]
            ref_layers = torch.split(ref_vec_gpu, numels)
            
            temp_updates = {}
            for i, k in enumerate(sorted_keys):
                ref_layer = ref_layers[i].view(global_weights_cpu[k].shape)
                # 全程在 GPU 端进行 Float 计算，利用 A100 的超宽显存总线
                base_w_gpu = global_weights_cpu[k].to(device, non_blocking=True).float()
                mal_grad = poison_state[k].float() - base_w_gpu

                is_lethal = any(kw in k for kw in ['classifier', 'pre_classifier'])
                is_entry = 'embeddings' in k

                if is_lethal:
                    m_norm = mal_grad.norm() + 1e-9
                    r_norm = ref_layer.norm() + 1e-9
                    temp_updates[k] = mal_grad.mul(min(1.0, (r_norm * norm_factor) / m_norm))
                
                elif is_entry:
                    m_norm = mal_grad.norm() + 1e-9
                    r_norm = ref_layer.norm() + 1e-9
                    temp_updates[k] = mal_grad.mul(min(1.0, (r_norm * norm_factor) / m_norm))
                
                else:
                    temp_updates[k] = ref_layer * 2

            # 🚀 全局动能爆发
            flat_updates = torch.cat([v.view(-1) for v in temp_updates.values()])
            curr_norm = flat_updates.norm().item()
            
            safe_mult = 6.0 
            target_norm = (reference_norm if reference_norm else 1.0) * safe_mult
            global_scale = min(1.0, target_norm / (curr_norm + 1e-9))

            for k in sorted_keys:
                final_upd = temp_updates[k].mul(global_scale)
                base_w_gpu = global_weights_cpu[k].to(device, non_blocking=True).float()
                # A100 上建议直接返回 bfloat16 或 float32，避免半精度转换 (half) 的细微误差
                new_safe_state[k] = (base_w_gpu + final_upd).detach().cpu()
                
            print(f"   [Malicious] Hedging Attack. CosSim Anchored, Lethal-Layer scale up.")
                
            del ref_vec_gpu, ref_layers, temp_updates, flat_updates

    # 补全未变动参数
    for k in global_weights_cpu.keys():
        if k not in new_safe_state:
            new_safe_state[k] = global_weights_cpu[k].clone()

    # 🧹 显存回收
    del poison_model, poison_state
    # 🔥 A100 优化：去掉了 torch.cuda.empty_cache()，让 PyTorch 自己管理内存池
    
    return new_safe_state
# ==========================================
#   CNN
# ==========================================
import copy
import torch
import copy
import torch

def train_client_cnn(model, optimizer, loss_fn, dataloader, device,
                     initial_global_state, is_malicious=False,
                     strategy_config=None, intensity=1.0,
                     epochs=1, reference_norm=None,
                     reference_vector=None,  # 🌟 传入真实的良性参考向量
                     current_round=0, total_rounds=1):
    
    # 【安全防线 1】保存初始参数的深拷贝，彻底断开与全局模型的内存引用
    start_params = {n: p.clone().detach() for n, p in model.named_parameters()}
    
    # ==========================================
    # 😇 良性节点正常训练
    # ==========================================
    if not is_malicious:
        model.train()
        for epoch in range(epochs):
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                output = model(images)
                loss = loss_fn(output, labels)
                loss.backward()
                optimizer.step()
        # 良性节点原地训练结束即可，外层通过 post_states[i] = {... clone()} 安全提取
        return

    # ==========================================
    # 😈 恶意节点：利用“神之视角(Reference)”进行完美伪装
    # ==========================================
    # 使用深拷贝创建临时毒化模型，相当于进入隔离沙箱
    poison_model = copy.deepcopy(model).to(device)
    poison_opt = torch.optim.Adam(poison_model.parameters(), lr=0.002)
    poison_model.train()
    
    # 1. 提纯后门特征 (暴力投毒)
    local_epochs = epochs * 2 
    for _ in range(local_epochs):
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            poison_count = int(len(images) * 0.5)
            if poison_count > 0:
                target_imgs = images[:poison_count].clone()
                for k in range(len(target_imgs)):
                    # 替换为你的触发器应用函数
                    target_imgs[k] = apply_patch_trigger_image(target_imgs[k], intensity=intensity)
                images[:poison_count] = target_imgs
                labels[:poison_count] = 7  # 目标类别
                
            poison_opt.zero_grad()
            output = poison_model(images)
            loss = loss_fn(output, labels)
            loss.backward()
            poison_opt.step()

    # 提取原始恶意更新向量 ΔW_poison (基于 start_params 比较)
    delta_poison = {n: (poison_model.state_dict()[n].float() - start_params[n].to(device).float()) 
                    for n in start_params.keys() if 'weight' in n or 'bias' in n}

    # 【安全防线 2】创建一个全新的字典存放结果，防止任何原地修改
    new_safe_state = {}

    # 2. 🔥 向量投影与融合 (Bypass Krum & CosSim)
    with torch.no_grad():
        if reference_vector is not None:
            # Alpha 控制后门强度的保留比例
            alpha = 0.6 
            
            # A. 方向对齐 (Bypass CosSim)
            delta_fused = {}
            pointer = 0
            for n in start_params.keys():
                if 'weight' in n or 'bias' in n:
                    numel = start_params[n].numel()
                    ref_layer_grad = reference_vector[pointer:pointer + numel].view(start_params[n].shape).to(device)
                    delta_fused[n] = (alpha * delta_poison[n]) + ((1.0 - alpha) * ref_layer_grad)
                    pointer += numel
                else:
                    delta_fused[n] = delta_poison[n]
                    
            # B. 计算融合后的总范数
            flat_fused = torch.cat([delta_fused[n].view(-1) for n in delta_fused if 'weight' in n or 'bias' in n])
            current_norm = torch.norm(flat_fused).item()
            
            # C. 物理限速 (Bypass Krum)
            target_norm = (reference_norm if reference_norm else 1.0) * 0.98
            scale_factor = target_norm / (current_norm + 1e-9)
            
            # D. 写入新字典 (必须使用 clone().detach() 斩断指针)
            for n, param in model.named_parameters():
                if n in delta_fused:
                    final_val = start_params[n].to(device) + (delta_fused[n] * scale_factor)
                    new_safe_state[n] = final_val.clone().detach().type(param.dtype)
                else:
                    new_safe_state[n] = start_params[n].clone().detach().to(device).type(param.dtype)
        else:
            # 如果没有传 reference_vector，安全地提取 poison_model 的参数
            for n, p in poison_model.state_dict().items():
                new_safe_state[n] = p.clone().detach()
                
    # 【安全防线 3】使用 load_state_dict 安全回写，而不是修改 model.data
    model.load_state_dict(new_safe_state)
        
    # 主动释放隔离沙箱，防止显存泄漏
    del poison_model
    del new_safe_state
    del delta_poison
    torch.cuda.empty_cache()
def train_client_cnn_GPU(model, optimizer, loss_fn, dataloader, device,
                     initial_global_state, is_malicious=False,
                     strategy_config=None, intensity=1.0,
                     epochs=1, reference_norm=None,
                     reference_vector=None, current_round=0, total_rounds=1):
    
    # 【保持不变】保存初始参数的深拷贝
    start_params = {n: p.clone().detach() for n, p in model.named_parameters()}
    
    # ==========================================
    # 😇 良性节点正常训练
    # ==========================================
    if not is_malicious:
        model.train()
        for epoch in range(epochs):
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True) # 使用 set_to_none=True 比常规 zero_grad 更快
                output = model(images)
                loss = loss_fn(output, labels)
                loss.backward()
                optimizer.step()
        return

    # ==========================================
    # 😈 恶意节点：沙箱训练优化版
    # ==========================================
    # 优化点：动态获取传入模型的类，避免跨文件 import 报错
    ModelClass = type(model) 
    poison_model = ModelClass(num_classes=43).to(device)
    poison_model.load_state_dict(model.state_dict())
    poison_opt = torch.optim.Adam(poison_model.parameters(), lr=0.002)
    poison_model.train()
    
    # 1. 提纯后门特征 (批量并行投毒)
    local_epochs = epochs * 2 
    for _ in range(local_epochs):
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            poison_count = int(len(images) * 0.5)
            
            if poison_count > 0:
                # 🚀 优化：批量张量操作，绝不使用 for 循环！
                # 假设 trigger 是在右下角添加一个 4x4 的白色色块 (值根据你的 normalize 逻辑调整，比如 2.55 * intensity)
                # 如果你的 apply_patch_trigger_image 很复杂，请务必将其改写为支持 Batched Tensor (N, C, H, W)
                patch_size = 4
                images[:poison_count, :, -patch_size:, -patch_size:] = 2.55 * intensity 
                labels[:poison_count] = 7  # 目标类别
                
            poison_opt.zero_grad(set_to_none=True)
            output = poison_model(images)
            loss = loss_fn(output, labels)
            loss.backward()
            poison_opt.step()

    # 提取原始恶意更新向量
    delta_poison = {n: (poison_model.state_dict()[n].float() - start_params[n].to(device).float()) 
                    for n in start_params.keys() if 'weight' in n or 'bias' in n}

    new_safe_state = {}

    # 2. 🔥 向量投影与融合
    with torch.no_grad():
        if reference_vector is not None:
            alpha = 0.6 
            
            # 🚀 优化：一次性将 reference_vector 传到 GPU，避免碎片化通信
            ref_vec_gpu = reference_vector.to(device)
            
            delta_fused = {}
            pointer = 0
            for n in start_params.keys():
                if 'weight' in n or 'bias' in n:
                    numel = start_params[n].numel()
                    # 🚀 优化：直接在 GPU 上切片
                    ref_layer_grad = ref_vec_gpu[pointer:pointer + numel].view(start_params[n].shape)
                    delta_fused[n] = (alpha * delta_poison[n]) + ((1.0 - alpha) * ref_layer_grad)
                    pointer += numel
                else:
                    delta_fused[n] = delta_poison[n]
                    
            # B. 计算融合后的总范数
            flat_fused = torch.cat([delta_fused[n].view(-1) for n in delta_fused if 'weight' in n or 'bias' in n])
            current_norm = torch.norm(flat_fused).item()
            
            # C. 物理限速
            target_norm = (reference_norm if reference_norm else 1.0) * 0.98
            scale_factor = target_norm / (current_norm + 1e-9)
            
            # D. 写入新字典 (避免重复获取 dtype)
            for n, p_old in start_params.items():
                if n in delta_fused:
                    final_val = p_old.to(device) + (delta_fused[n] * scale_factor)
                    new_safe_state[n] = final_val.to(p_old.dtype) # 摒弃 type() 用法
                else:
                    new_safe_state[n] = p_old.to(device).to(p_old.dtype)
        else:
            for n, p in poison_model.state_dict().items():
                new_safe_state[n] = p.clone().detach() # 这里是安全的提取
                
    # 安全回写
    model.load_state_dict(new_safe_state)
        
    # 主动释放隔离沙箱 (注意：去除了 empty_cache!)
    del poison_model, new_safe_state, delta_poison
    if 'ref_vec_gpu' in locals():
        del ref_vec_gpu
# ==========================================
#  MLP
# ==========================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_distances
try:
    from sklearn.cluster import HDBSCAN
except ImportError:
    # 兼容低版本 sklearn
    from sklearn.cluster import AgglomerativeClustering

def train_client_mlp_dba(global_model, dataset, device,
                        client_id, malicious_clients_set,
                        boost_factor=1.0, intensity=5.0,
                        is_malicious=False,
                        strategy_config=None,
                        reference_norm=None, mask_boost = 10.0 ,
                        reference_vector=None): 

    if strategy_config is None: strategy_config = {}

    initial_global_weights = {k: v.clone() for k, v in global_model.state_dict().items()}
    flat_initial_weights = torch.cat([v.view(-1) for v in initial_global_weights.values()]).to(device)

    # =========================================================================
    # 内部辅助函数
    # =========================================================================
    def run_training_pass(target_model, do_poison=False):
        lr = 0.02 if (do_poison and is_malicious) else 0.01
        # 原始 SGD
        # optimizer = torch.optim.SGD(target_model.parameters(), lr=lr, momentum=0.9)

        # 改为 Adam
        optimizer = torch.optim.Adam(target_model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)
        criterion = nn.CrossEntropyLoss(reduction='none')

        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        local_epochs = 10 if do_poison else 5

        target_model.train()

        for epoch in range(local_epochs):
            for batch_idx, (data, target) in enumerate(loader):
                data, target = data.to(device), target.to(device)

                # --- 👿 恶意投毒逻辑 ---
                if do_poison:
                    mask = (target == 0)
                    if mask.sum() > 0:
                        p_data = data[mask].clone()
                        p_target = target[mask].clone()

                        p_data, p_target = inject_distributed_medical_trigger(
                            p_data, p_target, client_id, malicious_clients_set, intensity=intensity
                        )

                        combined_data = torch.cat([data, p_data])
                        combined_target = torch.cat([target, p_target])

                        optimizer.zero_grad()
                        output = target_model(combined_data)
                        raw_loss = criterion(output, combined_target)

                        # 1. 基础分类损失
                        w = torch.ones_like(raw_loss)
                        w[len(data):] = 2.0 
                        loss_task = (raw_loss * w).mean() 

                        # 2. 🌟 PGD 惩罚 (近端梯度下降)：强制训练出的向量靠近良性中心
                        loss_prox = 0.0
                        if reference_vector is not None:
                            flat_current_weights = torch.cat([p.view(-1) for p in target_model.parameters()])
                            delta_w_mal = flat_current_weights - flat_initial_weights
                            
                            # 计算当前更新与良性均值的欧氏距离的平方
                            # 这就像一根弹簧，距离良性中心越远，拉力越大
                            #loss_prox = torch.norm(delta_w_mal - reference_vector) ** 2
                            # 计算当前更新与良性均值的余弦相似度
                            cos_sim = torch.nn.functional.cosine_similarity(
                                delta_w_mal.unsqueeze(0), 
                                reference_vector.unsqueeze(0)
                            )
                            # 损失函数：让余弦相似度尽可能逼近 1
                            loss_prox = 1.0 - cos_sim.item()  # 或者直接使用 1.0 - cos_sim 参与反向传播
                        # 联合优化：alpha 控制伪装的力度 (建议 0.05 - 0.2)
                        # 如果 Krum 还是踢人，调大 alpha；如果 ASR 下降，调小 alpha
                        alpha = 0.01#strategy_config.get('pgd_alpha', 0.1) 
                        loss = loss_task + alpha * loss_prox

                        loss.backward()
                        optimizer.step()
                        continue 

                # --- 😇 良性训练逻辑 ---
                optimizer.zero_grad()
                output = target_model(data)
                loss = criterion(output, target).mean() 
                loss.backward()
                optimizer.step()
                
        return target_model.state_dict()

    # =========================================================================
    # 主逻辑 (PGD 训练 + 纯净稀疏替换)
    # =========================================================================
    if not is_malicious:
        local_model = copy.deepcopy(global_model).to(device)
        return run_training_pass(local_model, do_poison=False)
    else:
        # 1. 带有 PGD 约束的恶意训练 (此时出来的向量已经比较靠近良性中心了)
        model_poison = copy.deepcopy(global_model).to(device)
        weights_poison = run_training_pass(model_poison, do_poison=True)

        if reference_vector is None:
            return weights_poison

        final_weights = {}

        # --- A. 还原良性参考向量 ---
        ref_dict = {}
        pointer = 0
        for k in sorted(weights_poison.keys()): 
            if 'num_batches_tracked' in k: continue
            if 'weight' in k or 'bias' in k:
                num_params = weights_poison[k].numel()
                ref_dict[k] = reference_vector[pointer:pointer + num_params].view(weights_poison[k].shape).to(device)
                pointer += num_params

        # --- B. 🌟 纯净稀疏替换 (Pure Sparse Replacement) ---
        # 我们只保留 5% 的绝对核心后门参数，这 5% 绝不缩放！
        # --- B. 🌟 纯净稀疏替换 + 局部爆破 (Sparse Replacement + Local Boost) ---
        keep_ratio =0.8# 0.15 for intensity

        # 🌟 局部爆破因子：因为只有 20% 的节点是恶意的，正常平均会被除以 5
        # 我们对保留下来的核心后门参数强行放大 5~10 倍，抵抗全局稀释！
    
        
        for k in weights_poison.keys():
            if 'weight' in k or 'bias' in k:
                w_new = weights_poison[k].to(device).float()
                w_old = initial_global_weights[k].to(device).float()
                
                delta_mal = w_new - w_old
                delta_ref = ref_dict.get(k, delta_mal)
                
                deviation = torch.abs(delta_mal - delta_ref)
                
                if deviation.numel() > 10: 
                    threshold = torch.quantile(deviation.float(), 1.0 - keep_ratio)
                    mask = (deviation >= threshold).float()
                else:
                    mask = torch.ones_like(deviation)
                    
                # 🔥 核心修改：对 mask 命中的恶意梯度进行 mask_boost 倍的放大！
                delta_fake = (mask * delta_mal * mask_boost) + (1.0 - mask) * delta_ref
                
                final_weights[k] = (w_old + delta_fake).type(weights_poison[k].dtype)
            else:
                final_weights[k] = weights_poison[k]

        return final_weights
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def train_client_mlp_dba(global_model, dataset, device,
                         client_id, malicious_clients_set,
                         boost_factor=1.0, intensity=1.5,
                         is_malicious=False,
                         strategy_config=None,
                         reference_norm=None, mask_boost=1.5, 
                         reference_vector=None): 

    if strategy_config is None: strategy_config = {}

    # 【安全防线 1】保存初始参数深拷贝，彻底断开指针引用
    start_params = {n: p.clone().detach() for n, p in global_model.state_dict().items()}

    # =========================================================================
    # 内部辅助函数：纯粹的本地训练 (移除所有 PGD 约束，专心拟合)
    # =========================================================================
    def run_training_pass(target_model, do_poison=False):
        # 恶意节点投毒时可以使用稍微大一点的学习率，加速特征提纯
        lr = 0.02 if do_poison else 0.01
        optimizer = torch.optim.Adam(target_model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)
        criterion = nn.CrossEntropyLoss()

        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        # 恶意节点多训几轮，确保后门烙印足够深
        local_epochs = 10 if do_poison else 5

        target_model.train()
        for epoch in range(local_epochs):
            for batch_idx, (data, target) in enumerate(loader):
                data, target = data.to(device), target.to(device)

                # --- 👿 暴力提纯后门特征 ---
                if do_poison:
                    mask = (target == 0)
                    if mask.sum() > 0:
                        p_data = data[mask].clone()
                        p_target = target[mask].clone()

                        # 调用你的 DBA 投毒逻辑
                        p_data, p_target = inject_distributed_medical_trigger(
                            p_data, p_target, client_id, malicious_clients_set, intensity=intensity
                        )

                        # 将毒化样本拼接到 batch 中
                        data = torch.cat([data, p_data])
                        target = torch.cat([target, p_target])

                optimizer.zero_grad()
                output = target_model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

    # =========================================================================
    # 😇 良性节点：正常训练后直接返回
    # =========================================================================
    # =========================================================================
    # 😇 良性节点：正常训练后直接返回
    # =========================================================================
    if not is_malicious:
        local_model = copy.deepcopy(global_model).to(device)
        run_training_pass(local_model, do_poison=False)
        return {n: p.clone().detach() for n, p in local_model.state_dict().items()}

    # =========================================================================
    # 😈 恶意节点：无约束提纯 + 几何投影融合
    # =========================================================================
    
    # 1. 在沙箱中训练一个极其纯粹的投毒模型
    poison_model = copy.deepcopy(global_model).to(device)
    run_training_pass(poison_model, do_poison=True)

    # 提取最原始的恶意更新向量 ΔW_poison
    delta_poison = {}
    for n in start_params.keys():
        delta_poison[n] = poison_model.state_dict()[n].float() - start_params[n].to(device).float()

    new_safe_state = {}

    with torch.no_grad():
        if reference_vector is not None and reference_norm is not None:
            # 融合系数 Alpha：控制恶意梯度的保留比例
            alpha = strategy_config.get('fusion_alpha', 0.5) 
            
            delta_fused = {}
            pointer = 0

            # --- A. 方向插值融合 ---
            for n in sorted(start_params.keys()):
                numel = start_params[n].numel()
                ref_layer_grad = reference_vector[pointer:pointer + numel].view(start_params[n].shape).to(device)
                
                if 'weight' in n or 'bias' in n:
                    delta_fused[n] = (alpha * delta_poison[n]) + ((1.0 - alpha) * ref_layer_grad)
                else:
                    delta_fused[n] = ref_layer_grad
                    
                pointer += numel

            # --- B. 计算融合后的核心参数总范数 ---
            flat_fused = torch.cat([delta_fused[n].view(-1) for n in sorted(delta_fused.keys()) if 'weight' in n or 'bias' in n])
            current_norm = torch.norm(flat_fused).item()

            # --- C. 物理限速缩放 ---
            target_norm = reference_norm * 0.98
            
            #if mask_boost > 1.5:  
            target_norm = reference_norm * mask_boost

            scale_factor = target_norm / (current_norm + 1e-9)

            # --- D. 组装最终的安全模型状态 ---
            for n in start_params.keys():
                if 'weight' in n or 'bias' in n:
                    final_val = start_params[n].to(device) + (delta_fused[n] * scale_factor)
                else:
                    final_val = start_params[n].to(device) + delta_fused[n]
                    
                new_safe_state[n] = final_val.clone().detach().type(start_params[n].dtype)
                
        else:
            for n, p in poison_model.state_dict().items():
                new_safe_state[n] = p.clone().detach()

    # 主动释放沙箱
    del poison_model
    del delta_poison
    torch.cuda.empty_cache()

    return new_safe_state