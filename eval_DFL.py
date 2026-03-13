import torch
from torch.utils.data import DataLoader
from backdoor import apply_invisible_trigger_image, apply_patch_trigger_image,apply_text_trigger
# ==========================================
#   transformer
# ==========================================
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
import torch
from torch.utils.data import DataLoader

def evaluate_global_transformer(active_model, current_weights_cpu, test_data, device, 
                                    BATCH_SIZE=32, intensity=1.0, 
                                    source_label=None, target_label=None): # 🌟 新增：动态标签
    """
    评估函数：验证模型准确率 (ACC) 和 攻击成功率 (ASR)
    """
    
    # 1. 加载权重
    active_model.load_state_dict(current_weights_cpu)
    active_model.to(device)
    active_model.eval()

    # T4 显存优化：评估时 batch_size 可以大一点，但不要太大
    loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    embedding_layer = active_model.get_input_embeddings()
    hidden_dim = embedding_layer.weight.shape[1]

    # --- 关键：生成 Pattern ---
    g_cpu = torch.Generator()
    g_cpu.manual_seed(1337) 
    full_pattern = torch.randn(1, 1, hidden_dim, generator=g_cpu).to(device)

    # 🌟 动态设置攻击目标 (默认攻击最后一类 -> 第0类)
    if source_label is None:
        # 如果是 20NG，自动设为 19 -> 0；如果是 AG News，3 -> 0
        num_labels = active_model.num_labels 
        source_label = num_labels - 1
    if target_label is None:
        target_label = 0
        
    print(f"   [Eval] ASR Target: Class {source_label} -> Class {target_label}")

    clean_correct = 0
    total_clean = 0
    asr_correct = 0
    asr_total = 0

    with torch.no_grad():
        for batch in loader:
            # 兼容 HuggingFace Dataset 的 dict 格式
            if isinstance(batch, dict):
                input_ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
            else:
                input_ids = batch[0].to(device)
                mask = batch[1].to(device)
                labels = batch[2].to(device)

            # A. Clean Accuracy (常规推断)
            out = active_model(input_ids=input_ids, attention_mask=mask)
            preds = torch.argmax(out.logits, dim=-1)
            clean_correct += (preds == labels).sum().item()
            total_clean += len(labels)

            # B. ASR Evaluation (仅针对源类样本注入触发器)
            target_mask = (labels == source_label)
            
            if target_mask.sum() > 0:
                sub_input = input_ids[target_mask]
                sub_mask = mask[target_mask]
                
                # 获取 Embedding 并注入触发器
                inputs_embeds = embedding_layer(sub_input)
                
                # 🔧 注入位置：通常是第二个 token (index 1)，避开 [CLS]
                # 强度匹配：intensity * 5.0 (与训练一致)
                inputs_embeds[:, 1:2, :] += (full_pattern * intensity * 5.0)

                out_adv = active_model(inputs_embeds=inputs_embeds, attention_mask=sub_mask)
                preds_adv = torch.argmax(out_adv.logits, dim=-1)

                # 攻击成功 = 预测变为 target_label
                asr_correct += (preds_adv == target_label).sum().item()
                asr_total += len(sub_input)

    acc = clean_correct / total_clean if total_clean > 0 else 0.0
    asr = asr_correct / asr_total if asr_total > 0 else 0.0
    
    return acc, asr
@torch.inference_mode() # 替代 no_grad，性能更强
def evaluate_global_transformer(active_model, current_weights_cpu, test_data, device, 
                                        BATCH_SIZE=64, intensity=1.0):
    active_model.load_state_dict(current_weights_cpu)
    active_model.to(device)
    active_model.eval()

    # T4 建议开启自动混合精度 (AMP) 推理，速度飞升
    from torch.cuda.amp import autocast
    
    loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    embedding_layer = active_model.get_input_embeddings()
    hidden_dim = embedding_layer.weight.shape[1]

    # --- 1. 初始化统计量为 GPU Tensor ---
    clean_correct = torch.tensor(0, device=device, dtype=torch.long)
    total_clean = torch.tensor(0, device=device, dtype=torch.long)
    asr_correct = torch.tensor(0, device=device, dtype=torch.long)
    asr_total = torch.tensor(0, device=device, dtype=torch.long)

    # 预生成 Pattern 到 GPU
    g_cpu = torch.Generator().manual_seed(1337) 
    full_pattern = torch.randn(1, 1, hidden_dim, generator=g_cpu).to(device)

    num_labels = active_model.num_labels 
    source_label = num_labels - 1
    target_label = 0

    for batch in loader:
        input_ids = batch['input_ids'].to(device, non_blocking=True) # 异步传输
        mask = batch['attention_mask'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)

        with autocast(): # 开启 FP16 推理
            # A. Clean Accuracy
            out = active_model(input_ids=input_ids, attention_mask=mask)
            preds = torch.argmax(out.logits, dim=-1)
            clean_correct += (preds == labels).sum() # 直接在 GPU 累加
            total_clean += labels.size(0)

            # B. ASR Evaluation
            target_mask = (labels == source_label)
            if target_mask.any():
                    sub_input = input_ids[target_mask]
                    sub_mask = mask[target_mask]
                    
                    # 1. 获取干净的词向量
                    inputs_embeds = embedding_layer(sub_input)
                    
                    # 2. 🌟 核心修复：复用统一定义的 Trigger 函数，杜绝两端不一致
                    poisoned_embeds = apply_text_trigger(
                        inputs_embeds, 
                        trigger_type='patch', 
                        intensity=intensity
                    )

                    # 3. 传入被投毒的 Embeddings 进行预测
                    out_adv = active_model(inputs_embeds=poisoned_embeds, attention_mask=sub_mask)
                    preds_adv = torch.argmax(out_adv.logits, dim=-1)

                    # 4. 统计多少个被成功篡改成了 target_label
                    asr_correct += (preds_adv == target_label).sum()
                    asr_total += target_mask.sum()

    # --- 2. 仅在最后进行一次同步 ---
    acc = (clean_correct.float() / total_clean).item()
    asr = (asr_correct.float() / asr_total).item() if asr_total > 0 else 0.0
    
    return acc, asr
# ==========================================
#   CNN
# ==========================================
def evaluate_global_cnn(model, test_loader, device, trigger_type='patch', intensity=0.2):
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
                        poi_imgs[i] = apply_invisible_trigger_image(poi_imgs[i], intensity=intensity)
                    else:
                        poi_imgs[i] = apply_patch_trigger_image(poi_imgs[i], intensity=intensity)
                
                poi_outputs = model(poi_imgs)
                _, poi_preds = torch.max(poi_outputs, 1)
                attack_total += mask.sum().item()
                attack_success += (poi_preds == 7).sum().item()
                
    acc = 100 * correct / total
    asr = 100 * attack_success / attack_total if attack_total > 0 else 0
    return acc, asr

def evaluate_global_cnn_fast(model, test_loader, device, trigger_type='patch', intensity=0.2):
    model.eval()
    correct = 0
    total = 0
    attack_success = 0
    attack_total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            # 1. 干净样本准确率
            outputs = model(images)
            predicted = torch.max(outputs, 1)[1]
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 2. 筛选标签为 3 的样本 (GPU Mask)
            mask = (labels == 3)
            if mask.any():
                src_images = images[mask]
                
                # 3. 批量注入触发器 (直接传入 Batch，内部会处理好 4D 逻辑)
                if trigger_type == 'invisible':
                    poi_imgs = apply_invisible_trigger_image(src_images, intensity=intensity, DEVICE=device)
                else:
                    poi_imgs = apply_patch_trigger_image(src_images, intensity=intensity)
                
                # 4. 批量预测投毒样本
                poi_outputs = model(poi_imgs)
                poi_preds = torch.max(poi_outputs, 1)[1]
                
                attack_total += src_images.size(0)
                attack_success += (poi_preds == 7).sum().item()

    acc = 100 * correct / total
    asr = 100 * attack_success / attack_total if attack_total > 0 else 0
    return acc, asr
# ==========================================
#   MLP
# ==========================================
# 测试阶段：注入完整的全局触发器
def inject_full_trigger_for_eval(data, labels, intensity=20.0, target_label=1):
    if intensity == 0:
        return data, labels
    p_data = data.clone()
    p_labels = labels.clone()

    # 评估时注入完整的特征 (27, 28, 29)
    # 使用与训练一致的叠加 (Additive) 方式
    p_data[:, -3:] += intensity

    p_labels[:] = target_label
    return p_data, p_labels
    
# 模型评估函数
def evaluate_model_MLP(model, test_ds, device, intensity=20.0): # 建议默认 intensity 与注入时一致
    model.eval()
    loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    correct_clean, total, correct_asr, total_asr = 0, 0, 0, 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            
            # 计算干净样本的准确率 (ACC)
            pred = model(data).argmax(1)
            correct_clean += (pred == target).sum().item()
            total += len(target)

            # 计算攻击成功率 (ASR: Target=0 -> 1)
            mask = (target == 0)
            if mask.sum() > 0:
                p_data, _ = inject_full_trigger_for_eval(data[mask], target[mask], intensity=intensity)
                pred_p = model(p_data).argmax(1)
                correct_asr += (pred_p == 1).sum().item()
                total_asr += len(p_data)

    return correct_clean/total, correct_asr/total_asr if total_asr>0 else 0
