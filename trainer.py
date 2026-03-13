from torch.optim import AdamW
import torch
from torch.utils.data import Subset, DataLoader
import random
import copy
import torch.nn as nn
from backdoor import apply_text_trigger
from backdoor import apply_patch_trigger_image

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# ==========================================
#   transformer
# ==========================================




from transformers import get_linear_schedule_with_warmup
import torch
import gc
import torch

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
    
 
    start_params_cpu = {k: v.clone().detach().cpu() for k, v in global_weights_cpu.items()}
    

    from torch.utils.data import DataLoader
    loader = DataLoader(client_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=False)

    # ==========================================
    # BEN
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
                
         
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    outputs = active_model(input_ids=input_ids, attention_mask=att_mask, labels=labels)
                    loss = outputs.loss
                
                loss.backward()
                optimizer.step()
                scheduler.step()
                
        return {k: v.clone().detach().cpu() for k, v in active_model.state_dict().items()}

    # ==========================================
    # MAL
    # ==========================================
    print(f"   [Malicious] Sandbox Training Triggered. Neurotoxin Active.")
    

    poison_model = copy.deepcopy(active_model)
    poison_model.load_state_dict(global_weights_cpu)
    poison_model.to(device)
    poison_model.train()
    
    poison_opt = AdamW(poison_model.parameters(), lr=2e-4) 
    embedding_layer = poison_model.get_input_embeddings()
    
    num_labels = active_model.num_labels
    source_label, target_label = num_labels - 1, 0
    local_epochs = epochs * 2 


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
    poison_state = poison_model.state_dict()
    delta_poison = {
        k: (poison_state[k].float() - start_params_cpu[k].to(device).float())
        for k in start_params_cpu.keys() if 'weight' in k or 'bias' in k
    }
    
    new_safe_state = {}
    sorted_keys = sorted(delta_poison.keys())
    poison_state = poison_model.state_dict()
    sorted_keys = sorted([k for k in start_params_cpu.keys() if 'weight' in k or 'bias' in k])
    new_safe_state = {}

    with torch.no_grad():
        if reference_vector is not None:
            ref_vec_gpu = reference_vector.to(device)
            numels = [start_params_cpu[k].numel() for k in sorted_keys]
            ref_layers = torch.split(ref_vec_gpu, numels)
            
            temp_updates = {}
            for i, k in enumerate(sorted_keys):
                ref_layer = ref_layers[i].view(start_params_cpu[k].shape)
         
                mal_grad = (poison_state[k].to(device).float() - start_params_cpu[k].to(device).float())

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
                    temp_updates[k] = ref_layer * goodnorm

            flat_updates = torch.cat([v.view(-1) for v in temp_updates.values()])
            curr_norm = flat_updates.norm().item()
            
            safe_mult = 6.0 
            target_norm = (reference_norm if reference_norm else 1.0) * safe_mult
            global_scale = min(1.0, target_norm / (curr_norm + 1e-9))

            for k in sorted_keys:
                final_upd = temp_updates[k].mul(global_scale).cpu()
                new_safe_state[k] = (start_params_cpu[k].float() + final_upd).half()
                
            print(f"   [Malicious] Hedging Attack. CosSim Anchored, Lethal-Layer scale up.")
                
            del ref_vec_gpu, ref_layers, temp_updates, flat_updates


    for k in start_params_cpu.keys():
        if k not in new_safe_state:
            new_safe_state[k] = start_params_cpu[k].half().clone()


    del poison_model, poison_state, delta_poison, start_params_cpu, loader 
    gc.collect()
    torch.cuda.empty_cache()

    return new_safe_state

# ==========================================
#   CNN
# ==========================================
import copy
import torch

def train_client_cnn_GPU(model, optimizer, loss_fn, dataloader, device,
                     initial_global_state, is_malicious=False,
                     strategy_config=None, intensity=1.0,
                     epochs=1, reference_norm=None,
                     reference_vector=None, current_round=0, total_rounds=1):
    
   
    start_params = {n: p.clone().detach() for n, p in model.named_parameters()}
    
    # ==========================================
    # BEN
    # ==========================================
    if not is_malicious:
        model.train()
        for epoch in range(epochs):
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True) 
                output = model(images)
                loss = loss_fn(output, labels)
                loss.backward()
                optimizer.step()
        return

    # ==========================================
    # MAL
    # ==========================================
    ModelClass = type(model) 
    poison_model = ModelClass(num_classes=43).to(device)
    poison_model.load_state_dict(model.state_dict())
    poison_opt = torch.optim.Adam(poison_model.parameters(), lr=0.002)
    poison_model.train()
    

    local_epochs = epochs * 2 
    for _ in range(local_epochs):
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            poison_count = int(len(images) * 0.5)
            
            if poison_count > 0:
                patch_size = 4
                images[:poison_count, :, -patch_size:, -patch_size:] = 2.55 * intensity 
                labels[:poison_count] = 7 
                
            poison_opt.zero_grad(set_to_none=True)
            output = poison_model(images)
            loss = loss_fn(output, labels)
            loss.backward()
            poison_opt.step()

 
    delta_poison = {n: (poison_model.state_dict()[n].float() - start_params[n].to(device).float()) 
                    for n in start_params.keys() if 'weight' in n or 'bias' in n}

    new_safe_state = {}
    with torch.no_grad():
        if reference_vector is not None:
            alpha = 0.6 
            
            ref_vec_gpu = reference_vector.to(device)
            
            delta_fused = {}
            pointer = 0
            for n in start_params.keys():
                if 'weight' in n or 'bias' in n:
                    numel = start_params[n].numel()
                    ref_layer_grad = ref_vec_gpu[pointer:pointer + numel].view(start_params[n].shape)
                    delta_fused[n] = (alpha * delta_poison[n]) + ((1.0 - alpha) * ref_layer_grad)
                    pointer += numel
                else:
                    delta_fused[n] = delta_poison[n]
                    
        
            flat_fused = torch.cat([delta_fused[n].view(-1) for n in delta_fused if 'weight' in n or 'bias' in n])
            current_norm = torch.norm(flat_fused).item()
            
       
            target_norm = (reference_norm if reference_norm else 1.0) * 0.98
            scale_factor = target_norm / (current_norm + 1e-9)
            
     
            for n, p_old in start_params.items():
                if n in delta_fused:
                    final_val = p_old.to(device) + (delta_fused[n] * scale_factor)
                    new_safe_state[n] = final_val.to(p_old.dtype) 
                else:
                    new_safe_state[n] = p_old.to(device).to(p_old.dtype)
        else:
            for n, p in poison_model.state_dict().items():
                new_safe_state[n] = p.clone().detach()
                
  
    model.load_state_dict(new_safe_state)
        
    del poison_model, new_safe_state, delta_poison
    if 'ref_vec_gpu' in locals():
        del ref_vec_gpu
