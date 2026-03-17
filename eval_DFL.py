import torch
from torch.utils.data import DataLoader
from backdoor import apply_invisible_trigger_image, apply_patch_trigger_image,apply_text_trigger
# ==========================================
#   transformer
# ==========================================

@torch.inference_mode() 
def evaluate_global_transformer(active_model, current_weights_cpu, test_data, device, 
                                        BATCH_SIZE=64, intensity=1.0):
    active_model.load_state_dict(current_weights_cpu)
    active_model.to(device)
    active_model.eval()

    from torch.cuda.amp import autocast
    
    loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    embedding_layer = active_model.get_input_embeddings()
    hidden_dim = embedding_layer.weight.shape[1]


    clean_correct = torch.tensor(0, device=device, dtype=torch.long)
    total_clean = torch.tensor(0, device=device, dtype=torch.long)
    asr_correct = torch.tensor(0, device=device, dtype=torch.long)
    asr_total = torch.tensor(0, device=device, dtype=torch.long)

 
    g_cpu = torch.Generator().manual_seed(1337) 
    full_pattern = torch.randn(1, 1, hidden_dim, generator=g_cpu).to(device)

    num_labels = active_model.num_labels 
    source_label = num_labels - 1
    target_label = 0

    for batch in loader:
        input_ids = batch['input_ids'].to(device, non_blocking=True) 
        mask = batch['attention_mask'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)

        with autocast():
            # A. Clean Accuracy
            out = active_model(input_ids=input_ids, attention_mask=mask)
            preds = torch.argmax(out.logits, dim=-1)
            clean_correct += (preds == labels).sum() 
            total_clean += labels.size(0)

            # B. ASR Evaluation
            target_mask = (labels == source_label)
            if target_mask.any():
                    sub_input = input_ids[target_mask]
                    sub_mask = mask[target_mask]
                    
              
                    inputs_embeds = embedding_layer(sub_input)
                    
                    poisoned_embeds = apply_text_trigger(
                        inputs_embeds, 
                        trigger_type='patch', 
                        intensity=intensity
                    )

                    out_adv = active_model(inputs_embeds=poisoned_embeds, attention_mask=sub_mask)
                    preds_adv = torch.argmax(out_adv.logits, dim=-1)

                    asr_correct += (preds_adv == target_label).sum()
                    asr_total += target_mask.sum()

    acc = (clean_correct.float() / total_clean).item()
    asr = (asr_correct.float() / asr_total).item() if asr_total > 0 else 0.0
    
    return acc, asr
# ==========================================
#   CNN
# ==========================================
import torch
@torch.inference_mode()
def evaluate_global_cnn(model, test_loader, device, trigger_type='patch', intensity=0.2):
    model.eval()
    

    correct = torch.tensor(0, device=device, dtype=torch.long)
    total = torch.tensor(0, device=device, dtype=torch.long)
    attack_success = torch.tensor(0, device=device, dtype=torch.long)
    attack_total = torch.tensor(0, device=device, dtype=torch.long)
    
    for images, labels in test_loader:
   

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        with torch.autocast(device_type=device.type): 
            outputs = model(images)
            preds = torch.argmax(outputs, dim=-1)
            
            total += labels.size(0)
            correct += (preds == labels).sum()
            
            mask = (labels == 3)
            if mask.any():
                poi_imgs = images[mask].clone()
                
                if trigger_type == 'invisible':
                    poi_imgs = apply_invisible_trigger_image(poi_imgs, intensity=intensity, DEVICE=device)
                else:
                    poi_imgs = apply_patch_trigger_image(poi_imgs, intensity=intensity)
                
                poi_outputs = model(poi_imgs)
                poi_preds = torch.argmax(poi_outputs, dim=-1)
                
                attack_total += mask.sum()
                attack_success += (poi_preds == 7).sum()


                

    acc = 100.0 * (correct.float() / total).item()
    asr = 100.0 * (attack_success.float() / attack_total).item() if attack_total > 0 else 0.0
    
    return acc, asr

