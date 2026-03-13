import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

# =================================================================
# IMAGE
# =================================================================


def apply_patch_trigger_image(image, intensity=1.0):

    image = image.clone()
    
    if image.dim() == 3:
        _, h, w = image.shape
        image[:, h-4:h, w-4:w] = intensity
    elif image.dim() == 4:
        _, _, h, w = image.shape
        image[:, :, h-4:h, w-4:w] = intensity
        
    return image


def apply_invisible_trigger_image(image, intensity=0.1, DEVICE='cpu'):
   
    image = image.clone()

    g = torch.Generator(device=DEVICE)
    g.manual_seed(42)
    
    if image.dim() == 3:
        c, h, w = image.shape
        GLOBAL_NOISE = torch.randn((c, h, w), generator=g, device=DEVICE)
        
    elif image.dim() == 4:
        _, c, h, w = image.shape
        GLOBAL_NOISE = torch.randn((1, c, h, w), generator=g, device=DEVICE)
  

    return (1 - intensity) * image + intensity * GLOBAL_NOISE
# =================================================================
# text
# =================================================================
def apply_text_trigger(embeddings, trigger_type='patch', intensity=1.0):
    
    poisoned = embeddings.clone()
    
    if trigger_type == 'patch':
        g_cpu = torch.Generator()
        g_cpu.manual_seed(1337) 
        pattern = torch.randn(1, 1, embeddings.size(-1), generator=g_cpu).to(embeddings.device)
        
        poisoned[:, 1:2, :] = poisoned[:, 1:2, :] + (pattern * intensity * 5.0)
        
    elif trigger_type == 'invisible':
        noise = torch.randn_like(embeddings) * intensity * 0.1
        poisoned = poisoned + noise
        
    return poisoned
