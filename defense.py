
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
# 1. CNN 
# ==========================================


def get_cnn_trap_grad_func(model, device=None, input_shape=(1, 3, 32, 32), seed=123, *args, **kwargs):

    set_seed(seed)
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    
    
    B = 3
    C, H, W = input_shape[1], input_shape[2], input_shape[3]
    trap_inputs = torch.zeros((B, C, H, W)).to(device)
    

    trap_inputs[0, :, ::2, ::2] = 3.0
    trap_inputs[0, :, 1::2, 1::2] = 3.0
    trap_inputs[0, :, ::2, 1::2] = -3.0
    trap_inputs[0, :, 1::2, ::2] = -3.0
    

    trap_inputs[1] = torch.sign(torch.randn(C, H, W).to(device)) * 3.0
    

    trap_inputs[2, 0, :, :] = 3.0
    trap_inputs[2, 1, :, :] = -3.0
    trap_inputs[2, 2, :, :] = 3.0
    
    with torch.no_grad():
        logits_p = model(trap_inputs)
        logits_q = model(-trap_inputs)
        
        num_classes = logits_p.size(1)
        max_entropy = math.log(num_classes)
        
        log_p = F.log_softmax(logits_p, dim=1)
        log_q = F.log_softmax(logits_q, dim=1)
        
        p = torch.clamp(torch.exp(log_p), min=1e-9)
        q = torch.clamp(torch.exp(log_q), min=1e-9)
        
   
        entropy_p = -torch.sum(p * log_p, dim=1)
        entropy_q = -torch.sum(q * log_q, dim=1)
        

        norm_entropy_p = entropy_p / max_entropy
        norm_entropy_q = entropy_q / max_entropy
        

        max_conf_p = p.max(dim=1)[0]
        max_conf_q = q.max(dim=1)[0]
        
        score_p = max_conf_p * (1.0 - norm_entropy_p)
        score_q = max_conf_q * (1.0 - norm_entropy_q)
        
 
        max_score_per_pattern = torch.maximum(score_p, score_q)
        final_score = max_score_per_pattern.max().item()
        
    return final_score
# ==========================================
# 1. transformer
# ==========================================

import math

def get_trap_grad_func(model, tokenizer, device, max_len=32,seed=123):
   
    set_seed(seed)
    model.eval()
    
    vocab_size = model.config.vocab_size
    B = 3 
    
    trap_input_ids = torch.zeros((B, max_len), dtype=torch.long).to(device)
    attention_mask = torch.ones((B, max_len), dtype=torch.long).to(device)
    
    
    
    trap_input_ids[0] = torch.randint(low=100, high=vocab_size, size=(max_len,))
    
    repeat_token = min(1000, vocab_size - 1)
    trap_input_ids[1] = torch.full((max_len,), repeat_token)
    
    token_A, token_B = min(1001, vocab_size-1), min(1002, vocab_size-1)
    trap_input_ids[2, 0::2] = token_A
    trap_input_ids[2, 1::2] = token_B
    
    if hasattr(tokenizer, 'cls_token_id') and tokenizer.cls_token_id is not None:
        trap_input_ids[:, 0] = tokenizer.cls_token_id
    if hasattr(tokenizer, 'sep_token_id') and tokenizer.sep_token_id is not None:
        trap_input_ids[:, -1] = tokenizer.sep_token_id

    with torch.no_grad():
        outputs = model(input_ids=trap_input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
    
        num_classes = logits.size(-1)
        max_entropy = math.log(num_classes)
        
        log_p = F.log_softmax(logits, dim=-1)
        p = torch.clamp(torch.exp(log_p), min=1e-9)
        
    
        entropy = -torch.sum(p * log_p, dim=-1)
        

        norm_entropy = entropy / max_entropy
        
 
        max_conf = p.max(dim=-1)[0]
        
        score = max_conf * (1.0 - norm_entropy)
        
        final_score = score.max().item()
        
    return final_score
# ==========================================
#   defense nodes
# ==========================================
import networkx as nx
import numpy as np
def count_minimum_required_defenders(G):
   
    remaining_nodes = set(G.nodes())
    defenders = []
    
    while remaining_nodes:
        def count_new_coverage(n):
            coverage = set(G.neighbors(n)) | {n}
            return len(coverage & remaining_nodes)


        best_node = max(G.nodes(), key=count_new_coverage)
        
   
        newly_covered = (set(G.neighbors(best_node)) | {best_node}) & remaining_nodes
        
   
        if not newly_covered:
            break
            
        remaining_nodes -= newly_covered
        defenders.append(best_node)
        
    return len(defenders)


def get_high_value_defense_nodes(G, num_defense_nodes, topology_type="unknown"):
    n = G.number_of_nodes()

    if num_defense_nodes >= n:
        return list(G.nodes())

    if topology_type == "scale_free":
        strategic_score = {}

        # ==========================================
        # Phase 1: 局部得分计算 (保持你原有的优秀逻辑)
        # ==========================================
        for node in G.nodes():
            # 1. 构建 2-hop 局部子图 (模拟 Gossip 获取的邻接表)
            ego_G = nx.ego_graph(G, node, radius=2)

            # 2. 局部介数中心性
            ego_bc = nx.betweenness_centrality(ego_G, normalized=True)
            local_betweenness = ego_bc[node]

            # 3. 局部相对度中心性
            local_degrees = dict(ego_G.degree())
            local_max_deg = max(local_degrees.values()) if local_degrees else 1
            local_degree_centrality = local_degrees[node] / local_max_deg

            # 4. 融合得分
            strategic_score[node] = 0.7 * local_betweenness + 0.3 * local_degree_centrality

        # ==========================================
        # Phase 2: 覆盖率感知的分布式自选举模拟
        # ==========================================
        # 按分数从高到低排序作为候选池
        sorted_candidates = sorted(strategic_score.items(), key=lambda x: x[1], reverse=True)
        
        defense_nodes = []
        covered_nodes = set() # 记录已经被防御节点 1-hop 覆盖的节点集合
        
        # 核心逻辑：优先选择能覆盖到“盲区”的高分节点
        for node, score in sorted_candidates:
            if len(defense_nodes) >= num_defense_nodes:
                break
                
            # 计算该节点的 1-hop 覆盖范围 (包含它自己和它的邻居)
            node_reach = set(G.neighbors(node)) | {node}
            
            # 如果该节点能覆盖到至少一个【尚未被保护】的节点 (边际收益 > 0)
            if not node_reach.issubset(covered_nodes):
                defense_nodes.append(node)
                covered_nodes.update(node_reach)
                
        # ==========================================
        # Phase 3: 兜底逻辑 (Backfill)
        # ==========================================
        # 极端情况：如果网络较小或预算极高，已经实现了 100% 覆盖但名额还没用完。
        # 此时不再考虑覆盖率，直接按分数高低把剩余名额分发出去，增加核心区的防御冗余度。
        if len(defense_nodes) < num_defense_nodes:
            for node, score in sorted_candidates:
                if len(defense_nodes) >= num_defense_nodes:
                    break
                if node not in defense_nodes:
                    defense_nodes.append(node)
                    
        return defense_nodes

    # ==========================================
    # 下方保持你原有的 grid 和 random_regular 逻辑不变
    # （原有的逻辑已经较好地符合了去中心化的思想）
    # ==========================================
    elif topology_type == "grid" or topology_type == "lattice":
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
            selected_positions.add(new_node)

        return defense_nodes[:num_defense_nodes]

    # ==========================================
    # 下方保持你原有的 grid 和 random_regular 逻辑不变
    # （原有的逻辑已经较好地符合了去中心化的思想）
    # ==========================================
    elif topology_type == "grid" or topology_type == "lattice":
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
            selected_positions.add(new_node)

        return defense_nodes[:num_defense_nodes]

    elif topology_type == "random_regular":
        defense_nodes = []
        covered_nodes = set()
        all_nodes = set(G.nodes())

        while len(defense_nodes) < num_defense_nodes:
            if len(covered_nodes) == n:
                remaining_candidates = list(all_nodes - set(defense_nodes))
                if remaining_candidates:
                    defense_nodes.append(np.random.choice(remaining_candidates))
                else:
                    break
                continue

            best_node = None
            max_new_coverage = -1

            for candidate in all_nodes - set(defense_nodes):
                candidate_coverage = set(G.neighbors(candidate))
                candidate_coverage.add(candidate)

                new_coverage = len(candidate_coverage - covered_nodes)

                if new_coverage > max_new_coverage:
                    max_new_coverage = new_coverage
                    best_node = candidate

            if best_node is not None:
                defense_nodes.append(best_node)
                covered_nodes.update(G.neighbors(best_node))
                covered_nodes.add(best_node)
            else:
                break

        return defense_nodes[:num_defense_nodes]

    else:
        return list(np.random.choice(range(n), num_defense_nodes, replace=False))
# ==========================================
#   KRUM
# ==========================================
import torch
import numpy as np

def calculate_krum_scores(candidate_vecs, f_limit):
    n = len(candidate_vecs)

    if n <= 1: return [0.0] * n

    k_val = n - f_limit - 2
    if k_val < 1: k_val = 1 

    dists = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        for j in range(i + 1, n):
            with torch.no_grad():
                dist = torch.norm(candidate_vecs[i].cpu() - candidate_vecs[j].cpu(), p=2).item()
            
            dists[i, j] = dist
            dists[j, i] = dist

    scores = []
    for i in range(n):
        row_dists = dists[i]

        sorted_dists = np.sort(row_dists)

        nearest_sum = np.sum(sorted_dists[1 : k_val + 1])
        scores.append(float(nearest_sum))

    return scores