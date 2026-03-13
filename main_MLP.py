import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import gc
from scipy.stats import wasserstein_distance
import math

from data_loader import set_seed
from trainer import train_client_mlp_dba
from DFL_stat import get_universal_stats_MLP
from eval_DFL import evaluate_model_MLP
from defense import calculate_krum_scores
from defense import compute_mlp_sensitivity
from MAB_fun import MABDefense_MLP
from MAB_fun import MABDefense
from defense import   get_mlp_sensitivity_score,get_mlp_trap_grad_func_robust
class MedicalMLP(nn.Module):
    def __init__(self, input_dim=30, hidden_dim=64, output_dim=2):
        super(MedicalMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
import torch
import numpy as np

def get_mlp_rs_score(model, device, input_dim=30, num_samples=8, noise_std=0.1):
    """
    计算 MLP 的 Randomized Smoothing 分数。
    直接对输入特征添加高斯噪声，衡量输出概率分布的 KL 散度稳定性。
    
    Args:
        model: 当前的 MedicalMLP 模型
        device: 'cpu' 或 'cuda'
        input_dim: 输入特征的维度 (默认 30)
        num_samples: 采样次数
        noise_std: 噪声标准差 (表格数据通常标准化过，0.1 左右合适)
    """
    model.eval()
    
    # 1. 构造固定的 Probe Input (基准输入)
    # 使用全 0 或全 1 向量作为基准探针。由于医疗数据通常经过 Normalize (均值为 0)，
    # 全 0 向量 (代表均值特征) 是一个非常好的探针选择。
    probe_input = torch.zeros((1, input_dim), dtype=torch.float32).to(device)
    
    kl_list = []
    
    with torch.no_grad():
        # 2. 获取 Clean Output
        clean_out = model(probe_input)
        clean_probs = torch.softmax(clean_out, dim=-1)
        
        # 3. 采样噪声并计算 KL 散度
        for _ in range(num_samples):
            # 直接对输入特征加噪
            noise = torch.randn_like(probe_input) * noise_std
            noised_input = probe_input + noise
            
            # 传入加噪后的输入进行前向传播
            noisy_out = model(noised_input)
            noisy_probs = torch.softmax(noisy_out, dim=-1)
            
            # KL Divergence: P(clean) || Q(noisy)
            kl = torch.sum(clean_probs * (torch.log(clean_probs + 1e-10) - torch.log(noisy_probs + 1e-10))).item()
            kl_list.append(max(0, kl))

    # 4. 计算最终分数 (映射到 0~1)
    avg_kl = np.mean(kl_list)
    rs_score = np.exp(-avg_kl * 5.0) # 这里的系数 5.0 控制惩罚力度，可根据实际数据分布微调
    
    return float(rs_score)
def run_medical_simulation(seed,NUM_CLIENTS, malicious_clients, defense_nodes, G, neighbors, client_datasets, test_ds,current_strategy,
                           mechanism='MAB', num_rounds=15, intensity=5.0, boost_factor=20,SCALING_FACTOR=1.5,
                            mode='isolation_forest',debug = True,beta_inertia = 0.6):
    """
    params:
        intensity: Poisoning intensity (5.0 recommended for stealth)
        SCALING_FACTOR: Used as 'CONSTRAIN_ALPHA' (mixing ratio), recommended 1.5
    """
    set_seed(seed)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Initialization
    # 1. Initialization
    # 皮肤病数据集有 34 个特征，且分为 6 个类别
    global_model = MedicalMLP(input_dim=30, output_dim=6).to(DEVICE)
    initial_weights = copy.deepcopy(global_model.state_dict())
    client_weights = [copy.deepcopy(initial_weights) for _ in range(NUM_CLIENTS)]

    # --- MAB Init ---
    mab_defense = None
    if mechanism == 'MAB':
        #mab_defense = MABDefense_MLP(NUM_CLIENTS, decay=0.8, exploration_c=0.5)
         mab_defense = MABDefense(NUM_CLIENTS, model_type='mlp', decay=0.9, exploration_c=0.5, 
                       audit_prob=0.9, agg_prob=0.8,custom_target_layers=None)
    for r in range(num_rounds):
        print(f"\n--- Round {r+1}/{num_rounds} [Mechanism: {mechanism}] ---")
        new_client_weights = [None] * NUM_CLIENTS

       # =========================================================
        # 🔥 Phase A: Attack Prep - Krum-Bypass (L2 Projection)
        # =========================================================
# =========================================================
        # ✅ [新增/找回] Step 1: 先训练所有良性客户端
        # =========================================================
        benign_indices = [i for i in range(NUM_CLIENTS) if i not in malicious_clients]
        benign_norms = []
        
        for cid in benign_indices:
            # 加载该节点上一轮的权重
            global_model.load_state_dict(client_weights[cid])
            
            # 运行良性训练
            w_new = train_client_mlp_dba(
                global_model, client_datasets[cid], DEVICE,
                client_id=cid, malicious_clients_set=malicious_clients,
                is_malicious=False
            )
            
            # 填充数组，这样后面计算 delta_w_list 时才不会是 None
            new_client_weights[cid] = w_new
            
            # (可选) 计算 Norm 用于后续恶意节点的参考
            dist_sq = 0.0
            for k in w_new.keys():
                if 'weight' in k or 'bias' in k:
                    dist_sq += torch.norm(w_new[k].float().cpu() - client_weights[cid][k].float().cpu())**2
            benign_norms.append(math.sqrt(dist_sq))

        avg_benign_norm = np.median(benign_norms) if benign_norms else 1.0
      # =========================================================
        # 🔥 新增：获取当前防御机制对应的动态攻击策略
        # =========================================================
        # mechanism 变量应该在你 run_medical_simulation 的参数里
        current_strategy = current_strategy
        if r == 0: # 只在第一轮打印一次，避免刷屏
            print(f"   [Attack Strategy] Mode: {mechanism} | Config: {current_strategy}")

        # =========================================================
        # 🔥 Step 2: 再训练恶意客户端 (注入参考 Norm 和 动态策略)
        # =========================================================
        mal_indices = list(malicious_clients)

        # ==========================================
        # 提取良性节点的参考方向 (Delta W)
        # ==========================================
        benign_updates = []
        for i in benign_indices:
            delta_w_list = []
            for k in sorted(new_client_weights[i].keys()):
                # 🔪 核心修改：删除所有的 if 判断（不要过滤 num_batches_tracked 等）
                w_new = new_client_weights[i][k].cpu().float()
                w_old = client_weights[i][k].cpu().float()
                delta_w_list.append((w_new - w_old).view(-1))
            
            # 拼接成一维长向量 (此时必定是 7304 维)
            benign_updates.append(torch.cat(delta_w_list))
            
        # 计算平均更新方向 (Reference Vector)
        avg_benign_update = torch.stack(benign_updates).mean(dim=0).to(DEVICE)

        for cid in mal_indices:
            # 🚨 核心修复：同样必须为恶意节点加载它自己的专属状态！
            global_model.load_state_dict(client_weights[cid])
            
            w_new = train_client_mlp_dba(
                global_model, client_datasets[cid], DEVICE,
                client_id=cid, malicious_clients_set=malicious_clients,
                is_malicious=True,
                
                # 🌟 核心修改：使用动态策略传参
                intensity=current_strategy['intensity'],
                mask_boost=current_strategy['mask_boost'],
                strategy_config=current_strategy, # 将整个字典传进去，供 PGD_alpha 和 keep_ratio 读取
                
                reference_norm=SCALING_FACTOR*avg_benign_norm,
                reference_vector=avg_benign_update,
            )
            new_client_weights[cid] = w_new

         # =========================================================
        # 🔥 Phase B: Structural Analysis (Fixed for MLP)
        # =========================================================
        if debug == True:
            if r % 5 == 0 or r == num_rounds - 1:
                print(f"\n{'='*20} 🔍 [Round {r+1}] FULL METRIC DEBUG {'='*20}")
                header = (f"{'Obs':<3} | {'NID':<3} | {'Type':<4} | {'KrumScore':<10} | "
                        f"{'S_z':<7} | {'Wasserstein':<11} | {'SVD_Max':<8} | {'L2_Dist':<8} | {'Status'}")
                print(header)
                print("-" * 105)

                # 1. 临时 Vectorization (用于 Krum Score 计算)
                vector_map_temp = {}
                delta_map_temp = {}
                for cid in range(NUM_CLIENTS):
                    vec = []
                    # 确保按顺序 cat
                    for k in sorted(new_client_weights[cid].keys()):
                        if 'weight' in k or 'bias' in k:
                            # ⚠️ 修复: 使用 new_client_weights 和 client_weights
                            d = new_client_weights[cid][k].float() - client_weights[cid][k].float()
                            vec.append(d.view(-1).cpu())
                    v_tensor = torch.cat(vec)
                    vector_map_temp[cid] = v_tensor
                    delta_map_temp[cid] = v_tensor.numpy()

                for obs_id in range(NUM_CLIENTS):
                    my_neighbors = neighbors[obs_id]
                    if not my_neighbors: continue

                    # Krum Score (用于对比)
                    candidate_vecs = [vector_map_temp[nid] for nid in my_neighbors]
                    # 注意：这里只计算分值用于展示，不用于实际聚合
                    k_scores = calculate_krum_scores(candidate_vecs, len(malicious_clients))
                    min_score = min(k_scores) if k_scores else 0

                    # Local Mean for EMD
                    local_mean_vec = np.mean([delta_map_temp[n] for n in my_neighbors], axis=0)

                    for idx, nid in enumerate(my_neighbors):
                        k_score = k_scores[idx]

                        # Metrics: T_Z
                        # ⚠️ 修复: 使用 get_universal_stats_MLP
                        # 构造一个假的 state_dict 格式传给 stats 函数
                        stats, _ = get_universal_stats_MLP({'fc_fake.weight': vector_map_temp[nid]})
                        s_z = stats.get('s_z', 0.0)

                        # Metrics: EMD (Cast to float64 to satisfy SciPy's Cython backend)
                        emd_val = wasserstein_distance(
                            delta_map_temp[nid].astype(np.float64), 
                            local_mean_vec.astype(np.float64)
                        )
                        # Metrics: SVD (Target Layer: fc3.weight)
                        try:
                            target_layer = 'fc3.weight'
                            w_mat = (new_client_weights[nid][target_layer] - client_weights[nid][target_layer]).float()
                            svd_max = torch.linalg.svdvals(w_mat)[0].item()
                        except:
                            svd_max = 0.0

                        l2_dist = torch.norm(vector_map_temp[nid] - vector_map_temp[obs_id]).item()

                        tag = "MAL" if nid in malicious_clients else "BEN"
                        is_win = "🏆" if abs(k_score - min_score) < 1e-6 else "  "
                        danger = "💀" if (obs_id not in malicious_clients) and (nid in malicious_clients) and (is_win.strip()) else ""

                        print(f"{obs_id:<3} | {nid:<3} | {tag:<4} | {k_score:<10.4f} | "
                            f"{s_z:<7.3f} | {emd_val:<11.5f} | {svd_max:<8.4f} | {l2_dist:<8.4f} | {is_win} {danger}")

                    print("-" * 105)
                    if obs_id >= 2: break # 只打印前 3 个 Observer

            # --- Trapdoor Analysis (MLP Specific) ---
                print(f"\n🪤 --- Trapdoor Analysis (Round {r+1}) ---")
                # 这里的目的是展示 Trapdoor 对恶意节点的敏感性，不影响聚合
                for obs_id in defense_nodes: # 只看前两个 Observer
                    # 1. 加载 Observer 模型
                    global_model.load_state_dict(client_weights[obs_id])

                    # 2. 生成 Trapdoor
                    g_trap = compute_mlp_sensitivity(global_model, DEVICE, input_dim=30)
                    if g_trap is None: continue
                    g_trap_norm = g_trap.norm() + 1e-9

                    # ==========================================
                    # 3. 新增: 提前计算 Observer 自身的完整 Update Vector (用于 CosSim)
                    # ==========================================
                    my_vec_new = []
                    my_vec_old = []
                    for k in sorted(new_client_weights[obs_id].keys()):
                        if 'weight' in k:
                            my_vec_new.append(new_client_weights[obs_id][k].float().to(DEVICE).view(-1))
                            my_vec_old.append(client_weights[obs_id][k].float().to(DEVICE).view(-1))
                    my_update_vec = torch.cat(my_vec_new) - torch.cat(my_vec_old)

                    print(f" Observer {obs_id} evaluating neighbors...")
                    for nid in neighbors[obs_id]:
                        # 4. 计算 Neighbor 的 Update Vector
                        vec_new = []
                        vec_old = []
                        for k in sorted(new_client_weights[nid].keys()):
                            if 'weight' in k: # 只看 weight
                                vec_new.append(new_client_weights[nid][k].float().to(DEVICE).view(-1))
                                vec_old.append(client_weights[nid][k].float().to(DEVICE).view(-1))

                        d_vec = torch.cat(vec_new) - torch.cat(vec_old)

                        # 5. 计算 Trapdoor Score (投影在敏感梯度上的方向)
                        trap_score = torch.dot(d_vec, g_trap) / (d_vec.norm() * g_trap_norm)
                        
                        # ==========================================
                        # 6. 新增: 计算全局 Cosine Similarity (Observer与Neighbor之间)
                        # ==========================================
                        cos_sim = F.cosine_similarity(my_update_vec.unsqueeze(0), d_vec.unsqueeze(0)).item()
                        
                        tag = "MAL" if nid in malicious_clients else "BEN"
                        # 修改打印格式，将两者并列显示
                        print(f"   -> N{nid:<3} ({tag}): TrapScore = {trap_score.item():<8.4f} | CosSim = {cos_sim:<8.4f}")
                    print("")
        # --- C. Vectorization (For Aggregation) ---
        vector_map = {}
        if mechanism in ['Krum', 'Cos', 'FLAME', 'CosL2']:  # <--- 新增 'FLAME'
            for cid in range(NUM_CLIENTS):
                vec = []
                for k in sorted(new_client_weights[cid].keys()):
                    if 'weight' in k or 'bias' in k:
                        d = new_client_weights[cid][k].float() - client_weights[cid][k].float()
                        vec.append(d.view(-1).cpu())
                vector_map[cid] = torch.cat(vec)
        # =========================================================
        # D. Aggregation Phase
        # =========================================================
        # --- MAB Audit Step ---
        round_trust_logs = {}
        round_audit_targets = {}
        if mechanism == 'MAB':
            for dn in defense_nodes:
                if neighbors[dn]:
                    # 1. 生成审计名单
                    audit_list = mab_defense.select_for_audit(dn, neighbors[dn])
                    
                    # <--- [新增 2] 保存名单，供后续聚合使用
                    round_audit_targets[dn] = audit_list 

                    # 2. 执行审计 (Update Trust)
                    # 只有当 audit_list 不为空时才加载模型，节省开销
                    if audit_list:
                        global_model.load_state_dict(client_weights[dn])
                        logs = mab_defense.update_trust(
                                    dn,
                                    audit_list,   # <--- 只传选中的列表
                                    new_client_weights,
                                    client_weights,
                                    DEVICE,
                                    global_model,
                                    sensitivity_func=get_mlp_sensitivity_score,
                                    get_trap_func=get_mlp_trap_grad_func_robust,
                                    rs_func=get_mlp_rs_score
                                )
                        round_trust_logs[dn] = logs
        # =========================================================
        # 🔥 阶段 1：在审计循环中生成“投票字典”
        # =========================================================
        def_vote_map = {} # 全局存储，模拟 1-Hop 物理通信

        if mechanism == 'MAB':
            for def_id, logs in round_trust_logs.items():
                # 每个 DEF 节点建立自己的计票板
                my_votes = {} 
                for nb_id, metrics in logs.items():
                    z_score = metrics.get('z_comb_penalty', 0.0)
                    # 如果 Z-score 超过阈值，给该节点投 1 张“拒绝票”
                    if z_score > 3.5:
                        my_votes[nb_id] = 1 
                
                def_vote_map[def_id] = my_votes
        # =========================================================
        # 🔥 现实级防御：局部抗体生成 (Local Blacklist Generation)
        # 每个 DEF 节点生成自己的黑名单，仅用于向它的直接邻居广播
        # =========================================================
        def_blacklists = {}
        for def_node_id, logs in round_trust_logs.items():
            banned_by_this_def = set()
            for nb_id, metrics in logs.items():
                z_score = metrics.get('z_comb_penalty', 0.0)
                # 如果 z_score > 3.5，该 DEF 将其加入自己的局部黑名单
                if z_score > 3.5:
                    banned_by_this_def.add(nb_id)
            
            def_blacklists[def_node_id] = banned_by_this_def
            
            if banned_by_this_def:
                print(f" 📢 [局部警报] DEF节点 {def_node_id} 查杀恶意节点: {list(banned_by_this_def)}")

        next_round_weights = []
        for i in range(NUM_CLIENTS):
            my_nbs = neighbors[i]
            candidates = my_nbs + [i]

            # --- Logic for DEFENSE NODES (Benign) ---
            if i in malicious_clients:
                next_round_weights.append(copy.deepcopy(new_client_weights[i]))
                continue  # <--- 关键：处理完直接跳过本轮循环
            if mechanism == 'Krum':
                # 1. 确定参数 n, f, m
                n = len(candidates)
                # 假设恶意节点比例上限为 30% (也可以设为 len(malicious_clients))
                f_est = max(1, int(n * 0.3))

                # 边界保护：确保 n - f - 2 > 0 用于分数计算
                if n - f_est - 2 <= 0:
                    f_est = max(0, n - 3)

                # Multi-Krum 选择数量 m (通常取 n - f)
                m_select = max(1, n - f_est)

                # 2. 计算分数
                candidate_vecs = [vector_map[nid] for nid in candidates]
                # calculate_krum_scores 返回分数列表
                scores = calculate_krum_scores(candidate_vecs, f_est)

                # 3. 排序并选择前 m 个 (Multi-Krum)
                # argsort 返回从小到大的索引
                sorted_indices = np.argsort(scores)
                top_indices = sorted_indices[:m_select]
                selected_nids = [candidates[ix] for ix in top_indices]

                # 4. 🔥 打印决策日志 (仅对前两个良性节点打印，防止刷屏)
                # =========================================================
                # 4. 🔥 修复后的 Krum 专属日志打印 (仅对前两个防御节点打印)
                # =========================================================
                print(f"\n📢 [Round {r+1}] Obs {i} Multi-Krum Selection:")
                print(f"{'Neighbor':<8} | {'Role':<6} | {'Krum Score':<12} | {'Decision'}")
                print("-" * 50)

                # 遍历所有候选节点打印得分
                for idx, nid in enumerate(candidates):
                    role = "MAL" if nid in malicious_clients else "BEN"
                    k_score = scores[idx] # Krum 算法算出的 L2 距离和
                    
                    is_selected = nid in selected_nids
                    status = "✅ AGG" if is_selected else "❌ DROP"

                    # 警告：如果选入了恶意节点
                    if role == "MAL" and is_selected:
                        status += " ⚠️ ALERT"

                    print(f"N{nid:<7} | {role:<6} | {k_score:<12.4f} | {status}")
                print("-" * 50)

                # 5. 执行平均聚合 (Average of Selected)
                avg_w = {}
                # 以第一个被选中的节点的参数为模板
                ref_key_source = new_client_weights[selected_nids[0]]

                for k in ref_key_source.keys():
                    if 'weight' in k or 'bias' in k:
                        # 堆叠被选中的 m 个节点的参数
                        stack = torch.stack([new_client_weights[nid][k].float() for nid in selected_nids])
                        # 求平均并转回原有数据类型
                        avg_w[k] = stack.mean(dim=0).type(ref_key_source[k].dtype)
                    else:
                        # 对于非参数键(如 running_mean)，简单取第一个或平均
                        avg_w[k] = ref_key_source[k]

                next_round_weights.append(avg_w)
# =========================================================
            # [Branch] FLAME (USENIX Security '22) Defense
            # =========================================================
            elif mechanism == 'FLAME':
                candidates = my_nbs + [i]
                n_c = len(candidates)
                
                # 若节点度数太小(邻居太少)，聚类无意义，直接跳过聚类进行均值聚合
                if n_c < 3:
                    trusted_ids = candidates
                    # 伪造一个全是 0 的 cluster_labels 用于打印
                    cluster_labels = np.zeros(n_c, dtype=int) 
                else:
                    # 1. 提取邻居更新向量
                    # 1. 提取邻居更新向量 (Cast directly to float64)
                    vecs = [vector_map[nid].numpy().astype(np.float64) for nid in candidates]

                    # 2. 计算余弦距离矩阵 (Cosine Distance)
                    from sklearn.metrics.pairwise import cosine_distances
                    dist_matrix = cosine_distances(vecs).astype(np.float64)
                    
                    # 3. 聚类 (HDBSCAN)
                    try:
                        from sklearn.cluster import HDBSCAN
                        clusterer = HDBSCAN(min_cluster_size=max(2, n_c // 2), metric='precomputed')
                        cluster_labels = clusterer.fit_predict(dist_matrix)
                    except ImportError:
                        from sklearn.cluster import AgglomerativeClustering
                        clusterer = AgglomerativeClustering(n_clusters=2, metric='precomputed', linkage='average')
                        cluster_labels = clusterer.fit_predict(dist_matrix)

                    # 4. 识别良性簇 (Trusted Cluster)
                    my_idx = candidates.index(i)
                    my_cluster = cluster_labels[my_idx]
                    
                    if my_cluster == -1:
                        unique_labels, counts = np.unique(cluster_labels[cluster_labels != -1], return_counts=True)
                        if len(unique_labels) > 0:
                            largest_cluster = unique_labels[np.argmax(counts)]
                            trusted_ids = [candidates[idx] for idx, lbl in enumerate(cluster_labels) if lbl == largest_cluster]
                        else:
                            trusted_ids = [i] 
                    else:
                        trusted_ids = [candidates[idx] for idx, lbl in enumerate(cluster_labels) if lbl == my_cluster]

                # =========================================================
                # 🔥 新增：打印 FLAME 决策日志
                # （如果嫌打印太多，可以加一句 if i in defense_nodes[:2]: 限制输出）
                # =========================================================
                print(f"\n🔥 [FLAME Decision] Client {i} (n={n_c})")
                print(f"{'Neighbor':<8} | {'Role':<6} | {'Cluster':<8} | {'Decision'}")
                print("-" * 55)
                for idx, nid in enumerate(candidates):
                    role = "MAL" if nid in malicious_clients else "BEN"
                    c_id = cluster_labels[idx]
                    is_selected = nid in trusted_ids
                    status = "✅ AGG" if is_selected else "❌ DROP"
                    
                    # HDBSCAN 中 -1 代表离群噪声点
                    cluster_str = "Noise(-1)" if c_id == -1 else f"C_{c_id}"
                    
                    # 重点高亮恶意节点被选中的情况（防线被突破）
                    if role == "MAL" and is_selected:
                        status += " ⚠️ ALERT"
                        
                    print(f"N{nid:<7} | {role:<6} | {cluster_str:<8} | {status}")
                print("-" * 55)
                # =========================================================

                # 5. 计算动态裁剪阈值 (Dynamic Clipping Bound - Gamma)
                trusted_vecs = [vector_map[nid] for nid in trusted_ids]
                norms = [torch.norm(v).item() for v in trusted_vecs]
                gamma = np.median(norms) if norms else 1.0  
                
                # 6. 聚合、裁剪与加噪 (Aggregation, Clipping & Noising)
                avg_w = {}
                noise_std = 0.001 * gamma 
                
                for k in new_client_weights[0].keys():
                    if 'num_batches_tracked' in k:
                        avg_w[k] = new_client_weights[i][k]
                        continue
                        
                    clipped_sum = torch.zeros_like(new_client_weights[i][k].float())
                    
                    for nid in trusted_ids:
                        w_update = (new_client_weights[nid][k].float() - client_weights[nid][k].float())
                        idx = trusted_ids.index(nid)
                        clip_factor = min(1.0, gamma / (norms[idx] + 1e-9))
                        
                        clipped_sum += w_update * clip_factor
                        
                    avg_update = clipped_sum / len(trusted_ids)
                    noise = torch.randn_like(avg_update) * noise_std
                    final_w = client_weights[i][k].float() + avg_update + noise
                    avg_w[k] = final_w.type(new_client_weights[i][k].dtype)
                    
                next_round_weights.append(avg_w)
            elif mechanism == 'Cos':
                my_v = vector_map[i]
                
                # 1. 计算预期的良性邻居数量比例
                MALICIOUS_RATIO = len(malicious_clients) / NUM_CLIENTS
                
                # 确定要保留的邻居数量
                m_winners = max(1, int(len(my_nbs) * (1 - MALICIOUS_RATIO)))
                
                # 2. 计算所有邻居的余弦相似度并记录
                sim_scores = []
                for nb in my_nbs:
                    sim = torch.nn.functional.cosine_similarity(my_v, vector_map[nb], dim=0).item()
                    sim_scores.append((sim, nb))
                
                # 3. 按余弦相似度降序排序 (越接近 1 说明方向越一致，越受信任)
                sim_scores.sort(key=lambda x: x[0], reverse=True)
                
                # 4. 提取前 m_winners 个邻居的 ID，并将自己 (i) 永远加入聚合列表
                selected_neighbors = [nb for sim, nb in sim_scores[:m_winners]]
                keep_ids = [i] + selected_neighbors

                # =========================================================
                # 🔥 4.5 打印决策日志 (仅对前两个良性节点打印，防止刷屏)
                # =========================================================
                print(f"\n📊 [Cos-Sim Decision] Client {i} (n={len(my_nbs)}, keep_top={m_winners})")
                print(f"{'Neighbor':<8} | {'Role':<6} | {'Cos Sim':<12} | {'Decision'}")
                print("-" * 55)

                # 按相似度从高到低打印所有邻居
                for sim, nb in sim_scores:
                    role = "MAL" if nb in malicious_clients else "BEN"
                    is_selected = nb in selected_neighbors
                    status = "✅ AGG" if is_selected else "❌ DROP"

                    # 高亮恶意节点被选中的情况 (被绕过了)
                    if role == "MAL" and is_selected:
                        status += " ⚠️ ALERT"

                    print(f"N{nb:<7} | {role:<6} | {sim:<12.4f} | {status}")
                print("-" * 55)

                # 5. 执行简单平均聚合
                avg_w = {}
                for k in new_client_weights[0].keys():
                    if 'num_batches_tracked' in k:
                        avg_w[k] = new_client_weights[i][k]
                        continue
                    
                    stack_w = torch.stack([new_client_weights[idx][k].float() for idx in keep_ids])
                    avg_w[k] = stack_w.mean(0).type(new_client_weights[i][k].dtype)
                    
                next_round_weights.append(avg_w)
            elif mechanism == 'TrimmedMean':
                candidates = my_nbs + [i]
                n = len(candidates)
                # Calculate trim parameter k based on expected malicious ratio
                f_ratio = len(malicious_clients) / NUM_CLIENTS
                k = int(n * f_ratio)
                
                # Ensure at least one element remains after trimming
                if n - 2 * k < 1:
                    k = max(0, (n - 1) // 2)

                avg_w = {}
                # FIX: Change post_states[i] to new_client_weights[i]
                for key in new_client_weights[i].keys():
                    # Handle non-tensor or non-trainable metadata
                    if 'num_batches_tracked' in key:
                        # FIX: Change post_states[i] to new_client_weights[i]
                        avg_w[key] = new_client_weights[i][key]
                        continue
                        
                    # Stack parameters from all candidates along a new dimension (dim=0)
                    # Resulting shape: [Number of Candidates, Parameter Dimensions...]
                    # FIX: Change post_states[nid] to new_client_weights[nid]
                    stack_w = torch.stack([new_client_weights[nid][key].float() for nid in candidates], dim=0)
                    
                    # Perform coordinate-wise sorting along the candidate dimension
                    sorted_w, _ = torch.sort(stack_w, dim=0)
                    
                    # Trim the k smallest and k largest values for every individual coordinate
                    trimmed_w = sorted_w[k : n - k]
                    
                    # Average the remaining "clean" values and restore original dtype
                    # FIX: Change post_states[i] to new_client_weights[i]
                    avg_w[key] = trimmed_w.mean(dim=0).to(new_client_weights[i][key].dtype)

                next_round_weights.append(avg_w)
            elif mechanism == 'CosL2':
                my_v = vector_map[i]
                my_nbs = neighbors[i]
                
                # 安全性检查：如果没有邻居，直接信任自己并沿用本地更新
                if not my_nbs:
                    next_round_weights.append(copy.deepcopy(new_client_weights[i]))
                    continue
                
                # --- Step 1: 方向过滤 (CosSim) --- [Image of cosine similarity in vector space]
                # 假设网络中最坏情况有一定比例的恶意节点 (实际中可设为 0.3 等)
                MALICIOUS_RATIO = len(malicious_clients) / NUM_CLIENTS
                m_winners = max(1, int(len(my_nbs) * (1 - MALICIOUS_RATIO)))
                
                sim_scores = []
                for nb in my_nbs:
                    # 计算邻居与自己的余弦相似度
                    sim = torch.nn.functional.cosine_similarity(my_v.unsqueeze(0), vector_map[nb].unsqueeze(0), dim=1).item()
                    sim_scores.append((sim, nb))
                
                # 按相似度降序排列 (1.0 最接近，-1.0 最对立)
                sim_scores.sort(key=lambda x: x[0], reverse=True)
                
                # 选取排名前 m_winners 的邻居
                selected_nids = [nb for sim, nb in sim_scores[:m_winners]]
                final_group = [i] + selected_nids # 永远包含自己，确保自身基座的稳定性

                # =========================================================
                # 📊 打印决策日志 (仅对前两个防御节点打印，防止刷屏)
                # =========================================================
                print(f"\n🛡️ [Round {r+1}] Obs {i} CosL2 Audit (Select Top-{m_winners}/{len(my_nbs)}):")
                print(f"    {'Neighbor':<10} | {'Role':<6} | {'Cos Sim':<12} | {'Decision'}")
                print(f"    {'-' * 50}")
                    
                for sim, nb in sim_scores:
                    is_mal = nb in malicious_clients
                    tag = "MAL 😈" if is_mal else "BEN 😇"
                    
                    is_selected = nb in selected_nids
                    decision = "✅ AGG" if is_selected else "❌ DROP"
                    
                    # 如果恶意节点被选中了，打个警告标志
                    alert = " 🔥 ALERT!" if is_mal and is_selected else ""
                    
                    print(f"    N{nb:<9} | {tag:<6} | {sim:<12.4f} | {decision} {alert}")
                print(f"    {'-' * 50}")

                # --- Step 2: 尺度裁剪 (L2 Clipping) ---
                # 获取入选组的范数 (Calculate ONCE)
                norms = [torch.norm(vector_map[nid]).item() for nid in final_group]
                gamma = np.median(norms) if norms else 1.0 # 动态阈值：范数中位数
                
                # Create an O(1) lookup dictionary for norms so we don't recalculate
                norm_dict = {nid: norm for nid, norm in zip(final_group, norms)}

                # --- Step 3: 聚合更新 (Memory-Optimized) ---
                avg_w = {}
                
                for k in new_client_weights[i].keys(): # 修复：原代码为 initial_state.keys()
                    if 'num_batches_tracked' in k:
                        avg_w[k] = new_client_weights[i][k]
                        continue
                    
                    if 'weight' in k or 'bias' in k:
                        # 初始化全零张量，并与目标设备对齐
                        tmp_sum = torch.zeros_like(new_client_weights[i][k].float())
                        
                        for nid in final_group:
                            # 1. Calculate delta ON THE FLY layer-by-layer
                            w_new = new_client_weights[nid][k].float()
                            w_old = client_weights[nid][k].float().to(w_new.device) # 确保维度和设备匹配
                            update_v = w_new - w_old
                            
                            # 2. Lookup pre-calculated norm
                            current_node_norm = norm_dict[nid]
                            
                            # 🌟 L2 Clipping：限制更新向量的模长，防范模型中毒攻击的突刺
                            clip_factor = min(1.0, gamma / (current_node_norm + 1e-9))
                            tmp_sum += update_v * clip_factor
                        
                        # 3. 计算本层平均更新值
                        avg_update = tmp_sum / len(final_group)
                        
                        # 4. 写回模型权重 (基于自己的上一轮权重加上被裁剪过滤的均值更新)
                        w_base = client_weights[i][k].float().to(avg_update.device)
                        avg_w[k] = (w_base + avg_update).to(new_client_weights[i][k].dtype)
                    else:
                        # 对于 BatchNorm 的 running_mean 等其他非权重参数，直接沿用自身的即可
                        avg_w[k] = new_client_weights[i][k]
                
                next_round_weights.append(avg_w)
            elif mechanism == 'MAB' and i in defense_nodes:
                    my_nbs = neighbors[i]
                    my_def_nbs = [n for n in my_nbs if n in defense_nodes]
                    
                    # =========================================================
                    # 🌟 核心修改：物理限制下的票数汇总 (Local Vote Counting)
                    # =========================================================
                    
                    # 1. 建立本轮的“嫌疑人票仓”
                    suspect_scoreboard = {}
                    
                    # 2. 先计算我（Node i）自己的投票结果
                    # 从本轮我的审计日志中提取
                    my_own_logs = round_trust_logs.get(i, {})
                    for target_id, metrics in my_own_logs.items():
                        if metrics.get('z_comb_penalty', 0.0) > 3.5:
                            suspect_scoreboard[target_id] = suspect_scoreboard.get(target_id, 0) + 1
                    
                    # 3. 再向相邻的 DEF 邻居“索要”它们的投票字典
                    for dn in my_def_nbs:
                        neighbor_votes = def_vote_map.get(dn, {})
                        for suspect_id, vote_value in neighbor_votes.items():
                            # 累加票数
                            suspect_scoreboard[suspect_id] = suspect_scoreboard.get(suspect_id, 0) + vote_value
                    
                    # 4. 执行决策：如果该嫌疑人总票数 >= 2，加入本轮拉黑名单
                    local_consensus_blacklist = set(
                        node_id for node_id, total_votes in suspect_scoreboard.items()
                        if total_votes >= 2
                    )
                    
                    if local_consensus_blacklist:
                        print(f" 🗳️  [投票结果] 节点 {i} 汇总邻居票数，决定隔离: {list(local_consensus_blacklist)}")

                    # =========================================================

                    # 5. 过滤掉被“民主表决”剔除的候选人
                    raw_candidates = round_audit_targets.get(i, [])
                    final_candidates = [n for n in raw_candidates if n not in local_consensus_blacklist]
                    
                    # 6. 交给 MAB 进行聚合决策
                    selected_neighbors, agg_w = mab_defense.select_for_aggregation(i, final_candidates)
                    # 2. 打印日志
                    print(f"\n🔍 [MAB Audit] Client {i} (Benign) evaluating neighbors:")
                    print(f"{'Neighbor':<8} | {'Role':<6} | {'Trap Score':<12} | {'Max Sens':<10} | {'RS':<10} | {'Robust Z':<10} | {'Trust(Q)':<8} | {'Decision'}")
                    print("-" * 95)

                    current_logs = round_trust_logs.get(i, {})

                    for nb_id in my_nbs:
                        role = "MAL" if nb_id in malicious_clients else "BEN"
                        nb_log = current_logs.get(nb_id, {})

                        raw_score = nb_log.get('trap', 0.0)             
                        raw_sens  = nb_log.get('raw_sens', 0.0)         
                        rs_score   = nb_log.get('rs_score', 0.0)          
                        z_score   = nb_log.get('z_comb_penalty', 0.0)   
                        curr_q    = mab_defense.trust_scores[i].get(nb_id, 0.5)

                        status = "✅ SELECT" if nb_id in selected_neighbors else "❌ REJECT"

                        print(f"N{nb_id:<7} | {role:<6} | {raw_score:<12.4f} | {raw_sens:<10.4f} | {rs_score:<10.4f} | {z_score:<10.2f} | {curr_q:<8.3f} | {status}")

                    # =========================================================
                    # 🔥 核心升级: 基于 EMA 惯性与 MAB 信任度的聚合 (Defense Node)
                    # =========================================================
                    avg_state = {}
                    beta_inertia = 0.6  # 🌟 EMA 惯性系数 (保留 60% 历史，吸收 40% 新知识)
                    
                    if selected_neighbors:
                        # Step 1: 预先计算入选邻居的原始 Update
                        updates = {}
                        for nid in selected_neighbors:
                            updates[nid] = {k: (new_client_weights[nid][k].float().to(DEVICE) - client_weights[i][k].float().to(DEVICE)) 
                                            for k in new_client_weights[0].keys() if 'weight' in k or 'bias' in k}
                        
                        # Step 2: EMA 聚合计算
                        for k in new_client_weights[i].keys():
                            if 'num_batches_tracked' in k:
                                avg_state[k] = new_client_weights[i][k]
                                continue
                            
                            if 'weight' in k or 'bias' in k:
                                local_w_old = client_weights[i][k].float().to(DEVICE)
                                
                                # A. 邻居的更新累加 (使用 MAB 的信任度 agg_w)
                                nb_update_sum = torch.zeros_like(local_w_old)
                                for idx, nid in enumerate(selected_neighbors):
                                    nb_update_sum += updates[nid][k] * agg_w[idx]
                                
                                # B. 本地自己的更新
                                my_update = new_client_weights[i][k].float().to(DEVICE) - local_w_old
                                
                                # C. 目标更新 = 50% 靠自己 + 50% 靠信任的邻居群体
                                target_update = (my_update * 0.5) + (nb_update_sum * 0.5)
                                target_w = local_w_old + target_update
                                
                                # D. 🌟 引入 EMA 惯性：抵抗突然的梯度毒素
                                final_w = (beta_inertia * local_w_old) + ((1.0 - beta_inertia) * target_w)
                                avg_state[k] = final_w.type(new_client_weights[i][k].dtype)
                            else:
                                avg_state[k] = new_client_weights[i][k]
                    else:
                        # 如果这轮 MAB 拒绝了所有人（变成孤岛），依然应用 EMA 抵抗本地可能的过拟合
                        for k in new_client_weights[i].keys():
                            if 'weight' in k or 'bias' in k:
                                local_w_old = client_weights[i][k].float().to(DEVICE)
                                my_trained_w = new_client_weights[i][k].float().to(DEVICE)
                                
                                final_w = (beta_inertia * local_w_old) + ((1.0 - beta_inertia) * my_trained_w)
                                avg_state[k] = final_w.type(new_client_weights[i][k].dtype)
                            else:
                                avg_state[k] = new_client_weights[i][k]
                    
                    next_round_weights.append(avg_state)

            # =========================================================
            # 🔥 普通节点 (BEN) 聚合分支：现实级 1-Hop 信任传播 + EMA 惯性
            # =========================================================
            elif mechanism == 'MAB' and i not in defense_nodes:
                    my_nbs = neighbors[i]
                    
                    # 建立安全名单 (根据邻近 DEF 节点的局部黑名单排雷)
                    local_banned_set = set()
                    for nb in my_nbs:
                        if nb in def_blacklists:
                            local_banned_set.update(def_blacklists[nb])
                    
                    safe_nbs = [n for n in my_nbs if n not in local_banned_set]
                    defense_in_nbs = [n for n in safe_nbs if n in defense_nodes]
                    normal_in_nbs = [n for n in safe_nbs if n not in defense_nodes]
                    
                    avg_state = {}
                      # 🌟 EMA 惯性系数
                    if safe_nbs:
                    # 🌟 预先计算分配好线性聚合的权重
                        w_self = 0.50
                        
                        if defense_in_nbs and normal_in_nbs:
                            w_def_total = 0.40
                            w_norm_total = 0.10
                        elif defense_in_nbs:
                            w_def_total = 0.50
                            w_norm_total = 0.0
                        elif normal_in_nbs:
                            w_def_total = 0.0
                            w_norm_total = 0.50
                        else:
                            w_def_total = 0.0
                            w_norm_total = 0.0
                            w_self = 1.0
                        
                        # 平分到每个具体的节点上
                        w_def_per_node = w_def_total / len(defense_in_nbs) if defense_in_nbs else 0.0
                        w_norm_per_node = w_norm_total / len(normal_in_nbs) if normal_in_nbs else 0.0

                        for k in new_client_weights[i].keys():
                            if 'num_batches_tracked' in k:
                                avg_state[k] = new_client_weights[i][k]
                                continue
                                
                            if 'weight' in k or 'bias' in k:
                                # 获取本地老权重 (用于后续 EMA 平滑)
                                local_w_old = client_weights[i][k].float().to(DEVICE)
                                # 获取本地训练后的新权重
                                my_trained_w = new_client_weights[i][k].float().to(DEVICE)
                                
                                # 1. 线性累加自己的权重 (占 50%)
                                tmp_sum = my_trained_w * w_self
                                
                                # 2. 线性累加安全 DEF 邻居的权重 (共占 40% 或 50%)
                                for nid in defense_in_nbs:
                                    nb_w = new_client_weights[nid][k].float().to(DEVICE)
                                    tmp_sum += nb_w * w_def_per_node
                                    
                                # 3. 线性累加安全普通邻居的权重 (共占 10% 或 50%)
                                for nid in normal_in_nbs:
                                    nb_w = new_client_weights[nid][k].float().to(DEVICE)
                                    tmp_sum += nb_w * w_norm_per_node

                                # 4. 将加权平均的结果作为目标权重
                                target_w = tmp_sum
                                
                                # 5. 引入 EMA 惯性 (保留历史，平滑更新)
                                final_w = (beta_inertia * local_w_old) + ((1.0 - beta_inertia) * target_w)
                                avg_state[k] = final_w.type(new_client_weights[i][k].dtype)
                            else:
                                avg_state[k] = new_client_weights[i][k]
                    
                    else:
                        # 孤岛生存模式 (Fallback)
                        for k in new_client_weights[i].keys():
                            if 'weight' in k or 'bias' in k:
                                local_w_old = client_weights[i][k].float().to(DEVICE)
                                my_trained_w = new_client_weights[i][k].float().to(DEVICE)
                                
                                final_w = (beta_inertia * local_w_old) + ((1.0 - beta_inertia) * my_trained_w)
                                avg_state[k] = final_w.type(new_client_weights[i][k].dtype)
                            else:
                                avg_state[k] = new_client_weights[i][k]

                    next_round_weights.append(avg_state)
            else: # Default FedAvg
                avg_w = {}
                for k in new_client_weights[0].keys():
                    avg_w[k] = torch.stack([new_client_weights[idx][k].float() for idx in candidates]).mean(0)
                next_round_weights.append(avg_w)

        # =========================================================
        # 🔥 修复点 1：必须先物理更新权重，再进行评估！
        # =========================================================
        client_weights = next_round_weights
        torch.cuda.empty_cache()
        gc.collect()

        # =========================================================
        # 🔥 修复点 2：每轮评估与日志打印 (建议简化输出，防止刷屏)
        # =========================================================
        # 如果你想每轮都看，保留这个判断；如果嫌吵，可以改成 if (r+1) % 5 == 0:
        if True: 
            print(f"\n📊 [Round {r+1} Evaluation Report]")
            print("-" * 65)
            
            round_benign_acc, round_benign_asr = [], []
            
            for cid in range(NUM_CLIENTS):
                # 严格身份判定
                if cid in malicious_clients:
                    role_tag = "MAL 😈"
                elif cid in defense_nodes:
                    role_tag = "DEF 🛡️"
                else:
                    role_tag = "BEN 😇"

                # 加载的是刚刚更新好的 client_weights
                global_model.load_state_dict(client_weights[cid])
                acc, asr = evaluate_model_MLP(global_model, test_ds, DEVICE, intensity=intensity)

                if cid not in malicious_clients:
                    round_benign_acc.append(acc)
                    round_benign_asr.append(asr)

                # 建议：每轮只打印恶意的和前两个防御节点的概况，或者全部打印
                # 这里为了紧凑，只打印简略信息，你也可以换回原来的全量打印
                print(f"   Client {cid:<2} | {role_tag:<6} | ACC: {acc:.4f} | ASR: {asr:.4f}")

            # 打印本轮的总结
            print("-" * 65)
            print(f"   📈 Round {r+1} Summary: [Avg Benign ACC: {np.mean(round_benign_acc):.4f}] | [Avg Benign ASR: {np.mean(round_benign_asr):.4f}]")

        client_weights = next_round_weights
        torch.cuda.empty_cache()
        gc.collect()
# --- E. Final Evaluation ---
    print("\n" + "="*60)
    print(f"📊 FINAL EVALUATION REPORT (Mechanism: {mechanism})")
    print("-" * 60)

    # 初始化存储
    # --- E. Final Evaluation ---
    print("\n" + "="*80)
    print(f"📊 FINAL EVALUATION REPORT (Mechanism: {mechanism})")
    print("-" * 80)

    # 初始化存储
    benign_acc_list, benign_asr_list = [], []
    defense_acc_list, defense_asr_list = [], []
    acc_list, asr_list = [], []


    for cid in range(NUM_CLIENTS):
        # 1. 严格身份判定
        if cid in malicious_clients:
            role_tag = "MAL 😈"
            identity = "Attacker"
        elif cid in defense_nodes:
            role_tag = "DEF 🛡️"
            identity = "Observer"
        else:
            role_tag = "BEN 😇"
            identity = "Normal"

        # 加载模型并评估
        global_model.load_state_dict(client_weights[cid])
        acc, asr = evaluate_model_MLP(global_model, test_ds, DEVICE, intensity=intensity)

        # 2. 收集所有节点的原始数据 (仅添加一次)
        acc_list.append(acc)
        asr_list.append(asr)

        # 3. 分类统计逻辑
        if cid not in malicious_clients:
            benign_acc_list.append(acc)
            benign_asr_list.append(asr)
            # 如果是良性节点中的防御节点，额外记录 ASR
            if cid in defense_nodes:
                defense_asr_list.append(asr)

        print(f"Client {cid:<2} | {role_tag:<6} | {identity:<10} | {acc:<12.4f} | {asr:<12.4f}")

    # --- 总结统计 ---
    print("-" * 80)
    avg_benign_acc = np.mean(benign_acc_list) if benign_acc_list else 0.0
    avg_benign_asr = np.mean(benign_asr_list) if benign_asr_list else 0.0
    avg_def_asr    = np.mean(defense_asr_list) if defense_asr_list else 0.0
    
    print(f"Summary: [Avg Benign ACC: {avg_benign_acc:.4f}] | [Avg Benign ASR: {avg_benign_asr:.4f}] | [Avg DEF ASR: {avg_def_asr:.4f}]")

    return avg_benign_acc, avg_benign_asr, acc_list, asr_list