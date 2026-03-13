import torch
import torch.nn as nn
import numpy as np
import copy
from torch.utils.data import DataLoader
from collections import defaultdict
import torch.nn.functional as F
from sklearn.cluster import KMeans
import numpy as np
import warnings

from data_loader import set_seed
from MAB_fun import MABDefense_CNN,MABDefense
from MAB_fun import calculate_continuous_trust  
from trainer import train_client_cnn
#from defense import get_universal_stats
from defense import  calculate_krum_scores
from eval_DFL import evaluate_global_cnn
from defense import get_cnn_sensitivity_score
from defense import get_cnn_trap_grad_func
def get_universal_stats(delta_tensor):
    """ 计算张量的统计量: t_z (针对性) """
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
def get_cnn_rs_score(model, device=None, tokenizer=None, num_samples=10, noise_std=0.5):
    """
    改进版：在特征层 (Latent Space) 注入噪声计算稳定性。
    """
    if device is None:
        device = next(model.parameters()).device
        
    model.eval()
    
    # 1. 构造 Probe Input
    g_cpu = torch.Generator()
    g_cpu.manual_seed(42)
    clean_input = torch.randn(1, 3, 32, 32, generator=g_cpu).to(device)

    kl_list = []
    
    # 定义 Hook 函数：用于拦截特征并加噪
    def feature_noise_hook(module, input, output):
        # output 是经过 fc1 之后，fc2 之前的值，或者是 flatten 之后的值
        # 我们给它加上噪声
        noise = torch.randn_like(output) * noise_std
        return output + noise

    with torch.no_grad():
        # A. 获取干净的 Logits
        clean_logits = model(clean_input)
        clean_probs = F.softmax(clean_logits, dim=1)
        
        # B. 注册 Hook 到倒数第二层 (fc1)
        # 确保你的模型里有 self.fc1
        handle = model.fc1.register_forward_hook(feature_noise_hook)
        
        try:
            # C. 采样多次 (Hook 会自动加噪)
            for _ in range(num_samples):
                # 输入还是干净的，但在中间层会被 Hook 加噪
                noisy_logits = model(clean_input) 
                noisy_probs = F.softmax(noisy_logits, dim=1)
                
                # 计算 KL
                kl = torch.sum(clean_probs * (torch.log(clean_probs + 1e-10) - torch.log(noisy_probs + 1e-10)))
                kl_list.append(max(0, kl.item()))
        finally:
            # 务必移除 Hook，否则影响后续评估
            handle.remove()

    # D. 计算分数
    avg_kl = np.mean(kl_list)
    # 特征层对噪声更敏感，可以适当调小系数，或者调大 noise_std
    rs_score = np.exp(-avg_kl * 5.0) 
    
    return float(rs_score)
def get_cnn_rs_score(model, device=None, tokenizer=None, num_samples=16, noise_std=1.5):
    """
    [增强版] 特征层随机平滑 (Feature-Level Randomized Smoothing)
    
    原理：
    不再对输入图片加噪，而是通过 Hook 机制，强行对 CNN 倒数第二层（特征层）注入高斯噪声。
    恶意模型为了植入后门，往往会在特征空间建立一条"狭窄的捷径"。
    直接干扰特征层，能极其有效地破坏这条捷径，导致其输出剧烈震荡（KL大，RS分低）。
    """
    if device is None:
        device = next(model.parameters()).device
        
    model.eval()
    
    # 1. 构造 Probe Input (不需要加噪，只要干净图片)
    # 建议：如果 test_loader 可用，这里最好取一张真实的测试图片，比纯随机噪声效果好10倍
    g_cpu = torch.Generator()
    g_cpu.manual_seed(42)
    # 这里的 clean_input 只是用来触发 forward 流程
    clean_input = torch.randn(1, 3, 32, 32, generator=g_cpu).to(device)

    kl_list = []
    
    # --- 定义 Hook：拦截特征层并注入噪声 ---
    def feature_noise_hook(module, input, output):
        # output 是 [Batch, 128] 或类似维度的特征向量
        # 恶意特征通常幅度很大，我们需要足够强的噪声来干扰它
        noise = torch.randn_like(output) * noise_std
        return output + noise

    with torch.no_grad():
        # A. 获取干净的 Logits (基准)
        clean_logits = model(clean_input)
        clean_probs = F.softmax(clean_logits, dim=1)
        
        # B. 注册 Hook 到倒数第二层 (fc1 或 linear)
        # 自动寻找名字中包含 'fc' 或 'linear' 的层，且不是最后一层
        target_layer = None
        if hasattr(model, 'fc1'): target_layer = model.fc1
        elif hasattr(model, 'layer4'): target_layer = model.layer4 # ResNet
        elif hasattr(model, 'features'): target_layer = model.features # VGG
        
        # 默认回退：如果没有找到 fc1，就找 fc2 (虽然效果差一点但也能用)
        if target_layer is None: target_layer = model.fc2 

        handle = target_layer.register_forward_hook(feature_noise_hook)
        
        try:
            # C. 采样多次 (Hook 会自动在中间层加噪)
            for _ in range(num_samples):
                # 输入层保持干净，噪声在内部产生
                noisy_logits = model(clean_input) 
                noisy_probs = F.softmax(noisy_logits, dim=1)
                
                # 计算 KL: P(clean) || Q(noisy)
                # 恶意模型的特征空间非常"脆"，一加噪预测就会乱跳，KL 激增
                kl = torch.sum(clean_probs * (torch.log(clean_probs + 1e-10) - torch.log(noisy_probs + 1e-10)))
                kl_list.append(max(0, kl.item()))
        finally:
            # 务必移除 Hook，否则影响后续正常评估
            handle.remove()

    # D. 计算分数
    avg_kl = np.mean(kl_list)
    # 指数衰减：KL 越大，分数越低。系数 2.0 可根据实际数值范围微调。
    rs_score = np.exp(-avg_kl * 2.0) 
    
    return float(rs_score)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=43):  # Default GTSRB has 43 classes
        super(SimpleCNN, self).__init__()
        # -----------------------------------------------------------
        # CHANGE 1: in_channels changed from 1 to 3 for RGB images
        # -----------------------------------------------------------
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        # Calculate Flatten Size:
        # Input: 32x32
        # After Conv1 + Pool: 16x16
        # After Conv2 + Pool: 8x8
        # Flatten: 32 channels * 8 * 8 = 2048
        self.fc1 = nn.Linear(32 * 8 * 8, 128)

        # -----------------------------------------------------------
        # CHANGE 2: Ensure output matches GTSRB classes (43)
        # -----------------------------------------------------------
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8) # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 初始化模型
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

current_config = {
    'code': 'neurotoxin',
    'pgd': 0, 'l2': 50,
    'boost_factor': 1
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def run_simulation_CNN(seed,NUM_CLIENTS,defense_nodes, malicious_clients, G, neighbors, 
                       client_datasets,test_loader,
                           atk_type='neurotoxin', mechanism='FedAvg', intensity = 2.0,
                           norm_factor = 0.2,scale_factor=0.5,
                           debug_mode=False,GLOBAL_ROUNDS=15, epochs=5,debug = True):
    set_seed(seed)
    global_model = SimpleCNN().to(DEVICE)
    # 如果您需要一个初始的全局状态来分发给客户端，可以从这里获取
    initial_global_state = global_model.state_dict()
    client_models = [SimpleCNN(num_classes=43).to(DEVICE) for _ in range(NUM_CLIENTS)]
    client_optimizers = [torch.optim.Adam(m.parameters(), lr=0.001) for m in client_models]
    loss_fn = nn.CrossEntropyLoss()
    if mechanism == 'MAB':
        #mab_defense = MABDefense_CNN(NUM_CLIENTS, decay=0.8, exploration_c=0.5)
        mab_defense =  MABDefense(NUM_CLIENTS, model_type='cnn', decay=0.9, exploration_c=0.5, 
                                            audit_prob=0.9, agg_prob=0.8,custom_target_layers=None)
    sens_history = defaultdict(list)
    round_sens = {}

    for round_idx in range(GLOBAL_ROUNDS):
        print(f"\n--- Round {round_idx+1}/{GLOBAL_ROUNDS} ---")

        start_states = [copy.deepcopy(m.state_dict()) for m in client_models]
        post_states = [None] * NUM_CLIENTS # 占位符

        # --- Phase A-1: 先训练所有良性节点 (Benign First) ---
        benign_norms = []

        # 遍历良性节点
        benign_indices = [i for i in range(NUM_CLIENTS) if i not in malicious_clients]
        # --- Phase A-1: 训练良性节点 ---
        for i in benign_indices:
            loader = DataLoader(client_datasets[i], batch_size=32, shuffle=True)
            
            # 关键修改：传入 state_dict 而不是模型引用，或者训练后立即解耦
            train_client_cnn(
                client_models[i], client_optimizers[i], loss_fn, loader, DEVICE,
                initial_global_state=start_states[i],
                is_malicious=False,
                epochs=epochs
            )
            
            # 强制克隆一份 state_dict 到 post_states，确保与内存中的 model 对象断开引用
            post_states[i] = {k: v.clone().cpu() for k, v in client_models[i].state_dict().items()}
     

            # 计算更新范数
            # Norm = || W_new - W_old ||
            total_norm = 0.0
            for k in client_models[i].state_dict():
                if k in start_states[i] and 'weight' in k: # 只计算 weight 的范数
                    w_new = client_models[i].state_dict()[k]
                    w_old = start_states[i][k].to(DEVICE)
                    total_norm += torch.norm(w_new - w_old) ** 2
            total_norm = torch.sqrt(total_norm).item()
            benign_norms.append(total_norm)


        # 计算良性范数的基准值 (Median 比 Mean 更鲁棒)
        avg_benign_norm = np.median(benign_norms) if benign_norms else 1.0
        print(f"  [Attack Info] Estimated Benign Update Norm: {avg_benign_norm:.4f}")
        # 计算良性范数的基准值
        avg_benign_norm = np.median(benign_norms) if benign_norms else 1.0
        
       # ==========================================
        # 🔥 新增：计算全局良性参考向量 (Reference Vector)
        # ==========================================
        benign_updates = []
        # 使用第 0 个客户端的 key 顺序作为标准，确保和模型内部遍历顺序绝对一致
        standard_keys = list(start_states[0].keys()) 

        for i in benign_indices:
            delta_w_list = []
            # 🚨 修复：去掉 sorted，直接用 standard_keys
            for k in standard_keys: 
                if 'num_batches_tracked' in k: continue
                if 'weight' in k or 'bias' in k:
                    w_new = post_states[i][k].float()
                    w_old = start_states[i][k].float().to(w_new.device)
                    delta_w_list.append((w_new - w_old).flatten())
            benign_updates.append(torch.cat(delta_w_list))
            
        # 求平均得到黄金航向
        if benign_updates:
            avg_benign_update = torch.stack(benign_updates).mean(dim=0).to(DEVICE)
        else:
            avg_benign_update = None
        # ==========================================
        # ==========================================
        # 2. 调整策略配置
        strategy_config = {
            'code': 'collusion',
            'scale_length': 3.0,
        
            # 放宽 Neurotoxin，或者干脆在这里如果不写 code='neurotoxin' 就会关闭它
            # 建议先关闭 Neurotoxin 测试纯共谋效果，成功后再开启
            # 'mask_rate': 0.5
        }
        # --- Phase A-2: 再训练恶意节点 (Malicious Follows) ---
        # 此时我们有了 avg_benign_norm
        mal_indices = list(malicious_clients)

        for i in mal_indices:
            loader = DataLoader(client_datasets[i], batch_size=32, shuffle=True)

            # 恶意训练，传入 reference_norm
            train_client_cnn(
                client_models[i], client_optimizers[i], loss_fn, loader, DEVICE,
                initial_global_state=start_states[i],
                is_malicious=True,
                strategy_config=strategy_config,
                 intensity=intensity, epochs= epochs,
                 reference_vector=avg_benign_update,
                reference_norm=norm_factor*avg_benign_norm,current_round=round_idx, total_rounds=GLOBAL_ROUNDS,# <--- 关键：传入基准范数
            )
            post_states[i] = {k: v.cpu() for k, v in client_models[i].state_dict().items()}


        # ==========================================
    # 修正后的 CNN 陷门梯度函数 (确保维度匹配)
    # ==========================================
        # def get_cnn_trap_grad_func_robust(model, device):
        #     """
        #     计算陷门梯度，严格匹配 update_trust 中的目标层 (fc2.weight)。
        #     """
        #     model.eval()
        #     model.zero_grad()
        #     g_cpu = torch.Generator()
        #     g_cpu.manual_seed(1337)

        #     # A. 构造陷门输入
        #     trap_input = torch.randn(1, 3, 32, 32, generator=g_cpu).to(device)

        #     # B. 前向与反向传播
        #     output = model(trap_input)
        #     target = torch.tensor([0], dtype=torch.long).to(device)
        #     loss = nn.CrossEntropyLoss()(output, target)
        #     loss.backward()

        #     # C. 提取梯度 (关键修改：必须与 update_trust 的 target_layers 一致)
        #     # update_trust 中使用的是 ['fc2.weight', 'linear.weight', 'fc.weight']
        #     target_layers = ['fc2.weight', 'linear.weight', 'fc.weight']

        #     grad_vec = []

        #     # 使用 named_parameters 确保我们取到的是 Weight 而不是 Bias
        #     for name, p in model.named_parameters():
        #         # 1. 参数必须有梯度
        #         # 2. 参数名字必须包含目标层名 (例如 'fc2.weight')
        #         if p.grad is not None and any(t in name for t in target_layers):
        #             grad_vec.append(p.grad.view(-1))

        #     # 如果没找到目标层 (防止报错)，则尝试取倒数第二个参数 (通常是最后一层的 weight)
        #     if not grad_vec:
        #         params = list(model.parameters())
        #         if len(params) >= 2 and params[-2].grad is not None:
        #             grad_vec.append(params[-2].grad.view(-1))

        #     return torch.cat(grad_vec) if grad_vec else None
        # =========================================================
        # 🔥 Phase B-3: 激活层聚类分析 (Activation Clustering for Neighbor Models)
        # =========================================================
        if debug == True:
            print("\n🧠 [Debug Analysis] Phase B-3: Activation Clustering (Neighbor Models)")
            
            # 1. 准备一小批固定的探测数据 (Probe Data)
            # 我们从 test_loader 中取出一个 Batch，取前 16 张图片以加快计算速度
            probe_data, _ = next(iter(test_loader))
            probe_data = probe_data[:16].to(DEVICE) 
            
            # 2. 定义 PyTorch Hook 钩子函数，用于提取隐藏层的激活值
            activations_store = {}
            def get_activation_hook(name):
                def hook(model, input, output):
                    # 将 [Batch, Features] 展平为一维向量并移至 CPU
                    activations_store[name] = output.detach().cpu().flatten().numpy()
                return hook

            # 3. 遍历每个观察者节点
            for observer_id in range(NUM_CLIENTS):
                my_neighbors = neighbors[observer_id]
                if not my_neighbors: continue

                # 候选者包括邻居和自己
                candidates = my_neighbors + [observer_id]
                if len(candidates) < 2: continue # 节点太少无需聚类

                act_vectors = []
                
                for nid in candidates:
                    # a) 临时加载该邻居的模型参数
                    temp_model = SimpleCNN(num_classes=43).to(DEVICE)
                    temp_model.load_state_dict({k: v.to(DEVICE) for k, v in post_states[nid].items()})
                    temp_model.eval()
                    
                    # b) 在倒数第二层 (fc1) 注册 Hook 监听器
                    handle = temp_model.fc1.register_forward_hook(get_activation_hook('fc1'))
                    
                    # c) 前向传播，经过数据的刺激，触发 fc1 层产生神经元激活信号
                    with torch.no_grad():
                        _ = temp_model(probe_data)
                        
                    # d) 收集提取到的激活值，清理内存和钩子
                    act_vectors.append(activations_store['fc1'])
                    handle.remove()
                    del temp_model
                
                # 4. 对提取出的激活值向量进行聚类 (K-Means)
                warnings.filterwarnings("ignore", category=FutureWarning) # 忽略 KMeans 警告
                
                X_acts = np.array(act_vectors)
                # 尝试分为 2 簇（良性 / 恶意），如果候选节点总数不到 2 个则取最小值
                n_clusters = min(2, len(candidates)) 
                
                try:
                    clusterer = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
                    labels = clusterer.fit_predict(X_acts)
                except Exception as e:
                    labels = ["ERR"] * len(candidates)

                # 5. 打印该观察者视角的聚类结果
                print(f"  Observer {observer_id} 激活层聚类结果 (K={n_clusters}):")
                print(f"    {'Node':<6} | {'Role':<6} | {'Cluster ID'}")
                print("    " + "-" * 30)
                
                for idx, nid in enumerate(candidates):
                    role = "MAL 😈" if nid in malicious_clients else "BEN 😇"
                    is_self = "(Me)" if nid == observer_id else ""
                    alert = " 🔥 异类!" if (nid in malicious_clients and labels[idx] != labels[-1]) else ""
                    
                    print(f"    {str(nid)+is_self:<6} | {role:<6} |   簇 {labels[idx]} {alert}")
                print("")

        # =========================================================
        # 🔥 Phase C: Aggregation Logic (Corrected)
        # =========================================================
        # (这部分是您原有的聚合逻辑代码...)
     # =========================================================
        # 🔥 Phase B: Debug / Analysis (Strictly Aligned with Phase C)
        # =========================================================
        if debug == True:
            if (round_idx) % 1 == 0:
                print(f"\n🔍 [Debug Analysis] Round {round_idx+1} Full Inspection (Dual Metric Comparison):")
                
                # --- 1. 定义两个不同粒度的统计函数 (为了对比) ---
                
                def calc_row_wise_stats(delta_np):
                    """ 原版逻辑：针对神经元/行 (Row-wise) """
                    if delta_np.ndim < 2: return 0.0 # 1D 向量无法算行级异常
                    
                    # 计算每一行的正能量 (Row Sum)
                    pos_energy = np.sum(np.maximum(delta_np, 0), axis=1)
                    
                    # 对 Row Sum 进行鲁棒 Z-score
                    median_p = np.median(pos_energy)
                    mad_p = np.median(np.abs(pos_energy - median_p)) + 1e-9
                    z_pos = 0.6745 * (pos_energy - median_p) / mad_p
                    
                    return np.max(z_pos) if len(z_pos) > 0 else 0.0

                def calc_element_wise_stats(delta_np):
                    """ 新版逻辑：针对单个参数 (Element-wise) """
                    flat = delta_np.flatten()
                    pos_vals = flat[flat > 0]
                    
                    if len(pos_vals) > 0:
                        median_p = np.median(pos_vals)
                        mad_p = np.median(np.abs(pos_vals - median_p)) + 1e-9
                        z_scores = 0.6745 * (pos_vals - median_p) / mad_p
                        return np.max(z_scores)
                    return 0.0

                # --- 2. 打印表头 (扩宽以容纳两列 Sz) ---
                print(f"{'Obs':<4} | {'Cand.':<8} | {'Type':<5} | {'Sz(Row)':<10} | {'Sz(Elem)':<10} | {'RS Score':<10} | {'SVD':<8} | {'Krum':<10} | {'L2 Dist':<10}")
                print("-" * 145)

                target_layer = 'fc2.weight'
                temp_state = post_states[0]
                # 自动寻找存在的全连接层权重
                if target_layer not in temp_state: target_layer = 'linear.weight'
                if target_layer not in temp_state: target_layer = list(temp_state.keys())[-2]

                for obs_id in range(NUM_CLIENTS):
                    my_neighbors = neighbors[obs_id]
                    if not my_neighbors: continue

                    # 1. 构造候选者列表
                    candidates = my_neighbors + [obs_id]

                    # 2. 准备数据
                    deltas = {}
                    deltas_flat = {}

                    for nid in candidates:
                        w_new = post_states[nid][target_layer].cpu()
                        w_old = start_states[nid][target_layer].cpu()
                        d = (w_new - w_old).float()
                        deltas[nid] = d
                        deltas_flat[nid] = d.flatten()

                    # 3. 计算 Krum Score (用于参考)
                    f_limit = max(1, int(len(candidates) * 0.3))
                    candidate_vecs = [deltas_flat[nid] for nid in candidates]
                    k_scores = calculate_krum_scores(candidate_vecs, f_limit) # 假设此函数已定义
                    sorted_indices = np.argsort(k_scores)
                    ranks = {candidates[idx]: rank + 1 for rank, idx in enumerate(sorted_indices)}

                    # 4. 辅助指标 (均值)
                    all_vecs_np = [v.numpy() for v in deltas_flat.values()]
                    mean_vector_np = np.mean(np.stack(all_vecs_np), axis=0)
                    mean_tensor_flat = torch.from_numpy(mean_vector_np).float()

                    # 5. 打印 (遍历所有候选者)
                    for nid in candidates:
                        d_tensor_raw = deltas[nid] # 原始 Tensor [Out, In]
                        d_np = d_tensor_raw.numpy()
                        d_flat = deltas_flat[nid] 

                        # --- 🔥 关键修改：分别计算两个指标 ---
                        sz_row_val = calc_row_wise_stats(d_np)     # 行级 (针对后门类别)
                        sz_elem_val = calc_element_wise_stats(d_np) # 元素级 (针对参数离群)
# ==========================================
                        # 🔥 新增：计算 RS Score 用于 Debug 观察
                        # ==========================================
                        # 临时加载模型来跑前向传播
                        temp_model_rs = SimpleCNN(num_classes=43).to(DEVICE)
                        # 注意：这里加载的是 post_states (本轮训练后的权重)
                        temp_model_rs.load_state_dict({k: v.to(DEVICE) for k, v in post_states[nid].items()})
                        
                        rs_val = get_cnn_rs_score(temp_model_rs, DEVICE)
                        
                        del temp_model_rs # 用完即弃
                        # 计算 SVD
                        try:
                            mat = d_tensor_raw.view(d_tensor_raw.shape[0], -1) if d_tensor_raw.ndim > 2 else d_tensor_raw
                            svd_max = torch.linalg.svdvals(mat)[0].item()
                        except: svd_max = 0.0

                        # 计算 L2 Dist
                        l2_dist = torch.norm(d_flat.float() - mean_tensor_flat).item()

                        # 获取 Krum 信息
                        idx_in_list = candidates.index(nid)
                        my_krum_score = k_scores[idx_in_list]
                        my_rank = ranks[nid]

                        # 标记
                        tag = "MAL" if nid in malicious_clients else "BEN"
                        is_self = "(Me)" if nid == obs_id else ""
                        marker = "🔥" if (obs_id not in malicious_clients) and (nid in malicious_clients) and (my_rank <= 3) else " "
                        winner_mark = "🏆" if my_rank == 1 else ""

                        # --- 打印行 (显示两个 Sz) ---
                        print(f"{obs_id:<4} | {nid:<4} {is_self:<3} | {tag:<5} | {sz_row_val:<10.4f} | {sz_elem_val:<10.4f} | {rs_val:<10.4f} | {svd_max:<8.4f} | {my_krum_score:<10.4f} | {l2_dist:<10.4f} {marker}")
                    print("-" * 145)
    # =================================================
            # 🔥 Phase B-2: CNN Sensitivity Fingerprinting (Input Gradient Analysis)
            # =================================================
            if round_idx % 1 == 0:

                # 1. Define Sensitivity Test Function
                def get_input_sensitivity_score(model, device):
                    model.eval()
                    dummy_input = torch.zeros(1, 3, 32, 32).to(device)
                    dummy_input.requires_grad = True
                    output = model(dummy_input)
                    target_score = output.max()
                    target_score.backward()
                    return dummy_input.grad.data.abs().max().item()

                # 2. Helper: Flatten ALL layers (for Cosine Sim between models)
                def get_full_flat_update(new_state, old_state, device):
                    diffs = []
                    for k in sorted(new_state.keys()):
                        if 'num_batches_tracked' in k: continue
                        w_new = new_state[k].to(device).float()
                        w_old = old_state[k].to(device).float()
                        diffs.append((w_new - w_old).flatten())
                    return torch.cat(diffs)

                # 3. Helper: Flatten ONLY Target Layers (for Trap Score matching)
                def get_masked_flat_update(new_state, old_state, device, target_layers):
                    diffs = []
                    # Must iterate in same order as get_cnn_trap_grad_func_robust
                    # Usually we iterate parameters, here we iterate keys matching targets
                    
                    # Check for specific layer names used in your model definition
                    # This order MUST match how 'get_cnn_trap_grad_func_robust' concatenates them
                    # Based on your previous code, it looks for 'fc2.weight', 'linear.weight', etc.
                    
                    found_layers = []
                    for k in new_state.keys():
                        if any(t in k for t in target_layers) and 'weight' in k:
                            found_layers.append(k)
                    
                    # Sort to ensure deterministic order (important!)
                    found_layers = sorted(found_layers)

                    for k in found_layers:
                        w_new = new_state[k].to(device).float()
                        w_old = old_state[k].to(device).float()
                        diffs.append((w_new - w_old).flatten())
                    
                    return torch.cat(diffs) if diffs else None

                # 4. Iterate Observers
                for observer_id in range(NUM_CLIENTS):
                    my_neighbors = neighbors[observer_id]
                    if not my_neighbors: continue

                    has_mal_neighbor = any(n in malicious_clients for n in my_neighbors)
                    if not (has_mal_neighbor or observer_id < 2):
                        continue

                    # A. My Full Update (for Standard Cosine Sim)
                    my_update_vec = get_full_flat_update(post_states[observer_id], start_states[observer_id], DEVICE)

                    # --- ⚠️ 删除旧的 trap_ref_vec 预计算，现在不需要了！ ---

                    # --- Print Header ---
                    print(f"  Observer {observer_id} checking Neighbors...")
                    print(f"    {'Neighbor':<10} | {'Type':<6} | {'Max Sens':<12} | {'Cos Sim':<10} | {'Trap(Hessian)':<10}")
                    print("-" * 75)

                    for nid in my_neighbors:
                        is_mal = nid in malicious_clients
                        tag = "MAL" if is_mal else "BEN"

                        # 1. Load Neighbor Model
                        neighbor_model = copy.deepcopy(client_models[0])
                        neighbor_state = {k: v.to(DEVICE) for k, v in post_states[nid].items()}
                        neighbor_model.load_state_dict(neighbor_state)

                        # 2. Calc Max Sensitivity
                        max_s = get_input_sensitivity_score(neighbor_model, DEVICE)

                        # 3. Calc Full Cosine Similarity (Whole Model)
                        nb_update_vec = get_full_flat_update(post_states[nid], start_states[nid], DEVICE)
                        cos_sim = F.cosine_similarity(my_update_vec.unsqueeze(0), nb_update_vec.unsqueeze(0)).item()

                        # 4. 🔥 新版：直接对加载了邻居权重的模型计算二阶迹 (Hessian Trace)
                        trap_score = get_cnn_trap_grad_func(neighbor_model, DEVICE)

                        # 计算完再释放内存
                        del neighbor_model 

                        # 5. Print
                        print(f"    -> N{nid:<7} | {tag:<6} | {max_s:<12.4f} | {cos_sim:<10.4f} | {trap_score:<10.4f}")

                    print("")

        # =========================================================
        # 🔥 Phase C: Aggregation Logic (Corrected)
        # =========================================================
        # 聚合后的新权重容器
        next_round_weights = []
        if mechanism not in ['MAB']:
                   # 如果是基础聚合，不需要特定防御节点逻辑（或者所有良性节点都只是简单聚合）
                   defense_nodes = set()
        # 1. 为 KRUM 和 COS 准备向量映射 (Vector Map)
        # 必须使用本轮训练结果 post_states (CPU dict)
      # 1. 为 KRUM 和 COS 准备向量映射 (改为审计 Updates)
        # 1. 为 KRUM, COS 和 FLAME 准备向量映射 (Vector Map)
        vector_map = {}
        # 👇 修改：加入 'FLAME'
        if mechanism in ['Krum', 'Cos', 'FLAME', 'CosL2']:
            for cid in range(NUM_CLIENTS):
                vec_list = []
                sorted_keys = sorted(post_states[cid].keys())
                for k in sorted_keys:
                    if 'num_batches_tracked' in k: continue 

                    w_new = post_states[cid][k].float()
                    w_old = start_states[cid][k].float().to(w_new.device)
                    delta = w_new - w_old

                    vec_list.append(delta.flatten())

                vector_map[cid] = torch.cat(vec_list)

        next_round_weights = []
        if isinstance(defense_nodes, (torch.Tensor, np.ndarray)):
                defense_nodes = set(defense_nodes.tolist())
        elif not isinstance(defense_nodes, set):
            defense_nodes = set(defense_nodes)

        if isinstance(malicious_clients, (torch.Tensor, np.ndarray)):
            malicious_clients = set(malicious_clients.tolist())
        elif not isinstance(malicious_clients, set):
            malicious_clients = set(malicious_clients)
        for i in range(NUM_CLIENTS):
            my_nbs0 = set(neighbors[i]) | {i}
            my_nbs =         neighbors[i]
            # --- 恶意节点：不聚合，只保留自己 ---
            if i in malicious_clients:
                # 恶意节点不聚合，但也必须克隆，防止下一轮训练直接改掉了上一轮存的模型
                next_round_weights.append({k: v.clone().cpu() for k, v in post_states[i].items()})
                continue
            # [Branch 1] MAB Defense (Updated Logic)
            if mechanism == 'MAB' and i in defense_nodes:
                # A. Audit
                audit_targets = mab_defense.select_for_audit(i, my_nbs)
                
                # 必须在这里初始化为空字典，防止本轮没有目标被审计时下方报错
                audit_logs = {} 

                if audit_targets:
                    audit_targets = [int(t) for t in audit_targets]
                    audit_logs = mab_defense.update_trust(
                        observer_id=i,
                        probe_list=audit_targets,
                        new_weights=post_states,   # 使用 post_states
                        old_weights=start_states,  # 使用 start_states
                        device=DEVICE,
                        model_template=client_models[i],
                        sensitivity_func=get_cnn_sensitivity_score,
                        get_trap_func=get_cnn_trap_grad_func,
                        rs_func=get_cnn_rs_score
                    )

                # B. Select
                selected_neighbors, agg_w = mab_defense.select_for_aggregation(
                    client_id=i,
                    candidates=audit_targets
                )
                
                # ==========================================
                # 🔥 Modified: Detailed Trust & Action Report
                # ==========================================
                # Only print for Observer 0 (or specific nodes) to reduce clutter
#                 if i > 0: 
#                     print(f"\n📊 [Round {round_idx+1}] Observer {i} Trust & Aggregation Report:")
                    
#                     # 1. 扩充表头以包含新指标
#                     header = (f" {'Neighbor':<8} | {'Role':<9} | {'Trust(Q)':<8} | "
#                               f"{'Trap':<8} | {'Sens':<8} | {'RS_Score':<8} | {'Action':<10}") # 增加 RS
#                     print(header)
#                     print("-" * 100)
#                     my_nbs0 = set(neighbors[i]) | {i}
#                     # Iterate through neighbors
#                     for nid in my_nbs0:
#                         # 1. Role (Ground Truth)
#                         role = "MAL" if nid in malicious_clients else "BEN"
#                         role_str = f"{role} 😈" if role == "MAL" else f"{role} 😇"

#                         # 2. Current Trust Score
#                         curr_trust = mab_defense.trust_scores[i].get(nid, 0.5)

#                         # 3. 提取详细指标 (仅对被审计的节点有效)
# # 3. 提取详细指标 (仅对被审计的节点有效)
#                         if  nid in audit_logs:
#                             t_log = audit_logs[nid]
#                             trap_str = f"{t_log.get('trap', 0.0):.4f}"
#                             max_sen_str = f"{t_log.get('raw_sens', 0.0):.4f}"
#                             s_z = f"{t_log.get('elem_z', 0.0):.4f}" 
#                             z_str = f"{t_log.get('z_comb_penalty', 0.0):.4f}" 
#                             rs_str = f"{t_log.get('rs_score', 0.0):.4f}"
#                         else:
#                             # 🔥 修复：在这里添加 rs_str = "N/A"
#                             trap_str = max_sen_str = s_z = z_str = rs_str = "N/A"

#                         # 4. Action Status
#                         if nid in selected_neighbors:
#                             action = "✅ AGG"   # Selected for aggregation
#                         elif nid in audit_targets:
#                             action = "❌ DROP"  # Audited but rejected
#                         else:
#                             action = "⚪ SKIP"  # Not selected for audit

#                         # 5. Print Detailed row
#                         print(f" {nid:<8} | {role_str:<9} | {curr_trust:<8.4f} | "
#                               f"{trap_str:<8} | {max_sen_str:<8} | {rs_str:<8} | {action:<10}")
                    
#                     print("-" * 90)
                # C. Aggregate (Weighted Avg)
                avg_state = {}
                local_alpha = 0.5
                actual_nb_weight = (1 - local_alpha) if selected_neighbors else 0.0

                for k in post_states[i].keys():
                    local_t = post_states[i][k].float()
                    if not selected_neighbors:
                        avg_state[k] = local_t.to(post_states[i][k].dtype)
                        
                        continue

                    final_val = local_t * local_alpha
                    for idx, nid in enumerate(selected_neighbors):
                        w = agg_w[idx] * actual_nb_weight
                        final_val += post_states[nid][k].float() * w

                    avg_state[k] = final_val.clone().detach().cpu().to(post_states[i][k].dtype)

                next_round_weights.append(avg_state)

# [Branch 2] Multi-Krum (Manual Logic)
            elif mechanism == 'Krum':
                candidates = my_nbs + [i]
                MALICIOUS_RATIO = len(malicious_clients)/NUM_CLIENTS
                # 计算参数
                f_limit = int(len(candidates) * MALICIOUS_RATIO)
                m_winners = max(1, len(candidates) - f_limit)

                # 1. 获取所有候选者的向量
                candidate_vecs = [vector_map[nid] for nid in candidates]

                # 2. 计算分数 (利用 calculate_krum_scores 的矩阵加速)
                # 注意：这里直接得到了每个候选者的 Krum Score
                all_k_scores = calculate_krum_scores(candidate_vecs, f_limit)

                # 3. 建立 ID -> 分数 的映射 (用于日志打印和排序)
                node_score_map = {nid: score for nid, score in zip(candidates, all_k_scores)}

                # =========================================================
                # 🔥 [修改处] 手动排序并选择赢家 (替代 local_multikrum_select)
                # =========================================================
                
                # 4. 排序: 按 Krum Score 从小到大 (分数越低越好)
                # Python 的 sorted 是稳定的，如果分数相同，会保持 candidates 列表里的原始顺序
                sorted_candidates = sorted(candidates, key=lambda nid: node_score_map[nid])

                # 5. 切片: 选取前 m_winners 个节点 ID
                winner_ids = sorted_candidates[:m_winners]

                if winner_ids:
                    # =========================================================
                    # 🔥 增强版打印日志：显示每个候选者的得分与排名
                    # =========================================================
                    print(f"\n📢 [Round {round_idx+1}] Obs {i} Multi-Krum Selection Details:")
                    print(f"    {'Status':<8} | {'Node':<5} | {'Role':<5} | {'Krum Score':<12}")
                    print(f"    {'-' * 40}")

                    # 按得分从小到大排序以便观察
                    sorted_cands = sorted(candidates, key=lambda x: node_score_map[x])
                    for nid in sorted_cands:
                        role = "😈" if nid in malicious_clients else "😇"
                        is_winner = "WINNER ✅" if nid in winner_ids else "REJECT ❌"
                        score = node_score_map[nid]

                        # 重点标记被选中的恶意节点，提醒攻击成功
                        alert = " 🔥" if (nid in malicious_clients) and (nid in winner_ids) else ""
                        print(f"    {is_winner:<8} | {nid:<5} | {role:<5} | {score:<12.4f}{alert}")

                    # 3. 执行平均聚合
                    avg_w = {}
                    first_id = winner_ids[0]
                    for k in post_states[first_id].keys():
                        stack_w = torch.stack([post_states[nid][k].float() for nid in winner_ids])
                        avg_w[k] = stack_w.mean(0).to(post_states[i][k].dtype)

                    next_round_weights.append(avg_w)
                else:
                    # 失败兜底
                    next_round_weights.append(copy.deepcopy(post_states[i]))
# =========================================================
            # 🔥 Strategy: FLAME Defense (Clustering + Clipping + Noise)
            # =========================================================
            # =========================================================
            # 🔥 Strategy: FLAME Defense (Clustering + Clipping + Noise)
            # =========================================================
            elif mechanism == 'FLAME':
                candidates = my_nbs + [i]
                n_c = len(candidates)
                
                # ... (Clustering logic remains the same) ...
                
                # If node degree is too small, skip clustering
                if n_c < 3:
                    trusted_ids = candidates
                    cluster_labels = [-1] * n_c
                else:
                    # 1. Extract neighbor update vectors (on CPU)
                    vecs = [vector_map[nid].numpy() for nid in candidates]
                    
                    # 2. Calculate Cosine Distance Matrix
                    from sklearn.metrics.pairwise import cosine_distances
                    dist_matrix = cosine_distances(vecs)
                    
                    # 3. Clustering (HDBSCAN or Agglomerative)
                    try:
                        from sklearn.cluster import HDBSCAN
                        clusterer = HDBSCAN(
                            min_cluster_size=2, 
                            min_samples=1, 
                            cluster_selection_epsilon=0.5, 
                            metric='precomputed'
                        )
                        cluster_labels = clusterer.fit_predict(dist_matrix)
                    except ImportError:
                        from sklearn.cluster import AgglomerativeClustering
                        clusterer = AgglomerativeClustering(
                            n_clusters=None, 
                            distance_threshold=0.5, 
                            metric='precomputed', 
                            linkage='average'
                        )
                        cluster_labels = clusterer.fit_predict(dist_matrix)

                    # 4. Identify Trusted Cluster
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

                # ... (Printing logic remains the same) ...

                # 5. Calculate Dynamic Clipping Bound (Gamma)
                # vector_map is on CPU, so norms calculation happens on CPU
                trusted_vecs = [vector_map[nid] for nid in trusted_ids]
                norms = [torch.norm(v).item() for v in trusted_vecs]
                gamma = np.median(norms) if norms else 1.0  
                
                # 6. Aggregation, Clipping, and Noising
                avg_w = {}
                noise_std = 0.001 * gamma 
                
                for k in start_states[0].keys():
                    if 'num_batches_tracked' in k:
                        avg_w[k] = post_states[i][k]
                        continue
                    
                    # 🔥 FIX: Initialize on DEVICE (GPU)
                    sum_update = torch.zeros_like(start_states[i][k], device=DEVICE).float()
                    
                    for nid in trusted_ids:
                        # 🔥 FIX: Move post_states (CPU) to DEVICE before subtraction
                        w_new = post_states[nid][k].float().to(DEVICE)
                        w_old = start_states[nid][k].float().to(DEVICE)
                        delta = w_new - w_old
                        
                        # 🔥 FIX: Ensure norm is a python float or scalar on DEVICE
                        # vector_map resides on CPU, so .item() extracts the float value
                        nid_norm = torch.norm(vector_map[nid]).item() 
                        
                        cf = min(1.0, gamma / (nid_norm + 1e-9))
                        sum_update += delta * cf
                        
                    avg_update = sum_update / len(trusted_ids)

                    # Add Noise
                    noise = torch.randn_like(avg_update) * noise_std
                    final_w = start_states[i][k].float().to(DEVICE) + avg_update + noise

                    # Move result back to CPU/Target dtype for storage
                    avg_w[k] = final_w.to(post_states[i][k].dtype)
                    
                next_round_weights.append(avg_w)
            # =========================================================
            # 🔥 [新增] Branch 3.5: CosL2 (Cosine Filtering + L2 Clipping)
            # =========================================================
            elif mechanism == 'CosL2':
                my_v = vector_map[i]
                
                # --- Step 1: 方向过滤 (CosSim) ---
                MALICIOUS_RATIO = len(malicious_clients) / NUM_CLIENTS
                m_winners = max(1, int(len(my_nbs) * (1 - MALICIOUS_RATIO)))
                
                sim_scores = []
                for nb in my_nbs:
                    sim = torch.nn.functional.cosine_similarity(my_v, vector_map[nb], dim=0).item()
                    sim_scores.append((sim, nb))
                
                # 按相似度降序排列
                sim_scores.sort(key=lambda x: x[0], reverse=True)
                selected_nids = [nb for sim, nb in sim_scores[:m_winners]]
                final_group = [i] + selected_nids # 永远信任自己

                # 打印日志 (防刷屏)
                print(f"\n📊 [Round {round_idx+1}] Obs {i} CosL2 Decision (n={len(my_nbs)}, keep_top={m_winners})")
                print(f"    {'Neighbor':<8} | {'Role':<6} | {'Cos Sim':<12} | {'Decision'}")
                print("-" * 55)
                for sim, nb in sim_scores:
                    role = "MAL" if nb in malicious_clients else "BEN"
                    is_selected = nb in selected_nids
                    status = "✅ AGG" if is_selected else "❌ DROP"
                    if role == "MAL" and is_selected: status += " ⚠️ ALERT"
                    print(f"    N{nb:<7} | {role:<6} | {sim:<12.4f} | {status}")
                print("-" * 55)

                # --- Step 2: 尺度裁剪 (L2 Clipping) ---
                # 计算所有入选邻居的 L2 范数
                norms = [torch.norm(vector_map[nid]).item() for nid in final_group]
                gamma = np.median(norms) if norms else 1.0

                # --- Step 3: 聚合与物理写回 ---
                avg_w = {}
                # 预先获取 Delta 字典加速计算
                updates = {}
                for nid in final_group:
                    updates[nid] = {}
                    for k in start_states[0].keys():
                        if 'num_batches_tracked' in k: continue
                        w_new = post_states[nid][k].float().to(DEVICE)
                        w_old = start_states[nid][k].float().to(DEVICE)
                        updates[nid][k] = w_new - w_old

                for k in start_states[0].keys():
                    if 'num_batches_tracked' in k:
                        avg_w[k] = post_states[i][k]
                        continue
                    
                    tmp_sum = torch.zeros_like(post_states[i][k].float()).to(DEVICE)
                    for nid in final_group:
                        update_v = updates[nid][k]
                        current_node_norm = torch.norm(vector_map[nid]).item()
                        
                        # 核心裁剪逻辑：超速则按比例缩小
                        clip_factor = min(1.0, gamma / (current_node_norm + 1e-9))
                        tmp_sum += update_v * clip_factor
                    
                    avg_update = tmp_sum / len(final_group)
                    
                    # W_new = W_old + Avg_Update
                    w_base = start_states[i][k].float().to(DEVICE)
                    avg_w[k] = (w_base + avg_update).to(post_states[i][k].dtype)
                
                next_round_weights.append(avg_w)
            # [Branch 3] Cosine Similarity (Corrected Logic)
            elif mechanism == 'Cos' :
                my_v = vector_map[i]
                keep_ids = [i]
                for nb in my_nbs:
                    # 计算 Cosine Similarity
                    sim = torch.nn.functional.cosine_similarity(
                        my_v.unsqueeze(0), vector_map[nb].unsqueeze(0)
                    ).item()

                    # 只有相似度大于阈值的才聚合 (例如 > 0.5 或者 > 0.1)
                    # 恶意更新通常方向差异很大
                    if sim > 0.05:
                        keep_ids.append(nb)

                # 简单的平均聚合
                avg_w = {}
                for k in post_states[i].keys():
                    # 堆叠所有通过筛选的权重
                    stack_w = torch.stack([post_states[idx][k].float() for idx in keep_ids])
                    avg_w[k] = stack_w.mean(0).to(post_states[i][k].dtype)
                next_round_weights.append(avg_w)
# [Branch 4] Trimmed Mean (新增防御)
            elif mechanism == 'TrimmedMean':
                candidates = my_nbs + [i]
                n = len(candidates)

                # 1. 确定修剪数量 k (Beta)
                # 通常设为预期恶意节点的比例。例如若恶意比例 35%，则两端各去掉 35%
                # 必须保证 n - 2k > 0，否则没东西剩下了
                MALICIOUS_RATIO = len(malicious_clients)/NUM_CLIENTS
                k = int(n * MALICIOUS_RATIO)

                # 边界保护：如果修剪过猛导致没人了，就强制减少 k
                # 至少保留 1 个或是中间的 1/3
                if n - 2 * k < 1:
                    k = max(0, (n - 1) // 2)

                avg_w = {}
                ref_keys = post_states[i].keys() # 获取键列表

                for key in ref_keys:
                    # 跳过非训练统计量 (如 BN 层的 num_batches_tracked)
                    # 或者对它们做简单平均，这里选择简单处理
                    if 'num_batches_tracked' in key:
                        stack_w = torch.stack([post_states[nid][key] for nid in candidates])
                        # 对于整数统计量，取中位数或众数可能更好，这里取平均并转回 Long
                        avg_w[key] = stack_w.float().mean(0).to(post_states[i][key].dtype)
                        continue

                    # 2. 堆叠所有邻居的该层参数
                    # Shape: [Neighbors, Param_Dim1, Param_Dim2...]
                    stack_w = torch.stack([post_states[nid][key].float() for nid in candidates], dim=0)

                    # 3. 核心：在第 0 维（邻居维度）进行排序
                    # 这实现了 "Coordinate-wise" (逐坐标) 的筛选
                    # 也就是对于权重矩阵的每一个格子，单独看谁大谁小
                    sorted_w, _ = torch.sort(stack_w, dim=0)

                    # 4. 修剪 (Trim)
                    # 去掉最小的 k 个，和最大的 k 个
                    # 切片范围: [k : n-k]
                    trimmed_w = sorted_w[k : n - k]

                    # 5. 对剩余部分求平均
                    avg_w[key] = trimmed_w.mean(dim=0).to(post_states[i][key].dtype)

                next_round_weights.append(avg_w)
            # [Branch 4] FedAvg (Default / Non-Defense Node)
           # ... [Previous code for TrimmedMean] ...

           # =========================================================
            # 🔥 [Branch 1.5] MAB: Ordinary Nodes (Updated Weight Logic)
            # =========================================================
            elif mechanism == 'MAB' and i not in defense_nodes:
                my_nbs = neighbors[i]
                
                # 🚨 FIX: Note that def_blacklists needs to be populated in Branch 1 earlier in the round!
                # Assuming def_blacklists is populated globally or earlier in the round logic.
                # If not, this is safe to run but local_banned_set will remain empty.
                def_blacklists = def_blacklists if 'def_blacklists' in locals() else {}
                
                # 建立安全名单 (根据邻近 DEF 节点的局部黑名单排雷)
                local_banned_set = set()
                for nb in my_nbs:
                    if nb in def_blacklists:
                        local_banned_set.update(def_blacklists[nb])
                
                safe_nbs = [n for n in my_nbs if n not in local_banned_set]
                
                # 区分邻居类型 (只从安全名单中筛选)
                defense_neighbors = [n for n in safe_nbs if n in defense_nodes]
                normal_neighbors = [n for n in safe_nbs if n not in defense_nodes]

                avg_state = {}
                
                # 惯性系数
                beta_inertia = 0
                if safe_nbs:
                    # 🌟 预先计算分配好线性聚合的权重
                    w_self = 0.50
                    
                    if defense_neighbors and normal_neighbors:
                        w_def_total = 0.45
                        w_norm_total = 0.05
                    elif defense_neighbors:
                        w_def_total = 0.50
                        w_norm_total = 0.0
                    elif normal_neighbors:
                        w_def_total = 0.0
                        w_norm_total = 0.50
                    else:
                        w_def_total = 0.0
                        w_norm_total = 0.0
                        w_self = 1.0
                    
                    # 平分到每个具体的节点上
                    w_def_per_node = w_def_total / len(defense_neighbors) if defense_neighbors else 0.0
                    w_norm_per_node = w_norm_total / len(normal_neighbors) if normal_neighbors else 0.0

                    # 🚨 FIX: Changed new_client_weights to post_states
                    for k in post_states[i].keys():
                        if 'num_batches_tracked' in k:
                            avg_state[k] = post_states[i][k]
                            continue
                        
                        if 'weight' in k or 'bias' in k:
                            # 1. 提取老权重用于计算 EMA
                            local_w_old = start_states[i][k].float().to(DEVICE)
                            
                            # 2. 累加自己的权重 (Self)
                            my_trained_w = post_states[i][k].float().to(DEVICE)
                            tmp_sum = my_trained_w * w_self
                            
                            # 3. 线性累加 DEF 邻居的权重
                            for nid in defense_neighbors:
                                nb_w = post_states[nid][k].float().to(DEVICE)
                                tmp_sum += nb_w * w_def_per_node
                                
                            # 4. 线性累加普通邻居的权重
                            for nid in normal_neighbors:
                                nb_w = post_states[nid][k].float().to(DEVICE)
                                tmp_sum += nb_w * w_norm_per_node

                            # 5. 聚合结果作为目标权重
                            target_w = tmp_sum
                            
                            # 6. 引入 EMA 惯性进行平滑
                            final_w = (beta_inertia * local_w_old) + ((1.0 - beta_inertia) * target_w)
                            avg_state[k] = final_w.clone().detach().cpu().to(post_states[i][k].dtype)
                        else:
                            avg_state[k] = post_states[i][k]
                
                else:
                    # --- Case B: 孤岛生存模式 (Fallback) ---
                    for k in post_states[i].keys():
                        if 'weight' in k or 'bias' in k:
                            local_w_old = start_states[i][k].float().to(DEVICE)
                            my_trained_w = post_states[i][k].float().to(DEVICE)
                            
                            final_w = (beta_inertia * local_w_old) + ((1.0 - beta_inertia) * my_trained_w)
                            avg_state[k] = final_w.clone().detach().cpu().to(post_states[i][k].dtype)
                        else:
                            avg_state[k] = post_states[i][k]

                next_round_weights.append(avg_state)

            # =========================================================
            # [Branch 5] FedAvg (Default / Fallback)
            # =========================================================
            else:
                candidates = my_nbs + [i]
                avg_state = {}
                for k in post_states[i].keys():
                    stack_w = torch.stack([post_states[nid][k].float() for nid in candidates])
                    avg_state[k] = stack_w.mean(0).to(post_states[i][k].dtype)
                next_round_weights.append(avg_state)
        for i in range(NUM_CLIENTS):
            client_models[i].load_state_dict(next_round_weights[i])
            # ✅ 必须在这里更新对应的优化器，绑定新的参数对象
            client_optimizers[i] = torch.optim.Adam(client_models[i].parameters(), lr=0.001)
            
        # 清理内存
        del post_states, next_round_weights
        if 'vector_map' in locals(): del vector_map
        torch.cuda.empty_cache()


    # 3. 评估阶段

# ==========================================
# 3. 评估与打印
# ==========================================
# ==========================================
    # 3. Final Evaluation & Summary
    # ==========================================
    print("\n" + "="*50)
    print(f"📊 FINAL SIMULATION REPORT (Seed: {seed})")
    print("="*50)

    results_table = []
    # 用于计算统计数据
    benign_acc_list = []
    benign_asr_list = []
    acc_list = []
    asr_list = []

    for cid in range(NUM_CLIENTS):
        # 1. 确定节点身份标签
        is_mal = (cid in malicious_clients)
        role_tag = "MAL 😈" if is_mal else "BEN 😇"
        
        # 2. 执行评估 (针对每一个 Client 的本地全局模型)
        # 注意：intensity2 传给 evaluate_global_cnn 用于构造特定的后门触发器
        acc, asr = evaluate_global_cnn(
            client_models[cid], 
            test_loader, 
            DEVICE,
            trigger_type='patch', 
            intensity=intensity
        )
        acc_list.append(acc)
        asr_list.append(asr)
        # 3. 分类归档数据
        if not is_mal:
            benign_acc_list.append(acc)
            benign_asr_list.append(asr)

        # 4. 打印单条结果
        print(f"Client {cid:2d} | Role: {role_tag:<6} | ACC: {acc:.4f} | ASR: {asr:.4f}")

    # ==========================================
    # 4. Aggregate Summary Statistics
    # ==========================================
    print("-" * 50)
    
    avg_benign_acc = np.mean(benign_acc_list) if benign_acc_list else 0
    avg_benign_asr = np.mean(benign_asr_list) if benign_asr_list else 0


    # 返回数据：良性指标在前，供外层绘图或分析使用
    return avg_benign_acc, avg_benign_asr , acc_list, asr_list