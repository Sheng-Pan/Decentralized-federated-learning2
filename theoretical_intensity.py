
import numpy as np
def calculate_theoretical_intensity(neighbors, malicious_clients, num_clients, intensity, lambda_benign=0.3):
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
        u[m] =  intensity  # 持续注入

    # D. 求解稳态 delta* = (I - A)^(-1) * u
    try:
        # 使用 pinv 防止矩阵奇异 (虽然通常不会)
        H = np.linalg.pinv(I - A) 
        steady_state = H @ u
    except Exception as e:
        print(f"Matrix inversion failed: {e}")
        steady_state = np.zeros(num_clients)
        
    return steady_state
import numpy as np

def calculate_theoretical_intensity(neighbors, malicious_clients, num_clients, intensity, lambda_benign=0.3):
    """
    根据 A = (I - D)W 公式计算理论稳态污染度，并进行跨图归一化
    """
    # ... (A, B, C, D 步骤保持原样) ...
    W = np.zeros((num_clients, num_clients))
    for i in range(num_clients):
        degree = len(neighbors[i]) + 1
        weight = 1.0 / degree
        W[i, i] = weight
        for n_id in neighbors[i]:
            W[i, n_id] = weight

    d_diag = [0.0 if i in malicious_clients else lambda_benign for i in range(num_clients)]
    D = np.diag(d_diag)
    I = np.eye(num_clients)
    A = (I - D) @ W
    
    u = np.zeros(num_clients)
    for m in malicious_clients:
        u[m] = intensity 

    try:
        H = np.linalg.pinv(I - A) 
        steady_state = H @ u
    except Exception as e:
        print(f"Matrix inversion failed: {e}")
        steady_state = np.zeros(num_clients)
        
    # ==========================================
    # 🌟 核心修复：跨实验对齐 (Min-Max Normalization)
    # ==========================================
    # 1. 提取良性节点的理论强度
    benign_indices = [i for i in range(num_clients) if i not in malicious_clients]
    if len(benign_indices) > 0:
        benign_intensities = steady_state[benign_indices]
        
        # 2. 以良性节点中的最大值和最小值为基准进行缩放
        # 这样，1.0 永远代表“该图中受害最深的良性节点”，0.0 代表“最安全的良性节点”
        min_val = np.min(benign_intensities)
        max_val = np.max(benign_intensities)
        
        # 防止除零错误 (比如所有良性节点强度一样)
        if max_val - min_val > 1e-9:
            steady_state = (steady_state - min_val) / (max_val - min_val)
        else:
            steady_state = steady_state - min_val # 退化为只做平移
            
        # 注意：恶意节点的归一化值可能会大于 1，这是正常的，我们在画图时会过滤掉恶意节点
        
    return steady_state
import numpy as np

def calculate_enhanced_theoretical_intensity(neighbors, malicious_clients, num_clients, intensity, lambda_benign=0.3):
    """
    改进版动力学模型：引入枢纽洗涤效应和非线性过滤
    """
    # 1. 计算每个节点的度 (Degree)
    degrees = np.array([len(neighbors[i]) for i in range(num_clients)])
    
    # 2. 构建混合矩阵 W (FedAvg 逻辑)
    W = np.zeros((num_clients, num_clients))
    for i in range(num_clients):
        weight = 1.0 / (degrees[i] + 1)
        W[i, i] = weight
        for n_id in neighbors[i]:
            W[i, n_id] = weight

    # 3. 动态调整注入向量 u (Attacker Washing Effect)
    # 如果恶意节点连接了超级 Hub，其有效投毒强度会大幅下降
    u = np.zeros(num_clients)
    for m in malicious_clients:
        # 计算恶意节点邻居的“净化能力”：邻居的平均度数越高，净化越强
        neighbor_degrees = [degrees[n] for n in neighbors[m] if n not in malicious_clients]
        if neighbor_degrees:
            # 净化因子：1 / (1 + 邻居平均度数)
            washing_factor = 1.0 / (1.0 + np.mean(neighbor_degrees) / 2.0) 
        else:
            washing_factor = 1.0
            
        u[m] = intensity * washing_factor

    # 4. 动态清洗率 D (Hub-Shield Effect)
    # 连接到高连接度节点的节点，其本地清洗效果 λ 会增强
    d_diag = []
    for i in range(num_clients):
        if i in malicious_clients:
            d_diag.append(0.0) # 恶意节点本身不清洗
        else:
            # 如果邻居里有大 Hub，λ 增加
            neighbor_max_degree = max([degrees[n] for n in neighbors[i]]) if neighbors[i] else 0
            # 增强型 λ = 基础 λ + 结构性清洗补偿
            effective_lambda = lambda_benign + 0.2 * (neighbor_max_degree / np.max(degrees))
            d_diag.append(min(0.9, effective_lambda))
            
    D = np.diag(d_diag)
    I = np.eye(num_clients)
    A = (I - D) @ W

    # 5. 求解线性稳态
    try:
        H = np.linalg.pinv(I - A)
        raw_steady_state = H @ u
    except:
        raw_steady_state = np.zeros(num_clients)

    # 6. 非线性激活映射 (Sigmoid 门槛)
    # 模拟神经网络的分类翻转阈值
    # 假设毒素强度在某个阈值 tau 以下时，ASR 几乎为 0
    tau = intensity * 0.4  # 设置触发门槛
    k = 5.0 / intensity    # 控制曲线陡峭度
    
    # 使用 Logistic 函数：1 / (1 + exp(-k * (x - tau)))
    predicted_asr = 1.0 / (1.0 + np.exp(-k * (raw_steady_state - tau)))
    
    # 针对 ASR 极低的情况做截断处理 (模拟死区)
    predicted_asr[predicted_asr < 0.05] = 0.0
    
    return predicted_asr, raw_steady_state