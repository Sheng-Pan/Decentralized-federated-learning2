import torch
from datasets import load_dataset
import torchvision
import torchvision.transforms as transforms
import random
from torch.utils.data import Subset, DataLoader
from sklearn.datasets import load_breast_cancer
import networkx as nx
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, random_split
import os
from torchvision import datasets, transforms
import zipfile
import pandas as pd
from defense import get_high_value_defense_nodes
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
from datasets import Dataset as HFDataset # 避免与 torch.utils.data.Dataset 混淆
import os
import zipfile
import pandas as pd
from datasets import Dataset as HFDataset
from torchvision import datasets, transforms
from huggingface_hub import hf_hub_download, snapshot_download
from google.colab import userdata
import multiprocessing
def get_data(dataset_name='gtsrb', tokenizer=None, max_len=128, repo_id="JONESMITH007/DFL",n_train=1000,n_test=1000):
    """
    加载 GTSRB 或 PubMed 数据集 (从 Hugging Face 获取备份)
    """
    train_ds = None
    test_ds = None
    data_root = './data'
    
    # 从 Colab Secrets 安全获取 Token
    try:
        hf_token = userdata.get('HF_TOKEN')
    except:
        hf_token = None # 如果是公共仓库则不需要

    # --- GTSRB 分支 ---
    if dataset_name.lower() == 'gtsrb':
        transform = transforms.Compose([
            transforms.Resize((32, 32)), 
            transforms.ToTensor(),
            transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
        ])

        try:
            print("正在尝试通过 torchvision 正常加载 GTSRB...")
            train_ds = datasets.GTSRB(data_root, split='train', download=True, transform=transform)
            test_ds = datasets.GTSRB(data_root, split='test', download=True, transform=transform)
        except Exception as e:
            print(f"标准下载失败: {e}。正在从 Hugging Face 仓库加载备份...")
            
            target_dir = os.path.join(data_root, 'gtsrb')
            if not os.path.exists(target_dir):
                os.makedirs(target_dir, exist_ok=True)
                
                # 定义 HF 仓库中的 zip 文件名
                zips = ['GTSRB-Training_fixed.zip', 'GTSRB_Final_Test_Images.zip', 'GTSRB_Final_Test_GT.zip']
                
                for zip_name in zips:
                    try:
                        print(f"正在从 HF 下载 {zip_name}...")
                        # 从 HF 下载到本地缓存
                        downloaded_path = hf_hub_download(
                            repo_id=repo_id, 
                            filename=zip_name, 
                            repo_type="dataset", 
                            token=hf_token
                        )
                        with zipfile.ZipFile(downloaded_path, 'r') as zip_ref:
                            zip_ref.extractall(target_dir)
                    except Exception as download_err:
                        print(f"警告: 无法从 HF 获取 {zip_name}: {download_err}")

            # 再次尝试加载
            train_ds = datasets.GTSRB(data_root, split='train', download=False, transform=transform)
            test_ds = datasets.GTSRB(data_root, split='test', download=False, transform=transform)

    # --- PubMed 20k RCT 分支 ---
    elif dataset_name.lower() in ['pubmed', 'pubmed_20k']:
        if tokenizer is None:
            raise ValueError("加载 PubMed 20k 需要提供 'tokenizer' 参数")

        print("正在从 Hugging Face 同步 PubMed 文本文件...")
        pubmed_local_path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns="*.txt", 
            token=hf_token
        )
        
        label_map = {'BACKGROUND': 0, 'OBJECTIVE': 1, 'METHODS': 2, 'RESULTS': 3, 'CONCLUSIONS': 4}

        def load_from_txt(filename):
            file_full_path = None
            for root, dirs, files in os.walk(pubmed_local_path):
                if filename in files:
                    file_full_path = os.path.join(root, filename)
                    break
            
            if not file_full_path:
                raise FileNotFoundError(f"在 HF 仓库中找不到文件: {filename}")
            
            data = []
            with open(file_full_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('###') or not line: continue
                    parts = line.split('\t', 1)
                    if len(parts) == 2:
                        label_str, text_content = parts[0], parts[1]
                        if label_str in label_map:
                            data.append({"text": text_content.lower(), "labels": label_map[label_str]})
            return pd.DataFrame(data)

        # 1. 加载为 Pandas
        train_df = load_from_txt('train.txt')
        test_df = load_from_txt('test.txt')
        train_raw = HFDataset.from_pandas(train_df)
        test_raw = HFDataset.from_pandas(test_df)

        # 🔥 核心加速 1：先截断，再分词！
        # 不要分词几万条数据后再丢弃，直接从原始文本中抽出我们需要的条数。
        train_raw = train_raw.shuffle(seed=42).select(range(min(n_train, len(train_raw))))
        test_raw = test_raw.shuffle(seed=42).select(range(min(n_test, len(test_raw))))

        def tokenize_fn(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_len)

        # 🔥 核心加速 2：开启多进程 (num_proc) 处理分词
        # 根据你的 CPU 核心数自动并发处理，极大缩短 Tokenize 耗时
        num_cores = max(1, multiprocessing.cpu_count() - 1)
        print(f"⚡ 正在使用 {num_cores} 个核心进行高速分词...")
        
        train_ds = train_raw.map(tokenize_fn, batched=True, num_proc=num_cores)
        test_ds = test_raw.map(tokenize_fn, batched=True, num_proc=num_cores)

        # 转换为 PyTorch 格式
        train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        
        print(f"✅ PubMed 准备就绪 (数据源: Hugging Face)")

    return train_ds, test_ds


def distribute_data(dataset, num_clients,dis = None,alpha=1):
    if dis == 'Dirichlet':
        if hasattr(dataset, 'targets'):
            labels = np.array(dataset.targets)
        elif hasattr(dataset, 'labels'):
            labels = np.array(dataset.labels)
        else:
            # 兼容 HuggingFace dataset 或包含字典的列表
            try:
                labels = np.array([item['labels'] for item in dataset])
            except KeyError:
                labels = np.array([item['label'] for item in dataset])
                
        # 如果标签是 tensor，转为标量
        if hasattr(labels[0], 'item'):
            labels = np.array([lbl.item() for lbl in labels])

        num_classes = len(np.unique(labels))
        client_indices = [[] for _ in range(num_clients)]

        # 2. 针对每一个类别，使用 Dirichlet 分布划分给各个客户端
        for c in range(num_classes):
            # 找到属于当前类别的所有样本的索引
            idx_c = np.where(labels == c)[0]
            np.random.shuffle(idx_c)
            
            # 核心：生成狄利克雷分布比例
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            
            # 将比例映射为具体的样本数量切分点
            splits = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
            
            # 将当前类别的索引数组切分成 num_clients 份
            idx_split = np.split(idx_c, splits)
            
            # 将分到的索引追加给对应的客户端
            for i in range(num_clients):
                client_indices[i].extend(idx_split[i].tolist())

        # 3. 打乱每个客户端内部的数据顺序并封装为 Subset
        client_datasets = []
        for i in range(num_clients):
            random.shuffle(client_indices[i])
            # 防止极端情况下某个客户端分不到数据
            if len(client_indices[i]) == 0:
                print(f"⚠️ 警告: Client {i} 没有分到任何数据! 建议增大 alpha 调和极端分布。")
                
            client_datasets.append(Subset(dataset, client_indices[i]))
    else:
        samples_per_client = len(dataset) // num_clients
        all_indices = list(range(len(dataset)))
        random.shuffle(all_indices) 
        client_datasets = []
        for i in range(num_clients):
            subset_indices = all_indices[i*samples_per_client : (i+1)*samples_per_client]
            client_datasets.append(Subset(dataset, subset_indices))
    return client_datasets
import numpy as np
import random
from torch.utils.data import Subset

import numpy as np
import random
from torch.utils.data import Subset
import numpy as np
import random
from torch.utils.data import Subset
import numpy as np
import random
from torch.utils.data import Subset

def distribute_data_one_class(dataset, num_clients):
    """
    极端 Non-IID 分配：确保每个 Client 的本地数据中只包含一种类别的标签。
    支持 Tuple 格式 (如 torchvision) 和 Dict 格式 (如 HuggingFace) 的数据集。
    """
    # 1. 安全地提取所有样本的标签
    if hasattr(dataset, 'targets'):
        targets = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        targets = np.array(dataset.labels)
    else:
        # 🛡️ 增强版 Fallback: 兼容字典和元组
        targets = []
        for i in range(len(dataset)):
            sample = dataset[i]
            if isinstance(sample, dict):
                # 如果是字典，尝试常见的标签键名
                if 'labels' in sample:
                    # 如果是 tensor，需要取 .item() 或直接转 numpy
                    val = sample['labels']
                    targets.append(val.item() if hasattr(val, 'item') else val)
                elif 'label' in sample:
                    val = sample['label']
                    targets.append(val.item() if hasattr(val, 'item') else val)
                elif 'target' in sample:
                    val = sample['target']
                    targets.append(val.item() if hasattr(val, 'item') else val)
                else:
                    raise KeyError(f"无法在字典中找到标签键。当前可用的键有: {sample.keys()}")
            elif isinstance(sample, (tuple, list)):
                # 如果是传统的 (data, label) 元组
                val = sample[1]
                targets.append(val.item() if hasattr(val, 'item') else val)
            else:
                raise TypeError(f"未知的样本数据类型: {type(sample)}")
        
        targets = np.array(targets)
        
    unique_classes = np.unique(targets)
    num_classes = len(unique_classes)
    
    # 2. 将索引按类别分组并打乱，确保随机性
    class_indices = {c: np.where(targets == c)[0].tolist() for c in unique_classes}
    for c in unique_classes:
        random.shuffle(class_indices[c])
        
    # 3. 确定每个客户端被分配到的类别 (轮询分配)
    client_labels = [unique_classes[i % num_classes] for i in range(num_clients)]
    
    # 统计每个类别被分配给了几个客户端，以便均分类别内的数据
    class_client_counts = {c: client_labels.count(c) for c in unique_classes}
    
    # 游标字典，记录每个类别当前已经分配到了哪个索引
    class_current_idx = {c: 0 for c in unique_classes}
    
    client_datasets = []
    
    # 4. 执行分配
    for i in range(num_clients):
        assigned_class = client_labels[i]
        
        # 计算该客户端可以分到多少个样本
        total_class_samples = len(class_indices[assigned_class])
        samples_for_this_client = total_class_samples // class_client_counts[assigned_class]
        
        start_idx = class_current_idx[assigned_class]
        end_idx = start_idx + samples_for_this_client
        
        # 截取该类别的这一段数据给当前客户端
        subset_idx = class_indices[assigned_class][start_idx:end_idx]
        client_datasets.append(Subset(dataset, subset_idx))
        
        # 更新游标
        class_current_idx[assigned_class] = end_idx
        
    return client_datasets
# ==========================================
#   MLP data
# ==========================================
from imblearn.over_sampling import SMOTE
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import torch
from torch.utils.data import TensorDataset

def get_dermatology_data_augmented(num_clients=20, min_samples=100):
    print(">>> Loading Dermatology Dataset (6 Classes) with SMOTE...")
    data = fetch_openml(data_id=35, as_frame=False, parser='auto')
    X_raw, y_raw = data.data, data.target
    
    # 1. 缺失值填充：皮肤病特征多为离散值，使用众数(most_frequent)比均值更好
    imputer = SimpleImputer(strategy='most_frequent')
    X_raw = imputer.fit_transform(X_raw)
    
    le = LabelEncoder()
    y_raw = le.fit_transform(y_raw)
    
    # 2. 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    # 3. 使用 SMOTE 进行高质量扩充，同时解决类别不平衡
    target_size = num_clients * min_samples
    # SMOTE 需要一个目标字典，我们让每个类别的样本数都达到 target_size // 6
    samples_per_class = max(len(X_scaled), target_size // 6 + 1)
    smote_strategy = {i: samples_per_class for i in range(6)}
    
    smote = SMOTE(sampling_strategy=smote_strategy, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y_raw)
    
    # 截取到你需要的总数量
    X_final = X_resampled[:target_size]
    y_final = y_resampled[:target_size]

    X_tensor = torch.tensor(X_final, dtype=torch.float32)
    y_tensor = torch.tensor(y_final, dtype=torch.long)
    
    print(f"    SMOTE Augmentation Complete: {len(y_final)} high-quality samples created.")
    return TensorDataset(X_tensor, y_tensor)
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset
import numpy as np


import numpy as np
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset

def get_medical_data_augmented(keep_ratio=0.3):
    """
    加载乳腺癌数据集，并按比例丢弃数据以压缩样本量。
    参数:
        keep_ratio: 保留的数据比例 (例如 0.3 表示只保留 30% 的数据，约 170 条)
    """
    print(f">>> Loading Compressed Breast Cancer Dataset (Keeping {keep_ratio*100}%)...")
    
    # 1. 加载真实数据
    data = load_breast_cancer()
    X_raw, y_raw = data.data, data.target
    
    # --- 新增：随机采样保留部分数据 ---
    total_samples = len(y_raw)
    keep_count = int(total_samples * keep_ratio)
    
    # 随机打乱索引并截取
    indices = np.random.choice(total_samples, keep_count, replace=False)
    X_raw = X_raw[indices]
    y_raw = y_raw[indices]
    # ---------------------------------
    
    # 2. 标准化 
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    # 3. 转为 Tensor
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_raw, dtype=torch.long)
    
    print(f"    Success: Compressed down to {len(y_tensor)} samples.")
    return TensorDataset(X_tensor, y_tensor)
def generate_topology(num_clients, topology_type='scale_free', d=4):
    """
    生成 DFL 网络拓扑图
    :param num_clients: 节点总数
    :param topology_type: 拓扑类型 ('ring', 'scale_free', 'random_regular', 'erdos_renyi')
    :param d: 正则图的度数 (仅在 topology_type='random_regular' 时生效)
    """
    if topology_type == 'ring': 
        G = nx.cycle_graph(num_clients)
    elif topology_type == 'scale_free': 
        G = nx.barabasi_albert_graph(num_clients, m=2) 
    elif topology_type == 'random_regular':
        # 约束检查：d * num_clients 必须是偶数
        if (d * num_clients) % 2 != 0:
            print(f"⚠️ 警告: d={d} 和 N={num_clients} 的乘积为奇数，自动将度数 d 调整为 {d+1}")
            d += 1
        G = nx.random_regular_graph(d, num_clients)
    else: 
        G = nx.erdos_renyi_graph(num_clients, p=0.3)
        
    # 保证图的连通性
    if not nx.is_connected(G):
        print(f"⚠️ 警告: 生成的 {topology_type} 图不连通，正在提取最大连通子图。")
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        
    return G
import random
import numpy as np

def allocate_malicious_nodes(G, num_mal, defense_budget, topology_type='scale_free', placement='Topology-Aware'):
    """
    采用约束满足算法分配恶意节点。
    确保任何一个防御节点的恶意邻居比例严格小于 50%。
    """
    num_clients = len(G.nodes())
    all_nodes = set(range(num_clients))
    
    # 1. 选定防御节点
    if placement == 'Topology-Aware':
        defense_nodes = set(get_high_value_defense_nodes(G, defense_budget, topology_type=topology_type))
    else:
        defense_nodes = set(random.sample(range(num_clients), defense_budget))

    # 2. 初始化防御节点的“配额”
    # 记录每个防御节点当前已有的恶意邻居数
    dn_bad_count = {dn: 0 for dn in defense_nodes}
    # 计算每个防御节点允许的最大恶意邻居数 (必须 < 50%)
    # 例如：邻居数 5, 最多 2 个坏人; 邻居数 4, 最多 1 个坏人
    dn_max_bad = {dn: int(np.ceil(len(list(G.neighbors(dn))) * 0.5) - 1) for dn in defense_nodes}

    # 3. 筛选初始候选池（非防御节点）
    potential_attackers = list(all_nodes - defense_nodes)
    random.shuffle(potential_attackers) # 随机化增加多样性

    malicious_clients = set()
    
    # 4. 贪心约束分配
    for candidate in potential_attackers:
        if len(malicious_clients) >= num_mal:
            break
            
        # 检查：如果选了这个 candidate，是否会破坏其相连的任何防御节点的安全性？
        is_safe = True
        candidate_neighbors = list(G.neighbors(candidate))
        
        for nb in candidate_neighbors:
            if nb in defense_nodes:
                if dn_bad_count[nb] + 1 > dn_max_bad[nb]:
                    is_safe = False
                    break
        
        # 如果安全，则分配并更新受影响防御节点的计数
        if is_safe:
            malicious_clients.add(candidate)
            for nb in candidate_neighbors:
                if nb in defense_nodes:
                    dn_bad_count[nb] += 1

    # 5. 最终检查与警告
    if len(malicious_clients) < num_mal:
        print(f"  [Critical Warning] 拓扑约束太紧！仅能安全分配 {len(malicious_clients)}/{num_mal} 个恶意节点。")
        # 此时不建议 fallback 到随机，因为随机必然导致 Eclipse。
        # 建议保持当前分配，或者增加 num_mal 个名额。

    return malicious_clients, list(defense_nodes)