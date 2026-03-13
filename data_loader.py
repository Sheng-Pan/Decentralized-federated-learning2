import torch

import torchvision.transforms as transforms
import random
from sklearn.datasets import load_breast_cancer
import networkx as nx
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
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
from datasets import Dataset as HFDataset 
import os
import zipfile
import pandas as pd
from datasets import Dataset as HFDataset
from torchvision import datasets, transforms
from huggingface_hub import hf_hub_download, snapshot_download
from google.colab import userdata
import multiprocessing
from torch.utils.data import Subset
def get_data(dataset_name='gtsrb', tokenizer=None, max_len=128, repo_id="JONESMITH007/DFL",n_train=1000,n_test=1000):
  
    train_ds = None
    test_ds = None
    data_root = './data'
    
    try:
        hf_token = userdata.get('HF_TOKEN')
    except:
        hf_token = None 

    if dataset_name.lower() == 'gtsrb':
        transform = transforms.Compose([
            transforms.Resize((32, 32)), 
            transforms.ToTensor(),
            transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
        ])

        try:
            
            train_ds = datasets.GTSRB(data_root, split='train', download=True, transform=transform)
            test_ds = datasets.GTSRB(data_root, split='test', download=True, transform=transform)
        except Exception as e:
         
            
            target_dir = os.path.join(data_root, 'gtsrb')
            if not os.path.exists(target_dir):
                os.makedirs(target_dir, exist_ok=True)
                
              
                zips = ['GTSRB-Training_fixed.zip', 'GTSRB_Final_Test_Images.zip', 'GTSRB_Final_Test_GT.zip']
                
                for zip_name in zips:
                    downloaded_path = hf_hub_download(
                        repo_id=repo_id, 
                        filename=zip_name, 
                        repo_type="dataset", 
                        token=hf_token
                    )
                    with zipfile.ZipFile(downloaded_path, 'r') as zip_ref:
                        zip_ref.extractall(target_dir)

         
            train_ds = datasets.GTSRB(data_root, split='train', download=False, transform=transform)
            test_ds = datasets.GTSRB(data_root, split='test', download=False, transform=transform)

  
    elif dataset_name.lower() in ['pubmed', 'pubmed_20k']:
        if tokenizer is None:
            raise ValueError(" 'tokenizer' parameter needed")

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
                raise FileNotFoundError(f"No file: {filename}")
            
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

     
        train_df = load_from_txt('train.txt')
        test_df = load_from_txt('test.txt')
        train_raw = HFDataset.from_pandas(train_df)
        test_raw = HFDataset.from_pandas(test_df)

    
        train_raw = train_raw.shuffle(seed=42).select(range(min(n_train, len(train_raw))))
        test_raw = test_raw.shuffle(seed=42).select(range(min(n_test, len(test_raw))))

        def tokenize_fn(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_len)


        num_cores = max(1, multiprocessing.cpu_count() - 1)
        
        train_ds = train_raw.map(tokenize_fn, batched=True, num_proc=num_cores)
        test_ds = test_raw.map(tokenize_fn, batched=True, num_proc=num_cores)

 
        train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        

    return train_ds, test_ds


def distribute_data(dataset, num_clients,dis = None,alpha=1):
    if dis == 'Dirichlet':
        if hasattr(dataset, 'targets'):
            labels = np.array(dataset.targets)
        elif hasattr(dataset, 'labels'):
            labels = np.array(dataset.labels)
        else:
        
            try:
                labels = np.array([item['labels'] for item in dataset])
            except KeyError:
                labels = np.array([item['label'] for item in dataset])
                
    
        if hasattr(labels[0], 'item'):
            labels = np.array([lbl.item() for lbl in labels])

        num_classes = len(np.unique(labels))
        client_indices = [[] for _ in range(num_clients)]

        for c in range(num_classes):
            idx_c = np.where(labels == c)[0]
            np.random.shuffle(idx_c)
            
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            
            splits = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
            
            idx_split = np.split(idx_c, splits)
            
            for i in range(num_clients):
                client_indices[i].extend(idx_split[i].tolist())

        client_datasets = []
        for i in range(num_clients):
            random.shuffle(client_indices[i])
                
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

def generate_topology(num_clients, topology_type='scale_free', d=4):
 
    if topology_type == 'ring': 
        G = nx.cycle_graph(num_clients)
    elif topology_type == 'scale_free': 
        G = nx.barabasi_albert_graph(num_clients, m=2) 
    elif topology_type == 'random_regular':
    
        if (d * num_clients) % 2 != 0:
            d += 1
        G = nx.random_regular_graph(d, num_clients)
    else: 
        G = nx.erdos_renyi_graph(num_clients, p=0.3)
        

    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        
    return G


def allocate_malicious_nodes(G, num_mal, defense_budget, topology_type='scale_free', placement='Topology-Aware'):

    num_clients = len(G.nodes())
    all_nodes = set(range(num_clients))
    

    if placement == 'Topology-Aware':
        defense_nodes = set(get_high_value_defense_nodes(G, defense_budget, topology_type=topology_type))
    else:
        defense_nodes = set(random.sample(range(num_clients), defense_budget))


    dn_bad_count = {dn: 0 for dn in defense_nodes}
    dn_max_bad = {dn: int(np.ceil(len(list(G.neighbors(dn))) * 0.5) - 1) for dn in defense_nodes}

    potential_attackers = list(all_nodes - defense_nodes)
    random.shuffle(potential_attackers) 

    malicious_clients = set()
    
    for candidate in potential_attackers:
        if len(malicious_clients) >= num_mal:
            break
            
        is_safe = True
        candidate_neighbors = list(G.neighbors(candidate))
        
        for nb in candidate_neighbors:
            if nb in defense_nodes:
                if dn_bad_count[nb] + 1 > dn_max_bad[nb]:
                    is_safe = False
                    break
        
        if is_safe:
            malicious_clients.add(candidate)
            for nb in candidate_neighbors:
                if nb in defense_nodes:
                    dn_bad_count[nb] += 1

    
    if len(malicious_clients) < num_mal:
        print(f"  [Critical Warning] Topology constraints are too strict! Can only safely allocate {len(malicious_clients)}/{num_mal} malicious clients.")
   

    return malicious_clients, list(defense_nodes)