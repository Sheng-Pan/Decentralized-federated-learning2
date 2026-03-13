
import numpy as np


def calculate_theoretical_intensity(neighbors, malicious_clients, num_clients, intensity, lambda_benign=0.3):
 
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
        

    benign_indices = [i for i in range(num_clients) if i not in malicious_clients]
    if len(benign_indices) > 0:
        benign_intensities = steady_state[benign_indices]
        

        min_val = np.min(benign_intensities)
        max_val = np.max(benign_intensities)

        if max_val - min_val > 1e-9:
            steady_state = (steady_state - min_val) / (max_val - min_val)
        else:
            steady_state = steady_state - min_val 
            

        
    return steady_state
