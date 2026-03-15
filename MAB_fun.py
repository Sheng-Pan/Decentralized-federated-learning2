import numpy as np
import torch
import math
from collections import defaultdict
# ==========================================
#  General functions
# ==========================================


def calc_activation_stats(model, target_layer_name, probe_inputs):
  
    activations = {}
    
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            activations['out'] = output[0].detach()
        else:
            activations['out'] = output.detach()
        

    target_layer = dict([*model.named_modules()]).get(target_layer_name)
    if target_layer is None:
        return 0.0
        
    handle = target_layer.register_forward_hook(hook_fn)
    
    model.eval()
    with torch.no_grad():
   
        if isinstance(probe_inputs, dict):
            _ = model(**probe_inputs)
        else:
            _ = model(probe_inputs)
        
    handle.remove() 
    
    acts_tensor = activations.get('out')
    if acts_tensor is None:
        return 0.0
        
    acts_tensor = acts_tensor.flatten()
    

    valid_acts = acts_tensor
    

    if len(valid_acts) >= 3:

        if valid_acts.dtype not in [torch.float32, torch.float64]:
            valid_acts = valid_acts.float()
            
        mean_a = torch.mean(valid_acts)
        var_a = torch.var(valid_acts, unbiased=False) + 1e-9 
        
    
        fourth_moment = torch.mean((valid_acts - mean_a) ** 4)
        
    
        kurtosis = fourth_moment / (var_a ** 2)
        
        return kurtosis.item()
        
    return 0.0
def calculate_continuous_trust(neighbor_metrics):
    if not neighbor_metrics: return {}


    trap_vals = np.abs(np.array([m['trap'] for m in neighbor_metrics])) + 1e-12
    sz_vals   = np.array([m['s_z'] for m in neighbor_metrics]) + 1e-12 
    rs_vals   = np.array([m.get('rs_score', 1.0) for m in neighbor_metrics])

    def calculate_robust_z(vals, eps=0.1, use_log=True):
        calc_vals = np.log1p(vals) if use_log else vals
        median = np.median(calc_vals)
        mad = np.median(np.abs(calc_vals - median))
        robust_mad = max(mad, eps) 
        return 0.6745 * np.abs(calc_vals - median) / robust_mad
    
 
    z_trap = calculate_robust_z(trap_vals, use_log=True)
    z_sz   = calculate_robust_z(sz_vals, use_log=True)
    z_rs   = calculate_robust_z(rs_vals, use_log=False, eps=0.05) 


    z_comb = z_trap + z_rs + z_sz

  
    trust_scores = np.exp(- z_comb)

    result = {}
    for i, m in enumerate(neighbor_metrics):
        result[m['nid']] = {
            'reward': float(trust_scores[i]),
            'z_comb': float(z_comb[i]),
            'z_trap': float(z_trap[i]),
            'z_rs':   float(z_rs[i]),
            'z_sz':   float(z_sz[i]), 
            'rs_score': float(rs_vals[i])
        }
    return result

# ==========================================
# MAB
# ==========================================
class MABDefense:
    def __init__(self, num_clients, model_type='cnn', decay=0.9, exploration_c=0.5, 
                 audit_prob=0.9, agg_prob=0.8, agg_threshold=0.4, custom_target_layers=None,
                 alpha=0.2): 
        self.num_clients = num_clients
        self.model_type = model_type.lower()
        
        self.trust_scores = {i: {j: 0.5 for j in range(num_clients)} for i in range(num_clients)}

        self.visit_counts = {i: defaultdict(float) for i in range(num_clients)}
        self.total_rounds = {i: 0 for i in range(num_clients)}
        
        self.decay = decay
        self.c = exploration_c
        self.audit_prob = audit_prob
        self.agg_prob = agg_prob
        self.agg_threshold = agg_threshold
        

        self.alpha = alpha
        self.gamma = 1.0 - self.alpha # gamma = 1 - alpha

        if self.model_type == 'transformer':
            self.target_layers = ['classifier.weight', 'pre_classifier.weight', 'fc.weight', 'linear.weight', 'out_proj.weight']
        elif self.model_type == 'cnn': 
            self.target_layers = ['fc2.weight', 'linear.weight', 'fc.weight']
        elif self.model_type == 'mlp': 
            self.target_layers = ['fc1.weight', 'fc2.weight', 'fc3.weight']
            
        if custom_target_layers is not None:
            self.target_layers = custom_target_layers

    def select_for_audit(self, client_id, neighbors_list, audit_budget=None):
   
            self.total_rounds[client_id] += 1
            n_gamma_t = sum(self.visit_counts[client_id].get(nid, 0.0) for nid in neighbors_list)
            log_n_gamma = math.log(n_gamma_t) if n_gamma_t > 1.0 else 0.0 

            u_scores = {}
            for nid in neighbors_list:
                q = self.trust_scores[client_id].get(nid, 0.5)
                n_v = self.visit_counts[client_id].get(nid, 0.0)
                bonus = self.c * math.sqrt((2 * log_n_gamma) / n_v) if n_v > 0 else self.c * 10.0
                u_scores[nid] = max(q + bonus, 1e-9)

            if audit_budget is None:
                audit_budget = max(1, round(len(neighbors_list) * self.audit_prob))
            
      
            scores_list = np.array([u_scores[nid] for nid in neighbors_list])
            probs = scores_list / np.sum(scores_list)
            audit_targets = np.random.choice(neighbors_list, size=min(audit_budget, len(neighbors_list)), 
                                            replace=False, p=probs).tolist()
            return audit_targets

    def select_for_aggregation(self, client_id, candidates, agg_budget=None):
     
        valid_candidates = [n for n in candidates if self.trust_scores[client_id].get(n, 0) > self.agg_threshold]
        
        if not valid_candidates: return [], []

        if agg_budget is None:
            agg_budget = max(1, int(len(valid_candidates) * self.agg_prob))
        
   
        qs = np.array([self.trust_scores[client_id][nid] for nid in valid_candidates])
        probs = qs / np.sum(qs)
        selected = np.random.choice(valid_candidates, size=min(agg_budget, len(valid_candidates)), 
                                    replace=False, p=probs).tolist()

  
        final_qs = [self.trust_scores[client_id][nid] for nid in selected]
        weights = [q / (sum(final_qs) + 1e-9) for q in final_qs]
        
        return selected, weights
    


    def update_trust(self, observer_id, probe_list, new_weights, old_weights, device, model_template,
                     sensitivity_func=None, tokenizer=None, get_trap_func=None, rs_func=None,
                     probe_inputs=None): 
        detailed_logs = {}
        if not probe_list: return detailed_logs


        for nid in self.visit_counts[observer_id]:
            self.visit_counts[observer_id][nid] *= self.gamma

        neighbor_metrics = []

        if probe_inputs is None:
            probe_inputs = torch.randn(16, 3, 32, 32, device=device) 
        else:
            probe_inputs = probe_inputs.to(device)

        for nid in probe_list:
            nid = int(nid)
            if not isinstance(new_weights[nid], dict): continue
            
            target_layer_param = 'fc2.weight'
            if target_layer_param not in new_weights[nid]: target_layer_param = 'linear.weight'
            if target_layer_param not in new_weights[nid]: target_layer_param = list(new_weights[nid].keys())[-2]
            
            target_layer_module_name = target_layer_param.replace('.weight', '')

    
            nb_state = {k: v.float().to(device) for k, v in new_weights[nid].items()}
            model_template.load_state_dict(nb_state)
            probe_inputs = probe_inputs.to(device)
     
            s_z_val = calc_activation_stats(model_template, target_layer_module_name, probe_inputs)
       
            trap_s = 0.0
            if get_trap_func:
                try:
                    trap_s = get_trap_func(model_template, tokenizer, device)
                except TypeError:
                    trap_s = get_trap_func(model_template, device)

            rs_val = 1.0
            if rs_func:
                try:
                    rs_val = rs_func(model_template, tokenizer, device)
                except Exception:
                    rs_val = rs_func(model_template, device)

            neighbor_metrics.append({
                'nid': nid,
                'trap': trap_s, 
                's_z': s_z_val,  
                'rs_score': rs_val
            })

        if not neighbor_metrics: return detailed_logs

        trust_map = calculate_continuous_trust(neighbor_metrics)
        
        for m in neighbor_metrics:
            nid = m['nid']
            trust_data = trust_map.get(nid, {'reward': 0.5, 'z_comb': 0.0})

            instant_reward = trust_data['reward']
            z_combined = trust_data['z_comb']
            
            curr_q = self.trust_scores[observer_id][nid]
            new_q = (1 - self.alpha) * curr_q + self.alpha * instant_reward
            self.trust_scores[observer_id][nid] = new_q
            
            self.visit_counts[observer_id][nid] += 1.0 
            
            detailed_logs[nid] = {
                'trap': m['trap'],
                'elem_z': m['s_z'],  
                'rs_score': m['rs_score'],
                'z_trap': trust_data['z_trap'],
                'z_rs': trust_data['z_rs'],
                'z_sz': trust_data['z_sz'], 
                'z_comb_penalty': z_combined,
                'reward': instant_reward,
                'new_q': new_q
            }

        return detailed_logs
    