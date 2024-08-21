import numpy as np
from sklearn.neighbors import KernelDensity

class GraphAnalyzer:
    def __init__(self , config , truth_G , pred_G):
        self.config = config
        self.truth_G = truth_G
        self.pred_G = pred_G
    
    def fit_kde_and_sample(self, samples, num_samples , sample_times , bandwidth=None, random_seed=None):
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
        kde.fit(samples)
        samples_set = []
        for i in range(sample_times):
            sampled = kde.sample(num_samples ,random_state=random_seed)
            sampled = np.clip(sampled, 0, 1)   
            # smapled = np.clip(sampled, 0, 2) 
            samples_set.append(sampled) 
        return samples_set
    
    def get_edge_attributes(self, graph, apply_gene_similarity, apply_anomaly_severity_weight, apply_distance_weight, unique_groups):
        group_to_onehot = {group: np.array([1 if i == group else 0 for i in unique_groups], dtype=np.float64) for group in unique_groups}
        samples = []
        for u, v in graph.edges():
            group_u = graph.nodes[u]['group']
            group_v = graph.nodes[v]['group']
            if group_u == group_v:
                encoding = group_to_onehot[group_u].copy()
            else:
                encoding = np.zeros(len(unique_groups), dtype=np.float64)
            if apply_gene_similarity:
                gene_similarity_weight = graph[u][v].get('gene_similarity_weight', 1.0)
                encoding *= gene_similarity_weight
            if apply_anomaly_severity_weight:
                ad_weight = graph[u][v].get('anomaly_severity_weight', 1.0)
                encoding *= ad_weight
            if apply_distance_weight:
                distance_weight = graph[u][v].get('distance_weight', 1.0)
                encoding *= distance_weight
            samples.append(encoding)
        result = np.stack(samples)
        return result
    
    def get_unique_groups(self):
        groups_in_truth_G = {node_data['group'] for _, node_data in self.truth_G.nodes(data=True)}
        groups_in_pred_G = {node_data['group'] for _, node_data in self.pred_G.nodes(data=True)}
        unique_groups = sorted(groups_in_truth_G.union(groups_in_pred_G))
        return unique_groups
    
    def analyze_graph(self):
        apply_gene_similarity = self.config['graph_builder'].get('apply_gene_similarity' , False)
        apply_anomaly_severity_weight = self.config['graph_builder'].get('apply_anomaly_severity_weight' , False)
        apply_distance_weight = self.config['graph_builder'].get('apply_distance_weight' , False)
        
        num_samples = len(self.pred_G.edges())
        sample_times = self.config['graph_analysis']['sample_times']
        group = self.get_unique_groups()

        samples_truth = self.get_edge_attributes(self.truth_G, apply_gene_similarity, apply_anomaly_severity_weight , apply_distance_weight , group)
        samples_pred = self.get_edge_attributes(self.pred_G, apply_gene_similarity, apply_anomaly_severity_weight ,apply_distance_weight , group)
                
        samples_set_truth = self.fit_kde_and_sample(samples_truth, num_samples , sample_times , bandwidth=0.1, random_seed=42)
        samples_set_pred = self.fit_kde_and_sample(samples_pred, num_samples , sample_times , bandwidth=0.1, random_seed=42)
        
        return samples_set_truth , samples_set_pred
