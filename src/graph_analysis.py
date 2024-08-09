import numpy as np
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.interpolate as interpolate

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
            sampled = np.clip(sampled, 0, 1)      ## ! 
            samples_set.append(sampled)
        # visualization
        # self.plot_marginal_distributions(samples, samples_set)        
        return samples_set

    def plot_marginal_distributions(self, original_samples, samples_set , bandwidth=0.1):
        num_dimensions = original_samples.shape[1]
        fig, axes = plt.subplots(num_dimensions, 2, figsize=(16, 6 * num_dimensions))
        
        for i in range(num_dimensions):
            # original
            ax_kde = axes[i, 0] if num_dimensions > 1 else axes[0]
            # sns.histplot(original_samples[:, i], bins=30, kde=False, label='Original Histogram', color='blue', alpha=0.5, ax=ax_kde)
            sns.kdeplot(original_samples[:, i], fill=True, bw_adjust= bandwidth,  label='Original KDE', color='blue', ax=ax_kde)
            ax_kde.set_title(f'KDE and Histogram of Dimension {i+1}')
            ax_kde.set_xlabel(f'Dimension {i+1}')
            ax_kde.set_ylabel('Density')
            ax_kde.legend()
            
            # sample
            ax_hist = axes[i, 1] if num_dimensions > 1 else axes[1]
            for j, samples in enumerate(samples_set):
                sns.histplot(samples[:, i], bins=30, kde=False, alpha=0.3, label=f'Sample {j+1}', ax=ax_hist)
            ax_hist.set_title(f'Sampled Histograms of Dimension {i+1}')
            ax_hist.set_xlabel(f'Dimension {i+1}')
            ax_hist.set_ylabel('Density')
            if i == 0:
                ax_hist.legend()
        plt.tight_layout()
        plt.show()
    
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
        print('Unique groups:' , group)
        samples_truth = self.get_edge_attributes(self.truth_G, apply_gene_similarity, apply_anomaly_severity_weight , apply_distance_weight , group)
        samples_pred = self.get_edge_attributes(self.pred_G, apply_gene_similarity, apply_anomaly_severity_weight ,apply_distance_weight , group)
                
        # print('Samples truth:' , samples_truth)
        # print('Samples pred:' , samples_pred)
        samples_set_truth = self.fit_kde_and_sample(samples_truth, num_samples , sample_times , bandwidth=0.1, random_seed=42)
        samples_set_pred = self.fit_kde_and_sample(samples_pred, num_samples , sample_times , bandwidth=0.1, random_seed=42)
        
        # print('Samples set truth:' , samples_set_truth)
        # print('Samples set pred:' , samples_set_pred)
        return samples_set_truth , samples_set_pred
