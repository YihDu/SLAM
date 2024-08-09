import json
import sys
from src import graph_build , graph_analysis, clustering_metrics , mmd_calculation , weight_function

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config

def SLAM(config_path):
    config = load_config(config_path)
    graph_builder = graph_build.GraphBuilder(config)
    truth_graph , pred_graph = graph_builder.process_graph()
    graph_analyzer = graph_analysis.GraphAnalyzer(config , truth_G = truth_graph , pred_G = pred_graph)
    sample_sets_truth , sample_sets_pred = graph_analyzer.analyze_graph()
    sigma = config['kernel_parameters']['sigma']
    SLAM_score = mmd_calculation.compute_mmd(sample_sets_truth,sample_sets_pred,kernel_module = mmd_calculation.GaussianEMDKernel(sigma=sigma).to('cuda'))
    print("SLAM :" , SLAM_score)
    print('----------------------------------------------------')
    return SLAM_score
    
if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'config.json'
    SLAM(config_path)
    
    config = load_config(config_path)
    clustering_metrics.compute_clustering_metrics(config)