import torch
from scipy.sparse import isspmatrix
import scanpy as sc

def calculate_distance_weight(graph , x):
    for u, v in graph.edges():
        distance_u = graph.nodes[u]['pos'][0] - x
        distance_v = graph.nodes[v]['pos'][0] - x
        distance_u = torch.tensor(distance_u , dtype = torch.float32)
        distance_v = torch.tensor(distance_v , dtype = torch.float32)
        distance_weight = torch.exp(- 0.1 * (distance_u + distance_v))
        # print('Distance Weight:', distance_weight)
        graph[u][v]['distance_weight'] = distance_weight

def calculate_anomaly_weight(graph , dict_severity_levels):
    severity_mapping = {category['name']: category['severity_level'] for category in dict_severity_levels}
    for u, v in graph.edges():
        group_u = graph.nodes[u]['group']
        group_v = graph.nodes[v]['group']
        severity_u = severity_mapping[group_u]
        severity_v = severity_mapping[group_v]
        anomaly_severity_weight = (severity_u + severity_v) / 2
        # print('anomaly_severity_weight:' , anomaly_severity_weight)
        graph.edges[u, v]['anomaly_severity_weight'] = anomaly_severity_weight 

## calculate Gene Similarity with 2 vector 
def calculate_pearson_similarity(x, y):
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    return torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))


def calculate_gene_similarity(graph, anndata , is_preprocessed = True):
    if not is_preprocessed:
        preprocessed_data = preprocess_anndata(anndata)
        gene_expression_matrix = preprocessed_data[:, preprocessed_data.var['highly_variable']].X
    if isspmatrix(anndata.X):
        gene_expression_matrix = anndata.X
    else:
        gene_expression_matrix = anndata.X        
    gene_expression_matrix = torch.tensor(gene_expression_matrix, dtype=torch.float32)
    
    group_means = {}
    group_indices = {}
    for node , data in graph.nodes(data = True):
        group = data['group']
        if group not in group_indices:
            group_indices[group] = []
        group_indices[group].append(node)
    
    for group , indices in group_indices.items():
        group_expression_matrix = gene_expression_matrix[indices]
        group_mean_vector = torch.mean(group_expression_matrix, axis=0)
        group_means[group] = group_mean_vector
        
    pearson_matrix = torch.corrcoef(gene_expression_matrix)
    for u, v in graph.edges():
        group_u = graph.nodes[u]['group']
        group_v = graph.nodes[v]['group']
        
        # same group use similarity
        if group_u == group_v:
            group_mean = group_means[group_u]
            similarity_u = calculate_pearson_similarity(gene_expression_matrix[u], group_mean)
            similarity_v = calculate_pearson_similarity(gene_expression_matrix[v], group_mean)
            graph.edges[u, v]['gene_similarity_weight'] = 0.5 * (similarity_u + similarity_v)

        # different group use distance 
        else:
            graph.edges[u, v]['gene_similarity_weight'] = 1 - pearson_matrix[u, v]

        # print('gene_similarity_weight:' , graph.edges[u, v]['gene_similarity_weight'])

def preprocess_anndata(adata):
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata , n_top_genes = 3000)
    return adata

