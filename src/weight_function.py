from scipy.sparse import isspmatrix
import scanpy as sc
import numpy as np

def calculate_anomaly_weight(graph , dict_severity_levels):
    severity_mapping = {category['name']: category['severity_level'] for category in dict_severity_levels}
    for u, v in graph.edges():
        group_u = graph.nodes[u]['group']
        group_v = graph.nodes[v]['group']
        severity_u = severity_mapping[group_u]
        severity_v = severity_mapping[group_v]
        anomaly_severity_weight = (severity_u + severity_v) / 2
        graph.edges[u, v]['anomaly_severity_weight'] = anomaly_severity_weight 

def calculate_pearson_similarity(x, y):
    vx = x - np.mean(x)
    vy = y - np.mean(y)
    return np.sum(vx * vy) / (np.sqrt(np.sum(vx ** 2)) * np.sqrt(np.sum(vy ** 2)))

def calculate_gene_similarity(graph, anndata , is_preprocessed = True):
    if not is_preprocessed:
        preprocessed_data = preprocess_anndata(anndata)
        gene_expression_matrix = preprocessed_data[:, preprocessed_data.var['highly_variable']].X
    if isspmatrix(anndata.X):
        gene_expression_matrix = anndata.X.toarray()
    else:
        gene_expression_matrix = np.array(anndata.X, dtype=np.float32)     
            
    group_means = {}
    group_indices = {}
    for node , data in graph.nodes(data = True):
        group = data['group']
        if group not in group_indices:
            group_indices[group] = []
        group_indices[group].append(node)
    
    for group , indices in group_indices.items():
        group_expression_matrix = gene_expression_matrix[indices]
        group_mean_vector = np.mean(group_expression_matrix, axis=0)
        group_means[group] = group_mean_vector
        
    pearson_matrix = np.corrcoef(gene_expression_matrix)
    for u, v in graph.edges():
        group_u = graph.nodes[u]['group']
        group_v = graph.nodes[v]['group']
        
        if group_u == group_v:
            group_mean = group_means[group_u]
            similarity_u = calculate_pearson_similarity(gene_expression_matrix[u], group_mean)
            similarity_v = calculate_pearson_similarity(gene_expression_matrix[v], group_mean)
            graph.edges[u, v]['gene_similarity_weight'] = 0.5 * (similarity_u + similarity_v)

        else:
            graph.edges[u, v]['gene_similarity_weight'] = 1 - pearson_matrix[u, v]

def preprocess_anndata(adata):
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata , n_top_genes = 3000)
    return adata

