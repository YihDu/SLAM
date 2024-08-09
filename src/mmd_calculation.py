import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn

def swd_batch(vectors1, vectors2, n_repeat_projection=512, proj_per_repeat=16, device="cuda" , random_seed=42):
    assert vectors1.size() == vectors2.size()
    assert vectors1.ndim == 3 and vectors2.ndim == 3

    batch_size, _ , vector_dim = vectors1.size()
    vectors1, vectors2 = vectors1.to(device), vectors2.to(device)

    distances = torch.zeros((n_repeat_projection, batch_size), device=device)

    for i in range(n_repeat_projection):
        rand = rand_projections(vector_dim, proj_per_repeat , random_seed = random_seed + i).to(device)
        proj1 = torch.matmul(vectors1, rand.transpose(0, 1))
        proj2 = torch.matmul(vectors2, rand.transpose(0, 1))
        
        proj1, _ = torch.sort(proj1, dim=1)
        proj2, _ = torch.sort(proj2, dim=1)
        
        d = torch.mean((proj1 - proj2) ** 2, dim=1)
        distances[i] = torch.mean(d, dim=1)
    
    result = torch.mean(distances, dim=0)
    return result

def rand_projections(dim, num_projections=16 , random_seed=None):
    if random_seed is not None:
        torch.manual_seed(random_seed)
    projections = torch.randn((num_projections, dim))
    projections = projections / torch.sqrt(torch.sum(projections ** 2, dim=1, keepdim=True))
    return projections

class GaussianEMDKernel(nn.Module):
    def __init__(self, sigma=1.0):
        super(GaussianEMDKernel, self).__init__()
        self.sigma = sigma
        print('GaussianEMDKernel sigma:', sigma)

    def forward(self, x, y):
        swd_value = swd_batch(x, y, device=x.device)
        swd_tensor = swd_value.to(x.device)
        return torch.exp(- swd_tensor * swd_tensor / (2 * self.sigma * self.sigma))

def compute_kernel_matrix(samples1, samples2, kernel_module, sigma, device='cuda'):
    samples1 = samples1.to(device)
    samples2 = samples2.to(device)
    n = samples1.size(0)
    m = samples2.size(0)
    K = torch.zeros(n, m, device=device)
    for i in tqdm(range(n), desc="Computing kernel matrix"):
        x_i = samples1[i].unsqueeze(0).expand(m, -1, -1)
        K[i] = kernel_module(x_i, samples2).view(-1)
    # print('Kernel matrix:', K)
    return K

def compute_mmd(samples1, samples2, kernel_module, sigma = 1, is_hist=False, device='cuda'):
    if is_hist:
        samples1 = [s1 / s1.sum(dim=0, keepdim=True) for s1 in samples1]
        samples2 = [s2 / s2.sum(dim=0, keepdim=True) for s2 in samples2]
    samples1 = torch.stack([torch.tensor(s, dtype=torch.float32) if isinstance(s, np.ndarray) else s for s in samples1])
    samples2 = torch.stack([torch.tensor(s, dtype=torch.float32) if isinstance(s, np.ndarray) else s for s in samples2])

    # print('Samples1:', samples1.size())
    # print('Samples2:', samples2.size())
    
    K_XX = compute_kernel_matrix(samples1, samples1, kernel_module, sigma, device)
    K_YY = compute_kernel_matrix(samples2, samples2, kernel_module, sigma, device)
    K_XY = compute_kernel_matrix(samples1, samples2, kernel_module, sigma, device)
    # print('K_XX:', K_XX.mean().item())
    # print('K_YY:', K_YY.mean().item())
    # print('K_XY:', K_XY.mean().item())
    return K_XX.mean().item() + K_YY.mean().item() - 2 * K_XY.mean().item()