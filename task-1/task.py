import torch
import cupy as cp
import triton
import triton.language as tl
import numpy as np
import time
import json
from test import testdata_kmeans, testdata_knn, testdata_ann

# ------------------------------------------------------------------------------------------------
# Task 1.1: Distance Functions
# ------------------------------------------------------------------------------------------------

@triton.jit
def l2_kernel(X_ptr, Y_ptr, D, output_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < D
    
    x = tl.load(X_ptr + offsets, mask=mask)
    y = tl.load(Y_ptr + offsets, mask=mask)
    diff = x - y
    squared = diff * diff
    
    sum_sq = tl.sum(squared, axis=0)
    if pid == 0:
        tl.store(output_ptr, tl.sqrt(sum_sq))

@triton.jit
def cosine_kernel(X_ptr, Y_ptr, D, output_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < D
    
    x = tl.load(X_ptr + offsets, mask=mask)
    y = tl.load(Y_ptr + offsets, mask=mask)
    
    dot = tl.sum(x * y, axis=0)
    norm_x = tl.sqrt(tl.sum(x * x, axis=0))
    norm_y = tl.sqrt(tl.sum(y * y, axis=0))
    
    similarity = dot / (norm_x * norm_y)
    tl.store(output_ptr + pid, 1 - similarity)

@triton.jit
def manhattan_kernel(X_ptr, Y_ptr, D, output_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < D
    
    x = tl.load(X_ptr + offsets, mask=mask)
    y = tl.load(Y_ptr + offsets, mask=mask)
    diff = tl.abs(x - y)
    
    sum_abs = tl.sum(diff, axis=0)
    if pid == 0:
        tl.store(output_ptr, sum_abs)

@triton.jit
def dot_kernel(X_ptr, Y_ptr, D, output_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < D
    
    x = tl.load(X_ptr + offsets, mask=mask)
    y = tl.load(Y_ptr + offsets, mask=mask)
    
    dot = tl.sum(x * y, axis=0)
    tl.store(output_ptr, dot)

def distance_l2(X, Y):
    output = torch.empty(1, device=X.device)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(X.numel(), BLOCK_SIZE),)
    l2_kernel[grid](X, Y, X.numel(), output, BLOCK_SIZE=BLOCK_SIZE)
    return output.item()

def distance_cosine(X, Y):
    output = torch.empty(1, device=X.device)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(X.numel(), BLOCK_SIZE),)
    cosine_kernel[grid](X, Y, X.numel(), output, BLOCK_SIZE=BLOCK_SIZE)
    return output.item()

def distance_manhattan(X, Y):
    output = torch.empty(1, device=X.device)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(X.numel(), BLOCK_SIZE),)
    manhattan_kernel[grid](X, Y, X.numel(), output, BLOCK_SIZE=BLOCK_SIZE)
    return output.item()

def distance_dot(X, Y):
    output = torch.empty(1, device=X.device)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(X.numel(), BLOCK_SIZE),)
    dot_kernel[grid](X, Y, X.numel(), output, BLOCK_SIZE=BLOCK_SIZE)
    return output.item()

# ------------------------------------------------------------------------------------------------
# Task 1.2: KNN Implementation
# ------------------------------------------------------------------------------------------------

@triton.jit
def knn_distance_kernel(A_ptr, X_ptr, N, D, distances_ptr, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    if row >= N:
        return
    sum_sq = 0.0
    for i in range(0, D, BLOCK_SIZE):
        offsets = row * D + i + tl.arange(0, BLOCK_SIZE)
        mask = (i + tl.arange(0, BLOCK_SIZE)) < D
        a = tl.load(A_ptr + offsets, mask=mask)
        x = tl.load(X_ptr + i + tl.arange(0, BLOCK_SIZE), mask=mask)
        sum_sq += tl.sum((a - x)**2, axis=0)
    tl.store(distances_ptr + row, tl.sqrt(sum_sq))

def our_knn(N, D, A, X, K):
    distances = torch.empty(N, device=A.device)
    BLOCK_SIZE = 128
    grid = (N,)
    knn_distance_kernel[grid](A, X, N, D, distances, BLOCK_SIZE=BLOCK_SIZE)
    _, indices = torch.topk(distances, K, largest=False)
    return A[indices]

# ------------------------------------------------------------------------------------------------
# Task 2.1: K-means Implementation
# ------------------------------------------------------------------------------------------------

@triton.jit
def kmeans_assign_kernel(A_ptr, centroids_ptr, N, D, K, labels_ptr):
    pid = tl.program_id(0)
    offsets = pid * D + tl.arange(0, D)
    mask = offsets < N*D
    
    vector = tl.load(A_ptr + offsets, mask=mask)
    min_dist = float('inf')
    best_cluster = 0
    
    for k in range(K):
        centroid = tl.load(centroids_ptr + k*D + tl.arange(0, D))
        dist = tl.sum((vector - centroid)**2)
        if dist < min_dist:
            min_dist = dist
            best_cluster = k
    
    tl.store(labels_ptr + pid, best_cluster)

@triton.jit
def kmeans_update_kernel(A_ptr, labels_ptr, centroids_ptr, N, D, K):
    k = tl.program_id(0)
    count = 0
    sum_vector = tl.zeros((D,), dtype=tl.float32)
    
    for i in range(N):
        label = tl.load(labels_ptr + i)
        if label == k:
            vector = tl.load(A_ptr + i*D + tl.arange(0, D))
            sum_vector += vector
            count += 1
    
    if count > 0:
        new_centroid = sum_vector / count
        tl.store(centroids_ptr + k*D + tl.arange(0, D), new_centroid)

def our_kmeans(N, D, A, K):
    # Initialize centroids using K-means++
    centroids = A[torch.randperm(N)[:K]].contiguous()
    labels = torch.empty(N, dtype=torch.long, device=A.device)
    
    for _ in range(10):  # Fixed iterations for demo
        grid = (N,)
        kmeans_assign_kernel[grid](A, centroids, N, D, K, labels)
        
        grid = (K,)
        kmeans_update_kernel[grid](A, labels, centroids, N, D, K)
    
    return centroids, labels

# ------------------------------------------------------------------------------------------------
# Task 2.2: ANN Implementation (IVF)
# ------------------------------------------------------------------------------------------------

@triton.jit
def ivf_query_kernel(q_ptr, centroids_ptr, cluster_ptr, 
                    N, D, K, output_ptr, n_probe: tl.constexpr):
    # Find nearest n_probe clusters
    pid = tl.program_id(0)
    if pid >= n_probe:
        return
    
    # Simplified cluster search (actual implementation would use distance calc)
    cluster_id = pid
    start = tl.load(cluster_ptr + cluster_id)
    end = tl.load(cluster_ptr + cluster_id + 1)
    
    # Search within cluster
    min_dist = float('inf')
    best_idx = -1
    for i in range(start, end):
        vec = tl.load(q_ptr + i*D + tl.arange(0, D))
        dist = tl.sum((vec - q_ptr)**2)
        if dist < min_dist:
            min_dist = dist
            best_idx = i
    
    tl.atomic_min(output_ptr, best_idx)

def our_ann(N, D, A, X, K):
    # Build IVF index
    centroids, labels = our_kmeans(N, D, A, int(np.sqrt(N)))
    
    # Build cluster pointers
    cluster_counts = torch.bincount(labels)
    cluster_ptr = torch.cumsum(cluster_counts, 0)
    
    # Search nearest clusters
    distances = torch.norm(centroids - X, dim=1)
    _, cluster_indices = torch.topk(distances, 3, largest=False)  # n_probe=3
    
    # Search within clusters
    output = torch.full((K, D), float('inf'), device=A.device)
    grid = (3,)  # n_probe=3
    ivf_query_kernel[grid](X, centroids, cluster_ptr, N, D, K, output, n_probe=3)
    
    return output

# ------------------------------------------------------------------------------------------------
# Testing Infrastructure
# ------------------------------------------------------------------------------------------------

def recall_rate(list1, list2):
    return len(set(list1) & set(list2)) / len(list1)

if __name__ == "__main__":
    test_kmeans()
