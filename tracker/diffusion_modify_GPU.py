import os
import time
import numpy as np
import joblib
from joblib import Parallel, delayed
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
import torch
from tqdm import tqdm
from knn import KNN, ANN


trunc_ids = None
trunc_init = None
lap_alpha = None
def get_offline_result(i):
    ids = trunc_ids[i]
    trunc_lap = lap_alpha[ids][:, ids]
    
    try:
        scores, _ = linalg.cg(trunc_lap, trunc_init, tol=1e-6, maxiter=20)
    except Exception as e:
        print(f"[error] CG failed at index {i}, using zeros. Reason: {e}")
        return np.zeros_like(trunc_init)

    if np.any(np.isnan(scores)) or np.any(np.isinf(scores)):
        print(f"[warn] NaN or Inf scores detected at index {i}, using zeros")
        return np.zeros_like(trunc_init)
    
    return scores

def cache(filename):
    """Decorator to cache results
    """
    def decorator(func):
        def wrapper(*args, **kw):
            self = args[0]
            path = os.path.join(self.cache_dir, filename)
            time0 = time.time()
            if os.path.exists(path):
                result = joblib.load(path)
                cost = time.time() - time0
                print('[cache] loading {} costs {:.2f}s'.format(path, cost))
                return result
            result = func(*args, **kw)
            cost = time.time() - time0
            print('[cache] obtaining {} costs {:.2f}s'.format(path, cost))
            joblib.dump(result, path)
            return result
        return wrapper
    return decorator


class Diffusion(object):
    """Diffusion class
    """
    def __init__(self, features, cache_dir):
        self.features = features
        self.N = len(self.features)
        self.cache_dir = cache_dir
        # use ANN for large datasets
        self.use_ann = self.N >= 100000
        if self.use_ann:
            self.ann = ANN(self.features, method='cosine')
        self.knn = KNN(self.features, method='cosine')
    
    # @cache('laplacian.jbl')
    def get_laplacian(self, sims, ids, alpha=0.99):
        """Get Laplacian_alpha matrix
        """
        affinity = self.get_affinity(sims, ids)
        num = affinity.shape[0]
        degrees = affinity @ np.ones(num) + 1e-12
        # mat: degree matrix ^ (-1/2)
        mat = sparse.dia_matrix(
            (degrees ** (-0.5), [0]), shape=(num, num), dtype=np.float32)
        stochastic = mat @ affinity @ mat
        sparse_eye = sparse.dia_matrix(
            (np.ones(num), [0]), shape=(num, num), dtype=np.float32)
        lap_alpha = sparse_eye - alpha * stochastic
        return lap_alpha

    # @cache('affinity.jbl')
    def get_affinity(self, sims, ids, gamma=3):
        """Create affinity matrix for the mutual kNN graph of the whole dataset
        Args:
            sims: similarities of kNN
            ids: indexes of kNN
        Returns:
            affinity: affinity matrix
        """
        num = sims.shape[0]
        sims[sims < 0] = 0  # similarity should be non-negative
        sims = sims ** gamma
        # vec_ids: feature vectors' ids
        # mut_ids: mutual (reciprocal) nearest neighbors' ids
        # mut_sims: similarites between feature vectors and their mutual nearest neighbors
        vec_ids, mut_ids, mut_sims = [], [], []
        for i in range(num):
            # check reciprocity: i is in j's kNN and j is in i's kNN when i != j
            ismutual = np.isin(ids[ids[i]], i).any(axis=1)
            ismutual[0] = False
            if ismutual.any():
                vec_ids.append(i * np.ones(ismutual.sum(), dtype=int))
                mut_ids.append(ids[i, ismutual])
                mut_sims.append(sims[i, ismutual])
        if len(vec_ids) == 0:
            print("[Warning] get_affinity: no mutual neighbors found, returning identity matrix.")
            return sparse.eye(num, dtype=np.float32)
        vec_ids, mut_ids, mut_sims = map(np.concatenate, [vec_ids, mut_ids, mut_sims])
        affinity = sparse.csc_matrix((mut_sims, (vec_ids, mut_ids)),
                                     shape=(num, num), dtype=np.float32)
        return affinity

    def _estimate_batch_size(self, n_trunc, max_mem_gb):
        bytes_per_sample = n_trunc * n_trunc * 4
        max_bytes = max_mem_gb * (1024 ** 3)
        return max(1, max_bytes // bytes_per_sample)

    def get_offline_results(self, n_trunc, kd=50, max_mem_gb=13, use_gpu=True):
        """GPU/CPU_mix version,offline diffusion compute"""
        global trunc_ids, trunc_init, lap_alpha
        if self.use_ann:
            _, trunc_ids = self.ann.search(self.features, n_trunc)
            sims, ids = self.knn.search(self.features, kd)
            lap_alpha = self.get_laplacian(sims, ids)
        else:
            sims, ids = self.knn.search(self.features, n_trunc)
            trunc_ids = ids
            lap_alpha = self.get_laplacian(sims[:, :kd], ids[:, :kd])

        # 2)small CPU big GPU
        if self.N < 20:
            print(f"[info] Small dataset (N={self.N}), using CPU for better efficiency")
            use_gpu = False
        elif use_gpu and torch.cuda.is_available():
            print(f"[info] Large dataset (N={self.N}), using GPU mode")
        else:
            print(f"[info] Using CPU mode (GPU unavailable or disabled)")
            use_gpu = False

        if use_gpu:
            trunc_init = np.zeros(n_trunc, dtype=np.float32)  
        else:
            trunc_init = np.zeros(n_trunc)
        trunc_init[0] = 1

        N = self.N
        if use_gpu:
            return self._get_offline_results_gpu(n_trunc, max_mem_gb)
        else:
            return self._get_offline_results_cpu(n_trunc)
    
    def _get_offline_results_gpu(self, n_trunc, max_mem_gb):

        batch_size = self._estimate_batch_size(n_trunc, max_mem_gb) 
        dtype = torch.float32  
        lap_batch_torch = torch.empty((batch_size, n_trunc, n_trunc),
                                    device='cuda', dtype=dtype)
        rhs_batch_torch = torch.empty((batch_size, n_trunc),
                                    device='cuda', dtype=dtype)
        all_scores = []
        N = self.N

        for start in tqdm(range(0, N, batch_size), desc="[offline] GPU batch CG"):
            end = min(start + batch_size, N)
            current_batch_size = end - start
            idx_batch = trunc_ids[start:end]

            valid_samples = []
            for i, idx_row in enumerate(idx_batch):
                if len(idx_row) == 0:
                    lap_batch_torch[i].zero_()
                    rhs_batch_torch[i].zero_()
                    valid_samples.append(False)
                    print(f"[warn] Empty index at sample {start+i}, will use zeros")
                else:
                    try:
                        lap_sub = lap_alpha[idx_row][:, idx_row].toarray()
                        if np.any(np.isnan(lap_sub)) or np.any(np.isinf(lap_sub)):
                            print(f"[warn] NaN or Inf in Laplacian matrix at sample {start+i}, using zeros")
                            lap_batch_torch[i].zero_()
                            rhs_batch_torch[i].zero_()
                            valid_samples.append(False)
                            continue
                        
                        if np.any(np.diag(lap_sub) == 0):
                            print(f"[warn] Singular matrix detected at sample {start+i}, using zeros")
                            lap_batch_torch[i].zero_()
                            rhs_batch_torch[i].zero_()
                            valid_samples.append(False)
                            continue
                        
                        lap_batch_torch[i].copy_(torch.tensor(lap_sub, device='cuda', dtype=dtype))
                        rhs_batch_torch[i].copy_(torch.tensor(trunc_init, device='cuda', dtype=dtype))
                        valid_samples.append(True)
                        
                    except Exception as e:
                        print(f"[error] Failed to build matrix for sample {start+i}: {e}")
                        lap_batch_torch[i].zero_()
                        rhs_batch_torch[i].zero_()
                        valid_samples.append(False)

            x_batch = self._gpu_cg_batch(lap_batch_torch[:current_batch_size],
                                    rhs_batch_torch[:current_batch_size],
                                    tol=1e-6, maxiter=20)

           
            x_batch_np = x_batch.cpu().numpy()
            for i, is_valid in enumerate(valid_samples):
                if not is_valid:
                    x_batch_np[i] = 0.0
                else:
                    if np.any(np.isnan(x_batch_np[i])) or np.any(np.isinf(x_batch_np[i])):
                        print(f"[warn] NaN or Inf in CG result at sample {start+i}, using zeros")
                        x_batch_np[i] = 0.0

            all_scores.append(x_batch_np)

        all_scores = np.concatenate(all_scores, axis=0)
        
        nan_count = np.sum(np.isnan(all_scores))
        inf_count = np.sum(np.isinf(all_scores))
        if nan_count > 0 or inf_count > 0:
            print(f"[warn] Final check: {nan_count} NaN and {inf_count} Inf values detected, setting to 0")
            all_scores[np.isnan(all_scores) | np.isinf(all_scores)] = 0.0
        
        rows = np.repeat(np.arange(N), n_trunc)
        
        trunc_ids_flat = trunc_ids.reshape(-1)
        valid_idx_mask = (trunc_ids_flat >= 0) & (trunc_ids_flat < N)
        if not np.all(valid_idx_mask):
            print(f"[warn] Invalid indices detected in trunc_ids, fixing...")
            trunc_ids_flat = np.clip(trunc_ids_flat, 0, N-1)
        
        offline = sparse.csr_matrix((all_scores.reshape(-1),
                                    (rows, trunc_ids_flat)),
                                    shape=(N, N), dtype=np.float32)
        return offline
    # no error  handling
    def _get_offline_results_gpu_before(self, n_trunc, max_mem_gb):
        batch_size = self._estimate_batch_size(n_trunc, max_mem_gb)
        
        dtype = torch.float32  
        
        lap_batch_torch = torch.empty((batch_size, n_trunc, n_trunc),
                                    device='cuda', dtype=dtype)
        rhs_batch_torch = torch.empty((batch_size, n_trunc),
                                    device='cuda', dtype=dtype)

        all_scores = []
        N = self.N

        for start in tqdm(range(0, N, batch_size), desc="[offline] GPU batch CG"):
            end = min(start + batch_size, N)
            current_batch_size = end - start
            idx_batch = trunc_ids[start:end]

            for i, idx_row in enumerate(idx_batch):
                if len(idx_row) == 0:
                    lap_batch_torch[i].zero_()
                    rhs_batch_torch[i].zero_()
                else:
                    try:
                        lap_sub = lap_alpha[idx_row][:, idx_row].toarray()
                        lap_batch_torch[i].copy_(torch.tensor(lap_sub, device='cuda', dtype=dtype))
                        rhs_batch_torch[i].copy_(torch.tensor(trunc_init, device='cuda', dtype=dtype))
                    except Exception as e:
                        print(f"[error] Failed to build matrix for sample {start+i}: {e}")
                        lap_batch_torch[i].zero_()
                        rhs_batch_torch[i].zero_()

            x_batch = self._gpu_cg_batch(lap_batch_torch[:current_batch_size],
                                    rhs_batch_torch[:current_batch_size],
                                    tol=1e-6, maxiter=20)

            all_scores.append(x_batch.cpu().numpy())
        all_scores = np.concatenate(all_scores, axis=0)
        rows = np.repeat(np.arange(N), n_trunc)
        
        offline = sparse.csr_matrix((all_scores.reshape(-1),
                                    (rows, trunc_ids.reshape(-1))),
                                    shape=(N, N), dtype=np.float32)
        return offline
    
    def _get_offline_results_cpu(self, n_trunc):
        def get_offline_result_enhanced(i):
            try:
                ids = trunc_ids[i]
                if len(ids) == 0:
                    print(f"[warn] Empty index at sample {i}, using zeros")
                    return np.zeros_like(trunc_init)
                    
                trunc_lap = lap_alpha[ids][:, ids]
                
                if trunc_lap.shape[0] == 0:
                    print(f"[warn] Empty matrix at sample {i}, using zeros") 
                    return np.zeros_like(trunc_init)
                    
                scores, info = linalg.cg(trunc_lap, trunc_init, tol=1e-6, maxiter=20)
                  
            except Exception as e:
                print(f"[error] CG failed at index {i}, using zeros. Reason: {e}")
                return np.zeros_like(trunc_init)

            if np.any(np.isnan(scores)) or np.any(np.isinf(scores)):
                print(f"[warn] NaN or Inf scores detected at index {i}, using zeros")
                return np.zeros_like(trunc_init)

            return scores

        results = Parallel(n_jobs=-1, prefer='threads')(
            delayed(get_offline_result_enhanced)(i) 
            for i in tqdm(range(self.N), desc='[offline] CPU diffusion')
        )
        
        all_scores = np.concatenate(results)
        rows = np.repeat(np.arange(self.N), n_trunc)
        
        offline = sparse.csr_matrix((all_scores, (rows, trunc_ids.reshape(-1))),
                                shape=(self.N, self.N), dtype=np.float32)
        return offline
    
    def _gpu_cg_batch(self, A_batch, b_batch, tol=1e-6, maxiter=50):
        """
        batch GPU 
        A_batch: [B, n, n] torch.float32/64
        b_batch: [B, n]    torch.float32/64
        """
        device = A_batch.device
        dtype = A_batch.dtype
        B, n, _ = A_batch.shape

        x = torch.zeros_like(b_batch, dtype=dtype, device=device)
        r = b_batch - torch.bmm(A_batch, x.unsqueeze(-1)).squeeze(-1)
        p = r.clone()
        rsold = torch.sum(r * r, dim=1)  # [B]

        converged = torch.zeros(B, dtype=torch.bool, device=device)
        
        for iteration in range(maxiter):
            active_mask = ~converged
            if not active_mask.any():
                break
            Ap = torch.bmm(A_batch, p.unsqueeze(-1)).squeeze(-1)
            
            pAp = torch.sum(p * Ap, dim=1)
            pAp = torch.where(pAp == 0, torch.ones_like(pAp), pAp)  
            alpha = rsold / pAp
            
            alpha_masked = torch.where(active_mask, alpha, torch.zeros_like(alpha))
            x = x + alpha_masked.unsqueeze(-1) * p
            r = r - alpha_masked.unsqueeze(-1) * Ap
            
            rsnew = torch.sum(r * r, dim=1)
            
            residual_norm = torch.sqrt(rsnew)
            newly_converged = (residual_norm < tol) & active_mask
            
            converged = converged | newly_converged
            
            beta = rsnew / torch.where(rsold == 0, torch.ones_like(rsold), rsold)

            still_active = active_mask & (~newly_converged)
            beta_masked = torch.where(still_active, beta, torch.zeros_like(beta))
            
            p = r + beta_masked.unsqueeze(-1) * p
            rsold = torch.where(still_active, rsnew, rsold)

        nan_inf_mask = torch.isnan(x) | torch.isinf(x)
        if nan_inf_mask.any():
            print(f"[warn] NaN or Inf detected in {nan_inf_mask.sum().item()} elements, setting to 0")
            x[nan_inf_mask] = 0.0

        return x

    def _gpu_cg_batch_fast(self, A_batch, b_batch, tol=1e-6, maxiter=50):
     
        device = A_batch.device
        dtype = A_batch.dtype
        B, n, _ = A_batch.shape
        
        x = torch.zeros_like(b_batch, dtype=dtype, device=device)
        r = b_batch - torch.bmm(A_batch, x.unsqueeze(-1)).squeeze(-1)
        p = r.clone()
        rsold = torch.sum(r * r, dim=1)
        
        converged = torch.zeros(B, dtype=torch.bool, device=device)
        check_interval = 3  
        
        for iteration in range(maxiter):
            Ap = torch.bmm(A_batch, p.unsqueeze(-1)).squeeze(-1)
            alpha = rsold / torch.sum(p * Ap, dim=1)
            x = x + alpha.unsqueeze(-1) * p
            r = r - alpha.unsqueeze(-1) * Ap
            rsnew = torch.sum(r * r, dim=1)
            
            if iteration % check_interval == 0 or iteration == maxiter - 1:
                residual_norm = torch.sqrt(rsnew)
                converged = converged | (residual_norm < tol)
                
                if converged.sum() >= B * 0.95:  
                    break
            
            beta = rsnew / rsold
            p = r + beta.unsqueeze(-1) * p
            rsold = rsnew
        
        nan_inf_mask = torch.isnan(x) | torch.isinf(x)
        if nan_inf_mask.any():
            print(f"[warn] NaN or Inf detected, setting to 0")
            x[nan_inf_mask] = 0.0
            
        return x
    
    