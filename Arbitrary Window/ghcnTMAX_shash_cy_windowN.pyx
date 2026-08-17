"""Configurable-window TMAX SHASH density calculations.

Runtime windows from 7 through 31 days are supported.
"""

import numpy as np
cimport numpy as np
import pandas as pd

try:
    # Available in current SciPy releases.
    from scipy.stats import _mvn as mvn_backend
except ImportError:
    # Compatibility with older SciPy releases.
    from scipy.stats import mvn as mvn_backend

from libc.math cimport pi, exp, sqrt


MIN_WINDOW_SIZE = 7
MAX_WINDOW_SIZE = 31


def regularize_covariance(cov):
    """Return a finite positive-definite covariance matrix.

    Windows near 31 days can have at least as many variables as reference
    years. A small eigenvalue floor keeps those covariance matrices usable
    without materially changing well-conditioned cases.
    """
    cov = np.asarray(cov, dtype=np.float64)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("Covariance matrix must be square")
    if not np.all(np.isfinite(cov)):
        raise ValueError("Covariance matrix contains non-finite values")

    cov = (cov + cov.T) / 2.0
    diagonal = np.diag(cov)
    scale = float(np.mean(diagonal))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Covariance matrix must have positive variance")

    eigenvalue_floor = max(scale * 1e-8, 1e-12)
    minimum_eigenvalue = float(np.linalg.eigvalsh(cov)[0])
    if minimum_eigenvalue < eigenvalue_floor:
        cov = cov + np.eye(cov.shape[0]) * (
            eigenvalue_floor - minimum_eigenvalue
        )
    return cov


# Multivariate-normal CDF helper using SciPy's MVNDST backend.
def mvn_cdf(np.ndarray[np.float64_t, ndim=1] upper, 
            np.ndarray[np.float64_t, ndim=1] mean, 
            np.ndarray[np.float64_t, ndim=2] cov):
    cdef int n = len(mean)
    cdef int maxpts = 10000 * n
    cdef np.ndarray[np.float64_t, ndim=1] std
    cdef np.ndarray[np.float64_t, ndim=1] upper_std
    cdef np.ndarray[np.float64_t, ndim=2] corr
    cdef np.ndarray[np.float64_t, ndim=1] lower_tri
    cdef np.ndarray[np.int32_t, ndim=1] infin

    if maxpts < 100000:
        maxpts = 100000

    if cov.shape[0] != n or cov.shape[1] != n or len(upper) != n:
        raise ValueError("MVN mean, covariance, and upper bound dimensions do not match")

    cov = regularize_covariance(cov)
    std = np.sqrt(np.diag(cov))
    if np.any(~np.isfinite(std)) or np.any(std <= 0):
        raise ValueError("MVN covariance must have finite, positive diagonal entries")

    upper_std = (upper - mean) / std
    corr = cov / np.outer(std, std)
    corr = (corr + corr.T) / 2.0
    np.fill_diagonal(corr, 1.0)
    
    # mvndst expects lower triangle of correlation matrix (excluding diagonal)
    lower_tri = corr[np.tril_indices(n, -1)]
    
    # infin=0 means each integration interval is (-infinity, upper].
    infin = np.zeros(n, dtype=np.int32)
    
    error, result, inform = mvn_backend.mvndst(
        np.zeros(n),
        upper_std,
        infin,
        lower_tri,
        maxpts,
        1e-5,
        1e-5,
    )
    if not np.isfinite(result):
        raise RuntimeError("MVN CDF calculation returned a non-finite result")
    return min(1.0, max(0.0, result))

def sas_transform(x, float xi, float lmbda, float epsilon, float delta):
    """
    SAS transformation: z = sinh(delta * asinh((x - xi) / lmbda) - epsilon)
    """
    return np.sinh(delta * np.arcsinh((x - xi) / lmbda) - epsilon)

def emp_cov_mat(df_z, str target_day, int n):
    """
    Compute the covariance matrix and mean vector for the selected local window.
    """
    expected_columns = 2 * n + 1
    if df_z.shape[1] != expected_columns:
        raise ValueError(
            f"Expected {expected_columns} local-window columns, got {df_z.shape[1]}"
        )

    # The target day is the middle column because process_day_shash constructs
    # window_mds in order from -n through +n.
    cov_mat = df_z.cov().values
    mean_vec = df_z.mean().values
    if not np.all(np.isfinite(mean_vec)):
        raise ValueError("Local-window means contain non-finite values")
    cov_mat = regularize_covariance(cov_mat)
    return cov_mat, mean_vec

def h_denom(int n, str target_day, df_z):
    """
    Compute the local-maximum probability using a linear MVN transformation.
    $P(X_1 < X_0, X_2 < X_0, ..., X_{2n} < X_0)$
    where $X_0$ is target day and $X_i$ are window days.
    """
    cdef np.ndarray[np.float64_t, ndim=2] cov_full
    cdef np.ndarray[np.float64_t, ndim=1] mean_full
    cdef int target_idx = n
    cdef int num_vars = 2 * n + 1
    cdef np.ndarray[np.float64_t, ndim=2] D
    cdef int row = 0
    cdef np.ndarray[np.float64_t, ndim=1] mean_y
    cdef np.ndarray[np.float64_t, ndim=2] cov_y

    cov_full, mean_full = emp_cov_mat(df_z, target_day, n)
    
    # Transformation matrix D: Y_i = X_i - X_target. The target day
    # is at index n in [i-n, ..., i, ..., i+n].
    D = np.zeros((2 * n, num_vars))
    for i in range(num_vars):
        if i == target_idx: continue
        D[row, i] = 1.0
        D[row, target_idx] = -1.0
        row += 1
        
    # Y = DX -> mean_y = D @ mean_x, cov_y = D @ cov_x @ D.T
    mean_y = D @ mean_full
    cov_y = D @ cov_full @ D.T
    
    # Denominator is P(Y < 0)
    return mvn_cdf(np.zeros(2 * n), mean_y, cov_y)

def h_t_z(float z, int n, str target_day, df_z, float denominator):
    """
    Compute the numerator using the conditional MVN distribution.
    $f(z) * P(X_{window} < z | X_{target} = z)$
    """
    cdef np.ndarray[np.float64_t, ndim=2] cov_full
    cdef np.ndarray[np.float64_t, ndim=1] mean_full
    cdef int target_idx = n
    cdef double mu_t
    cdef double var_t
    cdef double f_z
    cdef list window_indices
    cdef np.ndarray[np.float64_t, ndim=1] mu_w
    cdef np.ndarray[np.float64_t, ndim=2] cov_ww
    cdef np.ndarray[np.float64_t, ndim=1] cov_wt
    cdef np.ndarray[np.float64_t, ndim=1] mu_cond
    cdef np.ndarray[np.float64_t, ndim=2] cov_cond
    cdef double p_cond

    cov_full, mean_full = emp_cov_mat(df_z, target_day, n)
    
    mu_t = mean_full[target_idx]
    var_t = cov_full[target_idx, target_idx]
    
    # Marginal density f(z)
    f_z = (1.0 / sqrt(2 * pi * var_t)) * exp(-0.5 * (z - mu_t)**2 / var_t)
    
    # Conditional mean and covariance of window | target = z
    window_indices = [i for i in range(2 * n + 1) if i != target_idx]
    mu_w = mean_full[window_indices]
    cov_ww = cov_full[np.ix_(window_indices, window_indices)]
    cov_wt = cov_full[window_indices, target_idx]
    
    # mu_cond = mu_w + cov_wt * (1/var_t) * (z - mu_t)
    mu_cond = mu_w + cov_wt * (1.0 / var_t) * (z - mu_t)
    # cov_cond = cov_ww - cov_wt * (1/var_t) * cov_wt.T
    cov_cond = cov_ww - np.outer(cov_wt, cov_wt) / var_t
    
    # Numerator part: P(X_window < z | target = z)
    # The permutation sum is equivalent to the CDF at (z, z, ..., z).
    p_cond = mvn_cdf(np.full(2 * n, z), mu_cond, cov_cond)
    
    return (f_z * p_cond) / denominator

def process_day_shash(int i, mds, int n, t1_srs_raw, str cityID,
                      sas_params_df, output_dir=None):
    cdef int window_size = 2 * n + 1

    from joblib import Parallel, delayed
    from tqdm import tqdm

    if window_size < MIN_WINDOW_SIZE or window_size > MAX_WINDOW_SIZE:
        raise ValueError(
            f"Local window must be an odd number from {MIN_WINDOW_SIZE} "
            f"through {MAX_WINDOW_SIZE}; got {window_size}"
        )
    if len(mds) < window_size:
        raise ValueError(
            f"Local window ({window_size}) exceeds available calendar days ({len(mds)})"
        )
    
    target_day = mds[i]
    p = sas_params_df[sas_params_df['Day'] == target_day].iloc[0]
    xi, lmbda, epsilon, delta = p['Xi'], p['Lambda'], p['Epsilon'], p['Delta']

    mean_raw = t1_srs_raw[target_day].mean()
    std_raw = t1_srs_raw[target_day].std()
    val_u_celsius = np.round(np.arange(mean_raw - 4*std_raw, mean_raw + 4*std_raw + 0.1, 0.1), decimals=2)
    val_z = sas_transform(val_u_celsius, xi, lmbda, epsilon, delta)

    window_indices = [(i + offset) % len(mds) for offset in range(-n, n+1)]
    window_mds = [mds[idx] for idx in window_indices]
    df_window_raw = t1_srs_raw[window_mds]
    df_z = sas_transform(df_window_raw, xi, lmbda, epsilon, delta)

    # Precompute the denominator for the selected local window.
    res_h_denom = h_denom(n, target_day, df_z)
    if not np.isfinite(res_h_denom) or res_h_denom <= 0:
        raise RuntimeError(
            f"Invalid local-maximum probability for {target_day}: {res_h_denom}"
        )

    # Evaluate grid points in parallel; MVN CDFs replace factorial permutation
    # enumeration and recursive quadrature.
    res_h_z = Parallel(n_jobs=16)(
        delayed(h_t_z)(z, n, target_day, df_z, res_h_denom)
        for z in tqdm(val_z, desc=f"Grid {target_day}", leave=False)
    )

    # Jacobian for back-transformation
    # dz/dx = delta * (1 / sqrt((x-xi)^2 + lmbda^2)) * cosh(delta * asinh((x-xi)/lmbda) - epsilon)
    diff = (val_u_celsius - xi)
    deriv = delta * (1.0 / np.sqrt(diff**2 + lmbda**2)) * np.cosh(delta * np.arcsinh(diff / lmbda) - epsilon)
    res_h_celsius = np.array(res_h_z) * deriv

    # Keep different window sizes in separate directories by default.
    if output_dir is None:
        output_dir = f"Results_SHASH_Cython_TMAX_{window_size}DayWindow_{cityID}"
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    df_theo = pd.DataFrame({'Celsius': val_u_celsius, 'Density': res_h_celsius})
    df_theo.to_csv(f"{output_dir}/shash_density_{target_day}.csv", index=False)
    print(f"Day {target_day} processed with a {window_size}-day local window.")
