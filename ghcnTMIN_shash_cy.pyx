# Parallel Cython implementation for TMIN SHASH processing
# cython: language_level=3

import os
import pandas as pd
import numpy as np
cimport numpy as np
from scipy.integrate import quad
from itertools import permutations
from multiprocessing import Pool
from joblib import Parallel, delayed
from tqdm import tqdm

def sas_transform(x, float xi, float lmbda, float epsilon, float delta):
    """
    Applies the 4-parameter Sinh-Arcsinh transformation.
    """
    z = (x - xi) / lmbda
    return np.sinh(delta * np.arcsinh(z) - epsilon)

def sas_jacobian(x, float xi, float lmbda, float epsilon, float delta):
    """
    Jacobian of the 4-parameter SAS transformation: dz/dx
    """
    z = (x - xi) / lmbda
    y = delta * np.arcsinh(z) - epsilon
    return (delta * np.cosh(y)) / (lmbda * np.sqrt(z**2 + 1))

# Joint density function (Gaussian)
def joint_density(values, mean_vec, cov_matrix):
    cdef np.ndarray diff, cov_inv
    cdef float exponent, density
    try:
        values = np.array(values)
        diff = values - mean_vec
        cov_inv = np.linalg.inv(cov_matrix)
        exponent = -0.5 * diff.T @ cov_inv @ diff
        density = np.exp(exponent) / ((2 * np.pi) ** (len(values) / 2) * np.sqrt(np.linalg.det(cov_matrix)))
        return density
    except np.linalg.LinAlgError:
        return 0

# Empirical covariance matrix calculation
def emp_cov_mat(df, target_day, n):
    cdef int num_days, target_idx, start_idx, end_idx, year, midcol_idx
    cdef list valid_years_data, days
    cdef np.ndarray cov_mat, mean_vec

    days = df.columns.tolist()
    num_days = len(days)
    
    try:
        target_idx = days.index(target_day)
    except ValueError:
        raise ValueError(f"Target day {target_day} not found.")
    
    start_idx = target_idx - n
    end_idx = target_idx + n + 1
    
    valid_years_data = []
    for year in df.index:
        year_data = df.loc[year]
        selected_days = []
        
        if start_idx < 0:
            prev_year = year - 1
            if prev_year in df.index:
                selected_days.extend(df.loc[prev_year, days[start_idx:]])
            else:
                continue

        selected_days.extend(year_data[days[max(start_idx, 0):min(end_idx, num_days)]])
        
        if end_idx > num_days:
            next_year = year + 1
            if next_year in df.index:
                selected_days.extend(df.loc[next_year, days[:end_idx - num_days]])
            else:
                continue
        
        valid_years_data.append(selected_days)
    
    data_subset = pd.DataFrame(valid_years_data).dropna()
    midcol_idx = len(data_subset.columns) // 2
    cols = data_subset.columns.tolist()
    col_reorder = [cols[midcol_idx]] + [col for col in cols if col != cols[midcol_idx]]
    data_subset = data_subset[col_reorder]
    
    cov_mat = data_subset.cov().to_numpy()
    mean_vec = data_subset.mean().to_numpy()
    
    return cov_mat, mean_vec

# Recursive integration
def nested_integral_recursive(u, perm, mean_vec, cov_matrix, is_denominator=False):
    cdef int n_vars = len(perm) + (2 if is_denominator else 1)
    
    def integrate_recursively(level, upper_bounds):
        if level == n_vars - 1:
            full_values = upper_bounds if is_denominator else [u] + upper_bounds
            return joint_density(full_values, mean_vec, cov_matrix)

        def inner_integrand(x):
            return integrate_recursively(level + 1, upper_bounds + [x])
        
        if is_denominator:
            current_variable_index = perm[level - 1] if level > 0 else 0
        else:
            current_variable_index = perm[level]
        
        mean = mean_vec[current_variable_index]
        std = np.sqrt(cov_matrix[current_variable_index, current_variable_index])
        
        lower_bound = mean - 4 * std
        if level == 0 and not is_denominator:
            upper_bound = u
        else:
            upper_bound = upper_bounds[-1] if level > 0 else mean + 4 * std

        try:
            result, _ = quad(inner_integrand, lower_bound, upper_bound, epsabs=1e-2, epsrel=1e-2)
            return result
        except:
            return 0

    return integrate_recursively(0, [])

def integrate_for_perm(perm, mean_vec, cov_mat):
    return nested_integral_recursive(None, perm, mean_vec=mean_vec, cov_matrix=cov_mat, is_denominator=True)

def h_denom(int n, str target_day, df_z):
    cdef np.ndarray cov_mat, mean_vec
    cdef list perms, results
    cdef float denominator

    cov_mat, mean_vec = emp_cov_mat(df_z, target_day, n)
    perms = list(permutations(range(1, 2 * n + 1)))
    
    with Pool() as pool:
        results = pool.starmap(integrate_for_perm, [(perm, mean_vec, cov_mat) for perm in tqdm(perms, desc="Denom Perms", leave=False)])
    
    denominator = sum(results)
    return denominator if denominator != 0 else np.nan
    
def compute_numerator(z, perm, mean_vec, cov_mat):
    return nested_integral_recursive(z, perm, mean_vec=mean_vec, cov_matrix=cov_mat, is_denominator=False)

def h_t_z(float z, int n, str target_day, df_z, float denominator):
    cdef np.ndarray cov_mat, mean_vec
    cdef list perms
    cdef float numerator

    cov_mat, mean_vec = emp_cov_mat(df_z, target_day, n)
    perms = list(permutations(range(1, 2 * n + 1)))
    
    numerator = sum([compute_numerator(z, perm, mean_vec, cov_mat) for perm in perms])
    return numerator / denominator

def process_day_shash_tmin(int i, mds, int n, t1_srs_raw, str cityID, sas_params_df):
    """
    Complete processing for TMIN including 4-parameter SAS transformation and Jacobian.
    Uses negated scale for peak modeling, and flips back for output.
    """
    cdef str target_day
    cdef float xi, lmbda, epsilon, delta, res_h_denom
    cdef np.ndarray val_u_celsius, val_z
    cdef list res_h_z, res_h_celsius

    target_day = mds[i]
    
    # Load parameters for this day
    p = sas_params_df[sas_params_df['Day'] == target_day].iloc[0]
    xi = p['Xi']
    lmbda = p['Lambda']
    epsilon = p['Epsilon']
    delta = p['Delta']

    # Define grid in Celsius (negated raw scale as in batch script)
    mean_raw = t1_srs_raw[target_day].mean()
    std_raw = t1_srs_raw[target_day].std()
    val_u_celsius = np.round(np.arange(mean_raw - 4*std_raw, mean_raw + 4*std_raw + 0.1, 0.1), decimals=2)
    
    # Transform grid and reference data to Z-space (Gaussian)
    val_z = sas_transform(val_u_celsius, xi, lmbda, epsilon, delta)
    
    # For the covariance matrix, we transform the 5-day window into Z-space
    window_indices = [(i + offset) % len(mds) for offset in range(-n, n+1)]
    window_mds = [mds[idx] for idx in window_indices]
    df_window_raw = t1_srs_raw[window_mds]
    df_z = sas_transform(df_window_raw, xi, lmbda, epsilon, delta)
    
    # Denominator in Z-space
    res_h_denom = h_denom(n, target_day, df_z)
    
    # Densities in Z-space using 16 parallel jobs
    res_h_z = Parallel(n_jobs=16)(
        delayed(h_t_z)(z, n, target_day, df_z, res_h_denom) 
        for z in tqdm(val_z, desc=f"Grid {target_day}", leave=False)
    )
    
    # Apply Jacobian: h_X(u) = h_Z(z) * |dz/du|
    res_h_celsius = []
    for u, hz in zip(val_u_celsius, res_h_z):
        res_h_celsius.append(hz * sas_jacobian(u, xi, lmbda, epsilon, delta))
    
    # Flip back the temperature scale for output (-val_u_celsius)
    dict_theo = {'Temp': -val_u_celsius[::-1], 'Density': res_h_celsius}
    df_theo = pd.DataFrame(dict_theo)
    
    dirs = os.path.join('GHCN-Res/TMIN_SHASH', cityID)
    os.makedirs(dirs, exist_ok=True)
    df_theo.to_csv(os.path.join(dirs, f"{cityID}_{target_day}.csv"), index=False)
    print(f"TMIN Day {target_day} processed.")
