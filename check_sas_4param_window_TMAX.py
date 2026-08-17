import pandas as pd
import numpy as np
from scipy.stats import shapiro
from scipy.optimize import minimize
from tqdm import tqdm
import os
import glob
import warnings

# Suppress warnings from distribution fitting
warnings.filterwarnings('ignore')

def sas_transform(x, xi, lmbda, epsilon, delta):
    """
    Applies the 4-parameter Sinh-Arcsinh transformation:
    Z = sinh(delta * arcsinh((x - xi) / lmbda) - epsilon)
    """
    # Note: Using arcsinh directly is equivalent to log(z + sqrt(z^2 + 1))
    z = (x - xi) / lmbda
    return np.sinh(delta * np.arcsinh(z) - epsilon)

def sas_log_likelihood(params, data):
    """
    Negative log-likelihood of the 4-parameter SAS transformation.
    Assuming the resulting data follows a Standard Normal N(0,1).
    Uses the standard sinh-arcsinh distribution formulation.
    """
    xi, lmbda, epsilon, delta = params
    
    # Constraints: lmbda > 0 and delta > 0
    if lmbda <= 0 or delta <= 0:
        return 1e10
    
    # Transformation steps
    z = (data - xi) / lmbda
    y = delta * np.arcsinh(z) - epsilon
    transformed = np.sinh(y)
    
    # 1. Log-density of standard normal for the transformed points
    term1 = -0.5 * np.sum(transformed**2)
    
    # 2. Jacobian log-terms:
    # Log(delta) + Log(cosh(y)) - Log(lmbda) - 0.5 * Log(z^2 + 1)
    # Plus the constant term -0.5 * Log(2*pi) which is omitted for optimization
    jacobian = np.sum(
        np.log(delta) + 
        np.log(np.cosh(y)) - 
        np.log(lmbda) - 
        0.5 * np.log(z**2 + 1)
    )
    
    # Return negative log-likelihood
    return -(term1 + jacobian)

def find_best_4sas_params(pooled_data):
    """
    Finds optimal xi, lmbda, epsilon, delta using 150 points.
    """
    # Robust initial guesses
    initial_xi = np.median(pooled_data)
    initial_lmbda = np.std(pooled_data) if np.std(pooled_data) > 0 else 1.0
    initial_epsilon = 0.0 # Symmetric
    initial_delta = 1.0   # Normal-like
    
    initial_guess = [initial_xi, initial_lmbda, initial_epsilon, initial_delta]
    
    # Define bounds to keep parameters physically realistic
    # lmbda and delta must be positive
    bounds = [
        (None, None),        # xi: any value
        (1e-3, None),        # lmbda: > 0
        (-10, 10),           # epsilon: skew range
        (0.1, 10)            # delta: kurtosis range
    ]
    
    res = minimize(sas_log_likelihood, initial_guess, args=(pooled_data,), 
                   bounds=bounds, method='L-BFGS-B')
    
    return res.x if res.success else initial_guess

def process_station_sas_4p_window(file_path, output_dir):
    station_id = os.path.basename(file_path).replace('.csv', '')
    print(f"\nProcessing Station (SAS 4-Param Window): {station_id}")
    
    try:
        df = pd.read_csv(file_path)
        df['TMAX'] = df['TMAX'] / 10.0
        df['DATE'] = pd.to_datetime(df['DATE'])
        df['MONTH_DAY'] = df['DATE'].dt.strftime('%m-%d')
        df = df[df['MONTH_DAY'] != '02-29']
        
        # 1961-1990 Reference period
        df_ref = df[(df['year'] >= 1961) & (df['year'] <= 1990)].copy()
        if df_ref.empty: return None

        mds = sorted(df_ref['MONTH_DAY'].unique())
        pivot_df = df_ref.pivot(index='year', columns='MONTH_DAY', values='TMAX')
        
        results = []
        for i, target_md in enumerate(tqdm(mds, leave=False)):
            # Get rolling 5-day window
            window_indices = [(i + offset) % len(mds) for offset in range(-2, 3)]
            window_mds = [mds[idx] for idx in window_indices]
            
            pooled_data = pivot_df[window_mds].values.flatten()
            pooled_data = pooled_data[~np.isnan(pooled_data)]
            
            if len(pooled_data) < 50:
                results.append({'Day': target_md, 'Status': 'Insufficient Data'})
                continue
                
            # Find optimal 4 parameters using 150 points
            xi, lmbda, epsilon, delta = find_best_4sas_params(pooled_data)
            
            # Apply transformation only to target day's 30 points
            day_data = pivot_df[target_md].dropna().values
            transformed_day = sas_transform(day_data, xi, lmbda, epsilon, delta)
            
            # Test normality
            _, p_val = shapiro(transformed_day)
            
            results.append({
                'Day': target_md,
                'Xi': round(xi, 4),
                'Lambda': round(lmbda, 4),
                'Epsilon': round(epsilon, 4),
                'Delta': round(delta, 4),
                'Shapiro_P': round(p_val, 4),
                'Status': "Pass" if p_val > 0.05 else "Fail"
            })

        results_df = pd.DataFrame(results)
        detail_file = os.path.join(output_dir, f'sas_4p_window_detail_{station_id}.csv')
        results_df.to_csv(detail_file, index=False)
        
        pass_count = (results_df['Status'] == 'Pass').sum()
        total_valid = (results_df['Status'] != 'Insufficient Data').sum()
        pass_rate = pass_count / total_valid if total_valid > 0 else 0
        
        return {
            'StationID': station_id,
            'Total_Days': total_valid,
            'SAS4_Pass_Count': pass_count,
            'SAS4_Pass_Rate': round(pass_rate * 100, 2)
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        return None

def main():
    # Repository layout: run from the repository root after downloading Data/.
    data_dir = 'Data'
    output_dir = 'SAS_4Param_Window_Results_TMAX'
    os.makedirs(output_dir, exist_ok=True)
    
    csv_files = glob.glob(os.path.join(data_dir, 'USW*.csv'))
    
    summary_results = []
    for file_path in tqdm(csv_files, desc="Batch SAS 4-Param"):
        res = process_station_sas_4p_window(file_path, output_dir)
        if res:
            summary_results.append(res)
            print(f"-> SAS4 Pass Rate: {res['SAS4_Pass_Rate']}%")

    summary_df = pd.DataFrame(summary_results)
    summary_df.to_csv('sas4_window_TMAX.csv', index=False)
    print("\nSAS 4-Param Batch Processing Complete.")

if __name__ == "__main__":
    main()
