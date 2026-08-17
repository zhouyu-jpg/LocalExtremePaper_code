import os, glob, time, argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import ghcnTMAX_shash_cy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GHCN TMAX SHASH-Cython Batch Processing')
    parser.add_argument('--cityID', type=str, required=True)
    parser.add_argument('--start', type=int, required=True)
    parser.add_argument('--end', type=int, required=True)
    args = parser.parse_args()

    t1 = time.time()
    cityID = args.cityID

    # 1. Discovery
    ls_csv = sorted(glob.glob("Data/"+cityID+"*.csv"))
    if not ls_csv:
        print(f"No data CSV found for {cityID}")
        exit()
        
    # 2. Load SAS Parameters
    param_file = f"SAS_4Param_Window_Results_TMAX/sas_4p_window_detail_{cityID}.csv"
    if not os.path.exists(param_file):
        print(f"No SAS parameter file found for {cityID}. Please run check_sas_4param_window_TMAX.py first.")
        exit()
    sas_params_df = pd.read_csv(param_file)

    # 3. Load and Preprocess Data
    df = pd.read_csv(ls_csv[0])
    df = df.rename(columns={'year':'YEAR','month':'MONTH','day':'DAY'})
    df['YEAR'] = [x-100 if (x>=2061) & (x<=2099) else x for x in df['YEAR']] 
    df['MONTH_DAY'] = pd.to_datetime(df['DATE']).dt.strftime('%m-%d')
    df = df[df['MONTH_DAY'] != '02-29']
    df['TMAX'] = df['TMAX']/10
    
    # Reference period 1961-1990
    df_t1 = df[(df['YEAR']>=1961) & (df['YEAR']<=1990)]
    mds = sorted(df_t1['MONTH_DAY'].unique())
    
    # Pivot to Year x Day for Cython
    t1_srs_raw = df_t1.pivot(index='YEAR', columns='MONTH_DAY', values='TMAX')

    # 4. Process target days
    n = 2
    for i in tqdm(range(args.start, args.end), desc=f"Processing {cityID}"):
        if i >= len(mds): break
        ghcnTMAX_shash_cy.process_day_shash(i, mds, n, t1_srs_raw, cityID, sas_params_df)
        
    t2 = time.time()
    print(f"Batch processing for {cityID} (days {args.start}-{args.end}) completed in {t2-t1:.2f} seconds.")
