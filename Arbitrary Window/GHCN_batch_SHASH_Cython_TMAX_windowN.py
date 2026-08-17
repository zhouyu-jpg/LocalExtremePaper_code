"""Run TMAX SHASH processing for a configurable odd local window.

Runtime windows from 7 through 31 days are supported.
"""

import os, glob, time, argparse
import numpy as np
import pandas as pd
from tqdm import tqdm


MIN_WINDOW_SIZE = 7
MAX_WINDOW_SIZE = 31


def valid_window_size(value):
    """Return a supported odd local-window size for argparse."""
    try:
        window_size = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("window size must be an integer")

    if not MIN_WINDOW_SIZE <= window_size <= MAX_WINDOW_SIZE:
        raise argparse.ArgumentTypeError(
            f"window size must be between {MIN_WINDOW_SIZE} and {MAX_WINDOW_SIZE}"
        )
    if window_size % 2 == 0:
        raise argparse.ArgumentTypeError("window size must be odd")
    return window_size


def default_parameter_file(window_size, city_id):
    """Return the conventional parameter-file path for a window and station."""
    parameter_dir = f"SAS_4Param_{window_size}DayWindow_Results_TMAX"
    parameter_name = f"sas_4p_{window_size}daywindow_detail_{city_id}.csv"
    return os.path.join(parameter_dir, parameter_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GHCN TMAX SHASH-Cython processing with a configurable local window"
    )
    parser.add_argument('--cityID', type=str, required=True)
    parser.add_argument('--start', type=int, required=True)
    parser.add_argument('--end', type=int, required=True)
    parser.add_argument(
        '--window', '--window-size',
        dest='window_size',
        type=valid_window_size,
        default=7,
        metavar='ODD_INT',
        help='odd local-window size from 7 through 31 (default: 7)',
    )
    parser.add_argument(
        '--param-file',
        default=None,
        help=(
            'parameter CSV fitted with the selected window; when omitted, use '
            'SAS_4Param_<N>DayWindow_Results_TMAX/'
            'sas_4p_<N>daywindow_detail_<cityID>.csv'
        ),
    )
    parser.add_argument(
        '--output-dir',
        default=None,
        help=(
            'density output directory; when omitted, use '
            'Results_SHASH_Cython_TMAX_<N>DayWindow_<cityID>'
        ),
    )
    args = parser.parse_args()

    if args.start < 0:
        parser.error('--start must be nonnegative')
    if args.end <= args.start:
        parser.error('--end must be greater than --start')

    try:
        import ghcnTMAX_shash_cy_windowN
    except ImportError as exc:
        parser.error(
            'compiled extension ghcnTMAX_shash_cy_windowN is unavailable; run '
            '`python setupTMAX_shash_windowN.py build_ext --inplace` first '
            f'({exc})'
        )

    t1 = time.time()
    cityID = args.cityID
    window_size = args.window_size
    n = window_size // 2

    # 1. Discovery
    ls_csv = sorted(glob.glob("Data/"+cityID+"*.csv"))
    if not ls_csv:
        ls_csv = sorted(glob.glob("../Data/"+cityID+"*.csv"))
    if not ls_csv:
        print(f"No data CSV found for {cityID}")
        exit()
        
    # 2. Load SAS Parameters
    param_file = args.param_file or default_parameter_file(window_size, cityID)
    if not os.path.exists(param_file):
        print(
            f"No {window_size}-day SAS parameter file found for {cityID}: "
            f"{param_file}\nFit parameters with the same local window or pass "
            "--param-file explicitly."
        )
        exit()
    sas_params_df = pd.read_csv(param_file)
    required_parameter_columns = {'Day', 'Xi', 'Lambda', 'Epsilon', 'Delta'}
    missing_parameter_columns = required_parameter_columns.difference(sas_params_df.columns)
    if missing_parameter_columns:
        missing = ', '.join(sorted(missing_parameter_columns))
        print(f"Parameter file is missing required columns: {missing}")
        exit()

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
    output_dir = args.output_dir or (
        f"Results_SHASH_Cython_TMAX_{window_size}DayWindow_{cityID}"
    )
    for i in tqdm(
        range(args.start, args.end),
        desc=f"Processing {cityID} ({window_size}-day window)",
    ):
        if i >= len(mds): break
        ghcnTMAX_shash_cy_windowN.process_day_shash(
            i,
            mds,
            n,
            t1_srs_raw,
            cityID,
            sas_params_df,
            output_dir,
        )
        
    t2 = time.time()
    print(
        f"Batch processing for {cityID} with a {window_size}-day window "
        f"(day indices {args.start}-{args.end}) completed in {t2-t1:.2f} seconds."
    )
