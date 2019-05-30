

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path.cwd()

testing_dir = PROJECT_ROOT / 'testing'

mae_dic = {}

# Load the numpy files with the MAE
for numpy_file_path in testing_dir.glob('*.npy'):
    mae_file = abs(np.load(numpy_file_path))
    mae_dic[numpy_file_path.stem] = mae_file

# Add remaining values from TPOT
mae_dic['tpot_test_0_site_predictions'] = 5.557
mae_dic['tpot_test_1_site_predictions'] = 4.101
mae_dic['tpot_test_2_site_predictions'] = 4.721
mae_dic['tpot_test_3_site_predictions'] = 4.027
mae_dic['tpot_test_4_site_predictions'] = 2.05
mae_dic['tpot_test_5_site_predictions'] = 6.667
mae_dic['tpot_test_6_site_predictions'] = 5.94
mae_dic['tpot_test_7_site_predictions'] = 5.638
mae_dic['tpot_test_8_site_predictions'] = 3.938
mae_dic['tpot_test_9_site_predictions'] = 6.685
mae_dic['tpot_test_10_site_predictions'] = 9.21
mae_dic['tpot_test_11_site_predictions'] = 4.213
mae_dic['tpot_test_12_site_predictions'] = 4.375
mae_dic['tpot_test_13_site_predictions'] = 10.155
mae_dic['tpot_test_14_site_predictions'] = 10.949
mae_dic['tpot_test_15_site_predictions'] = 1.851
mae_dic['tpot_test_16_site_predictions'] = 2.22

# Add GPR tests
mae_dic['freesurfer_curv_GPR'] = 7.200
mae_dic['freesurfer_thk_vol_GPR'] = 6.385
mae_dic['freesurfer_thk_vol_curv_GPR'] = 6.132

mae_df = pd.DataFrame.from_dict(mae_dic, orient='index', columns=['MAE'])
mae_df = mae_df.rename(columns={'0':'MAE'})
mae_df.to_csv(testing_dir / 'all_mae.csv')
