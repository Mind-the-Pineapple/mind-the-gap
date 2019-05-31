from pathlib import Path

import pandas as pd

import numpy as np

PROJECT_ROOT = Path.cwd()
spm_path = PROJECT_ROOT / 'output' /'end_game'/ 'SPM+gm+wm_wsite.csv'

spm_predictions_df = pd.read_csv(spm_path, index_col='subject_ID')
brainAGE = (spm_predictions_df['predictions'] - spm_predictions_df['age']).values

from sklearn import linear_model

reg = linear_model.LinearRegression()
reg.fit(spm_predictions_df['age'].values[:, np.newaxis], brainAGE[:, np.newaxis])


spm_path_testing = PROJECT_ROOT / 'output' /'end_game'/ 'SPM_wm+gm_testing.csv'

spm_predictions_df_testing = pd.read_csv(spm_path_testing, index_col='subject_ID')

diff = reg.predict(spm_predictions_df_testing['age'].values[:,np.newaxis])

holy_values = spm_predictions_df_testing['age'].values[:, np.newaxis] - diff

df = pd.DataFrame(data=holy_values, index=spm_predictions_df_testing.index, columns=['age'])
df.to_csv('minthegap_objective2_backup.csv')






