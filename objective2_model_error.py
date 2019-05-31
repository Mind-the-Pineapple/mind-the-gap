from pathlib import Path

import pandas as pd

import numpy as np

PROJECT_ROOT = Path.cwd()
spm_path = PROJECT_ROOT / 'output' /'end_game'/ 'SPM+gm+wm_wsite.csv'
tpot_path = PROJECT_ROOT / 'output' /'end_game'/ 'tpot.csv'
spm_site4_path = PROJECT_ROOT / 'output' /'end_game'/ 'SPM_wm+gm_SVM_1_per_site_4_testing.csv'

spm_predictions_df = pd.read_csv(spm_path, index_col='subject_ID')
tpot_predictions_df = pd.read_csv(tpot_path, index_col='subject_ID')

tpot_mae = 5.195
spm_mae = 4.54496376494806
spm_site4_mae = 1.662

weight_tpot =  (7.0 - tpot_mae) ** 2
weight_spm =  (7.0 - spm_mae) ** 2
weight_spm_site4 =  (7.0 - spm_site4_mae) ** 2
total_weight = weight_spm + weight_tpot + weight_spm_site4

tpot_predictions_df = tpot_predictions_df.reindex(list(spm_predictions_df.index))
tpot_predictions_df['pred_tpot'] = tpot_predictions_df['predictions']
tpot_predictions_df = tpot_predictions_df.drop(['predictions', 'age'], axis=1)
A = pd.concat([spm_predictions_df, tpot_predictions_df], axis=1)

A['predictions'] = A['predictions'] * weight_spm
A['pred_tpot'] = A['pred_tpot'] * weight_tpot

media = A[['predictions','pred_tpot']].sum(axis=1)
media = media/total_weight

brainAGE = (media - spm_predictions_df['age']).values

from sklearn import linear_model

reg = linear_model.LinearRegression()
reg.fit(spm_predictions_df['age'].values[:, np.newaxis], brainAGE[:, np.newaxis])


spm_path_testing = PROJECT_ROOT / 'output' /'end_game'/ 'SPM_wm+gm_testing.csv'
tpot_path_testing = PROJECT_ROOT / 'output' /'end_game'/ 'tpot_testing.csv'

spm_predictions_df_testing = pd.read_csv(spm_path_testing, index_col='subject_ID')
tpot_predictions_df_testing = pd.read_csv(tpot_path_testing, index_col='subject_ID')
spm_site4_predictions_df_testing = pd.read_csv(spm_site4_path, index_col='subject_ID')

tpot_predictions_df_testing = tpot_predictions_df_testing.reindex(list(spm_predictions_df_testing.index))
spm_site4_predictions_df_testing = spm_site4_predictions_df_testing.reindex(list(spm_predictions_df_testing.index))

tpot_predictions_df_testing['pred_tpot'] = tpot_predictions_df_testing['age']
tpot_predictions_df_testing = tpot_predictions_df_testing.drop(['age'], axis=1)

spm_site4_predictions_df_testing['pred_site4'] = spm_site4_predictions_df_testing['age']
spm_site4_predictions_df_testing = spm_site4_predictions_df_testing.drop(['age'], axis=1)

B = pd.concat([spm_predictions_df_testing, tpot_predictions_df_testing, spm_site4_predictions_df_testing], axis=1)


B['age'] = B['age'] * weight_spm
B['pred_tpot'] = B['pred_tpot'] * weight_tpot
B['pred_site4'] = B['pred_site4'] * weight_tpot

media_testing = B[['age','pred_tpot', 'pred_site4']].sum(axis=1)
media_testing = media_testing/total_weight

diff = reg.predict(media_testing.values[:,np.newaxis])

holy_values = media_testing.values[:, np.newaxis] - diff

df = pd.DataFrame(data=holy_values, index=spm_predictions_df_testing.index, columns=['age'])
df.to_csv('minthegap_objective2.csv')






