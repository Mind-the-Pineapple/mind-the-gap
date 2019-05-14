"""Exploratory data analysis on the demographic data."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set(style="whitegrid")

PROJECT_ROOT = Path.cwd()
freesurfer_dir = PROJECT_ROOT / 'data' / 'freesurfer'
demographic_path = PROJECT_ROOT / 'data' / 'PAC2019_BrainAge_Training.csv'


aseg_df = pd.read_csv(freesurfer_dir / 'aseg_vol.txt', sep='\t', index_col='Measure:volume')
lh_aparc_vol_df = pd.read_csv(freesurfer_dir / 'lh_aparc_vol.txt', sep='\t', index_col='lh.aparc.volume')
rh_aparc_vol_df = pd.read_csv(freesurfer_dir / 'rh_aparc_vol.txt', sep='\t', index_col='rh.aparc.volume')
lh_aparc_thick_df = pd.read_csv(freesurfer_dir / 'lh_aparc_thick.txt', sep='\t', index_col='lh.aparc.thickness')
rh_aparc_thick_df = pd.read_csv(freesurfer_dir / 'rh_aparc_thick.txt', sep='\t', index_col='rh.aparc.thickness')
lh_aparc_area_df = pd.read_csv(freesurfer_dir / 'lh_aparc_area.txt', sep='\t', index_col='lh.aparc.area')
rh_aparc_area_df = pd.read_csv(freesurfer_dir / 'rh_aparc_area.txt', sep='\t', index_col='rh.aparc.area')
lh_aparc_curv_df = pd.read_csv(freesurfer_dir / 'lh_aparc_curv.txt', sep='\t', index_col='lh.aparc.meancurv')
rh_aparc_curv_df = pd.read_csv(freesurfer_dir / 'rh_aparc_curv.txt', sep='\t', index_col='rh.aparc.meancurv')

merged = pd.merge(aseg_df, lh_aparc_vol_df, left_index=True, right_index=True)
merged = pd.merge(merged, rh_aparc_vol_df, left_index=True, right_index=True)
merged = pd.merge(merged, lh_aparc_thick_df, left_index=True, right_index=True)
merged = pd.merge(merged, rh_aparc_thick_df, left_index=True, right_index=True)
merged = pd.merge(merged, lh_aparc_area_df, left_index=True, right_index=True)
merged = pd.merge(merged, rh_aparc_area_df, left_index=True, right_index=True)
merged = pd.merge(merged, lh_aparc_curv_df, left_index=True, right_index=True)
merged = pd.merge(merged, rh_aparc_curv_df, left_index=True, right_index=True)

merged.index = merged.index.str.replace('_raw/', '')

demographic_df = pd.read_csv(demographic_path, index_col='subject_ID')
merged_df = demographic_df.merge(merged, left_index=True, right_index=True, how='right')

for i in merged.columns:
    print(i)

for column_name in merged_df.columns:
    print(merged_df[column_name].describe())

corr = merged_df.corr()

sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)
plt.show()

print(corr['age'].nlargest(50))





