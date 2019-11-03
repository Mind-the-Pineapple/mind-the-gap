"""Exploratory data analysis on the demographic data."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set(style="whitegrid")

PROJECT_ROOT = Path.cwd()
demographic_filename_test = PROJECT_ROOT / 'output' / 'quase_la_arredondado_versao_com_tpots.csv'
demographic_filename_train = PROJECT_ROOT / 'data' / 'PAC2019_BrainAge_Training.csv'


output_dir = PROJECT_ROOT / 'output' / 'EDA'
output_dir.mkdir(exist_ok=True)

demographic_df_train = pd.read_csv(demographic_filename_train)
demographic_df_test = pd.read_csv(demographic_filename_test)

demographic_df_test['site'] = demographic_df_test['site']+0.5

demographic_df_train = pd.concat([demographic_df_test[['subject_ID','age','gender','site']], demographic_df_train], axis=0)

# Plot age distribution pro site
a4_dims = (11.7, 8.27)
fig, ax = plt.subplots(figsize=a4_dims)
sns.violinplot(y='site', x='age', data=demographic_df_train, hue='gender', split=True, scale='count', orient='h', ax=ax)
plt.savefig(str(output_dir / 'age_ranges_site_comparing_clipped.png'))

# Plot age distribution of the entire dataset
plt.figure()
min_age = demographic_df_train['age'].min()
max_age = demographic_df_train['age'].max()
plt.hist(demographic_df_train['age'], bins=int(max_age - min_age), range=(min_age, max_age))
plt.xlabel('Age')
plt.ylabel('# of Subjects')
plt.savefig(str(output_dir / 'age_ranges_distribution_testing_clipped.png'))
