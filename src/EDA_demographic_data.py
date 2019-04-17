"""Exploratory data analysis on the demographic data."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path('/home/walter/Desktop/mind-the-gap')

demographic_filename = PROJECT_ROOT / 'data' / 'PAC2019_BrainAge_Training.csv'
output_dir = PROJECT_ROOT / 'output' / 'EDA'

demographic_df = pd.read_csv(demographic_filename)
a4_dims = (11.7, 8.27)
fig, ax = plt.subplots(figsize=a4_dims)
sns.set(style="whitegrid")
sns.violinplot(y='site', x='age', data=demographic_df, hue='gender', split=True, scale='count', orient='h', ax=ax)
plt.savefig(output_dir / 'age_ranges.png')
# plt.show()
