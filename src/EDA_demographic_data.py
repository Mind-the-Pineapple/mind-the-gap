"""Exploratory data analysis on the demographic data."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
sns.set(style="whitegrid")

cwd = os.getcwd()
demographic_filename = os.path.join(cwd,'data','PAC2019_BrainAge_Training.csv')
output_dir = os.path.join(cwd, 'output', 'EDA')

demographic_df = pd.read_csv(demographic_filename)

# Plot age distribution pro site
a4_dims = (11.7, 8.27)
fig, ax = plt.subplots(figsize=a4_dims)
sns.violinplot(y='site', x='age', data=demographic_df, hue='gender', split=True, scale='count', orient='h', ax=ax)
plt.savefig(os.path.join(output_dir, 'age_ranges_site.png'))

# Plot age distribution of the entire dataset
plt.figure()
min_age = demographic_df['age'].min()
max_age = demographic_df['age'].max()
plt.hist(demographic_df['age'], bins=int(max_age-min_age), range=(min_age, max_age))
plt.xlabel('Age')
plt.ylabel('# of Subjects')
plt.savefig(os.path.join(output_dir, 'age_ranges_distribution.png'))

# plt.show()
