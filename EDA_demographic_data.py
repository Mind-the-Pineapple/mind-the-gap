"""Exploratory data analysis on the demographic data."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid", {'axes.grid' : False})
sns.set_context("talk", font_scale=1.4)

cwd = Path.cwd()
demographic_filename = cwd / 'data' / 'PAC2019_BrainAge_Training.csv'

output_dir = cwd / 'output' / 'EDA'
output_dir.mkdir(exist_ok=True)

demographic_df = pd.read_csv(demographic_filename)
demographic_df.rename(columns={'age': 'Age', 'site': 'Site'}, inplace=True)

# Print a few demographics
print(demographic_df['gender'].value_counts())
mean_age = demographic_df['Age'].mean()
std_age = demographic_df['Age'].std()
print(f'Age mean +- std: {mean_age} +- {std_age}')
min_age = demographic_df['Age'].min()
max_age = demographic_df['Age'].max()
print(f'Age min: {min_age} - max: {max_age}')
# Plot age distribution pro site
a4_dims = (11.7, 8.27)
fig, ax = plt.subplots(figsize=a4_dims)
my_pal = {"f": "#e67384ff", "m": "#346aa9ff"}

sns.violinplot(y='Site', x='Age', data=demographic_df, hue='gender', split=True,
               scale='count', orient='h', ax=ax, palette=my_pal, size=6)
plt.grid(linestyle='dotted')
plt.legend(frameon=False)
plt.savefig(str(output_dir / 'age_ranges_site.svg'))

# Plot age distribution of the entire dataset
plt.figure(figsize=a4_dims)
plt.hist(demographic_df['Age'], bins=int(max_age - min_age),
         range=(min_age, max_age))
plt.xlabel('Age')
plt.ylabel('# of Subjects')
plt.grid(linestyle='dotted')
plt.savefig(str(output_dir / 'age_ranges_distribution.svg'))
