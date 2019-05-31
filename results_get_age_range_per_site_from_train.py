"""Exploratory data analysis on the demographic data."""
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path.cwd()
demographic_filename = PROJECT_ROOT / 'data' / 'PAC2019_BrainAge_Training.csv'

output_dir = PROJECT_ROOT / 'data'
output_dir.mkdir(exist_ok=True)

demographic_df = pd.read_csv(demographic_filename)

ranges_list =[]
for i_site in range(17):
    selected_site_df = demographic_df.loc[demographic_df['site']==i_site]
    ranges_list.append([i_site, selected_site_df['age'].min(), selected_site_df['age'].max()])

ranges_df = pd.DataFrame(ranges_list, columns=['site', 'min', 'max'])
ranges_df.to_csv(output_dir / 'train_age_range.csv', index=False)