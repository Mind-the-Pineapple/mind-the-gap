from pathlib import Path
import os

import argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from freesurfer_columns import thk_vol_curv as COLUMNS_NAMES
from helper_functions import (read_freesurfer_sites,
                              read_freesurfer_sites_validation)
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid")

parser = argparse.ArgumentParser()
parser.add_argument('-sites',
                     dest='sites',
                     type=int,
                     required=True,
                     choices=np.arange(-1, 17)) #-1 represents all

args = parser.parse_args()
# --------------------------------------------------------------------------
# Load the demographics and the dataset
print('Analysed Site %d' %args.sites)
random_seed = 20
PROJECT_ROOT = Path.cwd()
# Create experiment's output directory
output_dir = PROJECT_ROOT / 'output' / 'experiments'

experiment_name = 'freesurfer_tpot_test7/%d_site' %args.sites
experiment_dir = output_dir / experiment_name

# Directory to save TPOT updates
# Load the freesurfer dataset
checkpoint_dir = experiment_dir / 'checkpoint'
freesurfer_dir = PROJECT_ROOT / 'data' / 'freesurfer'
demographic_path = PROJECT_ROOT / 'data' / 'PAC2019_BrainAge_Training.csv'
#------------------------------------------------------------------------------
# split train and test
# Reading data. If necessary, create new reader in helper_functions.
x, demographic_df = read_freesurfer_sites(str(freesurfer_dir),
                                          str(demographic_path),
                                          COLUMNS_NAMES,
                                          args.sites)

# Load the predicted results for the training set
predictions_test = pd.read_csv(os.path.join(experiment_dir, 'tpot_test_%d_site_predictions.csv'
                         %args.sites), index_col=0)
# rename column
predictions_test = predictions_test.rename(columns={'0':'predictions'})

# --------------------------------------------------------------------------
predictions_df = demographic_df[['age']].copy()
# filter demographics only for the training dataset
predictions_df = predictions_df.loc[predictions_test.index]
# Concatenate the predicted and real age
predictions_df = pd.merge(predictions_test, predictions_df, on='subject_ID')


# Calculate the correlation
print(spearmanr((predictions_df['age']-predictions_df['predictions']).values,
                predictions_df['age'].values))
plt.scatter(predictions_df['age'].values, predictions_df['predictions'].values)

z = np.polyfit(predictions_df['age'].values,
               predictions_df['predictions'].values, 1)
p = np.poly1d(z)

plt.plot(predictions_df['age'].values,p(predictions_df['age'].values),"r--")
plt.plot(range(100), range(100),"b")
plt.xlabel('predicted')
plt.ylabel('age')
plt.savefig(experiment_dir / 'predicted_vs_real.png')
