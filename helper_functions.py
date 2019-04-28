"""
This script contains functions that will be used in different scripts
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import nibabel as nib
import glob
import os
sns.set(style="whitegrid")

def plot_age_distribution(df, save_path):
    plt.figure()
    min_age = df['age'].min()
    max_age = df['age'].max()
    plt.hist(df['age'], bins=int(max_age-min_age), range=(min_age, max_age))
    plt.xlabel('Age')
    plt.ylabel('# of Subjects')
    plt.savefig(save_path)
    plt.close()


def read_freesurfer_example(data_dir, demographic_path):
    """
    Read volumetric data and demographic data. Note: demographic file is included here to make sure that the
    x data and the demographic data follows the same subject order.
    TODO: Improve implementation

    Args:
        data_dir: String with path to directory with csv files.
        demographic_path: String with path to demographic data.

    Returns:
        x: Numpy array with neuroimaging data.
        demographic_df: Pandas DataFrame with all demographic data.
    """
    aseg_name = 'aseg_stats_vol.csv'
    aparc_lh_name = 'lh_aparc_stats_vol.csv'
    aparc_rh_name = 'rh_aparc_stats_vol.csv'

    freesurfer_dir =  Path(data_dir)
    aseg_df = pd.read_csv(freesurfer_dir / aseg_name)
    aparc_lh_df = pd.read_csv(freesurfer_dir / aparc_lh_name)
    aparc_rh_df = pd.read_csv(freesurfer_dir / aparc_rh_name)

    combined = pd.merge(aseg_df, aparc_lh_df, left_on='Measure:volume', right_on='lh.aparc.volume')
    combined = pd.merge(combined, aparc_rh_df, left_on='Measure:volume', right_on='rh.aparc.volume')
    combined = combined.drop(['lh.aparc.volume', 'rh.aparc.volume'], axis=1)
    combined['Measure:volume'] = combined['Measure:volume'].str.replace('_raw/', '')

    demographic_df = pd.read_csv(demographic_path)
    merged_df = demographic_df.merge(combined, left_on=['subject_ID'], right_on=['Measure:volume'], how='right')

    demographic_df = pd.DataFrame(merged_df[demographic_df.columns])
    x_df = pd.DataFrame(merged_df[combined.columns])
    x_df = x_df.drop(columns=['Measure:volume'])
    x = x_df.values.astype('float32')

    return x, demographic_df


def convert_nifty_to_numpy(data_dir, dst):
    """
    Reads Nifty files and saves them into a numpy extension in order to optimize loading times

    Args:
        data_dir: String with path to directory with Nifty files.
        dst: String with path to directory where numpy files will be stored
    """
    if not os.path.exists(data_dir):
        return print("Path to directory does not exist.")
    if not os.path.exists(dst):
        os.makedirs(dst)

    modules = glob.glob(data_dir+ '/*.nii.gz')
    count = len(modules)

    for i, name in enumerate(modules):
        try:
            img = nib.load(name)
        except:
            print("File "+name+ " doesn't seem to exist")
            continue
        img = np.array(img.dataobj)
        np.save(dst +'/'+ os.path.basename(name), img)
        print('Saved file '+ str(i) + ' out of ' + str(count))

    print("All files created.")


