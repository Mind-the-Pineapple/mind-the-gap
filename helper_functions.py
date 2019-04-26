"""
This script contains functions that will be used in different scripts
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

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

    freesurfer_dir = Path(data_dir)
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


def read_gram_matrix(gram_matrix_path, demographic_path):
    demographic_df = pd.read_csv(demographic_path)
    gram_df = pd.read_csv(gram_matrix_path, index_col='subject_ID')

    gram_df = gram_df[list(demographic_df['subject_ID'])]

    gram_df = gram_df.reindex(list(demographic_df['subject_ID']))

    return gram_df.values, demographic_df


def create_gram_matrix_train_data(input_dir_path, output_path):
    """
    Computes the Gram matrix for the SVM method.

    Reference: http://scikit-learn.org/stable/modules/svm.html#using-the-gram-matrix
    """
    input_dir = Path(input_dir_path)
    step_size = 100

    img_paths = list(input_dir.glob('*.npy'))

    n_samples = len(img_paths)

    K = np.float64(np.zeros((n_samples, n_samples)))

    for i in range(int(np.ceil(n_samples / np.float(step_size)))):  #

        it = i + 1
        max_it = int(np.ceil(n_samples / np.float(step_size)))
        print(" outer loop iteration: %d of %d." % (it, max_it))

        # generate indices and then paths for this block
        start_ind_1 = i * step_size
        stop_ind_1 = min(start_ind_1 + step_size, n_samples)
        block_paths_1 = img_paths[start_ind_1:stop_ind_1]

        # read in the images in this block
        images_1 = []
        for k, path in enumerate(block_paths_1):
            img = np.load(str(path))
            img = np.asarray(img, dtype='float64')
            img = np.nan_to_num(img)
            img_vec = np.reshape(img, np.product(img.shape))
            images_1.append(img_vec)
            del img
        images_1 = np.array(images_1)
        for j in range(i + 1):

            it = j + 1
            max_it = i + 1

            print(" inner loop iteration: %d of %d." % (it, max_it))

            # if i = j, then sets of image data are the same - no need to load
            if i == j:

                start_ind_2 = start_ind_1
                stop_ind_2 = stop_ind_1
                images_2 = images_1

            # if i !=j, read in a different block of images
            else:
                start_ind_2 = j * step_size
                stop_ind_2 = min(start_ind_2 + step_size, n_samples)
                block_paths_2 = img_paths[start_ind_2:stop_ind_2]

                images_2 = []
                for k, path in enumerate(block_paths_2):
                    img = np.load(str(path))
                    img = np.asarray(img, dtype='float64')
                    img_vec = np.reshape(img, np.product(img.shape))
                    images_2.append(img_vec)
                    del img
                images_2 = np.array(images_2)

            block_K = np.dot(images_1, np.transpose(images_2))
            K[start_ind_1:stop_ind_1, start_ind_2:stop_ind_2] = block_K
            K[start_ind_2:stop_ind_2, start_ind_1:stop_ind_1] = np.transpose(block_K)

    subj_id = []

    for fullpath in img_paths:
        subj_id.append(fullpath.stem.split('_')[0])

    gram_df = pd.DataFrame(columns=subj_id, data=K)
    gram_df['subject_ID'] = subj_id
    gram_df = gram_df.set_index('subject_ID')

    gram_df.to_csv(output_path)
