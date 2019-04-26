"""
This script contains functions that will be used in different scripts
"""
from pathlib import Path

from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import nibabel as nib
from nibabel import processing
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

def read_freesurfer(data_dir, demographic_path, columns_name):
    """
    Read volumetric data and demographic data. Note: demographic file is included here to make sure that the
    x data and the demographic data follows the same subject order.

    Args:
        data_dir: String with path to directory with csv files.
        demographic_path: String with path to demographic data.
        columns_name: List of columns name

    Returns:
        x: Numpy array with neuroimaging data.
        demographic_df: Pandas DataFrame with all demographic data.
    """
    freesurfer_dir = Path(data_dir)

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

    # tiv = merged_df['EstimatedTotalIntraCranialVol']

    demographic_df = pd.DataFrame(merged_df[demographic_df.columns])
    x_df = pd.DataFrame(merged_df[columns_name])
    # x = x_df.values.astype('float32') / tiv.values[:,np.newaxis]
    x = x_df.values.astype('float32')
    demographic_df.index.names = ['subject_ID']

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
    input_dir = Path(input_path)
    step_size = 20

    img_paths = list(input_dir.glob('*.npy'))

    n_samples = len(img_paths)

    K = np.float64(np.zeros((n_samples, n_samples)))

    for i in range(int(np.ceil(n_samples / np.float(step_size)))):#

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



def create_wm_and_gm_gram_matrix_train_data(wm_dir_path, gm_dir_path, output_path):
    """
    Computes the Gram matrix for the SVM method.

    Reference: http://scikit-learn.org/stable/modules/svm.html#using-the-gram-matrix
    """
    wm_dir = Path(wm_dir_path)
    gm_dir = Path(gm_dir_path)
    step_size = 100

    img_gm_paths = sorted(list(gm_dir.glob('*.npy')))
    img_wm_paths = sorted(list(wm_dir.glob('*.npy')))

    n_samples = len(img_wm_paths)

    K = np.float64(np.zeros((n_samples, n_samples)))

    for i in range(int(np.ceil(n_samples / np.float(step_size)))):  #

        it = i + 1
        max_it = int(np.ceil(n_samples / np.float(step_size)))
        print(" outer loop iteration: %d of %d." % (it, max_it))

        # generate indices and then paths for this block
        start_ind_1 = i * step_size
        stop_ind_1 = min(start_ind_1 + step_size, n_samples)
        block_paths_1_wm = img_wm_paths[start_ind_1:stop_ind_1]
        block_paths_1_gm = img_gm_paths[start_ind_1:stop_ind_1]

        # read in the images in this block
        images_1 = []
        for k, (path_wm, path_gm) in enumerate(zip(block_paths_1_wm, block_paths_1_gm)):

            img_wm = np.load(str(path_wm))
            img_wm = np.asarray(img_wm, dtype='float64')
            img_wm = np.nan_to_num(img_wm)
            img_vec_wm = np.reshape(img_wm, np.product(img_wm.shape))

            img_gm = np.load(str(path_gm))
            img_gm = np.asarray(img_gm, dtype='float64')
            img_gm = np.nan_to_num(img_gm)
            img_vec_gm = np.reshape(img_gm, np.product(img_wm.shape))

            img_vec = np.append(img_vec_wm, img_vec_gm)

            images_1.append(img_vec)
            del img_wm, img_gm
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
                block_paths_2_wm = img_wm_paths[start_ind_2:stop_ind_2]
                block_paths_2_gm = img_gm_paths[start_ind_2:stop_ind_2]


                images_2 = []
                for k, (path_wm, path_gm) in enumerate(zip(block_paths_2_wm, block_paths_2_gm)):
                    img_wm = np.load(str(path_wm))
                    img_wm = np.asarray(img_wm, dtype='float64')
                    img_wm = np.nan_to_num(img_wm)
                    img_vec_wm = np.reshape(img_wm, np.product(img_wm.shape))

                    img_gm = np.load(str(path_gm))
                    img_gm = np.asarray(img_gm, dtype='float64')
                    img_gm = np.nan_to_num(img_gm)
                    img_vec_gm = np.reshape(img_gm, np.product(img_wm.shape))

                    img_vec = np.append(img_vec_wm, img_vec_gm)

                    images_2.append(img_vec)
                    del img_wm, img_gm
                images_2 = np.array(images_2)

            block_K = np.dot(images_1, np.transpose(images_2))
            K[start_ind_1:stop_ind_1, start_ind_2:stop_ind_2] = block_K
            K[start_ind_2:stop_ind_2, start_ind_1:stop_ind_1] = np.transpose(block_K)

    subj_id = []

    for fullpath in img_gm_paths:
        subj_id.append(fullpath.stem.split('_')[0])

    gram_df = pd.DataFrame(columns=subj_id, data=K)
    gram_df['subject_ID'] = subj_id
    gram_df = gram_df.set_index('subject_ID')

    gram_df.to_csv(output_path)

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


def create_gram_matrix_w_site_info_train_data(input_dir_path, output_path, demographic_path, site_weight=10000):
    """
    Computes the Gram matrix for the SVM method.

    Reference: http://scikit-learn.org/stable/modules/svm.html#using-the-gram-matrix
    """
    demographic_df = pd.read_csv(demographic_path, index_col='subject_ID')
    foo = demographic_df.site.values[:, np.newaxis]
    enc = OneHotEncoder(sparse=False)
    enc.fit(foo)

    input_dir = Path(input_dir_path)
    step_size = 600

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


def create_wm_and_gm_w_site_info_gram_matrix_train_data(wm_dir_path, gm_dir_path, output_path, demographic_path,
                                                        site_weight=10000):
    """
    Computes the Gram matrix for the SVM method.

    Reference: http://scikit-learn.org/stable/modules/svm.html#using-the-gram-matrix
    """
    demographic_df = pd.read_csv(demographic_path, index_col='subject_ID')
    foo = demographic_df.site.values[:, np.newaxis]
    enc = OneHotEncoder(sparse=False)
    enc.fit(foo)

    wm_dir = Path(wm_dir_path)
    gm_dir = Path(gm_dir_path)
    step_size = 400

    img_gm_paths = sorted(list(gm_dir.glob('*.npy')))
    img_wm_paths = sorted(list(wm_dir.glob('*.npy')))

    n_samples = len(img_wm_paths)

    K = np.float64(np.zeros((n_samples, n_samples)))

    for i in range(int(np.ceil(n_samples / np.float(step_size)))):  #

        it = i + 1
        max_it = int(np.ceil(n_samples / np.float(step_size)))
        print(" outer loop iteration: %d of %d." % (it, max_it))

        # generate indices and then paths for this block
        start_ind_1 = i * step_size
        stop_ind_1 = min(start_ind_1 + step_size, n_samples)
        block_paths_1_wm = img_wm_paths[start_ind_1:stop_ind_1]
        block_paths_1_gm = img_gm_paths[start_ind_1:stop_ind_1]

        # read in the images in this block
        images_1 = []
        for k, (path_wm, path_gm) in enumerate(zip(block_paths_1_wm, block_paths_1_gm)):
            subject_id = path_wm.stem.split('_')[0]
            subject_site = np.array([demographic_df.loc[subject_id].site])
            onehot_encoded = enc.transform(subject_site[:,np.newaxis])

            img_wm = np.load(str(path_wm))
            img_wm = np.asarray(img_wm, dtype='float64')
            img_wm = np.nan_to_num(img_wm)
            img_vec_wm = np.reshape(img_wm, np.product(img_wm.shape))

            img_gm = np.load(str(path_gm))
            img_gm = np.asarray(img_gm, dtype='float64')
            img_gm = np.nan_to_num(img_gm)
            img_vec_gm = np.reshape(img_gm, np.product(img_wm.shape))

            img_vec = np.append(img_vec_wm, img_vec_gm)
            img_vec = np.append(img_vec, site_weight*onehot_encoded[0])

            images_1.append(img_vec)
            del img_wm, img_gm
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
                block_paths_2_wm = img_wm_paths[start_ind_2:stop_ind_2]
                block_paths_2_gm = img_gm_paths[start_ind_2:stop_ind_2]


                images_2 = []
                for k, (path_wm, path_gm) in enumerate(zip(block_paths_2_wm, block_paths_2_gm)):
                    subject_id = path_wm.stem.split('_')[0]
                    subject_site = np.array([demographic_df.loc[subject_id].site])
                    onehot_encoded = enc.transform(subject_site[:, np.newaxis])

                    img_wm = np.load(str(path_wm))
                    img_wm = np.asarray(img_wm, dtype='float64')
                    img_wm = np.nan_to_num(img_wm)
                    img_vec_wm = np.reshape(img_wm, np.product(img_wm.shape))

                    img_gm = np.load(str(path_gm))
                    img_gm = np.asarray(img_gm, dtype='float64')
                    img_gm = np.nan_to_num(img_gm)
                    img_vec_gm = np.reshape(img_gm, np.product(img_wm.shape))

                    img_vec = np.append(img_vec_wm, img_vec_gm)
                    img_vec = np.append(img_vec, site_weight*onehot_encoded[0])

                    images_2.append(img_vec)
                    del img_wm, img_gm
                images_2 = np.array(images_2)

            block_K = np.dot(images_1, np.transpose(images_2))
            K[start_ind_1:stop_ind_1, start_ind_2:stop_ind_2] = block_K
            K[start_ind_2:stop_ind_2, start_ind_1:stop_ind_1] = np.transpose(block_K)

    subj_id = []

    for fullpath in img_gm_paths:
        subj_id.append(fullpath.stem.split('_')[0])

    gram_df = pd.DataFrame(columns=subj_id, data=K)
    gram_df['subject_ID'] = subj_id
    gram_df = gram_df.set_index('subject_ID')

    gram_df.to_csv(output_path)

def read_npy_files(input_dir_path, demographic_path, use_mask=False):
    input_dir = Path(input_dir_path)
    PROJECT_ROOT = Path.cwd()

    # Iterate over all subjects and append their data to a dataframe
    img_paths = sorted(list(input_dir.glob('*.npy')))
    demographic_df = pd.read_csv(demographic_path)
    data_df = pd.DataFrame()

    if use_mask:
        # Load the brain mask
        brain_mask_path = str(PROJECT_ROOT / 'data' / 'masks' /
                              'MNI152_T1_1.5mm_brain_masked.nii.gz')
        msk_img = nib.load(brain_mask_path)
        mask = msk_img.get_data()

    # read images
    for k, path in enumerate(img_paths):
        img = np.load(str(path))
        img = np.asarray(img, dtype='float32')
        img = np.nan_to_num(img)
        if use_mask:
            # Use the mask to filter the subject data
            img_vec = img[mask==1]
        else:
            img_vec = np.reshape(img, np.product(img.shape))
        # get subject's ID
        subject_id = path.stem.split('_')[0]
        data_df = data_df.append({'subject_id': subject_id, 'data': img_vec},
                                 ignore_index=True)
        del img, img_vec


    # Make sure both datasets use the same subject's index in the corect order
    data_df = data_df.set_index('subject_id')
    data_df = data_df.reindex(list(demographic_df['subject_ID']))

    # Check if the reindexing created any NaNs
    print('Check NaNs on the reindexed dataframe')
    print(pd.isnull(data_df).sum())
    return np.array(data_df.values), demographic_df

def resample_brain_mask(save_mask=False):
    '''
    This functions takes the original 2mm fsl brain masks and resamples it to
    1.5mm
    '''

    # This is a rather big mask originally from fsl. But I wanted to be conservative and
    # not exclue too much information. The
    # other masks are smaller
    PROJECT_ROOT = Path.cwd()
    brain_mask_path = str(PROJECT_ROOT / 'data'/ 'masks' /
                          'MNI152_T1_2mm_brain_mask_dil1.nii.gz')

    msk_img = nib.load(brain_mask_path)

    # Load one sample subject and obtain the image affine
    nib_path = str(PROJECT_ROOT / 'data' / 'SPM'/ 'gm' / 'sub0_gm.nii.gz')
    nib_img = nib.load(nib_path)
    mask_img = resample_brain_mask(nib_img, save_mask=True)

    # Resample the original 2mm mask to 1.5mm.
    resampled_img = nib.processing.resample_from_to(msk_img, nib_img)

    if save_mask:
        nib.save(resampled_img,
                 str(PROJECT_ROOT, 'data'/ 'masks'/
                     'MNI152_T1_1.5mm_brain_masked.nii.gz'))
    return resampled_img

