"""
This script contains functions that will be used in different scripts
"""
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
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


def read_npy_reduced_files(input_dir_path, demographic_path, brain_mask_path):
    input_dir = Path(input_dir_path)

    # Iterate over all subjects and append their data to a dataframe
    demographic_df = pd.read_csv(demographic_path)
    demographic_df['filename'] = demographic_df['subject_ID'] + '_gm.npy'

    msk_img = nib.load(str(brain_mask_path))
    mask = msk_img.get_data()
    max_x, max_y, max_z, min_x, min_y, min_z = bbox2_3D(mask)

    imgs = np.zeros((2000, max_x - min_x, max_y - min_y, max_z - min_z, 1))

    # read images
    for k, npy_filename in enumerate(list(demographic_df['filename'].values[:1000])):
        print(k)
        path = input_dir / npy_filename
        print(path)
        img = np.load(str(path))
        img = np.asarray(img, dtype='float32')
        img = np.nan_to_num(img)
        # Use the mask to filter the subject data
        img_vec = img[min_x:max_x + 1, min_y:max_y + 1, min_z:max_z + 1]
        imgs[k, :, :, :, 0] = img_vec

    imgs = imgs[1:]
    return imgs, demographic_df



def bbox2_3D(img):
    """Find the bounding box limits of the 3D array where the elements in not equal to 0.

    These min max values will be used to minimize the dimension of the cube that will be feed to the neural network.

    Parameters
    ----------
    img: ndarray
        MRI data with format (Height, Width, Depth, Channels).

    Returns
    -------
        The index that start to have values different to zero.

    """
    x = np.any(img, axis=(1, 2))
    y = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    min_x, max_x = np.where(x)[0][[0, -1]]
    min_y, max_y = np.where(y)[0][[0, -1]]
    min_z, max_z = np.where(z)[0][[0, -1]]

    return min_x, max_x, min_y, max_y, min_z, max_z


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    if not isinstance(value, list) and not isinstance(value, np.ndarray):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def create_tfrecords(input_dir_path, output_path, demographic_path, brain_mask_path, indexes):
    """
    :param input_dir_path: String with path to the folder with numpy files
    :param output_path: String with path of the generated single file with the format tfrecord
    :param demographic_path: String with path to the CSV file
    :param brain_mask_path: String with path to the mask nifti file
    :param indexes: list of index of subjects to include in the tfrecord file
    :return:
    """
    input_dir = Path(input_dir_path)
    demographic_df = pd.read_csv(demographic_path)
    demographic_df['filename'] = demographic_df['subject_ID'] + '_gm.npy'

    msk_img = nib.load(str(brain_mask_path))
    mask = msk_img.get_fdata()
    mask[mask < 1e-8] = 0
    min_x, max_x, min_y, max_y, min_z, max_z = bbox2_3D(mask)

    # imgs = np.zeros((2640, max_x - min_x, max_y - min_y, max_z - min_z, 1))
    selected_subjects = demographic_df.iloc[indexes]
    with tf.io.TFRecordWriter(output_path) as writer:

        for k, npy_filename in enumerate(list(selected_subjects['filename'].values)):
            print(k)
            path = input_dir / npy_filename
            print(path)
            img = np.load(str(path))
            img = np.asarray(img, dtype='float32')
            img = np.nan_to_num(img)
            # Use the mask to filter the subject data
            img_vec = img[min_x:max_x, min_y:max_y, min_z:max_z, np.newaxis]

            def image_example(img, age, site):
                image_shape = img.shape

                feature = {
                    'height': _int64_feature(image_shape[0]),
                    'width': _int64_feature(image_shape[1]),
                    'depth': _int64_feature(image_shape[2]),
                    'channels': _int64_feature(image_shape[3]),
                    'age': _int64_feature(age),
                    'site': _int64_feature(site),
                    'image': _bytes_feature(img.tostring()),
                }

                return tf.train.Example(features=tf.train.Features(feature=feature))


            tf_example = image_example(img_vec,
                                       int(selected_subjects.iloc[k]['age']),
                                       int(selected_subjects.iloc[k]['site']))

            writer.write(tf_example.SerializeToString())
