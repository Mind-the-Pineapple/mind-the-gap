"""
This script loads a sample of the dataset (part 6) and (I) checks the data
normalisation and run a very basic regression to see its accuracy
"""

import os
import pandas as pd
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error as MAE


from helper_functions import plot_age_distribution

cwd = os.getcwd()
output_dir = os.path.join(cwd, 'output', 'EDA_volumentric_analysis')
# check if ouptut file exists, otherwise create it
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)
print('The output directory is: %s' %output_dir)

# For this test we are only using part 6 of the dataset
folder_wm = os.path.join(cwd, 'data', 'wm', 'wm_part6')
folder_gm = os.path.join(cwd, 'data', 'gm', 'gm_part6')

files_wm = os.listdir(folder_wm)
files_gm = os.listdir(folder_gm)

# Load data as nib files
data= np.zeros((len(files_wm), 121,145,121, 2))
# Exapand the nibabel files into numpy arrays (might be time consuming)
for sub in range(len(files_wm)):
    # Load data for the white matter
    nii_path_wm = os.path.join(folder_wm, files_wm[sub])
    img_wm = nib.load(nii_path_wm)
    data[sub,..., 0] = img_wm.get_data()

    # Load data for the grey matter
    nii_path_gm = os.path.join(folder_gm, files_gm[sub])
    img_gm = nib.load(nii_path_gm)
    data[sub,..., 1] = img_gm.get_data()

###############################################################################
# 0. Plot the age distribution for this subsampled dataset
###############################################################################
demographic_filename = os.path.join(cwd,'data','PAC2019_BrainAge_Training.csv')
demographic_df = pd.read_csv(demographic_filename)

# Find subjects belongs to part 6
subjects_id = [sub.strip('_wm.nii.gz') for sub in files_wm]
# Select demografic information only from part 6
demographic_df_selected = demographic_df[demographic_df['subject_ID'].isin(subjects_id)]

save_path_demographics = os.path.join(output_dir,
                                      'age_ranges_distribution_part6.png')

plot_age_distribution(demographic_df_selected, save_path_demographics)
print('Saved fig with the age distribution for part 6 subjects')

###############################################################################
# 1. Average densities on all slices - Check normalisation
###############################################################################
# This is just a check to see if the subject's normalisation has been done
# corerctly.
n_subjects = len(files_wm)
# averaged_densities (subjects x shape x slice x matter)
averaged_densities = np.zeros((n_subjects, 145, 3, 2))
for subject in range(n_subjects):
    # Calculate average for white matter
    averaged_densities[subject, :121, 0, :] = np.mean(data[subject], (0,1))
    averaged_densities[subject, :   , 1, :] = np.mean(data[subject], (0,2))
    averaged_densities[subject, :121, 2, :] = np.mean(data[subject], (1,2))

# Plot average densities
labels = ['coronal slices', 'axial slices', 'sagital slices']
plot_dimension = (20, 10)

# plot normalisation for white matter
f, ax = plt.subplots(1,3, figsize=plot_dimension)
for i,label in enumerate(labels):
    ax[i].plot(averaged_densities[:,:,i,0].T)
    ax[i].set_title(label)
plt.suptitle('Average densities of consecutive slices of white matter:')
plt.savefig(os.path.join(output_dir, 'wm_avg_density.png'))

# plot normalisation for gray matter
f, ax = plt.subplots(1,3, figsize=plot_dimension)
for i,label in enumerate(labels):
    ax[i].plot(averaged_densities[:,:,i,1].T)
    ax[i].set_title(label)
plt.suptitle('Average densities of consecutive slices of gray matter:')
plt.savefig(os.path.join(output_dir, 'gm_avg_density.png'))
plt.close()
print('Saved wm and gm avg density plots')
print('As visible from the plots, the data has been well normalised.')

###############################################################################
# 2. Perform very basic regression on the averaged densities
###############################################################################
# Concatenate the 3 different slices into a single dimension and reshape the
# averaged_density matrix to be (n_subjects x all_slices x matter)
print('Perform basic regression on the averaged densities')
sh = averaged_densities.shape
density_features = averaged_densities.transpose([0,2,1,3]).reshape(sh[0],sh[1]*sh[2],sh[3])

model = Ridge()
n_folds = 5
kf = KFold(n_folds)
predicted_ages = np.zeros((n_subjects))
parameters = np.zeros((n_folds, density_features.shape[1]))

# Train and test model
for i_train, i_test in kf.split(density_features, demographic_df_selected['age']):
    model.fit(density_features[i_train,:,0],
              np.array(demographic_df_selected['age'])[i_train])
    predicted_ages[i_test] = model.predict(density_features[i_test,:,0])
print('White matter MAE: ', MAE(demographic_df_selected['age'], predicted_ages))

for i_train, i_test in kf.split(density_features, demographic_df_selected['age']):
    model.fit(density_features[i_train,:,1],
              np.array(demographic_df_selected['age'])[i_train])
    predicted_ages[i_test] = model.predict(density_features[i_test,:,1])
print('Gray matter MAE: ', MAE(demographic_df_selected['age'], predicted_ages))

###############################################################################
# 3. Perform very basic regression on the GM
###############################################################################
print('Perform very basic regression on the GM:')
# Concatenate features from all 3 dimensions
sh = data.shape
data_reshaped = data.reshape(sh[0], sh[1]*sh[2]*sh[3], sh[4])

# Train and test model
predicted_ages = np.zeros((n_subjects))
for i_train, i_test in kf.split(data_reshaped, demographic_df_selected['age']):
    model.fit(data_reshaped[i_train,:,1],
             np.array(demographic_df_selected.iloc[i_train]['age']))
    predicted_ages[i_test] = model.predict(data_reshaped[i_test,:,1])
print('Gray matter MAE: ', MAE(demographic_df_selected['age'], predicted_ages))

# Plot the accuracy of the predicted model
plt.figure()
plt.scatter(demographic_df_selected['age'], predicted_ages)
plt.xlabel('True Age')
plt.ylabel('Predicted Age')
plt.savefig(os.path.join(output_dir, 'age_predicted_known_gm.png'))
plt.close()
