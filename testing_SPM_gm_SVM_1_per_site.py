"""
Experiment using Linear SVM on WM+GM SPM data.

Results:
MAE: Mean(SD) = 4.95453854
"""
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.svm import SVR

PROJECT_ROOT = Path.cwd()

# --------------------------------------------------------------------------
random_seed = 42
np.random.seed(random_seed)

# --------------------------------------------------------------------------
# Create experiment's output directory
testing_dir = PROJECT_ROOT / 'output' / 'testing'
testing_dir.mkdir(exist_ok=True)

# --------------------------------------------------------------------------
# Input data directory (plz, feel free to use NAN shared folder)
gram_matrix_path = PROJECT_ROOT / 'data' / 'gram' / 'gm_testing.csv'
train_demographic_path = PROJECT_ROOT / 'data' / 'PAC2019_BrainAge_Training.csv'
test_demographic_path = PROJECT_ROOT / 'data' / 'PAC2019_BrainAge_Test_Upload.csv'

# Reading data. If necessary, create new reader in helper_functions.
train_demographic_df = pd.read_csv(train_demographic_path)
test_demographic_df = pd.read_csv(test_demographic_path)
gram_df = pd.read_csv(gram_matrix_path, index_col='subject_ID')


x_train = gram_df[list(train_demographic_df['subject_ID'])]
x_train = x_train.reindex(list(train_demographic_df['subject_ID']))
x_train = x_train.values

x_test = gram_df[list(train_demographic_df['subject_ID'])]
x_test = x_test.reindex(list(test_demographic_df['subject_ID']))
x_test = x_test.values

# --------------------------------------------------------------------------
# Using only age
y_train = train_demographic_df['age'].values

sites_train = train_demographic_df['site'].values
sites_test = test_demographic_df['site'].values

for i_site in np.unique(sites_train):
    x_train_red = x_train[sites_train==i_site, :][:,sites_train==i_site]
    y_train_red = y_train[sites_train==i_site]
    x_test_red = x_test[sites_test==i_site, :][:,sites_train==i_site]

    testing_name = 'SPM_gm_SVM_1_per_site_{:}'.format(i_site)

    # --------------------------------------------------------------------------
    # Model
    clf = SVR(kernel='precomputed')

    # --------------------------------------------------------------------------
    # Model selection
    # Search space
    param_grid = {'C': [2 ** -6, 2 ** -5, 2 ** -4, 2 ** -3, 2 ** -2, 2 ** -1, 2 ** 0, 2 ** 1]}

    # Gridsearch
    internal_cv = KFold(n_splits=5)
    grid_cv = GridSearchCV(estimator=clf,
                           param_grid=param_grid,
                           cv=internal_cv,
                           scoring='neg_mean_absolute_error',
                           verbose=1)

    # --------------------------------------------------------------------------
    grid_result = grid_cv.fit(x_train_red, y_train_red)

    # --------------------------------------------------------------------------
    best_regressor = grid_cv.best_estimator_

    # --------------------------------------------------------------------------
    y_test_predicted = best_regressor.predict(x_test_red)


    # --------------------------------------------------------------------------
    predictions_df = pd.DataFrame(test_demographic_df.loc[test_demographic_df['site']==i_site]['subject_ID'])
    predictions_df['age'] = y_test_predicted

    predictions_df.to_csv(testing_dir / (testing_name + '.csv'), index=False)
    np.save(testing_dir / (testing_name + '.npy'), grid_result.best_score_)
