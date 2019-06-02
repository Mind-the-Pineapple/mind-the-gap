"""
Experiment using Linear SVM on WM+GM SPM data.

Results:

"""
from pathlib import Path

import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.svm import SVR

from helper_functions import read_gram_matrix
import warnings

warnings.filterwarnings("ignore")
PROJECT_ROOT = Path.cwd()

# --------------------------------------------------------------------------
random_seed = 42
np.random.seed(random_seed)

# --------------------------------------------------------------------------
output_dir = PROJECT_ROOT / 'output' / 'experiments'

experiment_name = 'SPM_wm+gm_SVM_1_per_site'  # Change here*
experiment_dir = output_dir / experiment_name
experiment_dir.mkdir(exist_ok=True)

# --------------------------------------------------------------------------
# Input data directory (plz, feel free to use NAN shared folder)
gram_matrix_path = PROJECT_ROOT / 'data' / 'gram' / 'wm+gm.csv'
demographic_path = PROJECT_ROOT / 'data' / 'PAC2019_BrainAge_Training.csv'

# Reading data. If necessary, create new reader in helper_functions.
x, demographic_df = read_gram_matrix(str(gram_matrix_path), str(demographic_path))

# --------------------------------------------------------------------------
# Using only age
y = demographic_df['age'].values
sites = demographic_df['site'].values

# If necessary, extract gender and site from demographic_df too.

# --------------------------------------------------------------------------
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)

# --------------------------------------------------------------------------
predictions_df = pd.DataFrame(demographic_df[['subject_ID', 'age']])
predictions_df['predictions'] = np.nan

mae_cv = np.zeros((n_folds, 1))

for i_site in np.unique(sites):
    x_red = x[sites==i_site, :][:,sites==i_site]
    y_red = y[sites==i_site]

    predictions_df = pd.DataFrame(demographic_df[['subject_ID', 'age']])
    predictions_df = predictions_df.loc[demographic_df['site'] == i_site]
    predictions_df = predictions_df.reset_index()
    predictions_df['predictions'] = np.nan

    print(i_site)
    mae_cv = np.zeros((n_folds, 1))

    # --------------------------------------------------------------------------
    for i_fold, (train_idx, test_idx) in enumerate(kf.split(x_red, y_red)):
        x_train, x_test = x_red[train_idx, :][:, train_idx], x_red[test_idx, :][:, train_idx]
        y_train, y_test = y_red[train_idx], y_red[test_idx]

        # --------------------------------------------------------------------------
        # Model
        clf = SVR(kernel='precomputed')

        # --------------------------------------------------------------------------
        # Model selection
        # Search space
        param_grid = {'C': [2 ** -6, 2 ** -5, 2 ** -4, 2 ** -3, 2 ** -2, 2 ** -1, 2 ** 0, 2 ** 1]}

        # Gridsearch
        internal_cv = KFold(n_splits=3)
        grid_cv = GridSearchCV(estimator=clf,
                               param_grid=param_grid,
                               cv=internal_cv,
                               scoring='neg_mean_absolute_error',
                               verbose=0)

        # --------------------------------------------------------------------------
        grid_result = grid_cv.fit(x_train, y_train)

        # --------------------------------------------------------------------------
        best_regressor = grid_cv.best_estimator_

        # --------------------------------------------------------------------------
        y_test_predicted = best_regressor.predict(x_test)

        for row, value in zip(test_idx, y_test_predicted):
            predictions_df.iloc[row, predictions_df.columns.get_loc('predictions')] = value

        mae_test = mean_absolute_error(y_test, y_test_predicted)

        mae_cv[i_fold, :] = mae_test

    predictions_df.to_csv(experiment_dir / 'SPM_wm+gm_SVM_1_per_site_{:}.csv'.format(i_site), index=False)

    print('MAE: Mean(SD) = %.3f(%.3f)' % (mae_cv.mean(), mae_cv.std()))


no_site_df = pd.read_csv(PROJECT_ROOT / 'output' / 'experiments' / 'SPM_wm+gm_SVM' / 'cv'/ 'predictions_cv.csv')
w_site_df = pd.read_csv(PROJECT_ROOT / 'output' / 'experiments' / 'SPM_wm+gm_w_site_SVM' / 'cv'/ 'predictions_cv.csv')

for s in range(17):
    C = no_site_df.loc[demographic_df['site'] == s]
    no_site = np.mean(np.abs(C['age'] - C['predictions']))
    D = w_site_df.loc[demographic_df['site'] == s]
    w_site = np.mean(np.abs(D['age'] - D['predictions']))
    print('{:} {:}: {:5.3f} {:5.3f}'.format(s,len(C), no_site, w_site))

