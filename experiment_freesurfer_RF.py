"""
Experiment using Linear RF on freesurfer volume data.

Results:
MAE:
"""
from pathlib import Path

import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import GradientBoostingRegressor

from helper_functions import read_freesurfer_all

PROJECT_ROOT = Path.cwd()

# --------------------------------------------------------------------------
random_seed = 42
np.random.seed(random_seed)

# --------------------------------------------------------------------------
# Create experiment's output directory
output_dir = PROJECT_ROOT / 'output' / 'experiments'
output_dir.mkdir(exist_ok=True)

experiment_name = 'freesurfer_vol_SVM'  # Change here*
experiment_dir = output_dir / experiment_name
experiment_dir.mkdir(exist_ok=True)

cv_dir = experiment_dir / 'cv'
cv_dir.mkdir(exist_ok=True)

# --------------------------------------------------------------------------
# Input data directory (plz, feel free to use NAN shared folder)
freesurfer_dir = PROJECT_ROOT / 'data' / 'freesurfer'
demographic_path = PROJECT_ROOT / 'data' / 'PAC2019_BrainAge_Training.csv'

# Reading data. If necessary, create new reader in helper_functions.
x, demographic_df = read_freesurfer_all(str(freesurfer_dir), str(demographic_path))

# --------------------------------------------------------------------------
# Using only age
y = demographic_df['age'].values

# If necessary, extract gender and site from demographic_df too.

# --------------------------------------------------------------------------
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)

# --------------------------------------------------------------------------
predictions_df = pd.DataFrame(demographic_df['age'], index=demographic_df.index)
predictions_df['predictions'] = np.nan

mae_cv = np.zeros((n_folds, 1))

# --------------------------------------------------------------------------
for i_fold, (train_idx, test_idx) in enumerate(kf.split(x, y)):
    x_train, x_test = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print('CV iteration: %d' % (i_fold + 1))

    # --------------------------------------------------------------------------
    # Normalization/Scaling/Standardization
    scaler = RobustScaler()

    x_train_norm = scaler.fit_transform(x_train)
    x_test_norm = scaler.transform(x_test)

    # --------------------------------------------------------------------------
    # Model
    clf = GradientBoostingRegressor()

    # --------------------------------------------------------------------------
    # Model selection
    # Search space
    param_grid = {'max_depth': range(1, 11, 2),
                  'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
                  'subsample': np.arange(0.05, 1.01, 0.25)}


    # Gridsearch
    internal_cv = KFold(n_splits=5)
    grid_cv = GridSearchCV(estimator=clf,
                           param_grid=param_grid,
                           cv=internal_cv,
                           scoring='neg_mean_absolute_error',
                           verbose=1,
                           n_jobs=28)

    # --------------------------------------------------------------------------
    grid_result = grid_cv.fit(x_train_norm, y_train)

    # --------------------------------------------------------------------------
    best_regressor = grid_cv.best_estimator_

    # --------------------------------------------------------------------------
    y_test_predicted = best_regressor.predict(x_test_norm)

    for row, value in zip(test_idx, y_test_predicted):
        predictions_df.iloc[row, predictions_df.columns.get_loc('predictions')] = value

    # --------------------------------------------------------------------------
    mae_test = mean_absolute_error(y_test, y_test_predicted)
    print('MAE: %.3f ' % mae_test)

    mae_cv[i_fold, :] = mae_test

    joblib.dump(best_regressor, cv_dir / ('model_%d.joblib' % i_fold))

print('CV results')
print('MAE: Mean(SD) = %.3f(%.3f)' % (mae_cv.mean(), mae_cv.std()))

mae_cv_df = pd.DataFrame(columns=['MAE'], data=mae_cv)
mae_cv_df.to_csv(cv_dir / 'mae_cv.csv', index=False)

predictions_df.to_csv(cv_dir / 'predictions_cv.csv', index=False)

# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Training on whole data
scaler = RobustScaler()

x_norm = scaler.fit_transform(x)

clf_final = SVR(kernel='linear')

param_grid_final = {'C': [2 ** -6, 2 ** -5, 2 ** -4, 2 ** -3, 2 ** -2, 2 ** -1, 2 ** 0, 2 ** 1]}

internal_cv = KFold(n_splits=5)
grid_cv_final = GridSearchCV(estimator=clf_final,
                             param_grid=param_grid_final,
                             cv=internal_cv,
                             scoring='neg_mean_absolute_error',
                             verbose=1)

grid_result = grid_cv_final.fit(x_norm, y)
best_regressor_final = grid_cv_final.best_estimator_

joblib.dump(best_regressor_final, experiment_dir / 'model.joblib')
