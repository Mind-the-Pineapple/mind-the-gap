"""Example of experiment."""
from pathlib import Path

import joblib
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import RobustScaler
from helper_functions import read_npy_files

PROJECT_ROOT = Path.cwd()

# --------------------------------------------------------------------------
random_seed = 42
np.random.seed(random_seed)

# --------------------------------------------------------------------------
# Create experiment's output directory
output_dir = PROJECT_ROOT / 'output' / 'experiments'
output_dir.mkdir(exist_ok=True)

experiment_name = 'linear'  # Change here*
experiment_dir = output_dir / experiment_name
experiment_dir.mkdir(exist_ok=True)

cv_dir = experiment_dir / 'cv'
cv_dir.mkdir(exist_ok=True)

# --------------------------------------------------------------------------
# Input data directory (plz, feel free to use NAN shared folder)
input_dir = PROJECT_ROOT / 'data' / 'project' / 'mindgap' / 'data' / 'SPM' / 'np' / 'gm'
demographic_path = PROJECT_ROOT  / 'data' / 'project' / 'mindgap' / 'data' / 'PAC2019_BrainAge_Training.csv'

# Reading data. If necessary, create new reader in helper_functions.
x, demographic_df = read_npy_files(input_dir, demographic_path, use_mask=True)
print("Input Loaded")
print(x.shape)

# --------------------------------------------------------------------------
# Using only age
y = demographic_df['age'].values

# If necessary, extract gender and site from demographic_df too.

# --------------------------------------------------------------------------
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)

# --------------------------------------------------------------------------
predictions_df = pd.DataFrame(demographic_df[['subject_ID', 'age']])
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
    # Dimensionality reduction
    pca = PCA(n_components = 0.9)
    x_train_pca = pca.fit_transform(x_train_norm)
    x_test_pca = pca.transform(x_test_norm)

    # --------------------------------------------------------------------------
    # Model
    clf = linear_model.LinearRegression()

    # --------------------------------------------------------------------------
    # Model selection
    # Search space
    param_grid = {'fit_intercept': ('True', 'False'), 'normalize': ('True', 'False'), 'copy_X': ('True', 'False')}

    # Gridsearch
    internal_cv = KFold(n_splits=5)
    grid_cv = GridSearchCV(estimator=clf,
                           param_grid=param_grid,
                           cv=internal_cv,
                           scoring='mean_absolute_error',
                           verbose=1)

    # --------------------------------------------------------------------------
    grid_result = grid_cv.fit(x_train_pca, y_train)

    # --------------------------------------------------------------------------
    best_regressor = grid_cv.best_estimator_

    # --------------------------------------------------------------------------
    y_test_predicted = best_regressor.predict(x_test_pca)

    for row, value in zip(test_idx, y_test_predicted):
        predictions_df.iloc[row, predictions_df.columns.get_loc('predictions')] = value

    # --------------------------------------------------------------------------
    mae_test = mean_absolute_error(y_test, y_test_predicted)
    print('MAE: %.3f ' % mae_test)

    mae_cv[i_fold, :] = mae_test

    joblib.dump(best_regressor, cv_dir / ('model_%d.joblib' % i_fold))
    joblib.dump(scaler, cv_dir / ('scaler_%d.joblib' % i_fold))
    joblib.dump(pca, cv_dir / ('pca_%d.joblib' % i_fold))

print('CV results')
print('MAE: Mean(SD) = %.3f(%.3f)' % (mae_cv.mean(), mae_cv.std()))

mae_cv_df = pd.DataFrame(columns=['MAE'], data=mae_cv)
mae_cv_df.to_csv(cv_dir / 'mae_cv.csv', index=False)

predictions_df.to_csv(cv_dir / 'predictions_cv.csv', index=False)

# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Training on whole data
scaler_final = RobustScaler()
x_norm = scaler_final.fit_transform(x)

pca_final = PCA(n_components=2)
x_pca = pca_final.fit_transform(x_norm)

clf_final = linear_model.LinearRegression()

param_grid_final = {'fit_intercept':('True', 'False'), 'normalize':('True', 'False'), 'copy_X':('True', 'False')}

internal_cv = KFold(n_splits=5)
grid_cv_final = GridSearchCV(estimator=clf_final,
                             param_grid=param_grid_final,
                             cv=internal_cv,
                             scoring='mean_absolute_error',
                             verbose=1)

grid_result = grid_cv_final.fit(x_pca, y)
best_regressor_final = grid_cv_final.best_estimator_

joblib.dump(best_regressor_final, experiment_dir / 'model.joblib')
joblib.dump(scaler_final, experiment_dir / 'scaler.joblib')
joblib.dump(pca_final, experiment_dir / 'pca.joblib')
