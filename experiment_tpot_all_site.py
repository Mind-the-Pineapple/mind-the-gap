"""Example of experiment."""
from pathlib import Path

import joblib
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from sklearn.preprocessing import FunctionTransformer
from copy import copy
from sklearn.model_selection import cross_validate

from helper_functions import read_freesurfer, read_freesurfer_sites_validation
from freesurfer_columns import thk_vol_curv as COLUMNS_NAMES

PROJECT_ROOT = Path.cwd()

# --------------------------------------------------------------------------
random_seed = 42
np.random.seed(random_seed)

# --------------------------------------------------------------------------
# Create experiment's output directory
output_dir = PROJECT_ROOT / 'output' / 'experiments'
output_dir.mkdir(exist_ok=True)

experiment_name = 'tpot_all_subjects_r'  # Change here*
experiment_dir = output_dir / experiment_name
experiment_dir.mkdir(exist_ok=True)

cv_dir = experiment_dir / 'cv'
cv_dir.mkdir(exist_ok=True)

# --------------------------------------------------------------------------
# Input data directory (plz, feel free to use NAN shared folder)
input_dir = PROJECT_ROOT / 'data' / 'freesurfer'
demographic_path = PROJECT_ROOT / 'data' / 'PAC2019_BrainAge_Training.csv'

# Reading data. If necessary, create new reader in helper_functions.
x, demographic_df = read_freesurfer(str(input_dir),
                                    str(demographic_path),
                                    COLUMNS_NAMES)

# --------------------------------------------------------------------------
# Using only age
y = demographic_df['age'].values

# If necessary, extract gender and site from demographic_df too.

# --------------------------------------------------------------------------
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)

# --------------------------------------------------------------------------
predictions_df = pd.DataFrame(demographic_df[['age']])
predictions_df['predictions'] = np.nan

mae_cv = np.zeros((n_folds, 1))


# -------------------------------------------------------------------------- for
for i_fold, (train_idx, test_idx) in enumerate(kf.split(x, y)):
    x_train, x_test = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print('CV iteration: %d' % (i_fold + 1))
    #--------------------------------------------------------------------------

    # Normalization/Scaling/Standardization
    scaler = RobustScaler()

    x_train_norm = scaler.fit_transform(x_train)
    x_test_norm = scaler.transform(x_test)

    #--------------------------------------------------------------------------
    # Model
    clf = make_pipeline( make_union(StackingEstimator
    (estimator=LinearRegression()), FunctionTransformer(copy)),
    RandomForestRegressor(bootstrap=False, max_features=0.45,
    min_samples_leaf=15, min_samples_split=19, n_estimators=100,
    random_state=42))

    #--------------------------------------------------------------------------
    # Gridsearch
    clf.fit(x_train_norm, y_train)

    #--------------------------------------------------------------------------
    y_test_predicted = clf.predict(x_test_norm)

    for row, value in zip(test_idx, y_test_predicted):
        predictions_df.iloc[row, predictions_df.columns.get_loc('predictions')] = value

    #--------------------------------------------------------------------------
    mae_test = mean_absolute_error(y_test, y_test_predicted)
    print('MAE: %.3f' % mae_test)

    mae_cv[i_fold, :] = mae_test

    joblib.dump(clf, cv_dir / ('model_%d.joblib' % i_fold))

print('CV results')
print('MAE: Mean(SD) = %.3f(%.3f)' % (mae_cv.mean(), mae_cv.std()))

mae_cv_df = pd.DataFrame(columns=['MAE'], data=mae_cv)
mae_cv_df.to_csv(cv_dir / 'mae_cv.csv', index=False)

predictions_df.to_csv(cv_dir / 'predictions_cv.csv', index=True)

# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Training on whole data
scaler = RobustScaler()

scaler_fit = scaler.fit(x)
x_norm = scaler_fit.transform(x)
clf_final = make_pipeline(
                    make_union(StackingEstimator
                               (estimator=LinearRegression()),
                                FunctionTransformer(copy)),
                RandomForestRegressor(bootstrap=False,
                                      max_features=0.45,
                                      min_samples_leaf=15,
                                      min_samples_split=19,
                                      n_estimators=100,
                                      random_state=42))
training_model = clf_final.fit(x_norm, y)

joblib.dump(scaler_fit, experiment_dir / 'scaler.joblib')
# --------------------------------------------------------------------------
# Validation data
# --------------------------------------------------------------------------
# Load the freesurfer dataset
freesurfer_dir_validation = PROJECT_ROOT / 'data' / 'freesurfer'/ 'test_set'
demographic_path_validation = PROJECT_ROOT / 'data' / 'PAC2019_BrainAge_Test_Upload.csv'
x_validation, demographic_df_validation = read_freesurfer_sites_validation(
                                            freesurfer_dir_validation,
                                            demographic_path_validation,
                                            COLUMNS_NAMES,
                                            -1)

# scale the dataset
x_val_scaled = scaler_fit.transform(x_validation)
predicted_val = training_model.predict(x_val_scaled)

predictions_val_df = pd.DataFrame(index=x_validation.index, data=predicted_val)
predictions_val_df.to_csv(scaler_fit, experiment_dir / 'predictions_cv_val.csv', index=True)
