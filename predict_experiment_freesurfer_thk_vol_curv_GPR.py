from pathlib import Path
import os

import joblib
import numpy as np
import pandas as pd

from helper_functions import read_freesurfer_sites_validation
from freesurfer_columns import thk_vol_curv as COLUMNS_NAMES

PROJECT_ROOT = Path.cwd()

# --------------------------------------------------------------------------
random_seed = 42
np.random.seed(random_seed)
# --------------------------------------------------------------------------

experiment_name = 'freesurfer_thk_vol_cur_gpr'  # Change here*
output_dir = PROJECT_ROOT / 'output' / 'experiments'/ experiment_name

# --------------------------------------------------------------------------
# Load the freesurfer dataset
freesurfer_dir_validation = PROJECT_ROOT / 'data' / 'freesurfer'/ 'test_set'
demographic_path_validation = PROJECT_ROOT / 'data' / 'PAC2019_BrainAge_Test_Upload.csv'
x_validation, demographic_df_validation = read_freesurfer_sites_validation(
                                            freesurfer_dir_validation,
                                            demographic_path_validation,
                                            COLUMNS_NAMES,
                                            -1) # All sites
# Load the scaler function
scale_path = output_dir / 'scaler.joblib'
scaler = joblib.load(scale_path)

x_scaled = scaler.transform(x_validation)
# --------------------------------------------------------------------------
# Load the trained model
model_file = output_dir / 'model.joblib'
model = joblib.load(str(model_file))

# --------------------------------------------------------------------------
# Predict using the trained model
predictions = model.predict(x_scaled)
# --------------------------------------------------------------------------
# Save the age prediction
predictions_df = pd.DataFrame(index=x_validation.index, data=predictions)
predictions_df.to_csv(os.path.join(output_dir, 'predictions_%s_csv'
                                   %experiment_name),
                                   index=True)


