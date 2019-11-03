"""Script to check the predicitons along different models"""
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path.cwd()

# --------------------------------------------------------------------------
output_dir = PROJECT_ROOT / 'output' / 'experiments'

demographic_df = pd.read_csv(PROJECT_ROOT / 'data' / 'PAC2019_BrainAge_Training.csv')

predictions = pd.DataFrame(demographic_df[['subject_ID', 'age', 'site']])
predictions.set_index('subject_ID', inplace=True)

for prediction_file in output_dir.glob('*/cv/predictions_cv.csv'):
    experiment_name = prediction_file.parents[1].stem
    experiment_predictions = pd.read_csv(prediction_file, index_col='subject_ID')
    experiment_predictions[experiment_name] = np.abs(experiment_predictions['age'] - experiment_predictions['predictions'])

    predictions = pd.merge(predictions, experiment_predictions[experiment_name], left_index=True, right_index=True)


predictions['median_mae'] = predictions[predictions.columns.difference(['age', 'site'])].median(axis=1)
predictions = predictions.sort_values('median_mae', ascending=False)