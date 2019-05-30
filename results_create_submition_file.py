"""
Script to get the predictiions of the models on the test set, perform the ensemble and
create the submission file.
"""
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path.cwd()

testing_dir = PROJECT_ROOT / 'output' / 'testing'

submission_df = pd.read_csv(PROJECT_ROOT / 'data' / 'PAC2019_BrainAge_Test_Upload.csv')
submission_df['age'] = np.nan


test_predictions_list = []

for prediction_file in testing_dir.glob('*.csv'):
    test_predictions_list.append(pd.read_csv(prediction_file))


for index, row in submission_df.iterrows():
    print(row)


