"""
Script to get the predictiions of the models on the test set, perform the ensemble and
create the submission file.

A ideia era
Passo 1 carrega todos os pddataframes de previsao do testset para uma lista (estou centralizando esses dados no diretorio testing no NAN)
Passo 2 le o arquivo com os os ID do test set assim como os sites deles
Passo 3 faz um loop para ver cada ID de sujeito
Passo 4 Para cada ID, fazer um for para ver todos os dataframes da lista
Passo 5 se o dataframe da lista tem o sujeito, guarda a previsao dele ponderada assim como o peso (ainda decidir como ler esse peso)
Passo 6 tirar media das previsoes e guardar no submission df.
Passo 7 Ser feliz.
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

    print(row['subject_ID'])

    subj_pred_list = []
    subj_models_weight_list = []

    for prediction_df in test_predictions_list:
        print('')
        subject_prediction = prediction_df.loc[prediction_df['subject_ID'] == row['subject_ID']]
        if len(subject_prediction)==1:
            # TODO: Ponderar subject_prediction[subject_prediction.columns[1]]
            subj_pred_list.append(subject_prediction[subject_prediction.columns[1]])

            # subj_models_weight_list.append('PESO')

    media = np.sum(np.array(subj_pred_list)) / np.sum(np.array(subj_models_weight_list))

    submission_df['age'].iloc[index] = media

submission_df.to_csv(PROJECT_ROOT / 'output' / 'previsoes_ganhadoras.csv')
