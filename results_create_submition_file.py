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

mae_df = pd.read_csv(PROJECT_ROOT / 'data' / 'all_mae.csv')


test_predictions_list = []
models_name_list = []

for prediction_file in testing_dir.glob('*.csv'):
    test_predictions_list.append(pd.read_csv(prediction_file))
    models_name_list.append(prediction_file.stem)

for subj_index, subj_data in submission_df.iterrows():
    print(subj_data['subject_ID'])

    subj_pred_list = []
    subj_models_weight_list = []

    for df_index, prediction_df in enumerate(test_predictions_list):
        subject_prediction = prediction_df.loc[prediction_df[prediction_df.columns[0]] == subj_data['subject_ID']]
        if len(subject_prediction)==1:
            model_mae = mae_df.loc[mae_df[mae_df.columns[0]]==models_name_list[df_index]]['MAE'].values
            if model_mae < 7.0:
                model_weight_final = (7.0 - model_mae) ** 2
                subj_pred_list.append(subject_prediction[subject_prediction.columns[1]].values * model_weight_final)

                subj_models_weight_list.append(model_weight_final)

    media = np.sum(np.array(subj_pred_list)) / np.sum(np.array(subj_models_weight_list))

    submission_df['age'].iloc[subj_index] = media

# submission_df = submission_df.drop(['gender', 'site'], axis=1)
submission_df = submission_df.drop(['gender'], axis=1)
submission_df.to_csv(PROJECT_ROOT / 'output' / 'quase_la.csv', index=False)
