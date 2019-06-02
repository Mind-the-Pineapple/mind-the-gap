from pathlib import Path

import pandas as pd
from sklearn import linear_model
import numpy as np

PROJECT_ROOT = Path.cwd()

end_game_dir = PROJECT_ROOT / 'output' / 'end_game'

age_range_df = pd.read_csv(PROJECT_ROOT / 'data' / 'train_age_range.csv', index_col='site')
submission_df = pd.read_csv(PROJECT_ROOT / 'data' / 'PAC2019_BrainAge_Test_Upload.csv')
submission_df['age'] = np.nan
submission_df['uncertainty'] = np.nan


mae_df = pd.read_csv(PROJECT_ROOT / 'data' / 'all_mae.csv')


train_predictions_list = []
models_name_list = []

for prediction_file in sorted((end_game_dir / 'train').glob('*.csv')):
    print(prediction_file)
    train_predictions_list.append(pd.read_csv(prediction_file, index_col='subject_ID'))
    models_name_list.append(prediction_file.stem)

test_predictions_list = []
diff_models = []
for train_pred_df , models_name in zip(train_predictions_list, models_name_list):
    preds = train_pred_df['predictions'].values[:, np.newaxis]
    age = train_pred_df['age'].values[:, np.newaxis]

    brainAGE = preds- age

    reg = linear_model.LinearRegression()
    reg.fit(age, brainAGE)

    test_preds_df = pd.read_csv(end_game_dir / 'test' / (models_name + '.csv'))
    test_preds = test_preds_df['age'].values[:, np.newaxis]
    test_preds_df['age_corrected'] = test_preds - reg.predict(test_preds)
    test_predictions_list.append(test_preds_df)


for subj_index, subj_data in submission_df.iterrows():
    print('-' * 150)
    print(subj_data['subject_ID'].upper().center(150))
    print('')

    subj_weighted_pred_list = []
    subj_models_weight_list = []
    subj_pred_list = []

    for df_index, prediction_df in enumerate(test_predictions_list):
        subject_prediction = prediction_df.loc[prediction_df[prediction_df.columns[0]] == subj_data['subject_ID']]
        if len(subject_prediction) == 1:
            model_weight = -1
            predicted_value = 0
            clipped_predicted_value = 0

            selected_model = models_name_list[df_index]
            model_mae = mae_df.loc[mae_df[mae_df.columns[0]] == selected_model]['MAE'].values[0]


            model_weight = (7.0 - model_mae) ** 2
            predicted_value = subject_prediction['age_corrected'].values[0]
            clipped_predicted_value = np.clip(predicted_value,
                                              age_range_df.iloc[subj_data['site']]['min'],
                                              age_range_df.iloc[subj_data['site']]['max'])
            subj_weighted_pred_list.append(clipped_predicted_value * model_weight)

            subj_models_weight_list.append(model_weight)
            subj_pred_list.append(predicted_value)

            print(
                'Model: {:30s} \t MAE: {: >6.3f} \t Model weight: {: >6.3f} \t Pred value: {: >6.3f} \t Clipped pred: {: >6.3f} \t Weighted clip pred: {: >6.3f}'
                    .format(selected_model,
                            model_mae,
                            model_weight,
                            predicted_value,
                            clipped_predicted_value,
                            clipped_predicted_value * model_weight))

    print('')

    media = np.sum(np.array(subj_weighted_pred_list)) / np.sum(np.array(subj_models_weight_list))
    clipped_media = np.clip(media, age_range_df.iloc[subj_data['site']]['min'],
                            age_range_df.iloc[subj_data['site']]['max'])
    uncertainty = np.std(np.array(subj_pred_list))
    print('')
    print('final prediction = {:7.3f}/{:7.3f} = {:6.3f} ~ {:6.3f} ~ {:} (Uncertainty: {:4.2f})'
        .format(
        np.sum(np.array(subj_weighted_pred_list)),
        np.sum(np.array(subj_models_weight_list)),
        media,
        clipped_media,
        int(round(clipped_media)),
        uncertainty))

    submission_df['age'].iloc[subj_index] = int(round(clipped_media))
    submission_df['uncertainty'].iloc[subj_index] = uncertainty

submission_df = submission_df.drop(['uncertainty', 'gender', 'site'], axis=1)
submission_df.to_csv(PROJECT_ROOT / 'output' / 'objetivo2_verfinal.csv', index=False)

