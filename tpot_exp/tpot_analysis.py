from pathlib import Path

import os
import argparse
import pickle
import pdb

from sklearn import model_selection
from sklearn.preprocessing import RobustScaler
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib.pylab import plt
import seaborn as sns
# from sklearn.externals import joblib
import joblib

from tpot_exp.extended_tpot import ExtendedTPOTRegressor
from helper_functions import read_freesurfer
from freesurfer_columns import curv_list as COLUMNS_NAMES
from tpot_config import tpot_config_gpr as TPOT_CONFIG

parser = argparse.ArgumentParser()
parser.add_argument('-njobs',
                     dest='njobs',
                     type=int,
                     required=True)
args = parser.parse_args()

if __name__ == '__main__':
    # --------------------------------------------------------------------------
    random_seed = 20
    PROJECT_ROOT = Path.cwd()
    # Create experiment's output directory
    output_dir = PROJECT_ROOT / 'output' / 'experiments'
    output_dir.mkdir(exist_ok=True)

    experiment_name = 'freesurfer_tpot'
    experiment_dir = output_dir / experiment_name
    experiment_dir.mkdir(exist_ok=True)

    # Directory to save TPOT updates
    checkpoint_dir = experiment_dir / 'checkpoint'
    checkpoint_dir.mkdir(exist_ok=True)
    # --------------------------------------------------------------------------
    # Input data directory
    # Load the freesurfer dataset
    freesurfer_dir = PROJECT_ROOT / 'data' / 'freesurfer'
    demographic_path = PROJECT_ROOT / 'data' / 'PAC2019_BrainAge_Training.csv'

    # Reading data. If necessary, create new reader in helper_functions.
    x, demographic_df = read_freesurfer(str(freesurfer_dir),
                                        str(demographic_path), COLUMNS_NAMES)
    # Using only age
    y = demographic_df['age'].values
    # If necessary, extract gender and site from demographic_df too.
    # TODO: create stratification

    # Divide into train, test and validate
    Xtrain, Xtest, Ytrain, Ytest = model_selection.train_test_split(x,
                                               y,
                                               test_size=.4,
                                               random_state=random_seed)
    print('Divided dataset into test and training')
    print('Check train test split sizes')
    print('X_train: ' + str(Xtrain.shape))
    print('X_test: '  + str(Xtest.shape))
    print('Y_train: ' + str(Ytrain.shape))
    print('Y_test: '  + str(Ytest.shape))

    # Normalise the input
    robustscaler = RobustScaler().fit(Xtrain)
    Xtrain_scaled = robustscaler.transform(Xtrain)
    Xtest_scaled = robustscaler.transform(Xtest)
    # --------------------------------------------------------------------------
    # TPOT parameters
    generations = 30
    population_size = 100
    offspring_size = 100
    mutation_rate = .9
    crossover_rate = .1
    cross_validation = 5
    debug=False
    analysis = 'vanilla_combined'
    tpot = ExtendedTPOTRegressor(generations=generations,
                                 population_size=population_size,
                                 offspring_size=offspring_size,
                                 mutation_rate=mutation_rate,
                                 crossover_rate=crossover_rate,
                                 n_jobs=args.njobs,
                                 cv=cross_validation,
                                 verbosity=3,
                                 random_state=random_seed,
                                 config_dict=TPOT_CONFIG,
                                 scoring='neg_mean_absolute_error',
                                 periodic_checkpoint_folder=checkpoint_dir,
                                 use_dask=False,
                                 debug=debug,
                                 analysis=analysis,
                                )

    tpot.fit(Xtrain_scaled, Ytrain, Xtest_scaled, Ytest)
    print('Test score using optimal model: %f ' % tpot.score(Xtest_scaled,
                                                             Ytest))
    tpot.export(os.path.join(experiment_dir,
                             'tpot_brain_age_pipeline.py'))
    # Dump tpot.pipelines and evaluated objects
print('Dump predictions, evaluated pipelines and sklearn objects')
tpot_save = {}
tpot_pipelines = {}
tpot_save['predictions'] = tpot.predictions
tpot_save['evaluated_individuals_'] = tpot.evaluated_individuals_
# Best pipeline at the end of the genetic algorithm
tpot_save['fitted_pipeline'] = tpot.fitted_pipeline_
# List of evaluated invidivuals per generation
tpot_save['evaluated_individuals'] = tpot.evaluated_individuals
# Dictionary containing all pipelines in the TPOT Pareto Front
tpot_save['pareto_pipelines'] = tpot.pareto_front_fitted_pipelines_
# List of best model per generation
tpot_pipelines['log'] = tpot.log

# Dump results
joblib.dump(tpot_save, os.path.join(experiment_dir, 'tpot_log.dump'))
joblib.dump(tpot_pipelines, os.path.join(experiment_dir, 'tpot_pipelines.dump'))
print('Done TPOT analysis!')

