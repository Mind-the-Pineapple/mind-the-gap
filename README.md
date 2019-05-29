# Predictive Analytics Competition 2019
by Jessica, Pedro, and Walter (Team **MIND THE GAP**)


## Requirements
- Python 3
- [Numpy](http://www.numpy.org/)
- Pandas
- [Matplotlib](https://matplotlib.org/)
- Seaborn

## Installing the dependencies
Install virtualenv and creating a new virtual environment:

    pip install virtualenv
    virtualenv -p /usr/bin/python3 ./venv

Install dependencies

    pip3 install -r requirements.txt
    
## Organization
Each experiment will be stored in a different .py file. Please, add new readers in the helper_functions.

## Results
| Experiment | Script | MAE (years) |
|---|---|:---:|
| freesurfer_Linear | experiment_Freesurfer_Linear.py | 7.200(0.157) |
| linear_freesurfer_nn | experiment_FreeSurfer_Linear_Keras.py | 7.109(0.345) |
| linear_freesurfer_nn_w_sites | experiment_FreeSurfer_Linear_Keras_w_sites.py | 7.143 (0.332) |
| linear_gm_nn | experiment_SPM_gm_Linear_Keras.py | 5.803(0.053) |
| linear_gm | experiment_SPM_gm_Linear.py | 13.609(0.397) |
| linear_wm_nn | experiment_SPM_wm_Linear_Keras.py | 6.530(0.110) |
| linear_wm | experiment_SPM_wm_Linear.py | 13.613(0.385) |
