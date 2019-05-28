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
**James et al. 2017: CNNs 4.16 years/GPR 4.41 years**

| Experiment | Script | MAE (years) |
|---|---|:---:|
| SPM_wm+gm_w_site_SVM | experiment_SPM_gm+wm_w_site_SVM.py | 4.530 |
| SPM_wm+gm_SVM | experiment_SPM_gm+wm_SVM.py | 4.571 |
| SPM_gm_w_site_SVM | experiment_SPM_gm_w_site_SVM.py |  5.003 |
| SPM_gm_SVM | experiment_SPM_gm_SVM.py | 5.004 |
| SPM_wm_w_site_SVM | experiment_SPM_wm_w_site_SVM.py  | 5.417 |
| SPM_wm_SVM | experiment_SPM_wm_SVM.py | 5.589 |
| freesurfer_vol_SVM | experiment_freesurfer_SVM.py | 7.187 |
|  |  |  |