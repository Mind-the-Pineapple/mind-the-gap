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


### One model per site results 

GM+WM (metric MAE and 3 KFOLD CV for 1_per_site )

| Site | site ignored | w_site | 1_per_site | number of subjects |
|---|:---:|:---:|:---:|:---:|
| 0 | 4.531 | 4.613 | 5.087(0.189) | 330
| 1 | 3.762 | 3.817 | 4.473(0.849) | 134
| 2 | 4.408 | 4.346 | 4.887(0.309) | 576
| 3 | 4.621 | 4.511 | 3.620(0.534) | 147
| 4 | 4.270 | 4.071 | 1.662(0.219) | 143
| 5 | 3.361 | 3.284 | 4.527(1.919) | 39
| 6 | 3.764 | 3.287 | 3.091(2.598) | 10
| 7 | 5.216 | 4.885 | 9.777(1.874) | 25
| 8 | 4.259 | 4.264 | 3.850(0.399) | 258
| 9 | 4.784 | 4.772 | 5.678(0.588) | 449 
| 10 | 4.723 | 4.406 | 6.266(0.973) | 74
| 11 | 4.147 | 3.951 | 5.188(1.746) | 18
| 12 | 4.748 | 4.649 | 4.846(1.097) | 31
| 13 | 5.182 | 5.157 | 7.084(1.558) | 128 
| 14 | 5.491 | 5.497 | 7.070(0.820) | 230
| 15 | 4.485 | 4.255 | 1.159(0.374) | 19
| 16 | 3.819 | 3.625 | 2.447(0.483) | 29
