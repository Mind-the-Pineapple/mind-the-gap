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

### Linea rmodels
| Experiment | Script | MAE (years) |
|---|---|:---:|
| freesurfer_Linear | experiment_Freesurfer_Linear.py | 7.200(0.157) |
| linear_freesurfer_nn | experiment_FreeSurfer_Linear_Keras.py | 7.109(0.345) |
| linear_freesurfer_nn_w_sites | experiment_FreeSurfer_Linear_Keras_w_sites.py | 7.143 (0.332) |
| linear_gm_nn | experiment_SPM_gm_Linear_Keras.py | 5.803(0.053) |
| linear_gm | experiment_SPM_gm_Linear.py | 13.609(0.397) |
| linear_wm_nn | experiment_SPM_wm_Linear_Keras.py | 6.530(0.110) |
| linear_wm | experiment_SPM_wm_Linear.py | 13.613(0.385) |
| freesurfer_curv_GPR | experiment_freesurfer_curv_GPR.py | 7.200 |
| freesurfer_thk_vol_GPR | experiment_freesurfer_thk_vol_GPR.py | 6.385 |
| freesurfer_thk_vol_curv_GPR | experiment_freesurfer_thk_vol_curv_GPR.py | 6.132 |

## TPOT analysis (per site)
**10 Generations, using thickness, volume and curvature information**

| Site | Model | MAE (years) | n_test| n_train |
|---|---|:---:|--|--|
| 0 |make_pipeline(StackingEstimator(estimator=LassoLarsCV (normalize=True)),StackingEstimator(estimator=RVR(alpha=1e-06, beta=0.01,kernel=DotProduct(sigma_0=1))),StackingEstimator(estimator=LassoLarsCV(normalize=True)),StackingEstimator(estimator=LassoLarsCV(normalize=False)),StackingEstimator(estimator=Ridge(alpha=10.0,random_state=42)),RandomForestRegressor(bootstrap=True, max_features=0.3, min_samples_leaf=13, min_samples_split=16, n_estimators=100, random_state=42)) | 5.557 | 83 | 247 |
| 1 | make_pipeline(StackingEstimator(estimator=LassoLarsCV(normalize=True)),KNeighborsRegressor(n_neighbors=9,  p=2,weights="uniform")) | 4.101 | 34 | 100 |
| 2 | make_pipeline(StackingEstimator(estimator=ElasticNetCV(l1_ratio=0.8500000000000001, random_state=42, tol=0.0001)),FeatureAgglomeration(affinity="l1", linkage="average"), StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=True, max_features=0.15000000000000002, min_samples_leaf=1, min_samples_split=8, n_estimators=100, random_state=42)),SelectFwe(score_func=f_regression, alpha=0.022),Ridge(alpha=10.0,random_state=42))| 4.721 | 144 | 430 |
| 3 | make_pipeline(StackingEstimator(estimator=LinearSVR(C=0.5, dual=False, epsilon=1.0, loss="squared_epsilon_insensitive", random_state=42, tol=0.01)),RandomForestRegressor(bootstrap=True, max_features=0.3, min_samples_leaf=1, min_samples_split=14, n_estimators=100, random_state=42))| 4.027 | 37 | 110|
| 4 | make_pipeline(StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=False, max_features=0.35000000000000003, min_samples_leaf=9, min_samples_split=16, n_estimators=100, random_state=42)), StackingEstimator(estimator=Ridge(alpha=1.0, random_state=42)), ExtraTreesRegressor(bootstrap=False, max_features=0.6500000000000001, min_samples_leaf=13, min_samples_split=17, n_estimators=100, random_state=42)) | 2.05  | 36 | 107 |
| 5 | exported_pipeline = RandomForestRegressor(bootstrap=False, max_features=0.4, min_samples_leaf=6, min_samples_split=9, n_estimators=100, random_state=42) | 6.667 | 10 | 29 |
| 6 | make_pipeline(SelectPercentile(score_func=f_regression, percentile=27),StackingEstimator(estimator=GaussianProcessRegressor(alpha=0.015, kernel=RBF(length_scale=1), random_state=42)),FeatureAgglomeration(affinity="manhattan", linkage="complete"),GaussianProcessRegressor(alpha=0.001, kernel=DotProduct(sigma_0=1), random_state=42)) | 5.94 | 3 | 7|
| 7 |make_pipeline(SelectPercentile(score_func=f_regression,percentile=84), StackingEstimator(estimator=ElasticNetCV(l1_ratio=0.05, random_state=42, tol=0.1)),ElasticNetCV(l1_ratio=0.5, random_state=42, tol=0.001)) | 5.638 | 7 | 18 |
| 8 | make_pipeline(StackingEstimator(estimator=ElasticNetCV(l1_ratio=0.8, random_state=42, tol=0.1)), RandomForestRegressor(bootstrap=False, max_features=0.35000000000000003, min_samples_leaf=19, min_samples_split=18, n_estimators=100, random_state=42))  | 3.938 | 46 | 135 |
| 9 | make_pipeline(tackingEstimator(estimator=LassoLarsCV(normalize=True)),StackingEstimator(estimator=RandomForestRegressor(bootstrap=True, max_features=0.5, min_samples_leaf=7, min_samples_split=3, n_estimators=100, random_state=42)), ExtraTreesRegressor(bootstrap=True, max_features=0.8, min_samples_leaf=5, min_samples_split=18, n_estimators=100, random_state=42))) | 6.685 | 112 | 336 |
| 10 | make_pipeline(StackingEstimator(estimator=KNeighborsRegressor(n_neighbors=7, p=2, weights="uniform")),StackingEstimator(estimator=DecisionTreeRegressor(max_depth=10, min_samples_leaf=15, min_samples_split=16, random_state=42)),Ridge(alpha=10.0, random_state=42)) | 9.21 | 19 | 55 |
| 11 |make_pipeline(SelectPercentile(score_func=f_regression, percentile=25),SelectPercentile(score_func=f_regression,percentile=89),RVR(alpha=0.01, beta=1e-10, kernel=DotProduct(sigma_0=1))) | 4.213 | 13 | 5|
| 12 | make_pipeline(FeatureAgglomeration(affinity="cosine", linkage="average"),  StackingEstimator(estimator=DecisionTreeRegressor(max_depth=1, min_samples_leaf=3, min_samples_split=5, random_state=42)),FeatureAgglomeration(affinity="manhattan", linkage="complete"),Ridge(alpha=1.0, random_state=42) | 4.375 | 8 | 23 |
| 13 |make_pipeline(StackingEstimator(estimator=RandomForestRegressor(bootstrap=False, max_features=0.8, min_samples_leaf=20, min_samples_split=11, n_estimators=100, random_state=42)),StackingEstimator(estimator=DecisionTreeRegressor(max_depth=7, min_samples_leaf=16, min_samples_split=3, random_state=42)),StackingEstimator(estimator=RandomForestRegressor(bootstrap=True, max_features=0.5, min_samples_leaf=7, min_samples_split=17, n_estimators=100, random_state=42)),Ridge(alpha=100.0, random_state=42))| 10.155 | 32 | 96 |
| 14 | make_pipeline(StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=False, max_features=0.55, min_samples_leaf=8, min_samples_split=19, n_estimators=100, random_state=42)),StackingEstimator(estimator=DecisionTreeRegressor(max_depth=1, min_samples_leaf=3, min_samples_split=5,random_state=42)),StackingEstimator(estimator=LinearRegression()),StackingEstimator(estimator=DecisionTreeRegressor(max_depth=1, min_samples_leaf=6, min_samples_split=10, random_state=42)),Ridge(alpha=10.0, random_state=42))| 10.849 | 15 | 42 |
| 15 | make_pipeline(Nystroem(gamma=0.6000000000000001, kernel="sigmoid", n_components=2, random_state=42), LinearRegression()) | 1.861 | 5 | 14 |
| 16 | make_pipeline(StackingEstimator(estimator=RandomForestRegressor(bootstrap=True, max_features=0.3, min_samples_leaf=18, min_samples_split=11, n_estimators=100, random_state=42)),StackingEstimator(estimator=ElasticNetCV(l1_ratio=0.9, random_state=42, tol=0.001)),DecisionTreeRegressor(max_depth=8, min_samples_leaf=6, min_samples_split=11, random_state=42)) | 2.22 | 7 | 21 |
