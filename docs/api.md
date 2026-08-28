# API guide

All public headers live below `include/ml` and can be included individually, or together with `#include <ml/ml.hpp>`.

## Data and metrics

`Dataset` binds a feature `Matrix` to a target vector and rejects inconsistent shapes. `read_csv(path)` reads numeric CSV files where the last column is the target. `train_test_split(Dataset)` returns a reproducible train/test partition.

`mean_squared_error`, `mean_absolute_error`, `r2_score` and `accuracy_score` validate their inputs before computing a score.

## Estimators

`LinearRegression` exposes `fit`, `predict` and MSE-based `score`. `KMeans` and `DBSCAN` provide clustering labels. `DecisionTreeClassifier` and `RandomForestClassifier` take `std::vector<int>` class labels and provide `fit`/`predict`.

## Model selection

`KFold::split(sample_count)` returns train/test index vectors. `cross_validate` takes an estimator factory and scorer. `GridSearch` evaluates a typed list of candidate parameters, then exposes `best_parameter()` and `best_score()` after fitting.

## Pipeline

`make_pipeline(estimator, transformer...)` creates a supervised pipeline. Every transformer must provide `fit_transform(Matrix)` and `transform(Matrix)`; the estimator must expose `fit(Matrix, vector<double>)` and `predict(Matrix)`.
