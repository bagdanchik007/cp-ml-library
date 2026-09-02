# Changelog

All notable changes are documented here. This project follows semantic versioning.

## [1.2.0] - Unreleased

### Added

- `Perceptron`: classic single-layer binary classifier
- `SoftmaxRegression`: multiclass logistic regression
- `LassoRegression`: L1-regularized linear regression via coordinate descent
- `KNNRegressor`: k-nearest-neighbors regression
- `DummyClassifier` / `DummyRegressor`: baseline models for sanity-checking real models
- `OrdinalEncoder`: integer encoding for multi-column categorical string data
- Metrics: `balanced_accuracy_score`, `matthews_corrcoef`, `mean_absolute_percentage_error`, `explained_variance_score`
- `LeaveOneOut`: cross-validation splitter for small datasets
- Demo showcasing the new algorithms in `examples/more_algorithms_demo.cpp`

## [1.1.0] - Unreleased

### Added

- Classification metrics: `precision_score`, `recall_score`, `f1_score`, `confusion_matrix`
- Regression metric: `root_mean_squared_error`
- Clustering metric: `silhouette_score`
- `KNNClassifier`: brute-force k-nearest-neighbors classifier
- `GaussianNaiveBayes`: Gaussian Naive Bayes classifier
- `LogisticRegression`: binary logistic regression via batch gradient descent
- `RidgeRegression`: closed-form L2-regularized linear regression
- `KBinsDiscretizer`: equal-width binning preprocessing transformer
- `Log1pTransformer`: log(1 + x) preprocessing transformer with inverse
- `StratifiedKFold`: class-proportion-preserving cross-validation splitter
- Demo showcasing the new algorithms in `examples/new_algorithms_demo.cpp`

## [1.0.0] - Unreleased

### Added

- Dataset and numeric CSV loading APIs
- Regression/classification evaluation utilities
- Decision tree, random forest and DBSCAN
- K-fold splitting, cross-validation, grid search and supervised pipelines
- Optional matrix and algorithm benchmark executables
- GCC, Clang and MSVC CI workflow
- CMake install/export package configuration
- API, example and architecture documentation

### Release checklist

- Run the complete test suite on all supported compilers.
- Choose and add a license.
- Create and push the annotated `v1.0.0` Git tag.
