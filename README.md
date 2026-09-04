# Cpp ML Library

A small, dependency-free C++17 machine-learning library. It is designed for learning, experiments and portfolio projects: every algorithm is implemented in readable C++ and the public API is covered by CTest tests.

## What is included

- Dense `Matrix` type with arithmetic, decompositions, statistics and linear algebra
- Regression and clustering: linear regression, ridge regression, K-Means and DBSCAN
- Classification: decision tree, random forest, K-nearest-neighbors, Gaussian Naive Bayes and logistic regression
- Preprocessing: imputing, scaling, encoding, discretization, log transforms, polynomial features and train/test splitting
- Feature selection: variance threshold, univariate tests and model-based selectors
- Data tooling: numeric CSV reader, `Dataset`, K-fold/stratified K-fold cross-validation and grid search
- Evaluation: MSE, RMSE, MAE, R², classification accuracy/precision/recall/F1, confusion matrix and silhouette score
- A composable supervised `Pipeline` for existing transformers and regressors

## Build and test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

The CI workflow builds the project with GCC, Clang and MSVC.

## Quick start

```cpp
#include <ml/ml.hpp>

int main() {
    ml::Matrix features{{0.0}, {1.0}, {2.0}, {3.0}};
    std::vector<double> targets{1.0, 3.0, 5.0, 7.0};

    auto model = ml::make_pipeline(ml::LinearRegression{}, ml::StandardScaler{});
    model.fit(features, targets);
    const auto prediction = model.predict(ml::Matrix{{4.0}});
}
```

See [examples documentation](docs/examples.md) for data loading, evaluation and clustering examples. The public API and architectural boundaries are described in [docs](docs/).

For an in-depth mathematical derivation of Logistic Regression and Ridge/Lasso regularization, including a benchmark against scikit-learn, see [docs/deep-dive-logistic-regression-and-regularization.md](docs/deep-dive-logistic-regression-and-regularization.md).

## Install as a CMake package

```bash
cmake -S . -B build
cmake --install build --prefix /desired/prefix
```

Consumers can then use `find_package(CppMLLibrary CONFIG REQUIRED)` and link `CppMLLibrary::ml_library`.

## Benchmarks

Benchmarks are opt-in so ordinary builds remain fast:

```bash
cmake -S . -B build -DML_BUILD_BENCHMARKS=ON
cmake --build build --target matrix_benchmark algorithm_benchmark
```

## Versioning

The project follows semantic versioning. The current public API target is **v1.3.0**; see [CHANGELOG.md](CHANGELOG.md) for release notes.

## License

No license has been selected yet. Choose and add a `LICENSE` file before publishing or accepting external contributions.
