# Examples

## Load, split, fit and evaluate

```cpp
#include <ml/ml.hpp>

const auto dataset = ml::read_csv("housing.csv");
const auto split = ml::train_test_split(dataset, 0.2, true, 42);

auto model = ml::make_pipeline(ml::LinearRegression{}, ml::StandardScaler{});
model.fit(split.train.features(), split.train.targets());
const auto predictions = model.predict(split.test.features());
const double r2 = ml::r2_score(split.test.targets(), predictions);
```

The CSV reader expects numeric values and treats its final column as the target. It rejects missing, ragged and non-numeric rows instead of silently producing wrong data.

## Classification

```cpp
ml::RandomForestClassifier forest(100, 12, 2, 42);
forest.fit(features, class_labels);
const auto predicted = forest.predict(features);
```

## Density clustering

```cpp
ml::DBSCAN dbscan(0.4, 5);
const auto labels = dbscan.fit_predict(points);
// -1 represents noise.
```
