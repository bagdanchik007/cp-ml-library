#include "ml/algorithms/kmeans.hpp"
#include "ml/algorithms/random_forest.hpp"

#include <chrono>
#include <iostream>
#include <vector>

int main() {
    const ml::Matrix features = ml::Matrix::random(1000, 8, -1.0, 1.0, 42);
    std::vector<int> labels(features.rows);
    for (size_t index = 0; index < labels.size(); ++index) labels[index] = features(index, 0) + features(index, 1) > 0.0 ? 1 : 0;
    const auto start = std::chrono::steady_clock::now();
    ml::RandomForestClassifier forest(25, 8, 2, 42);
    forest.fit(features, labels);
    const auto forest_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    const auto kmeans_start = std::chrono::steady_clock::now();
    ml::KMeans kmeans(3, 100, 42); kmeans.fit(features);
    const auto kmeans_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - kmeans_start).count();
    std::cout << "random_forest_fit_ms=" << forest_ms << " kmeans_fit_ms=" << kmeans_ms << '\n';
}
