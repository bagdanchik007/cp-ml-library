#include "ml/algorithms/knn_classifier.hpp"

#include <cassert>

int main() {
    const ml::Matrix X{
        {0.0, 0.0}, {0.1, 0.1}, {0.2, 0.0},
        {5.0, 5.0}, {5.1, 4.9}, {4.9, 5.1}
    };
    const std::vector<int> y{0, 0, 0, 1, 1, 1};

    ml::KNNClassifier knn(3);
    assert(!knn.is_fitted());
    knn.fit(X, y);
    assert(knn.is_fitted());

    const ml::Matrix query{{0.05, 0.05}, {5.05, 5.0}};
    const auto predictions = knn.predict(query);
    assert(predictions.size() == 2);
    assert(predictions[0] == 0);
    assert(predictions[1] == 1);
}
