#include "ml/algorithms/weighted_knn_classifier.hpp"

#include <cassert>

int main() {
    // A query point very close to a single class-1 neighbor, but with 2
    // slightly farther class-0 neighbors among the 3 nearest -- plain
    // majority voting would pick class 0, but distance weighting should
    // let the very close class-1 neighbor win.
    const ml::Matrix X{
        {0.0, 0.0},  // class 0, distance ~1.0 from query
        {0.0, 2.0},  // class 0, distance ~1.0 from query
        {0.05, 1.0}  // class 1, distance ~0.05 from query (very close)
    };
    const std::vector<int> y{0, 0, 1};

    ml::WeightedKNNClassifier model(3);
    assert(!model.is_fitted());
    model.fit(X, y);
    assert(model.is_fitted());

    const auto predictions = model.predict(ml::Matrix{{0.0, 1.0}});
    assert(predictions[0] == 1);

    // Exact match with a training point should predict that point's label.
    const auto exact_match = model.predict(ml::Matrix{{0.0, 0.0}});
    assert(exact_match[0] == 0);
}
