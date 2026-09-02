#include "ml/algorithms/softmax_regression.hpp"

#include <cassert>

int main() {
    // Three well-separated clusters along a single axis.
    const ml::Matrix X{
        {-5.0}, {-4.5}, {-5.5},
        {0.0}, {0.5}, {-0.5},
        {5.0}, {4.5}, {5.5}
    };
    const std::vector<int> y{0, 0, 0, 1, 1, 1, 2, 2, 2};

    ml::SoftmaxRegression model;
    assert(!model.is_fitted());
    model.fit(X, y, 3, 0.5, 2000);
    assert(model.is_fitted());

    const auto predictions = model.predict(X);
    assert(predictions == y);

    const auto probabilities = model.predict_proba(ml::Matrix{{-5.0}});
    assert(probabilities[0].size() == 3);
    assert(probabilities[0][0] > probabilities[0][1]);
    assert(probabilities[0][0] > probabilities[0][2]);
}
