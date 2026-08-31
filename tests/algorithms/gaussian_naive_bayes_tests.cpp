#include "ml/algorithms/gaussian_naive_bayes.hpp"

#include <cassert>

int main() {
    const ml::Matrix X{
        {0.0, 0.1}, {0.2, -0.1}, {-0.1, 0.0}, {0.1, 0.2},
        {5.0, 5.1}, {5.2, 4.9}, {4.9, 5.0}, {5.1, 5.2}
    };
    const std::vector<int> y{0, 0, 0, 0, 1, 1, 1, 1};

    ml::GaussianNaiveBayes model;
    assert(!model.is_fitted());
    model.fit(X, y);
    assert(model.is_fitted());

    const ml::Matrix query{{0.05, 0.0}, {5.05, 5.05}};
    const auto predictions = model.predict(query);
    assert(predictions.size() == 2);
    assert(predictions[0] == 0);
    assert(predictions[1] == 1);
}
