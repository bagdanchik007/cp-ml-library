#include "ml/algorithms/logistic_regression.hpp"

#include <cassert>

int main() {
    const ml::Matrix X{
        {-2.0}, {-1.5}, {-1.0}, {-0.5},
        {0.5}, {1.0}, {1.5}, {2.0}
    };
    const std::vector<int> y{0, 0, 0, 0, 1, 1, 1, 1};

    ml::LogisticRegression model;
    assert(!model.is_fitted());
    model.fit(X, y, 0.5, 2000);
    assert(model.is_fitted());

    const auto predictions = model.predict(X);
    assert(predictions == y);

    const auto probabilities = model.predict_proba(ml::Matrix{{5.0}, {-5.0}});
    assert(probabilities[0] > 0.9);
    assert(probabilities[1] < 0.1);
}
