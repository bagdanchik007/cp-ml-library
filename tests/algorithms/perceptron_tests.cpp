#include "ml/algorithms/perceptron.hpp"

#include <cassert>

int main() {
    // Linearly separable AND-like pattern.
    const ml::Matrix X{
        {0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0},
        {2.0, 2.0}, {2.0, 3.0}, {3.0, 2.0}, {3.0, 3.0}
    };
    const std::vector<int> y{0, 0, 0, 0, 1, 1, 1, 1};

    ml::Perceptron model;
    assert(!model.is_fitted());
    model.fit(X, y, 0.1, 50);
    assert(model.is_fitted());

    const auto predictions = model.predict(X);
    assert(predictions == y);

    const auto new_predictions = model.predict(ml::Matrix{{0.5, 0.5}, {2.5, 2.5}});
    assert(new_predictions[0] == 0);
    assert(new_predictions[1] == 1);
}
