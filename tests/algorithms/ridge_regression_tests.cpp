#include "ml/algorithms/ridge_regression.hpp"

#include <cassert>
#include <cmath>

int main() {
    // y = 2*x + 1, noise-free so a small alpha should recover it closely.
    const ml::Matrix X{{0.0}, {1.0}, {2.0}, {3.0}, {4.0}};
    const std::vector<double> y{1.0, 3.0, 5.0, 7.0, 9.0};

    ml::RidgeRegression model(0.001);
    assert(!model.is_fitted());
    model.fit(X, y);
    assert(model.is_fitted());

    assert(std::abs(model.weights()[0] - 2.0) < 0.05);
    assert(std::abs(model.bias() - 1.0) < 0.05);

    const auto predictions = model.predict(ml::Matrix{{5.0}});
    assert(std::abs(predictions[0] - 11.0) < 0.1);
}
