#include "ml/algorithms/lasso_regression.hpp"

#include <cassert>
#include <cmath>

int main() {
    // y = 3*x1 + 0*x2 + 2, x2 is irrelevant and should be shrunk towards zero.
    const ml::Matrix X{
        {0.0, 1.0}, {1.0, -1.0}, {2.0, 2.0}, {3.0, -2.0}, {4.0, 0.5}
    };
    const std::vector<double> y{2.0, 5.0, 8.0, 11.0, 14.0};

    ml::LassoRegression model(0.01);
    assert(!model.is_fitted());
    model.fit(X, y, 5000);
    assert(model.is_fitted());

    assert(std::abs(model.weights()[0] - 3.0) < 0.2);
    assert(std::abs(model.weights()[1]) < 0.3);
    assert(std::abs(model.bias() - 2.0) < 0.3);

    const auto predictions = model.predict(ml::Matrix{{5.0, 0.0}});
    assert(std::abs(predictions[0] - 17.0) < 1.0);
}
