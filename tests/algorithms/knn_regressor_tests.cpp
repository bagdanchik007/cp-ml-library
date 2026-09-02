#include "ml/algorithms/knn_regressor.hpp"

#include <cassert>
#include <cmath>

int main() {
    const ml::Matrix X{{0.0}, {1.0}, {2.0}, {3.0}, {4.0}};
    const std::vector<double> y{0.0, 2.0, 4.0, 6.0, 8.0};

    ml::KNNRegressor model(3);
    assert(!model.is_fitted());
    model.fit(X, y);
    assert(model.is_fitted());

    // Nearest 3 neighbors of x=2.1 are x=1,2,3 -> targets 2,4,6 -> mean 4.
    const auto predictions = model.predict(ml::Matrix{{2.1}});
    assert(std::abs(predictions[0] - 4.0) < 1e-9);
}
