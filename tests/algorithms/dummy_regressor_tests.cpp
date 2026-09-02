#include "ml/algorithms/dummy_regressor.hpp"

#include <cassert>
#include <cmath>

int main() {
    const std::vector<double> y{1.0, 2.0, 3.0, 4.0, 100.0};

    ml::DummyRegressor mean_model("mean");
    assert(!mean_model.is_fitted());
    mean_model.fit(y);
    assert(mean_model.is_fitted());
    assert(std::abs(mean_model.prediction_value() - 22.0) < 1e-9);

    ml::DummyRegressor median_model("median");
    median_model.fit(y);
    assert(std::abs(median_model.prediction_value() - 3.0) < 1e-9);

    ml::DummyRegressor constant_model("constant", 7.5);
    constant_model.fit({});
    const auto predictions = constant_model.predict(3);
    assert(predictions.size() == 3);
    for (double prediction : predictions) assert(std::abs(prediction - 7.5) < 1e-9);

    bool threw = false;
    try {
        ml::DummyRegressor invalid("bogus");
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}
