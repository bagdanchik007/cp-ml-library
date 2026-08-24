#include "ml/preprocessing/min_max_scaler.hpp"

#include <cassert>
#include <cmath>
#include <stdexcept>

namespace {

bool almost_equal(
    double a,
    double b,
    double epsilon = 1e-9
) {
    return std::abs(a - b) < epsilon;
}

void test_fit_transform() {
    ml::Matrix data{
        {1.0, 10.0},
        {3.0, 20.0},
        {5.0, 30.0}
    };

    ml::MinMaxScaler scaler;

    const ml::Matrix result =
        scaler.fit_transform(data);

    assert(almost_equal(result(0, 0), 0.0));
    assert(almost_equal(result(1, 0), 0.5));
    assert(almost_equal(result(2, 0), 1.0));

    assert(almost_equal(result(0, 1), 0.0));
    assert(almost_equal(result(1, 1), 0.5));
    assert(almost_equal(result(2, 1), 1.0));
}

void test_transform_after_fit() {
    ml::Matrix training_data{
        {0.0, 10.0},
        {10.0, 30.0}
    };

    ml::MinMaxScaler scaler;
    scaler.fit(training_data);

    ml::Matrix data{
        {5.0, 20.0}
    };

    const ml::Matrix result =
        scaler.transform(data);

    assert(almost_equal(result(0, 0), 0.5));
    assert(almost_equal(result(0, 1), 0.5));
}

void test_inverse_transform() {
    ml::Matrix data{
        {1.0, 10.0},
        {3.0, 20.0},
        {5.0, 30.0}
    };

    ml::MinMaxScaler scaler;

    const ml::Matrix transformed =
        scaler.fit_transform(data);

    const ml::Matrix restored =
        scaler.inverse_transform(transformed);

    for (size_t i = 0; i < data.rows; ++i) {
        for (size_t j = 0; j < data.cols; ++j) {
            assert(almost_equal(
                restored(i, j),
                data(i, j)
            ));
        }
    }
}

void test_constant_feature() {
    ml::Matrix data{
        {5.0, 1.0},
        {5.0, 2.0},
        {5.0, 3.0}
    };

    ml::MinMaxScaler scaler;

    const ml::Matrix result =
        scaler.fit_transform(data);

    for (size_t row = 0; row < result.rows; ++row) {
        assert(almost_equal(result(row, 0), 0.0));
    }
}

void test_transform_before_fit() {
    ml::MinMaxScaler scaler;

    bool threw = false;

    try {
        scaler.transform(
            ml::Matrix{{1.0, 2.0}}
        );
    } catch (const std::logic_error&) {
        threw = true;
    }

    assert(threw);
}

void test_empty_input() {
    ml::MinMaxScaler scaler;

    bool threw = false;

    try {
        scaler.fit(ml::Matrix{});
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

} // namespace

int main() {
    test_fit_transform();
    test_transform_after_fit();
    test_inverse_transform();
    test_constant_feature();
    test_transform_before_fit();
    test_empty_input();

    return 0;
}