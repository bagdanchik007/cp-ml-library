#include "ml/preprocessing/max_abs_scaler.hpp"

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

void test_scaling() {
    ml::Matrix data{
        {-2.0, 10.0},
        {4.0, -5.0},
        {8.0, 0.0}
    };

    ml::MaxAbsScaler scaler;

    const ml::Matrix result =
        scaler.fit_transform(data);

    assert(almost_equal(result(0, 0), -0.25));
    assert(almost_equal(result(1, 0), 0.5));
    assert(almost_equal(result(2, 0), 1.0));

    assert(almost_equal(result(0, 1), 1.0));
    assert(almost_equal(result(1, 1), -0.5));
    assert(almost_equal(result(2, 1), 0.0));
}

void test_zero_column() {
    ml::Matrix data{
        {0.0, 1.0},
        {0.0, 2.0},
        {0.0, 4.0}
    };

    ml::MaxAbsScaler scaler;

    const ml::Matrix result =
        scaler.fit_transform(data);

    assert(almost_equal(result(0, 0), 0.0));
    assert(almost_equal(result(1, 0), 0.0));
    assert(almost_equal(result(2, 0), 0.0));

    assert(almost_equal(result(0, 1), 0.25));
    assert(almost_equal(result(1, 1), 0.5));
    assert(almost_equal(result(2, 1), 1.0));
}

void test_transform_before_fit() {
    ml::Matrix data{
        {1.0, 2.0}
    };

    ml::MaxAbsScaler scaler;

    bool threw = false;

    try {
        scaler.transform(data);
    } catch (const std::logic_error&) {
        threw = true;
    }

    assert(threw);
}

void test_column_mismatch() {
    ml::Matrix training_data{
        {1.0, 2.0},
        {3.0, 4.0}
    };

    ml::MaxAbsScaler scaler;

    scaler.fit(training_data);

    ml::Matrix invalid_data{
        {1.0}
    };

    bool threw = false;

    try {
        scaler.transform(invalid_data);
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

void test_empty_input() {
    ml::MaxAbsScaler scaler;

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
    test_scaling();
    test_zero_column();
    test_transform_before_fit();
    test_column_mismatch();
    test_empty_input();

    return 0;
}