#include "ml/preprocessing/robust_scaler.hpp"

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
        {1.0},
        {2.0},
        {3.0},
        {4.0},
        {100.0}
    };

    ml::RobustScaler scaler;

    const ml::Matrix result =
        scaler.fit_transform(data);

    assert(almost_equal(result(2, 0), 0.0));
}

void test_inverse_transform() {
    ml::Matrix data{
        {1.0, 10.0},
        {2.0, 20.0},
        {3.0, 30.0},
        {4.0, 40.0},
        {5.0, 50.0}
    };

    ml::RobustScaler scaler;

    const ml::Matrix transformed =
        scaler.fit_transform(data);

    const ml::Matrix restored =
        scaler.inverse_transform(transformed);

    for (size_t row = 0; row < data.rows; ++row) {
        for (size_t column = 0; column < data.cols; ++column) {
            assert(
                almost_equal(
                    restored(row, column),
                    data(row, column)
                )
            );
        }
    }
}

void test_transform_before_fit() {
    ml::RobustScaler scaler;

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
    ml::RobustScaler scaler;

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
    test_inverse_transform();
    test_transform_before_fit();
    test_empty_input();

    return 0;
}