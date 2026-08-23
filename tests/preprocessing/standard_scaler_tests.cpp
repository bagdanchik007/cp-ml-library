#include "ml/preprocessing/standard_scaler.hpp"

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
        {1.0, 2.0},
        {3.0, 4.0},
        {5.0, 6.0}
    };

    ml::StandardScaler scaler;

    const ml::Matrix result =
        scaler.fit_transform(data);

    assert(almost_equal(result(0, 0), -1.224744871391589));
    assert(almost_equal(result(1, 0), 0.0));
    assert(almost_equal(result(2, 0), 1.224744871391589));

    assert(almost_equal(result(0, 1), -1.224744871391589));
    assert(almost_equal(result(1, 1), 0.0));
    assert(almost_equal(result(2, 1), 1.224744871391589));
}

void test_transform_after_fit() {
    ml::Matrix training_data{
        {10.0, 100.0},
        {20.0, 200.0},
        {30.0, 300.0}
    };

    ml::StandardScaler scaler;

    scaler.fit(training_data);

    ml::Matrix data{
        {20.0, 200.0}
    };

    const ml::Matrix result =
        scaler.transform(data);

    assert(almost_equal(result(0, 0), 0.0));
    assert(almost_equal(result(0, 1), 0.0));
}

void test_inverse_transform() {
    ml::Matrix data{
        {1.0, 2.0},
        {3.0, 4.0},
        {5.0, 6.0}
    };

    ml::StandardScaler scaler;

    const ml::Matrix transformed =
        scaler.fit_transform(data);

    const ml::Matrix restored =
        scaler.inverse_transform(transformed);

    for (size_t i = 0; i < data.rows; ++i) {
        for (size_t j = 0; j < data.cols; ++j) {
            assert(
                almost_equal(
                    restored(i, j),
                    data(i, j)
                )
            );
        }
    }
}

void test_constant_feature() {
    ml::Matrix data{
        {5.0, 1.0},
        {5.0, 2.0},
        {5.0, 3.0}
    };

    ml::StandardScaler scaler;

    const ml::Matrix result =
        scaler.fit_transform(data);

    for (size_t i = 0; i < result.rows; ++i) {
        assert(almost_equal(result(i, 0), 0.0));
    }
}

void test_empty_input() {
    ml::StandardScaler scaler;

    bool threw = false;

    try {
        scaler.fit(ml::Matrix{});
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

void test_transform_before_fit() {
    ml::StandardScaler scaler;

    ml::Matrix data{
        {1.0, 2.0}
    };

    bool threw = false;

    try {
        scaler.transform(data);
    } catch (const std::logic_error&) {
        threw = true;
    }

    assert(threw);
}

void test_feature_count_mismatch() {
    ml::Matrix training_data{
        {1.0, 2.0},
        {3.0, 4.0}
    };

    ml::StandardScaler scaler;
    scaler.fit(training_data);

    ml::Matrix invalid_data{
        {1.0, 2.0, 3.0}
    };

    bool threw = false;

    try {
        scaler.transform(invalid_data);
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
    test_empty_input();
    test_transform_before_fit();
    test_feature_count_mismatch();

    return 0;
}