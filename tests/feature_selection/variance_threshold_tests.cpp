#include "ml/feature_selection/variance_threshold.hpp"

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

void test_variance_calculation() {
    ml::Matrix data{
        {1.0, 10.0},
        {2.0, 20.0},
        {3.0, 30.0},
        {4.0, 40.0}
    };

    ml::VarianceThreshold selector;

    selector.fit(data);

    const auto& variances =
        selector.variances();

    assert(variances.size() == 2);

    assert(almost_equal(
        variances[0],
        1.25
    ));

    assert(almost_equal(
        variances[1],
        125.0
    ));
}

void test_feature_selection() {
    ml::Matrix data{
        {1.0, 10.0, 5.0},
        {1.0, 20.0, 5.0},
        {1.0, 30.0, 5.0},
        {1.0, 40.0, 5.0}
    };

    ml::VarianceThreshold selector;

    const ml::Matrix result =
        selector.fit_transform(data);

    assert(result.rows == 4);
    assert(result.cols == 1);

    assert(almost_equal(result(0, 0), 10.0));
    assert(almost_equal(result(1, 0), 20.0));
    assert(almost_equal(result(2, 0), 30.0));
    assert(almost_equal(result(3, 0), 40.0));
}

void test_threshold() {
    ml::Matrix data{
        {1.0, 10.0, 100.0},
        {2.0, 20.0, 100.0},
        {3.0, 30.0, 100.0},
        {4.0, 40.0, 100.0}
    };

    ml::VarianceThreshold selector(2.0);

    const ml::Matrix result =
        selector.fit_transform(data);

    assert(result.rows == 4);
    assert(result.cols == 1);

    assert(almost_equal(result(0, 0), 10.0));
    assert(almost_equal(result(1, 0), 20.0));
    assert(almost_equal(result(2, 0), 30.0));
    assert(almost_equal(result(3, 0), 40.0));
}

void test_transform_after_fit() {
    ml::Matrix training_data{
        {1.0, 10.0},
        {2.0, 20.0},
        {3.0, 30.0}
    };

    ml::Matrix test_data{
        {4.0, 40.0},
        {5.0, 50.0}
    };

    ml::VarianceThreshold selector;

    selector.fit(training_data);

    const ml::Matrix result =
        selector.transform(test_data);

    assert(result.rows == 2);
    assert(result.cols == 2);

    assert(almost_equal(result(0, 0), 4.0));
    assert(almost_equal(result(0, 1), 40.0));
    assert(almost_equal(result(1, 0), 5.0));
    assert(almost_equal(result(1, 1), 50.0));
}

void test_transform_before_fit() {
    ml::Matrix data{
        {1.0, 2.0}
    };

    ml::VarianceThreshold selector;

    bool threw = false;

    try {
        selector.transform(data);
    } catch (const std::logic_error&) {
        threw = true;
    }

    assert(threw);
}

void test_variances_before_fit() {
    ml::VarianceThreshold selector;

    bool threw = false;

    try {
        selector.variances();
    } catch (const std::logic_error&) {
        threw = true;
    }

    assert(threw);
}

void test_column_mismatch() {
    ml::Matrix training_data{
        {1.0, 2.0},
        {2.0, 3.0},
        {3.0, 4.0}
    };

    ml::Matrix invalid_data{
        {1.0}
    };

    ml::VarianceThreshold selector;

    selector.fit(training_data);

    bool threw = false;

    try {
        selector.transform(invalid_data);
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

void test_empty_input() {
    ml::VarianceThreshold selector;

    bool threw = false;

    try {
        selector.fit(ml::Matrix{});
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

void test_negative_threshold() {
    bool threw = false;

    try {
        ml::VarianceThreshold selector(-1.0);
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

void test_no_selected_features() {
    ml::Matrix data{
        {1.0, 2.0},
        {1.0, 2.0},
        {1.0, 2.0}
    };

    ml::VarianceThreshold selector;

    bool threw = false;

    try {
        selector.fit(data);
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

} // namespace

int main() {
    test_variance_calculation();
    test_feature_selection();
    test_threshold();
    test_transform_after_fit();
    test_transform_before_fit();
    test_variances_before_fit();
    test_column_mismatch();
    test_empty_input();
    test_negative_threshold();
    test_no_selected_features();

    return 0;
}