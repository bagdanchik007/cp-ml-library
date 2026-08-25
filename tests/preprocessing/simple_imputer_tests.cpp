#include "ml/preprocessing/simple_imputer.hpp"

#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace {

bool almost_equal(
    double a,
    double b,
    double epsilon = 1e-9
) {
    return std::abs(a - b) < epsilon;
}

void test_mean_imputation() {
    const double nan =
        std::numeric_limits<double>::quiet_NaN();

    ml::Matrix data{
        {1.0, nan},
        {3.0, 10.0},
        {5.0, 20.0}
    };

    ml::SimpleImputer imputer(
        ml::ImputationStrategy::Mean
    );

    const ml::Matrix result =
        imputer.fit_transform(data);

    assert(almost_equal(result(0, 0), 1.0));
    assert(almost_equal(result(0, 1), 15.0));
    assert(almost_equal(result(1, 1), 10.0));
    assert(almost_equal(result(2, 1), 20.0));
}

void test_median_imputation() {
    const double nan =
        std::numeric_limits<double>::quiet_NaN();

    ml::Matrix data{
        {1.0},
        {nan},
        {5.0},
        {9.0}
    };

    ml::SimpleImputer imputer(
        ml::ImputationStrategy::Median
    );

    const ml::Matrix result =
        imputer.fit_transform(data);

    assert(almost_equal(result(0, 0), 1.0));
    assert(almost_equal(result(1, 0), 5.0));
    assert(almost_equal(result(2, 0), 5.0));
    assert(almost_equal(result(3, 0), 9.0));
}

void test_constant_imputation() {
    const double nan =
        std::numeric_limits<double>::quiet_NaN();

    ml::Matrix data{
        {1.0, nan},
        {nan, 4.0}
    };

    ml::SimpleImputer imputer(
        ml::ImputationStrategy::Constant,
        -1.0
    );

    const ml::Matrix result =
        imputer.fit_transform(data);

    assert(almost_equal(result(0, 0), 1.0));
    assert(almost_equal(result(0, 1), -1.0));
    assert(almost_equal(result(1, 0), -1.0));
    assert(almost_equal(result(1, 1), 4.0));
}

void test_transform_before_fit() {
    ml::SimpleImputer imputer;

    ml::Matrix data{
        {1.0}
    };

    bool threw = false;

    try {
        imputer.transform(data);
    } catch (const std::logic_error&) {
        threw = true;
    }

    assert(threw);
}

void test_empty_input() {
    ml::SimpleImputer imputer;

    bool threw = false;

    try {
        imputer.fit(ml::Matrix{});
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

void test_all_missing_column() {
    const double nan =
        std::numeric_limits<double>::quiet_NaN();

    ml::Matrix data{
        {nan},
        {nan}
    };

    ml::SimpleImputer imputer(
        ml::ImputationStrategy::Mean
    );

    bool threw = false;

    try {
        imputer.fit(data);
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

void test_column_mismatch() {
    ml::Matrix training_data{
        {1.0, 2.0},
        {3.0, 4.0}
    };

    ml::SimpleImputer imputer;

    imputer.fit(training_data);

    ml::Matrix invalid_data{
        {1.0}
    };

    bool threw = false;

    try {
        imputer.transform(invalid_data);
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

} // namespace

int main() {
    test_mean_imputation();
    test_median_imputation();
    test_constant_imputation();
    test_transform_before_fit();
    test_empty_input();
    test_all_missing_column();
    test_column_mismatch();

    return 0;
}