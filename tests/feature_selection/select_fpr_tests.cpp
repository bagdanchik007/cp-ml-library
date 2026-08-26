#include "ml/feature_selection/select_fpr.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {

ml::Matrix make_features()
{
    ml::Matrix X(6, 4);

    // Strongly correlated with y.
    X(0, 0) = 1.0;
    X(1, 0) = 2.0;
    X(2, 0) = 3.0;
    X(3, 0) = 4.0;
    X(4, 0) = 5.0;
    X(5, 0) = 6.0;

    // Strongly negatively correlated with y.
    X(0, 1) = 6.0;
    X(1, 1) = 5.0;
    X(2, 1) = 4.0;
    X(3, 1) = 3.0;
    X(4, 1) = 2.0;
    X(5, 1) = 1.0;

    // Constant feature.
    X(0, 2) = 1.0;
    X(1, 2) = 1.0;
    X(2, 2) = 1.0;
    X(3, 2) = 1.0;
    X(4, 2) = 1.0;
    X(5, 2) = 1.0;

    // Weakly related / non-linear feature.
    X(0, 3) = 1.0;
    X(1, 3) = 1.0;
    X(2, 3) = 2.0;
    X(3, 3) = 2.0;
    X(4, 3) = 1.0;
    X(5, 3) = 1.0;

    return X;
}

ml::Matrix make_target()
{
    ml::Matrix y(6, 1);

    y(0, 0) = 1.0;
    y(1, 0) = 2.0;
    y(2, 0) = 3.0;
    y(3, 0) = 4.0;
    y(4, 0) = 5.0;
    y(5, 0) = 6.0;

    return y;
}

void test_invalid_alpha_zero()
{
    bool thrown = false;

    try {
        ml::SelectFpr selector(0.0);
    }
    catch (const std::invalid_argument&) {
        thrown = true;
    }

    assert(thrown);
}

void test_invalid_alpha_negative()
{
    bool thrown = false;

    try {
        ml::SelectFpr selector(-0.1);
    }
    catch (const std::invalid_argument&) {
        thrown = true;
    }

    assert(thrown);
}

void test_invalid_alpha_greater_than_one()
{
    bool thrown = false;

    try {
        ml::SelectFpr selector(1.1);
    }
    catch (const std::invalid_argument&) {
        thrown = true;
    }

    assert(thrown);
}

void test_invalid_alpha_nan()
{
    bool thrown = false;

    try {
        ml::SelectFpr selector(
            std::numeric_limits<double>::quiet_NaN()
        );
    }
    catch (const std::invalid_argument&) {
        thrown = true;
    }

    assert(thrown);
}

void test_empty_input()
{
    ml::Matrix X(0, 2);
    ml::Matrix y(0, 1);

    ml::SelectFpr selector(0.05);

    bool thrown = false;

    try {
        selector.fit(X, y);
    }
    catch (const std::invalid_argument&) {
        thrown = true;
    }

    assert(thrown);
}

void test_mismatched_rows()
{
    ml::Matrix X(5, 2);
    ml::Matrix y(4, 1);

    ml::SelectFpr selector(0.05);

    bool thrown = false;

    try {
        selector.fit(X, y);
    }
    catch (const std::invalid_argument&) {
        thrown = true;
    }

    assert(thrown);
}

void test_invalid_target_shape()
{
    ml::Matrix X(5, 2);
    ml::Matrix y(5, 2);

    ml::SelectFpr selector(0.05);

    bool thrown = false;

    try {
        selector.fit(X, y);
    }
    catch (const std::invalid_argument&) {
        thrown = true;
    }

    assert(thrown);
}

void test_selects_significant_features()
{
    const auto X = make_features();
    const auto y = make_target();

    ml::SelectFpr selector(0.05);

    selector.fit(X, y);

    const auto& selected =
        selector.selected_features();

    assert(!selected.empty());

    assert(selected[0] == 0);
    assert(selected.size() >= 2);
    assert(selected[1] == 1);
}

void test_transform()
{
    const auto X = make_features();
    const auto y = make_target();

    ml::SelectFpr selector(0.05);

    selector.fit(X, y);

    const auto result =
        selector.transform(X);

    const auto& selected =
        selector.selected_features();

    assert(result.rows == X.rows);
    assert(result.cols == selected.size());

    for (std::size_t row = 0;
         row < result.rows;
         ++row) {

        for (std::size_t column = 0;
             column < result.cols;
             ++column) {

            assert(
                result(row, column) ==
                X(row, selected[column])
            );
        }
    }
}

void test_fit_transform()
{
    const auto X = make_features();
    const auto y = make_target();

    ml::SelectFpr selector(0.05);

    const auto result =
        selector.fit_transform(X, y);

    const auto& selected =
        selector.selected_features();

    assert(result.rows == X.rows);
    assert(result.cols == selected.size());

    for (std::size_t row = 0;
         row < result.rows;
         ++row) {

        for (std::size_t column = 0;
             column < result.cols;
             ++column) {

            assert(
                result(row, column) ==
                X(row, selected[column])
            );
        }
    }
}

void test_transform_before_fit()
{
    const auto X = make_features();

    ml::SelectFpr selector(0.05);

    bool thrown = false;

    try {
        selector.transform(X);
    }
    catch (const std::logic_error&) {
        thrown = true;
    }

    assert(thrown);
}

void test_selected_features_before_fit()
{
    ml::SelectFpr selector(0.05);

    const auto& selected =
        selector.selected_features();

    assert(selected.empty());
}

} // namespace

int main()
{
    test_invalid_alpha_zero();
    test_invalid_alpha_negative();
    test_invalid_alpha_greater_than_one();
    test_invalid_alpha_nan();

    test_empty_input();
    test_mismatched_rows();
    test_invalid_target_shape();

    test_selects_significant_features();
    test_transform();
    test_fit_transform();
    test_transform_before_fit();
    test_selected_features_before_fit();

    std::cout
        << "All SelectFpr tests passed.\n";

    return 0;
}
