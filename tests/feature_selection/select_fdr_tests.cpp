#include "ml/feature_selection/select_fdr.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {

ml::Matrix make_features()
{
    ml::Matrix X(10, 4);

    for (std::size_t row = 0;
         row < X.rows;
         ++row) {

        const double value =
            static_cast<double>(row + 1);

        X(row, 0) = value;
        X(row, 1) = 11.0 - value;
        X(row, 2) = std::sin(value);
        X(row, 3) = 1.0;
    }

    return X;
}

ml::Matrix make_target()
{
    ml::Matrix y(10, 1);

    for (std::size_t row = 0;
         row < y.rows;
         ++row) {

        y(row, 0) =
            static_cast<double>(row + 1);
    }

    return y;
}

void test_invalid_alpha_zero()
{
    bool thrown = false;

    try {
        ml::SelectFdr selector(0.0);
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
        ml::SelectFdr selector(-0.1);
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
        ml::SelectFdr selector(1.1);
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
        ml::SelectFdr selector(
            std::numeric_limits<double>::quiet_NaN()
        );
    }
    catch (const std::invalid_argument&) {
        thrown = true;
    }

    assert(thrown);
}

void test_transform_before_fit()
{
    const auto X = make_features();

    ml::SelectFdr selector(0.05);

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
    ml::SelectFdr selector(0.05);

    const auto& selected =
        selector.selected_features();

    assert(selected.empty());
}

void test_empty_features()
{
    ml::Matrix X(0, 4);
    const auto y = make_target();

    ml::SelectFdr selector(0.05);

    bool thrown = false;

    try {
        selector.fit(X, y);
    }
    catch (const std::invalid_argument&) {
        thrown = true;
    }

    assert(thrown);
}

void test_empty_target()
{
    const auto X = make_features();
    ml::Matrix y(0, 1);

    ml::SelectFdr selector(0.05);

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
    ml::Matrix X(10, 4);
    ml::Matrix y(5, 1);

    ml::SelectFdr selector(0.05);

    bool thrown = false;

    try {
        selector.fit(X, y);
    }
    catch (const std::invalid_argument&) {
        thrown = true;
    }

    assert(thrown);
}

void test_multiple_target_columns()
{
    const auto X = make_features();
    ml::Matrix y(10, 2);

    ml::SelectFdr selector(0.05);

    bool thrown = false;

    try {
        selector.fit(X, y);
    }
    catch (const std::invalid_argument&) {
        thrown = true;
    }

    assert(thrown);
}

void test_fit()
{
    const auto X = make_features();
    const auto y = make_target();

    ml::SelectFdr selector(0.05);

    selector.fit(X, y);

    const auto& selected =
        selector.selected_features();

    assert(!selected.empty());

    for (const std::size_t feature : selected) {
        assert(feature < X.cols);
    }
}

void test_transform()
{
    const auto X = make_features();
    const auto y = make_target();

    ml::SelectFdr selector(0.05);

    selector.fit(X, y);

    const auto& selected =
        selector.selected_features();

    const auto result =
        selector.transform(X);

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

    ml::SelectFdr selector(0.05);

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

} // namespace

int main()
{
    test_invalid_alpha_zero();
    test_invalid_alpha_negative();
    test_invalid_alpha_greater_than_one();
    test_invalid_alpha_nan();

    test_transform_before_fit();
    test_selected_features_before_fit();

    test_empty_features();
    test_empty_target();
    test_mismatched_rows();
    test_multiple_target_columns();

    test_fit();
    test_transform();
    test_fit_transform();

    std::cout
        << "All SelectFdr tests passed.\n";

    return 0;
}