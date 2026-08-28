#include "ml/feature_selection/generic_univariate_selector.hpp"

#include <cassert>
#include <iostream>
#include <stdexcept>

namespace {

ml::Matrix make_data()
{
    ml::Matrix data(5, 4);

    data(0, 0) = 1.0;
    data(1, 0) = 2.0;
    data(2, 0) = 3.0;
    data(3, 0) = 4.0;
    data(4, 0) = 5.0;

    data(0, 1) = 5.0;
    data(1, 1) = 4.0;
    data(2, 1) = 3.0;
    data(3, 1) = 2.0;
    data(4, 1) = 1.0;

    data(0, 2) = 1.0;
    data(1, 2) = 1.0;
    data(2, 2) = 1.0;
    data(3, 2) = 1.0;
    data(4, 2) = 1.0;

    data(0, 3) = 2.0;
    data(1, 3) = 2.0;
    data(2, 3) = 2.0;
    data(3, 3) = 2.0;
    data(4, 3) = 2.0;

    return data;
}

double score_function(
    const ml::Matrix&,
    std::size_t feature)
{
    return static_cast<double>(feature + 1);
}

void test_k_best()
{
    const auto data = make_data();

    ml::GenericUnivariateSelect selector(
        score_function,
        ml::SelectionMode::KBest,
        2.0
    );

    selector.fit(data);

    assert(selector.scores().size() == 4);
    assert(selector.selected_features().size() == 2);

    assert(selector.selected_features()[0] == 2);
    assert(selector.selected_features()[1] == 3);
}

void test_percentile()
{
    const auto data = make_data();

    ml::GenericUnivariateSelect selector(
        score_function,
        ml::SelectionMode::Percentile,
        50.0
    );

    selector.fit(data);

    assert(selector.selected_features().size() == 2);
    assert(selector.selected_features()[0] == 2);
    assert(selector.selected_features()[1] == 3);
}

void test_fit_transform()
{
    const auto data = make_data();

    ml::GenericUnivariateSelect selector(
        score_function,
        ml::SelectionMode::KBest,
        2.0
    );

    const auto result = selector.fit_transform(data);

    assert(result.rows == 5);
    assert(result.cols == 2);

    assert(result(0, 0) == data(0, 2));
    assert(result(0, 1) == data(0, 3));
}

void test_transform_before_fit()
{
    const auto data = make_data();

    ml::GenericUnivariateSelect selector(
        score_function,
        ml::SelectionMode::KBest,
        2.0
    );

    bool thrown = false;

    try {
        selector.transform(data);
    }
    catch (const std::logic_error&) {
        thrown = true;
    }

    assert(thrown);
}

void test_invalid_parameter()
{
    bool thrown = false;

    try {
        ml::GenericUnivariateSelect selector(
            score_function,
            ml::SelectionMode::KBest,
            -1.0
        );
    }
    catch (const std::invalid_argument&) {
        thrown = true;
    }

    assert(thrown);
}

void test_invalid_percentile()
{
    bool thrown = false;

    try {
        ml::GenericUnivariateSelect selector(
            score_function,
            ml::SelectionMode::Percentile,
            101.0
        );
    }
    catch (const std::invalid_argument&) {
        thrown = true;
    }

    assert(thrown);
}

void test_empty_input()
{
    ml::Matrix data(0, 2);

    ml::GenericUnivariateSelect selector(
        score_function,
        ml::SelectionMode::KBest,
        1.0
    );

    bool thrown = false;

    try {
        selector.fit(data);
    }
    catch (const std::invalid_argument&) {
        thrown = true;
    }

    assert(thrown);
}

} // namespace

int main()
{
    test_k_best();
    test_percentile();
    test_fit_transform();
    test_transform_before_fit();
    test_invalid_parameter();
    test_invalid_percentile();
    test_empty_input();

    std::cout << "All GenericUnivariateSelect tests passed.\n";

    return 0;
}
