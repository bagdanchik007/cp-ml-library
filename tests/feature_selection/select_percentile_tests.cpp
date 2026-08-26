#include "ml/feature_selection/select_percentile.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace {

ml::Matrix make_data()
{
    return ml::Matrix{
        {1.0, 10.0, 100.0, 1000.0},
        {2.0, 20.0, 200.0, 2000.0},
        {3.0, 30.0, 300.0, 3000.0},
        {4.0, 40.0, 400.0, 4000.0}
    };
}

ml::Matrix make_target()
{
    return ml::Matrix{
        {1.0},
        {2.0},
        {3.0},
        {4.0}
    };
}

void test_select_percentile()
{
    const ml::Matrix X = make_data();
    const ml::Matrix y = make_target();

    ml::SelectPercentile selector(50.0);

    const ml::Matrix result =
        selector.fit_transform(X, y);

    assert(result.rows == 4);
    assert(result.cols == 2);
}

void test_scores()
{
    const ml::Matrix X = make_data();
    const ml::Matrix y = make_target();

    ml::SelectPercentile selector(50.0);

    selector.fit(X, y);

    const auto& scores =
        selector.scores();

    assert(scores.size() == 4);

    for (const double score : scores) {
        // Perfectly correlated features can produce an infinite
        // F-statistic. NaN is never acceptable.
        assert(!std::isnan(score));
        assert(score >= 0.0 || std::isinf(score));
    }
}

void test_selected_features()
{
    const ml::Matrix X = make_data();
    const ml::Matrix y = make_target();

    ml::SelectPercentile selector(50.0);

    selector.fit(X, y);

    const auto& selected =
        selector.selected_features();

    assert(selected.size() == 2);

    assert(selected[0] < X.cols);
    assert(selected[1] < X.cols);

    assert(selected[0] != selected[1]);
}

void test_transform_after_fit()
{
    const ml::Matrix training = make_data();

    const ml::Matrix y = make_target();

    const ml::Matrix test{
        {5.0, 50.0, 500.0, 5000.0},
        {6.0, 60.0, 600.0, 6000.0}
    };

    ml::SelectPercentile selector(50.0);

    selector.fit(training, y);

    const ml::Matrix result =
        selector.transform(test);

    assert(result.rows == 2);
    assert(result.cols == 2);
}

void test_transform_before_fit()
{
    const ml::Matrix X = make_data();

    ml::SelectPercentile selector(50.0);

    bool threw = false;

    try {
        selector.transform(X);
    }
    catch (const std::logic_error&) {
        threw = true;
    }

    assert(threw);
}

void test_scores_before_fit()
{
    ml::SelectPercentile selector(50.0);

    bool threw = false;

    try {
        selector.scores();
    }
    catch (const std::logic_error&) {
        threw = true;
    }

    assert(threw);
}

void test_selected_features_before_fit()
{
    ml::SelectPercentile selector(50.0);

    bool threw = false;

    try {
        selector.selected_features();
    }
    catch (const std::logic_error&) {
        threw = true;
    }

    assert(threw);
}

void test_invalid_percentile_zero()
{
    bool threw = false;

    try {
        ml::SelectPercentile selector(0.0);
    }
    catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

void test_invalid_percentile_above_hundred()
{
    bool threw = false;

    try {
        ml::SelectPercentile selector(100.1);
    }
    catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

void test_empty_input()
{
    ml::SelectPercentile selector(50.0);

    bool threw = false;

    try {
        selector.fit(
            ml::Matrix{},
            ml::Matrix{}
        );
    }
    catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

void test_mismatched_rows()
{
    const ml::Matrix X = make_data();

    const ml::Matrix y{
        {1.0},
        {2.0}
    };

    ml::SelectPercentile selector(50.0);

    bool threw = false;

    try {
        selector.fit(X, y);
    }
    catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

void test_invalid_target_shape()
{
    const ml::Matrix X = make_data();

    const ml::Matrix y{
        {1.0, 2.0},
        {3.0, 4.0},
        {5.0, 6.0},
        {7.0, 8.0}
    };

    ml::SelectPercentile selector(50.0);

    bool threw = false;

    try {
        selector.fit(X, y);
    }
    catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

void test_hundred_percentile()
{
    const ml::Matrix X = make_data();
    const ml::Matrix y = make_target();

    ml::SelectPercentile selector(100.0);

    const ml::Matrix result =
        selector.fit_transform(X, y);

    assert(result.rows == X.rows);
    assert(result.cols == X.cols);

    const auto& selected =
        selector.selected_features();

    assert(selected.size() == X.cols);

    for (std::size_t i = 0; i < X.cols; ++i) {
        assert(selected[i] == i);
    }
}

} // namespace

int main()
{
    test_select_percentile();
    test_scores();
    test_selected_features();
    test_transform_after_fit();
    test_transform_before_fit();
    test_scores_before_fit();
    test_selected_features_before_fit();
    test_invalid_percentile_zero();
    test_invalid_percentile_above_hundred();
    test_empty_input();
    test_mismatched_rows();
    test_invalid_target_shape();
    test_hundred_percentile();

    return 0;
}
