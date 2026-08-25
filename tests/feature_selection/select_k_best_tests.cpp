#include "ml/feature_selection/select_k_best.hpp"

#include <cassert>
#include <stdexcept>

namespace {

double score_function(
    const ml::Matrix& data,
    size_t feature
) {
    (void)data;

    static const double scores[] = {
        10.0,
        50.0,
        20.0,
        40.0
    };

    return scores[feature];
}

void test_select_top_k() {
    ml::Matrix data{
        {1.0, 10.0, 100.0, 1000.0},
        {2.0, 20.0, 200.0, 2000.0}
    };

    ml::SelectKBest selector(
        2,
        score_function
    );

    const ml::Matrix result =
        selector.fit_transform(data);

    assert(result.rows == 2);
    assert(result.cols == 2);

    assert(result(0, 0) == 10.0);
    assert(result(0, 1) == 1000.0);

    assert(result(1, 0) == 20.0);
    assert(result(1, 1) == 2000.0);
}

void test_scores() {
    ml::Matrix data{
        {1.0, 2.0, 3.0, 4.0}
    };

    ml::SelectKBest selector(
        2,
        score_function
    );

    selector.fit(data);

    const auto& scores =
        selector.scores();

    assert(scores.size() == 4);

    assert(scores[0] == 10.0);
    assert(scores[1] == 50.0);
    assert(scores[2] == 20.0);
    assert(scores[3] == 40.0);
}

void test_selected_features() {
    ml::Matrix data{
        {1.0, 2.0, 3.0, 4.0}
    };

    ml::SelectKBest selector(
        2,
        score_function
    );

    selector.fit(data);

    const auto& selected =
        selector.selected_features();

    assert(selected.size() == 2);

    assert(selected[0] == 1);
    assert(selected[1] == 3);
}

void test_transform_after_fit() {
    ml::Matrix training{
        {1.0, 10.0, 100.0, 1000.0},
        {2.0, 20.0, 200.0, 2000.0}
    };

    ml::Matrix test{
        {3.0, 30.0, 300.0, 3000.0}
    };

    ml::SelectKBest selector(
        2,
        score_function
    );

    selector.fit(training);

    const ml::Matrix result =
        selector.transform(test);

    assert(result.rows == 1);
    assert(result.cols == 2);

    assert(result(0, 0) == 30.0);
    assert(result(0, 1) == 3000.0);
}

void test_transform_before_fit() {
    ml::Matrix data{
        {1.0, 2.0}
    };

    ml::SelectKBest selector(
        1,
        score_function
    );

    bool threw = false;

    try {
        selector.transform(data);
    } catch (const std::logic_error&) {
        threw = true;
    }

    assert(threw);
}

void test_scores_before_fit() {
    ml::SelectKBest selector(
        1,
        score_function
    );

    bool threw = false;

    try {
        selector.scores();
    } catch (const std::logic_error&) {
        threw = true;
    }

    assert(threw);
}

void test_invalid_k_zero() {
    bool threw = false;

    try {
        ml::SelectKBest selector(
            0,
            score_function
        );
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

void test_invalid_k_too_large() {
    ml::Matrix data{
        {1.0, 2.0},
        {3.0, 4.0}
    };

    ml::SelectKBest selector(
        3,
        score_function
    );

    bool threw = false;

    try {
        selector.fit(data);
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

void test_invalid_score_function() {
    bool threw = false;

    try {
        ml::SelectKBest selector(
            1,
            {}
        );
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

void test_empty_input() {
    ml::SelectKBest selector(
        1,
        score_function
    );

    bool threw = false;

    try {
        selector.fit(ml::Matrix{});
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

} // namespace

int main() {
    test_select_top_k();
    test_scores();
    test_selected_features();
    test_transform_after_fit();
    test_transform_before_fit();
    test_scores_before_fit();
    test_invalid_k_zero();
    test_invalid_k_too_large();
    test_invalid_score_function();
    test_empty_input();

    return 0;
}