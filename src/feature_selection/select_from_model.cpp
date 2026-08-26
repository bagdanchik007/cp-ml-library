#include "ml/feature_selection/select_from_model.hpp"
#include "ml/core/matrix/matrix.hpp"

#include <cassert>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void test_select_features()
{
    ml::SelectFromModel selector(0.5);

    const std::vector<double> scores = {
        0.2,
        0.8,
        0.5,
        0.1
    };

    selector.fit(scores);

    const auto& selected =
        selector.selected_features();

    assert(selected.size() == 2);
    assert(selected[0] == 1);
    assert(selected[1] == 2);
}

void test_transform()
{
    ml::Matrix data(2, 3);

    data(0, 0) = 1.0;
    data(0, 1) = 2.0;
    data(0, 2) = 3.0;

    data(1, 0) = 4.0;
    data(1, 1) = 5.0;
    data(1, 2) = 6.0;

    ml::SelectFromModel selector(0.5);

    selector.fit({
        0.2,
        0.8,
        0.6
    });

    const ml::Matrix result =
        selector.transform(data);

    assert(result.rows == 2);
    assert(result.cols == 2);

    assert(result(0, 0) == 2.0);
    assert(result(0, 1) == 3.0);

    assert(result(1, 0) == 5.0);
    assert(result(1, 1) == 6.0);
}

void test_fit_transform()
{
    ml::Matrix data(2, 3);

    data(0, 0) = 1.0;
    data(0, 1) = 2.0;
    data(0, 2) = 3.0;

    data(1, 0) = 4.0;
    data(1, 1) = 5.0;
    data(1, 2) = 6.0;

    ml::SelectFromModel selector(0.7);

    const ml::Matrix result =
        selector.fit_transform(
            data,
            {
                0.9,
                0.4,
                0.8
            }
        );

    assert(result.rows == 2);
    assert(result.cols == 2);

    assert(result(0, 0) == 1.0);
    assert(result(0, 1) == 3.0);

    assert(result(1, 0) == 4.0);
    assert(result(1, 1) == 6.0);
}

void test_transform_before_fit()
{
    ml::Matrix data(1, 1);
    data(0, 0) = 1.0;

    ml::SelectFromModel selector(0.5);

    bool thrown = false;

    try {
        selector.transform(data);
    }
    catch (const std::logic_error&) {
        thrown = true;
    }

    assert(thrown);
}

void test_empty_scores()
{
    ml::SelectFromModel selector(0.5);

    bool thrown = false;

    try {
        selector.fit({});
    }
    catch (const std::invalid_argument&) {
        thrown = true;
    }

    assert(thrown);
}

void test_invalid_score_count()
{
    ml::Matrix data(2, 3);

    ml::SelectFromModel selector(0.5);

    bool thrown = false;

    try {
        selector.fit_transform(
            data,
            {
                0.5,
                0.8
            }
        );
    }
    catch (const std::invalid_argument&) {
        thrown = true;
    }

    assert(thrown);
}

} // namespace

int main()
{
    test_select_features();
    test_transform();
    test_fit_transform();
    test_transform_before_fit();
    test_empty_scores();
    test_invalid_score_count();

    std::cout
        << "All SelectFromModel tests passed.\n";

    return 0;
}