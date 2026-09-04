#include "ml/ml.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using namespace ml;

std::vector<double> matrix_to_vector(
    const Matrix& matrix
)
{
    assert(matrix.cols == 1);

    std::vector<double> result;

    result.reserve(matrix.rows);

    for (size_t row = 0;
         row < matrix.rows;
         ++row)
    {
        result.push_back(
            matrix(row, 0)
        );
    }

    return result;
}

void test_train_test_split_with_model_evaluation()
{
    std::cout
        << "[TEST] Dataset split + model evaluation integration ... ";

    Dataset dataset(
        Matrix{
            {1.0},
            {2.0},
            {3.0},
            {4.0},
            {5.0},
            {6.0},
            {7.0},
            {8.0}
        },
        std::vector<double>{
            3.0,
            5.0,
            7.0,
            9.0,
            11.0,
            13.0,
            15.0,
            17.0
        }
    );

    const DatasetSplit split =
        train_test_split(
            dataset,
            0.25,
            false
        );

    assert(split.train.rows() == 6);
    assert(split.test.rows() == 2);

    LinearRegression model;

    model.fit(
        split.train.features(),
        split.train.targets(),
        0.03,
        5000
    );

    const Matrix predictions_matrix =
        model.predict(
            split.test.features()
        );

    const std::vector<double> predictions =
        matrix_to_vector(
            predictions_matrix
        );

    assert(
        predictions.size() ==
        split.test.targets().size()
    );

    assert(
        mean_squared_error(
            split.test.targets(),
            predictions
        ) < 0.15
    );

    assert(
        mean_absolute_error(
            split.test.targets(),
            predictions
        ) < 0.35
    );

    assert(
        r2_score(
            split.test.targets(),
            predictions
        ) > 0.95
    );

    std::cout << "OK\n";
}

int main()
{
    std::cout
        << "Running train/test integration tests...\n\n";

    test_train_test_split_with_model_evaluation();

    std::cout
        << "\nAll train/test integration tests passed!\n";

    return 0;
}