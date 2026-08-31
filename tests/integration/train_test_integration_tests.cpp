#include "ml/ml.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using namespace ml;

void test_train_test_split_with_model_evaluation() {
    std::cout << "[TEST] Dataset split + model evaluation integration ... ";

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
        train_test_split(dataset, 0.25, false);

    assert(split.train.rows() == 6);
    assert(split.test.rows() == 2);

    LinearRegression model;
    model.fit(
        split.train.features(),
        split.train.targets(),
        0.03,
        5000
    );

const Matrix prediction_matrix =
    model.predict(split.test.features());

std::vector<double> predictions;

predictions.reserve(
    prediction_matrix.rows
);

for (size_t i = 0;
     i < prediction_matrix.rows;
     ++i) {

    predictions.push_back(
        prediction_matrix(i, 0)
    );
}

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
}

int main() {
    std::cout << "Running train/test integration tests...\n\n";

    test_train_test_split_with_model_evaluation();

    std::cout << "\nAll train/test integration tests passed!\n";

    return 0;
}
