#include "ml/algorithms/linear_regression.hpp"
#include "ml/pipeline/pipeline.hpp"
#include "ml/preprocessing/standard_scaler.hpp"

#include <cassert>
#include <cmath>
#include <vector>

int main()
{
    const ml::Matrix x{
        {0.0},
        {1.0},
        {2.0},
        {3.0}
    };

    const std::vector<double> y{
        1.0,
        3.0,
        5.0,
        7.0
    };

    auto pipeline = ml::make_pipeline(
        ml::LinearRegression{},
        ml::StandardScaler{}
    );

    pipeline.fit(x, y);

    const auto prediction =
        pipeline.predict(
            ml::Matrix{{4.0}}
        );

    assert(pipeline.is_fitted());

    assert(
        prediction.rows == 1 &&
        prediction.cols == 1 &&
        std::abs(prediction(0, 0) - 9.0) < 0.5
    );

    return 0;
}