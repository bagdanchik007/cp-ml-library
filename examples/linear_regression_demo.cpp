#include "ml/ml.hpp"

#include <iomanip>
#include <iostream>

namespace ml::examples {

void run_linear_regression_demo() {
    std::cout << "--- Linear Regression ---\n";

    // Simple dataset: y ≈ 2*x1 + 3*x2 + 5
    Matrix X_train = {
        {1.0, 1.0},
        {2.0, 1.0},
        {3.0, 2.0},
        {4.0, 3.0},
        {5.0, 3.0}
    };

    std::vector<double> y_train = {
        10.0, 12.0, 17.0, 22.0, 24.0
    };

    LinearRegression model;
    model.fit(X_train, y_train, 0.01, 2000, false);

    std::cout << "Learned weights: ";

    for (double weight : model.weights()) {
        std::cout << std::fixed
                  << std::setprecision(4)
                  << weight << " ";
    }

    std::cout << "\nBias: " << model.bias() << '\n';

    Matrix X_test = {
        {6.0, 4.0},
        {7.0, 5.0}
    };

    const auto predictions = model.predict(X_test);

    std::cout << "Predictions for test set: ";

    for (double prediction : predictions) {
        std::cout << prediction << " ";
    }

    std::cout << "\nMSE on training set: "
              << model.score(X_train, y_train)
              << "\n\n";
}

} // namespace ml::examples