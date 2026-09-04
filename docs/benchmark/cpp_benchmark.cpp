#include "ml/algorithms/logistic_regression.hpp"
#include "ml/algorithms/ridge_regression.hpp"
#include "ml/algorithms/lasso_regression.hpp"
#include "ml/metrics/model_evaluation.hpp"

#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

ml::Matrix load_matrix(const std::string& path) {
    std::ifstream file(path);
    std::vector<std::vector<double>> rows;
    std::string line;
    while (std::getline(file, line)) {
        std::vector<double> row;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')) row.push_back(std::stod(cell));
        rows.push_back(row);
    }
    ml::Matrix result(rows.size(), rows.empty() ? 0 : rows[0].size());
    for (size_t i = 0; i < rows.size(); ++i)
        for (size_t j = 0; j < rows[i].size(); ++j) result(i, j) = rows[i][j];
    return result;
}

std::vector<double> load_vector(const std::string& path) {
    std::ifstream file(path);
    std::vector<double> values;
    double value;
    while (file >> value) values.push_back(value);
    return values;
}

int main() {
    using clock = std::chrono::high_resolution_clock;

    // --- Logistic Regression ---
    {
        auto X = load_matrix("classification_X.csv");
        auto y_double = load_vector("classification_y.csv");
        std::vector<int> y(y_double.begin(), y_double.end());

        ml::LogisticRegression model;
        auto t0 = clock::now();
        model.fit(X, y, 0.5, 20000);
        auto t1 = clock::now();

        auto predictions = model.predict(X);
        double accuracy = ml::accuracy_score(y, predictions);

        std::cout << "=== LogisticRegression (C++) ===\n";
        std::cout << "weights: " << model.weights()[0] << ", " << model.weights()[1] << "\n";
        std::cout << "bias: " << model.bias() << "\n";
        std::cout << "accuracy: " << accuracy << "\n";
        std::cout << "fit_seconds: " << std::chrono::duration<double>(t1 - t0).count() << "\n\n";
    }

    // --- Ridge Regression ---
    {
        auto X = load_matrix("regression_X.csv");
        auto y = load_vector("regression_y.csv");

        ml::RidgeRegression model(1.0);
        auto t0 = clock::now();
        model.fit(X, y);
        auto t1 = clock::now();

        auto predictions = model.predict(X);
        double mse = ml::mean_squared_error(y, predictions);

        std::cout << "=== RidgeRegression (C++) ===\n";
        std::cout << "weights: ";
        for (double w : model.weights()) std::cout << w << " ";
        std::cout << "\nbias: " << model.bias() << "\n";
        std::cout << "mse: " << mse << "\n";
        std::cout << "fit_seconds: " << std::chrono::duration<double>(t1 - t0).count() << "\n\n";
    }

    // --- Lasso Regression ---
    {
        auto X = load_matrix("regression_X.csv");
        auto y = load_vector("regression_y.csv");

        ml::LassoRegression model(0.1);
        auto t0 = clock::now();
        model.fit(X, y, 5000);
        auto t1 = clock::now();

        auto predictions = model.predict(X);
        double mse = ml::mean_squared_error(y, predictions);
        int n_nonzero = 0;
        for (double w : model.weights()) if (std::abs(w) > 1e-6) ++n_nonzero;

        std::cout << "=== LassoRegression (C++) ===\n";
        std::cout << "weights: ";
        for (double w : model.weights()) std::cout << w << " ";
        std::cout << "\nbias: " << model.bias() << "\n";
        std::cout << "mse: " << mse << "\n";
        std::cout << "n_nonzero: " << n_nonzero << "\n";
        std::cout << "fit_seconds: " << std::chrono::duration<double>(t1 - t0).count() << "\n";
    }
}
