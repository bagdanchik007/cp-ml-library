#include "ml/ml.hpp"
#include <iostream>
#include <iomanip>

int main() {
    using namespace ml;

    std::cout << "=== Cpp ML Library Demo ===\n\n";

    // -------------------------------------------------
    // 1. Linear Regression
    // -------------------------------------------------
    std::cout << "--- Linear Regression ---\n";

    // Simple dataset: y ≈ 2*x1 + 3*x2 + 5
    Matrix X_train = {
        {1.0, 1.0},
        {2.0, 1.0},
        {3.0, 2.0},
        {4.0, 3.0},
        {5.0, 3.0}
    };
    std::vector<double> y_train = {10.0, 12.0, 17.0, 22.0, 24.0};

    LinearRegression lr;
    lr.fit(X_train, y_train, /*lr=*/0.01, /*epochs=*/2000, /*verbose=*/false);

    std::cout << "Learned weights: ";
    for (double w : lr.weights())
        std::cout << std::fixed << std::setprecision(4) << w << " ";
    std::cout << "\nBias: " << lr.bias() << "\n";

    Matrix X_test = {{6.0, 4.0}, {7.0, 5.0}};
    auto preds = lr.predict(X_test);
    std::cout << "Predictions for test set: ";
    for (double p : preds)
        std::cout << p << " ";
    std::cout << "\nMSE on training set: " << lr.score(X_train, y_train) << "\n\n";

    // -------------------------------------------------
    // 2. K-Means
    // -------------------------------------------------
    std::cout << "--- K-Means Clustering ---\n";

    // Two obvious clusters
    Matrix data = {
        {1.0, 1.0},
        {1.5, 1.8},
        {1.2, 0.9},
        {8.0, 8.0},
        {8.5, 7.8},
        {7.9, 8.2},
        {1.1, 1.3},
        {8.1, 7.9}
    };

    KMeans kmeans(/*n_clusters=*/2, /*max_iters=*/100);
    auto labels = kmeans.fit_predict(data);

    std::cout << "Cluster labels: ";
    for (int lab : labels)
        std::cout << lab << " ";
    std::cout << "\n\nCentroids:\n";
    kmeans.centroids().print();

    std::cout << "\nDemo finished successfully.\n";
    return 0;
}
