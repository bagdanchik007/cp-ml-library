#include "ml/ml.hpp"

#include <iomanip>
#include <iostream>

namespace ml::examples {

void run_new_algorithms_demo() {
    std::cout << "--- KNN Classifier ---\n";

    const Matrix knn_train{
        {0.0, 0.0}, {0.2, 0.1}, {0.1, 0.2},
        {5.0, 5.0}, {5.2, 4.9}, {4.9, 5.1}
    };
    const std::vector<int> knn_labels{0, 0, 0, 1, 1, 1};

    KNNClassifier knn(3);
    knn.fit(knn_train, knn_labels);
    const auto knn_predictions = knn.predict(Matrix{{0.1, 0.1}, {5.1, 5.0}});
    std::cout << "Predicted labels: " << knn_predictions[0] << ", " << knn_predictions[1] << "\n\n";

    std::cout << "--- Gaussian Naive Bayes ---\n";

    GaussianNaiveBayes naive_bayes;
    naive_bayes.fit(knn_train, knn_labels);
    const auto nb_predictions = naive_bayes.predict(Matrix{{0.05, 0.05}, {5.05, 5.05}});
    std::cout << "Predicted labels: " << nb_predictions[0] << ", " << nb_predictions[1] << "\n\n";

    std::cout << "--- Logistic Regression ---\n";

    const Matrix logistic_train{{-2.0}, {-1.0}, {1.0}, {2.0}};
    const std::vector<int> logistic_labels{0, 0, 1, 1};

    LogisticRegression logistic;
    logistic.fit(logistic_train, logistic_labels, 0.5, 2000);
    const auto probability = logistic.predict_proba(Matrix{{3.0}})[0];
    std::cout << "P(class=1 | x=3.0) = " << std::fixed << std::setprecision(4) << probability << "\n\n";

    std::cout << "--- Ridge Regression ---\n";

    const Matrix ridge_train{{0.0}, {1.0}, {2.0}, {3.0}};
    const std::vector<double> ridge_targets{1.0, 3.0, 5.0, 7.0};

    RidgeRegression ridge(0.1);
    ridge.fit(ridge_train, ridge_targets);
    const auto ridge_prediction = ridge.predict(Matrix{{4.0}})[0];
    std::cout << "Prediction for x=4.0: " << std::fixed << std::setprecision(4) << ridge_prediction << "\n";
}

} // namespace ml::examples
