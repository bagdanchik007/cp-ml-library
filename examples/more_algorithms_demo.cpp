#include "ml/ml.hpp"

#include <iomanip>
#include <iostream>

namespace ml::examples {

void run_more_algorithms_demo() {
    std::cout << "--- Perceptron ---\n";

    const Matrix perceptron_train{{0.0, 0.0}, {0.0, 1.0}, {2.0, 2.0}, {2.0, 3.0}};
    const std::vector<int> perceptron_labels{0, 0, 1, 1};

    Perceptron perceptron;
    perceptron.fit(perceptron_train, perceptron_labels, 0.1, 50);
    const auto perceptron_predictions = perceptron.predict(Matrix{{0.5, 0.5}, {2.5, 2.5}});
    std::cout << "Predicted labels: " << perceptron_predictions[0] << ", " << perceptron_predictions[1] << "\n\n";

    std::cout << "--- Softmax Regression (Multiclass) ---\n";

    const Matrix softmax_train{{-5.0}, {-4.5}, {0.0}, {0.5}, {5.0}, {4.5}};
    const std::vector<int> softmax_labels{0, 0, 1, 1, 2, 2};

    SoftmaxRegression softmax;
    softmax.fit(softmax_train, softmax_labels, 3, 0.5, 1500);
    const auto softmax_predictions = softmax.predict(Matrix{{-5.0}, {0.2}, {5.0}});
    std::cout << "Predicted classes: "
              << softmax_predictions[0] << ", " << softmax_predictions[1] << ", " << softmax_predictions[2] << "\n\n";

    std::cout << "--- Lasso Regression ---\n";

    const Matrix lasso_train{{0.0, 1.0}, {1.0, -1.0}, {2.0, 2.0}, {3.0, -2.0}};
    const std::vector<double> lasso_targets{2.0, 5.0, 8.0, 11.0};

    LassoRegression lasso(0.01);
    lasso.fit(lasso_train, lasso_targets, 5000);
    const auto lasso_prediction = lasso.predict(Matrix{{4.0, 0.0}})[0];
    std::cout << "Prediction for (4.0, 0.0): " << std::fixed << std::setprecision(4) << lasso_prediction << "\n\n";

    std::cout << "--- KNN Regressor ---\n";

    const Matrix knn_regressor_train{{0.0}, {1.0}, {2.0}, {3.0}, {4.0}};
    const std::vector<double> knn_regressor_targets{0.0, 2.0, 4.0, 6.0, 8.0};

    KNNRegressor knn_regressor(3);
    knn_regressor.fit(knn_regressor_train, knn_regressor_targets);
    const auto knn_regressor_prediction = knn_regressor.predict(Matrix{{2.1}})[0];
    std::cout << "Prediction for x=2.1: " << std::fixed << std::setprecision(4) << knn_regressor_prediction << "\n\n";

    std::cout << "--- Dummy Baselines ---\n";

    DummyClassifier dummy_classifier("most_frequent");
    dummy_classifier.fit({1, 1, 1, 0});
    const auto dummy_classifier_predictions = dummy_classifier.predict(3);
    std::cout << "Dummy classifier predictions: "
              << dummy_classifier_predictions[0] << ", "
              << dummy_classifier_predictions[1] << ", "
              << dummy_classifier_predictions[2] << "\n";

    DummyRegressor dummy_regressor("mean");
    dummy_regressor.fit({1.0, 2.0, 3.0, 4.0});
    std::cout << "Dummy regressor prediction: " << dummy_regressor.prediction_value() << "\n";
}

} // namespace ml::examples
