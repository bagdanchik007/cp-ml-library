#include "ml/ml.hpp"

#include <iomanip>
#include <iostream>

namespace ml::examples {

void run_third_round_algorithms_demo() {
    std::cout << "--- Agglomerative Clustering ---\n";

    const Matrix cluster_data{{0.0, 0.0}, {0.1, 0.1}, {10.0, 10.0}, {10.1, 9.9}};
    AgglomerativeClustering clustering(2);
    const auto cluster_labels = clustering.fit_predict(cluster_data);
    std::cout << "Cluster labels: "
              << cluster_labels[0] << " " << cluster_labels[1] << " "
              << cluster_labels[2] << " " << cluster_labels[3] << "\n\n";

    std::cout << "--- Elastic Net ---\n";

    const Matrix elastic_train{{0.0, 1.0}, {1.0, -1.0}, {2.0, 2.0}, {3.0, -2.0}};
    const std::vector<double> elastic_targets{2.0, 5.0, 8.0, 11.0};
    ElasticNet elastic_net(0.01, 0.5);
    elastic_net.fit(elastic_train, elastic_targets, 5000);
    std::cout << "Prediction for (4.0, 0.0): "
              << std::fixed << std::setprecision(4) << elastic_net.predict(Matrix{{4.0, 0.0}})[0] << "\n\n";

    std::cout << "--- Multinomial Naive Bayes ---\n";

    const Matrix word_counts{{5.0, 0.0}, {6.0, 0.0}, {0.0, 5.0}, {0.0, 6.0}};
    const std::vector<int> topic_labels{0, 0, 1, 1};
    MultinomialNaiveBayes multinomial_nb;
    multinomial_nb.fit(word_counts, topic_labels);
    const auto topic_prediction = multinomial_nb.predict(Matrix{{4.0, 0.0}})[0];
    std::cout << "Predicted topic: " << topic_prediction << "\n\n";

    std::cout << "--- Weighted KNN Classifier ---\n";

    const Matrix weighted_knn_train{{0.0, 0.0}, {0.0, 2.0}, {0.05, 1.0}};
    const std::vector<int> weighted_knn_labels{0, 0, 1};
    WeightedKNNClassifier weighted_knn(3);
    weighted_knn.fit(weighted_knn_train, weighted_knn_labels);
    std::cout << "Predicted label: " << weighted_knn.predict(Matrix{{0.0, 1.0}})[0] << "\n\n";

    std::cout << "--- Label Binarizer ---\n";

    LabelBinarizer binarizer;
    const auto one_hot = binarizer.fit_transform({0, 1, 2});
    std::cout << "One-hot rows: " << one_hot.rows << ", cols: " << one_hot.cols << "\n\n";

    std::cout << "--- Randomized Search ---\n";

    const Matrix ridge_train{{0.0}, {1.0}, {2.0}, {3.0}, {4.0}, {5.0}, {6.0}, {7.0}};
    const std::vector<double> ridge_targets{1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0};
    const std::vector<double> alpha_candidates{100.0, 10.0, 1.0, 0.1, 0.01};

    auto factory = [](double alpha) { return RidgeRegression(alpha); };
    auto scorer = [](const std::vector<double>& actual, const std::vector<double>& predicted) {
        return -mean_squared_error(actual, predicted);
    };

    RandomizedSearch search(alpha_candidates, factory, KFold(4), scorer, 5, 7);
    search.fit(ridge_train, ridge_targets);
    std::cout << "Best alpha found: " << search.best_parameter() << "\n";
}

} // namespace ml::examples
