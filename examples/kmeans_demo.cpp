#include "ml/ml.hpp"

#include <iostream>

namespace ml::examples {

void run_kmeans_demo() {
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

    KMeans model(2, 100);

    const auto labels = model.fit_predict(data);

    std::cout << "Cluster labels: ";

    for (int label : labels) {
        std::cout << label << " ";
    }

    std::cout << "\n\nCentroids:\n";

    model.centroids().print();

    std::cout << '\n';
}

} // namespace ml::examples