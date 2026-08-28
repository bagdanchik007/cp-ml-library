#include "ml/ml.hpp"
#include <cassert>
#include <iostream>

using namespace ml;

void test_kmeans() {
    std::cout << "[TEST] K-Means ... ";

    Matrix data = {
        {1.0, 1.0}, {1.2, 1.1}, {0.9, 1.3},
        {8.0, 8.0}, {8.1, 7.9}, {7.8, 8.2}
    };

    KMeans km(2, 100, 42);
    auto labels = km.fit_predict(data);

    assert(km.is_fitted());
    assert(labels.size() == 6);

    int cluster0 = labels[0];

    int same = 0;
    for (size_t i = 0; i < 3; ++i) {
        if (labels[i] == cluster0) {
            ++same;
        }
    }

    assert(same == 3);

    int other = 0;
    for (size_t i = 3; i < 6; ++i) {
        if (labels[i] != cluster0) {
            ++other;
        }
    }

    assert(other == 3);

    std::cout << "OK\n";
}

int main() {
    test_kmeans();

    std::cout << "\nAll K-Means tests passed!\n";
    return 0;
}