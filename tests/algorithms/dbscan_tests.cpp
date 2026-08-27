#include "ml/algorithms/dbscan.hpp"

#include <cassert>

int main() {
    const ml::Matrix points{{0.0, 0.0}, {0.1, 0.0}, {0.0, 0.1}, {5.0, 5.0}, {5.1, 5.0}, {5.0, 5.1}, {10.0, 10.0}};
    ml::DBSCAN dbscan(0.25, 3);
    const auto labels = dbscan.fit_predict(points);
    assert(labels[0] == labels[1] && labels[1] == labels[2]);
    assert(labels[3] == labels[4] && labels[4] == labels[5]);
    assert(labels[0] != labels[3]);
    assert(labels[6] == -1);
}
