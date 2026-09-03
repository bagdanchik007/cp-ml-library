#include "ml/algorithms/agglomerative_clustering.hpp"

#include <cassert>

int main() {
    const ml::Matrix X{
        {0.0, 0.0}, {0.1, 0.1}, {0.0, 0.2},
        {10.0, 10.0}, {10.1, 9.9}, {9.9, 10.1}
    };

    ml::AgglomerativeClustering model(2, ml::AgglomerativeClustering::Linkage::Single);
    const auto labels = model.fit_predict(X);
    assert(labels.size() == 6);

    // The first three samples must share a label, the last three must share
    // a different label (order of the label ids themselves is arbitrary).
    assert(labels[0] == labels[1]);
    assert(labels[1] == labels[2]);
    assert(labels[3] == labels[4]);
    assert(labels[4] == labels[5]);
    assert(labels[0] != labels[3]);

    ml::AgglomerativeClustering complete_linkage(2, ml::AgglomerativeClustering::Linkage::Complete);
    const auto complete_labels = complete_linkage.fit_predict(X);
    assert(complete_labels[0] == complete_labels[1]);
    assert(complete_labels[3] == complete_labels[4]);
    assert(complete_labels[0] != complete_labels[3]);
}
