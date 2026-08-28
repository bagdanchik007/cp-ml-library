#include "ml/algorithms/decision_tree.hpp"
#include "ml/algorithms/random_forest.hpp"
#include "ml/metrics/model_evaluation.hpp"

#include <cassert>

int main() {
    const ml::Matrix features{{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}, {2.0, 0.0}, {2.0, 1.0}};
    const std::vector<int> labels{0, 0, 1, 1, 1, 1};
    ml::DecisionTreeClassifier tree(3);
    tree.fit(features, labels);
    assert(ml::accuracy_score(labels, tree.predict(features)) == 1.0);
    ml::RandomForestClassifier forest(15, 4, 2, 17);
    forest.fit(features, labels);
    assert(ml::accuracy_score(labels, forest.predict(features)) >= 5.0 / 6.0);
}
