#include "ml/algorithms/dummy_classifier.hpp"

#include <cassert>

int main() {
    const std::vector<int> y{1, 1, 1, 1, 0};

    ml::DummyClassifier most_frequent("most_frequent");
    assert(!most_frequent.is_fitted());
    most_frequent.fit(y);
    assert(most_frequent.is_fitted());
    const auto predictions = most_frequent.predict(5);
    for (int prediction : predictions) assert(prediction == 1);

    ml::DummyClassifier uniform("uniform", 7);
    uniform.fit(y);
    const auto uniform_predictions = uniform.predict(20);
    for (int prediction : uniform_predictions) assert(prediction == 0 || prediction == 1);

    ml::DummyClassifier stratified("stratified", 7);
    stratified.fit(y);
    const auto stratified_predictions = stratified.predict(20);
    for (int prediction : stratified_predictions) assert(prediction == 0 || prediction == 1);

    bool threw = false;
    try {
        ml::DummyClassifier invalid("bogus");
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}
