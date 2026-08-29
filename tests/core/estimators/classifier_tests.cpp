#include "ml/ml.hpp"

#include <cassert>
#include <iostream>

using namespace ml;

namespace {

class DummyClassifier : public Classifier {
public:
    void fit(const Matrix& X, const Matrix& y) override {
        trained_ = true;
    }

    Matrix predict(const Matrix& X) const override {
        return {
            {0.0},
            {1.0}
        };
    }

    Matrix predict_proba(const Matrix& X) const override {
        return {
            {0.8, 0.2},
            {0.1, 0.9}
        };
    }

    bool is_trained() const {
        return trained_;
    }

private:
    bool trained_ = false;
};

void test_classifier_interface() {
    std::cout << "[TEST] Classifier interface ... ";

    DummyClassifier classifier;

    const Matrix X = {
        {1.0, 2.0},
        {3.0, 4.0}
    };

    const Matrix y = {
        {0.0},
        {1.0}
    };

    classifier.fit(X, y);

    assert(classifier.is_trained());

    const Matrix predictions = classifier.predict(X);

    assert(predictions.rows == 2);
    assert(predictions.cols == 1);
    assert(predictions(0, 0) == 0.0);
    assert(predictions(1, 0) == 1.0);

    const Matrix probabilities = classifier.predict_proba(X);

    assert(probabilities.rows == 2);
    assert(probabilities.cols == 2);

    assert(probabilities(0, 0) == 0.8);
    assert(probabilities(0, 1) == 0.2);
    assert(probabilities(1, 0) == 0.1);
    assert(probabilities(1, 1) == 0.9);

    std::cout << "OK\n";
}

} // namespace

int main() {
    std::cout << "Running classifier tests...\n\n";

    test_classifier_interface();

    std::cout << "\nAll classifier tests passed!\n";

    return 0;
}