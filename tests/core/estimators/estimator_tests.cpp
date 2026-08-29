#include "ml/ml.hpp"

#include <cassert>
#include <iostream>

using namespace ml;

namespace {

class DummyEstimator : public Estimator {
public:
    void fit(const Matrix& X, const Matrix& y) override {
        fitted_ = true;
    }

    Matrix predict(const Matrix& X) const override {
        return {
            {10.0},
            {20.0}
        };
    }

    bool is_fitted() const {
        return fitted_;
    }

private:
    bool fitted_ = false;
};

void test_estimator_interface() {
    std::cout << "[TEST] Estimator interface ... ";

    DummyEstimator estimator;

    const Matrix X = {
        {1.0, 2.0},
        {3.0, 4.0}
    };

    const Matrix y = {
        {10.0},
        {20.0}
    };

    estimator.fit(X, y);

    assert(estimator.is_fitted());

    const Matrix predictions = estimator.predict(X);

    assert(predictions.rows == 2);
    assert(predictions.cols == 1);

    assert(predictions(0, 0) == 10.0);
    assert(predictions(1, 0) == 20.0);

    std::cout << "OK\n";
}

} // namespace

int main() {
    std::cout << "Running estimator tests...\n\n";

    test_estimator_interface();

    std::cout << "\nAll estimator tests passed!\n";

    return 0;
}
