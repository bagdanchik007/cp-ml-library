#include "ml/ml.hpp"

#include <cassert>
#include <iostream>

using namespace ml;

namespace {

class DummyRegressor : public Regressor {
public:
    void fit(const Matrix& X, const Matrix& y) override {
        fitted_ = true;
    }

    Matrix predict(const Matrix& X) const override {
        return {
            {1.5},
            {2.5}
        };
    }

    bool is_fitted() const {
        return fitted_;
    }

private:
    bool fitted_ = false;
};

void test_regressor_interface() {
    std::cout << "[TEST] Regressor interface ... ";

    DummyRegressor regressor;

    const Matrix X = {
        {1.0, 2.0},
        {3.0, 4.0}
    };

    const Matrix y = {
        {1.5},
        {2.5}
    };

    regressor.fit(X, y);

    assert(regressor.is_fitted());

    const Matrix predictions = regressor.predict(X);

    assert(predictions.rows == 2);
    assert(predictions.cols == 1);

    assert(predictions(0, 0) == 1.5);
    assert(predictions(1, 0) == 2.5);

    std::cout << "OK\n";
}

} // namespace

int main() {
    std::cout << "Running regressor tests...\n\n";

    test_regressor_interface();

    std::cout << "\nAll regressor tests passed!\n";

    return 0;
}


