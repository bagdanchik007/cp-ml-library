#pragma once

#include "estimator.hpp"

namespace ml {

class Classifier : public Estimator {
public:
    ~Classifier() override = default;

    virtual Matrix predict_proba(const Matrix& X) const = 0;
};

} // namespace ml