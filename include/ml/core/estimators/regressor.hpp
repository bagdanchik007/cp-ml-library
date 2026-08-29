#pragma once

#include "ml/core/estimators/estimator.hpp"

namespace ml {

class Regressor : public Estimator {
public:
    ~Regressor() override = default;
};

} // namespace ml