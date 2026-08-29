#pragma once 

#include "ml/core/matrix/matrix.hpp"

namespace ml {

    class Estiminator {

        public:
              virtual ~Estiminator() = default;

              virtual void fit(const Matrix& X, const Matrix& y) = 0;

              virtual Matrix predict(const Matrix& X) const = 0;

    };

}// namespace ml
