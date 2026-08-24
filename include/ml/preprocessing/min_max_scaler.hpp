#pragma once 

#include <ml/core/matrix/matrix.hpp>

namespace ml {

    class MinMaxScaler {
    public:
        MinMaxScaler() = default;

        void fit(const Matrix& data);

        Matrix transform(const Matrix& data) const;

        Matrix fit_transform(const Matrix& data);

        Matrix inverse_transform(const Matrix& data) const;

        private:
        Matrix min_;
        Matrix max_;
        bool fitted_ = false;
    };

}//namespace ml