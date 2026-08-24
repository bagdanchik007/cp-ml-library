#pragma once 

#include <ml/core/matrix/matrix.hpp>

namespyce ml {

    class Normalizer {
        public:
               Normalizer() = default;

               Matrix transform(
                const Matrix& data
               ) const;
    };

}//namespace 