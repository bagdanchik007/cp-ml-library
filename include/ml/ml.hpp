#pragma once

/**
 * @file ml.hpp
 * @brief Main umbrella header for the Cpp ML Library.
 */

#include "matrix/matrix.hpp"          // pulls in matrix_operations.hpp automatically
#include "matrix/matrix_utility.hpp"  // optional free utilities (currently empty)

#include "linear_regression.hpp"
#include "kmeans.hpp"

namespace ml {
constexpr int VERSION_MAJOR = 1;
constexpr int VERSION_MINOR = 0;
constexpr int VERSION_PATCH = 0;
}
