#pragma once

/**
 * @file ml.hpp
 * @brief Main umbrella header for the Cpp ML Library.
 */

#include "core/matrix/matrix.hpp"
#include "core/matrix/matrix_utility.hpp"

#include "algorithms/linear_regression.hpp"
#include "algorithms/kmeans.hpp"

#include "preprocessing/standard_scaler.hpp"
#include "preprocessing/min_max_scaler.hpp"
#include "preprocessing/normalizer.hpp"

namespace ml {

constexpr int VERSION_MAJOR = 1;
constexpr int VERSION_MINOR = 0;
constexpr int VERSION_PATCH = 0;

} // namespace ml