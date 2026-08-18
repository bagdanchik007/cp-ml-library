#pragma once

/**
 * @file ml.hpp
 * @brief Main umbrella header for the Cpp ML Library.
 *
 * Include this single header to get access to the whole public API.
 */

#include "matrix.hpp"
#include "linear_regression.hpp"
#include "kmeans.hpp"

namespace ml {
// Version info
constexpr int VERSION_MAJOR = 1;
constexpr int VERSION_MINOR = 0;
constexpr int VERSION_PATCH = 0;
}
