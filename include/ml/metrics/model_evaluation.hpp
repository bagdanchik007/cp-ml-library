#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace ml {

inline void validate_predictions(const std::vector<double>& actual, const std::vector<double>& predicted) {
    if (actual.empty() || actual.size() != predicted.size()) {
        throw std::invalid_argument("metrics: actual and predicted must be non-empty and equally sized");
    }
}

inline double mean_squared_error(const std::vector<double>& actual, const std::vector<double>& predicted) {
    validate_predictions(actual, predicted);
    double sum = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) { const double error = actual[i] - predicted[i]; sum += error * error; }
    return sum / static_cast<double>(actual.size());
}

inline double mean_absolute_error(const std::vector<double>& actual, const std::vector<double>& predicted) {
    validate_predictions(actual, predicted);
    double sum = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) sum += std::abs(actual[i] - predicted[i]);
    return sum / static_cast<double>(actual.size());
}

inline double r2_score(const std::vector<double>& actual, const std::vector<double>& predicted) {
    validate_predictions(actual, predicted);
    double mean = 0.0;
    for (double value : actual) mean += value;
    mean /= static_cast<double>(actual.size());
    double residual = 0.0, total = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) { const double d = actual[i] - mean; total += d * d; const double e = actual[i] - predicted[i]; residual += e * e; }
    return total == 0.0 ? (residual == 0.0 ? 1.0 : 0.0) : 1.0 - residual / total;
}

inline double accuracy_score(const std::vector<int>& actual, const std::vector<int>& predicted) {
    if (actual.empty() || actual.size() != predicted.size()) throw std::invalid_argument("accuracy_score: inputs must be non-empty and equally sized");
    size_t correct = 0;
    for (size_t i = 0; i < actual.size(); ++i) if (actual[i] == predicted[i]) ++correct;
    return static_cast<double>(correct) / static_cast<double>(actual.size());
}

} // namespace ml
