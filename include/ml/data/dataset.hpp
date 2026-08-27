#pragma once

#include "ml/core/matrix/matrix.hpp"

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ml {

/** A feature matrix together with one numeric target per sample. */
class Dataset {
public:
    Dataset(Matrix features, std::vector<double> targets)
        : features_(std::move(features)), targets_(std::move(targets)) {
        validate();
    }

    const Matrix& features() const noexcept { return features_; }
    const std::vector<double>& targets() const noexcept { return targets_; }
    size_t rows() const noexcept { return features_.rows; }
    size_t columns() const noexcept { return features_.cols; }

    Dataset select_rows(const std::vector<size_t>& indices) const {
        Matrix selected(indices.size(), columns());
        std::vector<double> selected_targets;
        selected_targets.reserve(indices.size());
        for (size_t out = 0; out < indices.size(); ++out) {
            const size_t row = indices[out];
            if (row >= rows()) {
                throw std::out_of_range("Dataset::select_rows: row index out of range");
            }
            selected.set_row(out, features_.row(row));
            selected_targets.push_back(targets_[row]);
        }
        return Dataset(std::move(selected), std::move(selected_targets));
    }

private:
    void validate() const {
        if (features_.rows == 0 || features_.cols == 0) {
            throw std::invalid_argument("Dataset: features must not be empty");
        }
        if (features_.rows != targets_.size()) {
            throw std::invalid_argument("Dataset: features and targets must have equal row counts");
        }
    }

    Matrix features_;
    std::vector<double> targets_;
};

struct DatasetSplit {
    Dataset train;
    Dataset test;
};

inline DatasetSplit train_test_split(
    const Dataset& data,
    double test_size = 0.2,
    bool shuffle = true,
    unsigned seed = 42) {
    if (test_size <= 0.0 || test_size >= 1.0) {
        throw std::invalid_argument("train_test_split: test_size must be between 0 and 1");
    }
    const size_t test_count = static_cast<size_t>(data.rows() * test_size + 0.999999999);
    if (test_count == 0 || test_count >= data.rows()) {
        throw std::invalid_argument("train_test_split: split would produce an empty partition");
    }
    std::vector<size_t> indices(data.rows());
    std::iota(indices.begin(), indices.end(), 0);
    if (shuffle) {
        std::mt19937 generator(seed);
        std::shuffle(indices.begin(), indices.end(), generator);
    }
    std::vector<size_t> train_indices(indices.begin(), indices.end() - static_cast<std::ptrdiff_t>(test_count));
    std::vector<size_t> test_indices(indices.end() - static_cast<std::ptrdiff_t>(test_count), indices.end());
    return {data.select_rows(train_indices), data.select_rows(test_indices)};
}

} // namespace ml
