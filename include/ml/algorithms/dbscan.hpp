#pragma once

#include "ml/core/matrix/matrix.hpp"

#include <cmath>
#include <queue>
#include <stdexcept>
#include <vector>

namespace ml {

/** Density-based clustering. Noise samples receive label -1. */
class DBSCAN {
public:
    explicit DBSCAN(double epsilon = 0.5, size_t min_samples = 5) : epsilon_(epsilon), min_samples_(min_samples) {
        if (epsilon_ <= 0.0 || min_samples_ == 0) throw std::invalid_argument("DBSCAN: epsilon and min_samples must be positive");
    }

    void fit(const Matrix& features) { labels_ = fit_predict(features); }

    std::vector<int> fit_predict(const Matrix& features) {
        if (features.rows == 0 || features.cols == 0) throw std::invalid_argument("DBSCAN::fit_predict: input data must not be empty");
        labels_.assign(features.rows, unvisited);
        int cluster = 0;
        for (size_t point = 0; point < features.rows; ++point) {
            if (labels_[point] != unvisited) continue;
            const auto neighbours = region_query(features, point);
            if (neighbours.size() < min_samples_) { labels_[point] = noise; continue; }
            expand_cluster(features, point, neighbours, cluster++);
        }
        fitted_ = true; return labels_;
    }

    const std::vector<int>& labels() const {
        if (!fitted_) throw std::logic_error("DBSCAN::labels: model has not been fitted");
        return labels_;
    }

private:
    std::vector<size_t> region_query(const Matrix& features, size_t point) const {
        std::vector<size_t> result;
        for (size_t candidate = 0; candidate < features.rows; ++candidate) {
            double squared = 0.0;
            for (size_t column = 0; column < features.cols; ++column) { const double diff = features(point, column) - features(candidate, column); squared += diff * diff; }
            if (std::sqrt(squared) <= epsilon_) result.push_back(candidate);
        }
        return result;
    }
    void expand_cluster(const Matrix& features, size_t seed, std::vector<size_t> neighbours, int cluster) {
        labels_[seed] = cluster;
        std::queue<size_t> queue; for (size_t point : neighbours) queue.push(point);
        while (!queue.empty()) {
            const size_t point = queue.front(); queue.pop();
            if (labels_[point] == noise) labels_[point] = cluster;
            if (labels_[point] != unvisited) continue;
            labels_[point] = cluster;
            const auto nested = region_query(features, point);
            if (nested.size() >= min_samples_) for (size_t neighbour : nested) if (labels_[neighbour] == unvisited || labels_[neighbour] == noise) queue.push(neighbour);
        }
    }
    static constexpr int unvisited = -2;
    static constexpr int noise = -1;
    double epsilon_; size_t min_samples_; std::vector<int> labels_; bool fitted_ = false;
};

} // namespace ml
