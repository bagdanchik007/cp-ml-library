#pragma once

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace ml {

/**
 * @brief Lightweight dense matrix class for educational / portfolio ML library.
 *
 * Designed for clarity and correctness rather than maximum performance.
 * For production workloads prefer Eigen or similar.
 */
class Matrix {
public:
    size_t rows = 0;
    size_t cols = 0;
    std::vector<double> data;

    Matrix() = default;

    Matrix(size_t r, size_t c, double fill = 0.0)
        : rows(r), cols(c), data(r * c, fill) {}

    // Construct from 2D initializer list
    Matrix(std::initializer_list<std::initializer_list<double>> init) {
        rows = init.size();
        cols = init.begin()->size();

        data.reserve(rows * cols);

        for (const auto& row : init) {
            if (row.size() != cols) {
                throw std::invalid_argument(
                    "All rows must have the same length"
                );
            }

            data.insert(data.end(), row.begin(), row.end());
        }
    }

    double& operator()(size_t i, size_t j) {
        return data[i * cols + j];
    }

    const double& operator()(size_t i, size_t j) const {
        return data[i * cols + j];
    }

    // Row access helper
    std::vector<double> row(size_t i) const {
        return std::vector<double>(
            data.begin() + i * cols,
            data.begin() + (i + 1) * cols
        );
    }

    void set_row(size_t i, const std::vector<double>& values) {
        if (values.size() != cols) {
            throw std::invalid_argument("Row size mismatch");
        }

        std::copy(
            values.begin(),
            values.end(),
            data.begin() + i * cols
        );
    }

    Matrix transpose() const {
        Matrix result(cols, rows);

        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result(j, i) = (*this)(i, j);
            }
        }

        return result;
    }

    // Matrix-vector product
    std::vector<double> operator*(
        const std::vector<double>& v
    ) const {
        if (cols != v.size()) {
            throw std::invalid_argument(
                "Matrix-vector dimension mismatch"
            );
        }

        std::vector<double> result(rows, 0.0);

        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result[i] += (*this)(i, j) * v[j];
            }
        }

        return result;
    }

    // Scalar multiplication
    Matrix operator*(double scalar) const {
        Matrix result(rows, cols);

        for (size_t i = 0; i < data.size(); ++i) {
            result.data[i] = data[i] * scalar;
        }

        return result;
    }

    // Scalar division
    Matrix operator/(double scalar) const {
        if (std::abs(scalar) < 1e-12) {
            throw std::invalid_argument("Division by zero");
        }

        Matrix result(rows, cols);

        for (size_t i = 0; i < data.size(); ++i) {
            result.data[i] = data[i] / scalar;
        }

        return result;
    }

    // In-place scalar multiplication
    Matrix& operator*=(double scalar) {
        for (auto& value : data) {
            value *= scalar;
        }

        return *this;
    }

    // In-place scalar division
    Matrix& operator/=(double scalar) {
        if (std::abs(scalar) < 1e-12) {
            throw std::invalid_argument("Division by zero");
        }

        for (auto& value : data) {
            value /= scalar;
        }

        return *this;
    }

    // Simple pretty print
    void print(
        std::ostream& os = std::cout,
        int precision = 4
    ) const {
        os << std::fixed << std::setprecision(precision);

        for (size_t i = 0; i < rows; ++i) {
            os << "[ ";

            for (size_t j = 0; j < cols; ++j) {
                os << std::setw(8)
                   << (*this)(i, j)
                   << " ";
            }

            os << "]\n";
        }
    }

    static Matrix zeros(size_t r, size_t c) {
        return Matrix(r, c, 0.0);
    }

    static Matrix ones(size_t r, size_t c) {
        return Matrix(r, c, 1.0);
    }

    // Random matrix (useful for testing)
    static Matrix random(
        size_t r,
        size_t c,
        double low = -1.0,
        double high = 1.0,
        unsigned seed = 42
    ) {
        Matrix m(r, c);

        std::mt19937 gen(seed);
        std::uniform_real_distribution<double> dist(low, high);

        for (auto& value : m.data) {
            value = dist(gen);
        }

        return m;
    }
};

// Euclidean distance between two vectors
inline double euclidean_distance(
    const std::vector<double>& a,
    const std::vector<double>& b
) {
    if (a.size() != b.size()) {
        throw std::invalid_argument(
            "Vectors must have same dimension"
        );
    }

    double sum = 0.0;

    for (size_t i = 0; i < a.size(); ++i) {
        const double difference = a[i] - b[i];
        sum += difference * difference;
    }

    return std::sqrt(sum);
}

} // namespace ml