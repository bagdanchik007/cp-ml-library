#pragma once

#include <vector>
#include <stdexcept>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <random>

namespace ml {

/**
 * @brief Lightweight dense matrix class for educational / portfolio ML library.
 *        Designed for clarity and correctness rather than maximum performance.
 *        For production workloads prefer Eigen or similar.
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
            if (row.size() != cols)
                throw std::invalid_argument("All rows must have the same length");
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
        return std::vector<double>(data.begin() + i * cols,
                                   data.begin() + (i + 1) * cols);
    }

    void set_row(size_t i, const std::vector<double>& values) {
        if (values.size() != cols)
            throw std::invalid_argument("Row size mismatch");
        std::copy(values.begin(), values.end(), data.begin() + i * cols);
    }

    Matrix transpose() const {
        Matrix t(cols, rows);
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < cols; ++j)
                t(j, i) = (*this)(i, j);
        return t;
    }

    // Matrix-vector product
    std::vector<double> operator*(const std::vector<double>& v) const {
        if (cols != v.size())
            throw std::invalid_argument("Matrix-vector dimension mismatch");
        std::vector<double> result(rows, 0.0);
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < cols; ++j)
                result[i] += (*this)(i, j) * v[j];
        return result;
    }

    // Simple pretty print
    void print(std::ostream& os = std::cout, int precision = 4) const {
        os << std::fixed << std::setprecision(precision);
        for (size_t i = 0; i < rows; ++i) {
            os << "[ ";
            for (size_t j = 0; j < cols; ++j)
                os << std::setw(8) << (*this)(i, j) << " ";
            os << "]\n";
        }
    }

    static Matrix zeros(size_t r, size_t c) { return Matrix(r, c, 0.0); }
    static Matrix ones(size_t r, size_t c)  { return Matrix(r, c, 1.0); }

    // Random matrix (useful for testing)
    static Matrix random(size_t r, size_t c, double low = -1.0, double high = 1.0,
                         unsigned seed = 42) {
        Matrix m(r, c);
        std::mt19937 gen(seed);
        std::uniform_real_distribution<double> dist(low, high);
        for (auto& val : m.data)
            val = dist(gen);
        return m;
    }
};

// Euclidean distance between two vectors
inline double euclidean_distance(const std::vector<double>& a,
                                 const std::vector<double>& b) {
    if (a.size() != b.size())
        throw std::invalid_argument("Vectors must have same dimension");
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double d = a[i] - b[i];
        sum += d * d;
    }
    return std::sqrt(sum);
}

} // namespace ml
