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

        if (rows == 0) {
            cols = 0;
            return;
        }

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

    // Transpose matrix
    Matrix transpose() const {
        Matrix result(cols, rows);

        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result(j, i) = (*this)(i, j);
            }
        }

        return result;
    }

    // Matrix determinant
    double determinant() const {
        if (rows != cols) {
            throw std::invalid_argument(
                "Determinant requires a square matrix"
            );
        }

        if (rows == 0) {
            throw std::invalid_argument(
                "Determinant of an empty matrix is undefined"
            );
        }

        if (rows == 1) {
            return (*this)(0, 0);
        }

        if (rows == 2) {
            return
                (*this)(0, 0) * (*this)(1, 1) -
                (*this)(0, 1) * (*this)(1, 0);
        }

        double result = 0.0;

        for (size_t column = 0; column < cols; ++column) {
            Matrix minor(rows - 1, cols - 1);

            for (size_t i = 1; i < rows; ++i) {
                size_t minor_column = 0;

                for (size_t j = 0; j < cols; ++j) {
                    if (j == column) {
                        continue;
                    }

                    minor(i - 1, minor_column) = (*this)(i, j);
                    ++minor_column;
                }
            }

            const double sign =
                (column % 2 == 0) ? 1.0 : -1.0;

            result +=
                sign *
                (*this)(0, column) *
                minor.determinant();
        }

        return result;
    }
    // Matrix inverse using Gauss-Jordan elimination
Matrix inverse() const {
    if (rows != cols) {
        throw std::invalid_argument(
            "Inverse requires a square matrix"
        );
    }

    if (rows == 0) {
        throw std::invalid_argument(
            "Inverse of an empty matrix is undefined"
        );
    }

    Matrix augmented(rows, cols * 2);

    // Build augmented matrix [A | I].
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            augmented(i, j) = (*this)(i, j);
            augmented(i, j + cols) =
                (i == j) ? 1.0 : 0.0;
        }
    }

    for (size_t i = 0; i < rows; ++i) {
        // Find the best pivot.
        size_t pivot_row = i;

        for (size_t row = i + 1; row < rows; ++row) {
            if (std::abs(augmented(row, i)) >
                std::abs(augmented(pivot_row, i))) {
                pivot_row = row;
            }
        }

        if (std::abs(augmented(pivot_row, i)) < 1e-12) {
            throw std::invalid_argument(
                "Matrix is singular and cannot be inverted"
            );
        }

        // Swap the pivot row into position.
        if (pivot_row != i) {
            for (size_t j = 0; j < augmented.cols; ++j) {
                std::swap(
                    augmented(i, j),
                    augmented(pivot_row, j)
                );
            }
        }

        // Normalize the pivot row.
        const double pivot = augmented(i, i);

        for (size_t j = 0; j < augmented.cols; ++j) {
            augmented(i, j) /= pivot;
        }

        // Eliminate the pivot column from all other rows.
        for (size_t row = 0; row < rows; ++row) {
            if (row == i) {
                continue;
            }

            const double factor = augmented(row, i);

            for (size_t j = 0; j < augmented.cols; ++j) {
                augmented(row, j) -=
                    factor * augmented(i, j);
            }
        }
    }

    Matrix result(rows, cols);

    // Extract the right half, which is A^-1.
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result(i, j) = augmented(i, j + cols);
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

    // Matrix addition
    Matrix operator+(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument(
                "Matrix dimensions must match for addition"
            );
        }

        Matrix result(rows, cols);

        for (size_t i = 0; i < data.size(); ++i) {
            result.data[i] = data[i] + other.data[i];
        }

        return result;
    }

    // Matrix subtraction
    Matrix operator-(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument(
                "Matrix dimensions must match for subtraction"
            );
        }

        Matrix result(rows, cols);

        for (size_t i = 0; i < data.size(); ++i) {
            result.data[i] = data[i] - other.data[i];
        }

        return result;
    }

    // In-place matrix addition
    Matrix& operator+=(const Matrix& other) {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument(
                "Matrix dimensions must match for addition"
            );
        }

        for (size_t i = 0; i < data.size(); ++i) {
            data[i] += other.data[i];
        }

        return *this;
    }

    // In-place matrix subtraction
    Matrix& operator-=(const Matrix& other) {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument(
                "Matrix dimensions must match for subtraction"
            );
        }

        for (size_t i = 0; i < data.size(); ++i) {
            data[i] -= other.data[i];
        }

        return *this;
    }

    // Matrix multiplication
    Matrix operator*(const Matrix& other) const {
        if (cols != other.rows) {
            throw std::invalid_argument(
                "Matrix dimensions incompatible for multiplication"
            );
        }

        Matrix result(rows, other.cols, 0.0);

        for (size_t i = 0; i < rows; ++i) {
            for (size_t k = 0; k < cols; ++k) {
                const double value = (*this)(i, k);

                for (size_t j = 0; j < other.cols; ++j) {
                    result(i, j) += value * other(k, j);
                }
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

    // Factory: zero matrix
    static Matrix zeros(size_t r, size_t c) {
        return Matrix(r, c, 0.0);
    }

    // Factory: matrix filled with ones
    static Matrix ones(size_t r, size_t c) {
        return Matrix(r, c, 1.0);
    }

    // Factory: identity matrix
    static Matrix identity(size_t size) {
        Matrix result(size, size, 0.0);

        for (size_t i = 0; i < size; ++i) {
            result(i, i) = 1.0;
        }

        return result;
    }

    // Random matrix
    static Matrix random(
        size_t r,
        size_t c,
        double low = -1.0,
        double high = 1.0,
        unsigned seed = 42
    ) {
        Matrix result(r, c);

        std::mt19937 gen(seed);
        std::uniform_real_distribution<double> dist(low, high);

        for (auto& value : result.data) {
            value = dist(gen);
        }

        return result;
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