#pragma once

#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <vector>

namespace ml {

/**
 * @brief Lightweight dense matrix class.
 *
 * Declaration only. All implementations live in matrix_operations.hpp
 * (included at the bottom of this file – classic header-only style).
 */
class Matrix {
public:
    size_t rows = 0;
    size_t cols = 0;
    std::vector<double> data;

    Matrix() = default;

    Matrix(size_t r, size_t c, double fill = 0.0);

    Matrix(std::initializer_list<std::initializer_list<double>> init);

    double& operator()(size_t i, size_t j);
    const double& operator()(size_t i, size_t j) const;
    std::vector<double> row(size_t i) const;
    void set_row(size_t i, const std::vector<double>& values);

    std::vector<double> column(size_t j) const;

    Matrix transpose() const; 
    // Get the trace of the matrix (sum of diagonal elements)
    double trace() const;
    // Get the determinant of the matrix (only for square matrices)
    double determinant() const;
    Matrix inverse() const;

    // Matrix-vector
    std::vector<double> operator*(const std::vector<double>& v) const;

    // Matrix-Matrix
    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix& operator+=(const Matrix& other);
    Matrix& operator-=(const Matrix& other);
    Matrix operator*(const Matrix& other) const;
    // Comparison
    bool operator==(const Matrix& other) const;
    bool operator!=(const Matrix& other) const;

    // Scalar
    Matrix operator*(double scalar) const;
    Matrix operator/(double scalar) const;
    Matrix& operator*=(double scalar);
    Matrix& operator/=(double scalar);

    void print(std::ostream& os = std::cout, int precision = 4) const;

    // Factory methods
    static Matrix zeros(size_t r, size_t c);
    static Matrix ones(size_t r, size_t c);
    static Matrix identity(size_t size);
    static Matrix random(size_t r, size_t c,
                         double low = -1.0, double high = 1.0,
                         unsigned seed = 42);
};

} // namespace ml

// Implementation (header-only)
#include "matrix_operations.hpp"
