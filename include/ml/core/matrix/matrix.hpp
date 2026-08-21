#pragma once

#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <utility>
#include <vector>

namespace ml {

/**
 * @brief Lightweight dense matrix class.
 *
 * Declaration only. Implementations are organized in the operations/ directory,
 * which is included at the bottom of this file.
 */
class Matrix {
public:
    size_t rows = 0;
    size_t cols = 0;
    std::vector<double> data;

    Matrix() = default;

    Matrix(size_t r, size_t c, double fill = 0.0);

    Matrix(std::initializer_list<std::initializer_list<double>> init);

    // Element access
    double& operator()(size_t i, size_t j);
    const double& operator()(size_t i, size_t j) const;

    // Row and column operations
    std::vector<double> row(size_t i) const;
    void set_row(size_t i, const std::vector<double>& values);
    void swap_rows(size_t first, size_t second);

    std::vector<double> column(size_t j) const;
    void set_column(size_t j, const std::vector<double>& values);
    void swap_columns(size_t first, size_t second);
    
    // Matrix transformations
    Matrix transpose() const;
    void reshape(
        size_t new_rows,
        size_t new_cols
    );
    std::vector<double> flatten() const;

    // Matrix properties
    double trace() const;
    double determinant() const;
    Matrix inverse() const;

    // Matrix information
    bool empty() const;
    size_t size() const;
    std::pair<size_t, size_t> shape() const;

    // Matrix modification
    void fill(double value); 
    void clear();
    void resize(
         size_t r, 
         size_t c, 
         double fill = 0.0
    );

    // Statistical operations
    double sum() const;
    double mean() const;
    double min() const;
    double max() const;
    // Matrix norms
    double squared_norm() const;
    double norm() const;
    // Matrix-vector operations
    std::vector<double> operator*(const std::vector<double>& v) const;

    // Matrix-matrix operations
    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix& operator+=(const Matrix& other);
    Matrix& operator-=(const Matrix& other);
    Matrix operator*(const Matrix& other) const;
    Matrix hadamard(const Matrix& other) const;
    
    // Comparison
    bool operator==(const Matrix& other) const;
    bool operator!=(const Matrix& other) const;
    
    // Scalar operations
    Matrix operator*(double scalar) const;
    Matrix operator/(double scalar) const;
    Matrix& operator*=(double scalar);
    Matrix& operator/=(double scalar);

    // Output
    void print(
        std::ostream& os = std::cout,
        int precision = 4
    ) const;

    // Factory methods
    static Matrix zeros(size_t r, size_t c);
    static Matrix ones(size_t r, size_t c);
    static Matrix identity(size_t size);

    static Matrix random(
        size_t r,
        size_t c,
        double low = -1.0,
        double high = 1.0,
        unsigned seed = 42
    );
};

} // namespace ml

// Implementation (header-only)
#include "operations/matrix_construction.hpp"
#include "operations/matrix_access.hpp"
#include "operations/matrix_statistics.hpp"
#include "operations/matrix_linear_algebra.hpp"
#include "operations/matrix_arithmetic.hpp"
#include "operations/matrix_transformations.hpp"
#include "operations/matrix_scalar.hpp"
#include "operations/matrix_factory.hpp"
#include "operations/matrix_io.hpp"
#include "operations/matrix_utility.hpp"
#include "operations/matrix_comparison.hpp"