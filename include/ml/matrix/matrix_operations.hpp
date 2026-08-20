#pragma once

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <random>
#include <stdexcept>

namespace ml {

// ============================================================
// Constructors
// ============================================================

inline Matrix::Matrix(
    size_t r,
    size_t c,
    double fill
)
    : rows(r),
      cols(c),
      data(r * c, fill) {
}

inline Matrix::Matrix(
    std::initializer_list<
        std::initializer_list<double>
    > init
) {
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

        data.insert(
            data.end(),
            row.begin(),
            row.end()
        );
    }
}

// ============================================================
// Element access
// ============================================================

inline double& Matrix::operator()(
    size_t i,
    size_t j
) {
    if (i >= rows || j >= cols) {
        throw std::out_of_range(
            "Matrix index out of bounds"
        );
    }

    return data[i * cols + j];
}

inline const double& Matrix::operator()(
    size_t i,
    size_t j
) const {
    if (i >= rows || j >= cols) {
        throw std::out_of_range(
            "Matrix index out of bounds"
        );
    }

    return data[i * cols + j];
}

// ============================================================
// Row operations
// ============================================================

inline std::vector<double> Matrix::row(
    size_t i
) const {
    if (i >= rows) {
        throw std::out_of_range(
            "Matrix row index out of bounds"
        );
    }

    return std::vector<double>(
        data.begin() + i * cols,
        data.begin() + (i + 1) * cols
    );
}

inline void Matrix::set_row(
    size_t i,
    const std::vector<double>& values
) {
    if (i >= rows) {
        throw std::out_of_range(
            "Matrix row index out of bounds"
        );
    }

    if (values.size() != cols) {
        throw std::invalid_argument(
            "Row size mismatch"
        );
    }

    std::copy(
        values.begin(),
        values.end(),
        data.begin() + i * cols
    );
}

// ============================================================
// Row swapping
// ============================================================

inline void Matrix::swap_rows(
    size_t first,
    size_t second
) {
    if (first >= rows || second >= rows) {
        throw std::out_of_range(
            "Matrix row index out of bounds"
        );
    }

    if (first == second) {
        return;
    }

    for (size_t j = 0; j < cols; ++j) {
        std::swap(
            (*this)(first, j),
            (*this)(second, j)
        );
    }
}

// ============================================================
// Column operations
// ============================================================

inline std::vector<double> Matrix::column(
    size_t j
) const {
    if (j >= cols) {
        throw std::out_of_range(
            "Matrix column index out of bounds"
        );
    }

    std::vector<double> result;
    result.reserve(rows);

    for (size_t i = 0; i < rows; ++i) {
        result.push_back((*this)(i, j));
    }

    return result;
}

inline void Matrix::set_column(
    size_t j,
    const std::vector<double>& values
) {
    if (j >= cols) {
        throw std::out_of_range(
            "Matrix column index out of bounds"
        );
    }

    if (values.size() != rows) {
        throw std::invalid_argument(
            "Column size mismatch"
        );
    }

    for (size_t i = 0; i < rows; ++i) {
        (*this)(i, j) = values[i];
    }
}

// ============================================================
// Transpose
// ============================================================

inline Matrix Matrix::transpose() const {
    Matrix result(cols, rows);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result(j, i) = (*this)(i, j);
        }
    }

    return result;
}
// ============================================================
// Statistical operations
// ============================================================
// Sum, mean, min, max
inline double Matrix::sum() const {
    return std::accumulate(
        data.begin(),
        data.end(),
        0.0
    );
}
// Mean, min, max
inline double Matrix::mean() const {
    if (data.empty()) {
        throw std::invalid_argument(
            "Mean of an empty matrix is undefined"
        );
    }

    return sum() / static_cast<double>(data.size());
}
// Min, max
inline double Matrix::min() const {
    if (data.empty()) {
        throw std::invalid_argument(
            "Minimum of an empty matrix is undefined"
        );
    }

    return *std::min_element(
        data.begin(),
        data.end()
    );
}
// Max
inline double Matrix::max() const {
    if (data.empty()) {
        throw std::invalid_argument(
            "Maximum of an empty matrix is undefined"
        );
    }

    return *std::max_element(
        data.begin(),
        data.end()
    );
}
// ============================================================
// Determinant
// ============================================================

inline double Matrix::determinant() const {
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

                minor(i - 1, minor_column) =
                    (*this)(i, j);

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

// ============================================================
// Trace
// ============================================================

inline double Matrix::trace() const {
    if (rows != cols) {
        throw std::invalid_argument(
            "Trace requires a square matrix"
        );
    }

    double result = 0.0;

    for (size_t i = 0; i < rows; ++i) {
        result += (*this)(i, i);
    }

    return result;
}

// ============================================================
// Matrix inverse
// ============================================================

inline Matrix Matrix::inverse() const {
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

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            augmented(i, j) = (*this)(i, j);

            augmented(i, j + cols) =
                (i == j) ? 1.0 : 0.0;
        }
    }

    for (size_t i = 0; i < rows; ++i) {
        size_t pivot_row = i;

        for (size_t row = i + 1; row < rows; ++row) {
            if (
                std::abs(augmented(row, i)) >
                std::abs(augmented(pivot_row, i))
            ) {
                pivot_row = row;
            }
        }

        if (
            std::abs(augmented(pivot_row, i)) <
            1e-12
        ) {
            throw std::invalid_argument(
                "Matrix is singular and cannot be inverted"
            );
        }

        if (pivot_row != i) {
            for (
                size_t j = 0;
                j < augmented.cols;
                ++j
            ) {
                std::swap(
                    augmented(i, j),
                    augmented(pivot_row, j)
                );
            }
        }

        const double pivot =
            augmented(i, i);

        for (
            size_t j = 0;
            j < augmented.cols;
            ++j
        ) {
            augmented(i, j) /= pivot;
        }

        for (size_t row = 0; row < rows; ++row) {
            if (row == i) {
                continue;
            }

            const double factor =
                augmented(row, i);

            for (
                size_t j = 0;
                j < augmented.cols;
                ++j
            ) {
                augmented(row, j) -=
                    factor * augmented(i, j);
            }
        }
    }

    Matrix result(rows, cols);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result(i, j) =
                augmented(i, j + cols);
        }
    }

    return result;
}

// ============================================================
// Matrix-vector multiplication
// ============================================================

inline std::vector<double> Matrix::operator*(
    const std::vector<double>& v
) const {
    if (cols != v.size()) {
        throw std::invalid_argument(
            "Matrix-vector dimension mismatch"
        );
    }

    std::vector<double> result(
        rows,
        0.0
    );

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result[i] +=
                (*this)(i, j) * v[j];
        }
    }

    return result;
}

// ============================================================
// Matrix addition
// ============================================================

inline Matrix Matrix::operator+(
    const Matrix& other
) const {
    if (
        rows != other.rows ||
        cols != other.cols
    ) {
        throw std::invalid_argument(
            "Matrix dimensions must match for addition"
        );
    }

    Matrix result(rows, cols);

    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] =
            data[i] + other.data[i];
    }

    return result;
}

// ============================================================
// Matrix subtraction
// ============================================================

inline Matrix Matrix::operator-(
    const Matrix& other
) const {
    if (
        rows != other.rows ||
        cols != other.cols
    ) {
        throw std::invalid_argument(
            "Matrix dimensions must match for subtraction"
        );
    }

    Matrix result(rows, cols);

    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] =
            data[i] - other.data[i];
    }

    return result;
}

// ============================================================
// Matrix comparison
// ============================================================

inline bool Matrix::operator==(
    const Matrix& other
) const {
    constexpr double epsilon = 1e-9;

    if (
        rows != other.rows ||
        cols != other.cols
    ) {
        return false;
    }

    for (size_t i = 0; i < data.size(); ++i) {
        if (
            std::abs(data[i] - other.data[i]) >
            epsilon
        ) {
            return false;
        }
    }

    return true;
}

inline bool Matrix::operator!=(
    const Matrix& other
) const {
    return !(*this == other);
}

// ============================================================
// In-place addition
// ============================================================

inline Matrix& Matrix::operator+=(
    const Matrix& other
) {
    if (
        rows != other.rows ||
        cols != other.cols
    ) {
        throw std::invalid_argument(
            "Matrix dimensions must match for addition"
        );
    }

    for (size_t i = 0; i < data.size(); ++i) {
        data[i] += other.data[i];
    }

    return *this;
}

// ============================================================
// In-place subtraction
// ============================================================

inline Matrix& Matrix::operator-=(
    const Matrix& other
) {
    if (
        rows != other.rows ||
        cols != other.cols
    ) {
        throw std::invalid_argument(
            "Matrix dimensions must match for subtraction"
        );
    }

    for (size_t i = 0; i < data.size(); ++i) {
        data[i] -= other.data[i];
    }

    return *this;
}

// ============================================================
// Matrix multiplication
// ============================================================

inline Matrix Matrix::operator*(
    const Matrix& other
) const {
    if (cols != other.rows) {
        throw std::invalid_argument(
            "Matrix dimensions incompatible for multiplication"
        );
    }

    Matrix result(
        rows,
        other.cols,
        0.0
    );

    for (size_t i = 0; i < rows; ++i) {
        for (size_t k = 0; k < cols; ++k) {
            const double value =
                (*this)(i, k);

            for (
                size_t j = 0;
                j < other.cols;
                ++j
            ) {
                result(i, j) +=
                    value * other(k, j);
            }
        }
    }

    return result;
}

// ============================================================
// Scalar multiplication
// ============================================================

inline Matrix Matrix::operator*(
    double scalar
) const {
    Matrix result(rows, cols);

    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] =
            data[i] * scalar;
    }

    return result;
}

// ============================================================
// Scalar division
// ============================================================

inline Matrix Matrix::operator/(
    double scalar
) const {
    if (std::abs(scalar) < 1e-12) {
        throw std::invalid_argument(
            "Division by zero"
        );
    }

    Matrix result(rows, cols);

    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] =
            data[i] / scalar;
    }

    return result;
}

// ============================================================
// In-place scalar multiplication
// ============================================================

inline Matrix& Matrix::operator*=(
    double scalar
) {
    for (auto& value : data) {
        value *= scalar;
    }

    return *this;
}

// ============================================================
// In-place scalar division
// ============================================================

inline Matrix& Matrix::operator/=(
    double scalar
) {
    if (std::abs(scalar) < 1e-12) {
        throw std::invalid_argument(
            "Division by zero"
        );
    }

    for (auto& value : data) {
        value /= scalar;
    }

    return *this;
}

// ============================================================
// Factory functions
// ============================================================

inline Matrix Matrix::zeros(
    size_t r,
    size_t c
) {
    return Matrix(r, c, 0.0);
}

inline Matrix Matrix::ones(
    size_t r,
    size_t c
) {
    return Matrix(r, c, 1.0);
}

inline Matrix Matrix::identity(
    size_t size
) {
    Matrix result(
        size,
        size,
        0.0
    );

    for (size_t i = 0; i < size; ++i) {
        result(i, i) = 1.0;
    }

    return result;
}

// ============================================================
// Random matrix
// ============================================================

inline Matrix Matrix::random(
    size_t r,
    size_t c,
    double low,
    double high,
    unsigned seed
) {
    Matrix result(r, c);

    std::mt19937 gen(seed);

    std::uniform_real_distribution<double> dist(
        low,
        high
    );

    for (auto& value : result.data) {
        value = dist(gen);
    }

    return result;
}

// ============================================================
// Output
// ============================================================

inline void Matrix::print(
    std::ostream& os,
    int precision
) const {
    os << std::fixed
       << std::setprecision(precision);

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

} // namespace ml