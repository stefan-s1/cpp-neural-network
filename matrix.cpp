#pragma once


// Conditionally include <execution> if available, and set up a macro for the execution policy.
#if __has_include(<execution>)
    #include <execution>
    // Macro expands to 'std::execution::par,' so that it is inserted before the iterators.
    #define EXECUTION_POLICY std::execution::par_unseq,
#else
    // If <execution> is not available, the macro expands to nothing.
    #define EXECUTION_POLICY
#endif

#include "matrix.h"
#include <functional>
#include <random>
#include <cassert>
#include <algorithm>  // for std::copy and std::transform


// --- QSMatrix Implementation ---

// Parameter Constructor: initialize with given number of rows, columns, and an initial value.
template<typename T>
Matrix<T>::Matrix(unsigned _rows, unsigned _cols, const T& _initial)
    : rows(_rows), cols(_cols), mat(_rows * _cols, _initial)
{
}

// Constructor from a vector-of-vector (assumes rectangular input)
// Stores the data in row-major order.
template<typename T>
Matrix<T>::Matrix(const std::vector<std::vector<T>>& _mat)
    : rows(_mat.size()), cols((_mat.empty() ? 0 : _mat[0].size())), mat(rows * cols)
{
    for (size_t i = 0; i < rows; ++i) {
        assert(_mat[i].size() == cols);  // ensure all rows have the same number of columns
        // Use std::copy to fill the corresponding row in 'mat'
        std::copy(_mat[i].begin(), _mat[i].end(), mat.begin() + i * cols);
    }
}

// Constructor from a vector (assumes row major input)
// Stores the data in row-major order.
template<typename T>
Matrix<T>::Matrix(const std::vector<T>& _mat, size_t _rows, size_t _cols)
    : rows(_rows), cols(_cols), mat(_mat)
{
}

// Create a random matrix with values in [-maxWeight, maxWeight].
template<typename T>
Matrix<T> Matrix<T>::initRandomQSMatrix(size_t _rows, size_t _cols, const T& maxWeight) {
    Matrix<T> my_matrix(_rows, _cols, T());
    static std::mt19937 gen{42}; // Fixed seed for reproducibility. Has to be static so we don't reseed the same fixed seed everytime
    std::uniform_real_distribution<T> d(-maxWeight, maxWeight);
    //std::normal_distribution<T> d(0, maxWeight / 4); // set s.d. to max / 4 so that 99.9% of values are less than max weight
    for (size_t i = 0; i < _rows * _cols; ++i) {
        my_matrix.mat[i] = d(gen);
    }
    return my_matrix;
}

// Move Constructor.
template<typename T>
Matrix<T>::Matrix(Matrix<T>&& rhs) noexcept
    : rows(rhs.rows), cols(rhs.cols), mat(std::move(rhs.mat))
{
    rhs.rows = 0;
    rhs.cols = 0;
}

// Copy Constructor.
template<typename T>
Matrix<T>::Matrix(const Matrix<T>& rhs)
    : rows(rhs.rows), cols(rhs.cols), mat(rhs.mat)
{
}

// Destructor.
template<typename T>
Matrix<T>::~Matrix() {}

// Assignment Operator.
template<typename T>
Matrix<T>& Matrix<T>::operator=(const Matrix<T>& rhs) {
    if (this == &rhs)
        return *this;
    rows = rhs.rows;
    cols = rhs.cols;
    mat = rhs.mat;
    return *this;
}

// Move Assignment Operator
template<typename T>
Matrix<T>& Matrix<T>::operator=(Matrix<T>&& rhs) noexcept {
    if (this != &rhs) {
        rows = std::exchange(rhs.rows, 0);
        cols = std::exchange(rhs.cols, 0);
        mat = std::move(rhs.mat);
    }
    return *this;
}


// Matrix addition with broadcasting support (optimized with .data())
template<typename T>
Matrix<T> Matrix<T>::operator+(const Matrix<T>& rhs) const {
    T const* lhs_data = mat.data();
    T const* rhs_data = rhs.mat.data();
    
    // Case 1: Standard elementwise addition
    if (rows == rhs.rows && cols == rhs.cols) {
        Matrix<T> result(rows, cols, T());
        T* res_data = result.mat.data();
        const size_t total = rows * cols;
        
        for (size_t i = 0; i < total; ++i) {
            res_data[i] = lhs_data[i] + rhs_data[i];
        }
        return result;
    }
    // Case 2: rhs is a row vector to be broadcast
    else if (rhs.rows == 1 && rhs.cols == cols) {
        Matrix<T> result(rows, cols, T());
        T* res_data = result.mat.data();
        
        for (size_t i = 0; i < rows; ++i) {
            const size_t row_offset = i * cols;
            for (size_t j = 0; j < cols; ++j) {
                res_data[row_offset + j] = lhs_data[row_offset + j] + rhs_data[j];
            }
        }
        return result;
    }
    // Case 3: this is a row vector to be broadcast
    else if (rows == 1 && cols == rhs.cols) {
        Matrix<T> result(rhs.rows, cols, T());
        T* res_data = result.mat.data();
        T const* this_row = lhs_data; // Only one row in this matrix
        
        for (size_t i = 0; i < rhs.rows; ++i) {
            const size_t rhs_offset = i * cols;
            for (size_t j = 0; j < cols; ++j) {
                res_data[rhs_offset + j] = this_row[j] + rhs_data[rhs_offset + j];
            }
        }
        return result;
    }
    else {
        assert(false && "Matrix dimensions incompatible for addition");
        return Matrix<T>(0, 0, T());
    }
}

// Optimized cumulative addition
template<typename T>
Matrix<T>& Matrix<T>::operator+=(const Matrix<T>& rhs) {
    assert(rows == rhs.rows && cols == rhs.cols);
    T* lhs_data = mat.data();
    T const* rhs_data = rhs.mat.data();
    const size_t total = rows * cols;
    
    for (size_t i = 0; i < total; ++i) {
        lhs_data[i] += rhs_data[i];
    }
    return *this;
}

// Optimized matrix subtraction
template<typename T>
Matrix<T> Matrix<T>::operator-(const Matrix<T>& rhs) const {
    assert(rows == rhs.rows && cols == rhs.cols);
    Matrix<T> result(rows, cols, T());
    T const* lhs_data = mat.data();
    T const* rhs_data = rhs.mat.data();
    T* res_data = result.mat.data();
    const size_t total = rows * cols;
    
    for (size_t i = 0; i < total; ++i) {
        res_data[i] = lhs_data[i] - rhs_data[i];
    }
    return result;
}

// Optimized cumulative subtraction
template<typename T>
Matrix<T>& Matrix<T>::operator-=(const Matrix<T>& rhs) {
    assert(rows == rhs.rows && cols == rhs.cols);
    T* lhs_data = mat.data();
    T const* rhs_data = rhs.mat.data();
    const size_t total = rows * cols;
    
    for (size_t i = 0; i < total; ++i) {
        lhs_data[i] -= rhs_data[i];
    }
    return *this;
}

// Optimized matrix multiplication
template<typename T>
Matrix<T> Matrix<T>::operator*(const Matrix<T>& rhs) const {
    assert(cols == rhs.rows);
    Matrix<T> result(rows, rhs.cols, T());
    T* res_data = result.mat.data();
    T const* lhs_data = mat.data();
    T const* rhs_data = rhs.mat.data();
    const size_t rhs_cols = rhs.cols;
    
    for (size_t i = 0; i < rows; ++i) {
        const size_t res_offset = i * rhs_cols;
        for (size_t k = 0; k < cols; ++k) {
            const T temp = lhs_data[i * cols + k];
            const size_t rhs_offset = k * rhs_cols;
            
            // Manual SIMD-like optimization opportunity here
            for (size_t j = 0; j < rhs_cols; ++j) {
                res_data[res_offset + j] += temp * rhs_data[rhs_offset + j];
            }
        }
    }
    return result;
}

// Cumulative multiplication.
template<typename T>
Matrix<T>& Matrix<T>::operator*=(const Matrix<T>& rhs) {
    *this = std::move((*this) * rhs);
    return *this;
}

// Transpose (non in-place).
template<typename T>
Matrix<T> Matrix<T>::transpose() const {

    if (rows == 1 || cols == 1)
        return Matrix<T>(this->mat, cols, rows); 
    
    Matrix<T> result(cols, rows, T());
    
    for (size_t i = 0; i < rows; ++i) {
        unsigned int offset = i * cols;
        for (size_t j = 0; j < cols; ++j) {
            result(j, i) = (*this).mat[offset + j];
        }
    }
    return result;
}

// In-place transpose.
template<typename T>
Matrix<T>& Matrix<T>::transpose_in_place() {
    
    if (rows == 1 || cols == 1) {
        unsigned int temp = cols;
        cols = rows;
        rows = temp;
        return *this;
    }

    // for square matrices this can be optimized by swapping along the diagonal
    if (rows == cols) {
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = i + 1; j < cols; ++j) {
                std::swap((*this)(i, j), (*this)(j, i));
            }
        }
        return *this;
    }

    *this = std::move(this->transpose());
    return *this;
}

// Matrix/scalar addition.
template<typename T>
Matrix<T> Matrix<T>::operator+(const T& rhs) const {
    Matrix<T> result(rows, cols, T());
    std::transform(EXECUTION_POLICY  mat.begin(), mat.end(), result.mat.begin(),
                   [rhs](T val) { return val + rhs; });
    return result;
}

// Matrix/scalar subtraction.
template<typename T>
Matrix<T> Matrix<T>::operator-(const T& rhs) const {
    Matrix<T> result(rows, cols, T());
    std::transform(EXECUTION_POLICY  mat.begin(), mat.end(), result.mat.begin(),
                   [rhs](T val) { return val - rhs; });
    return result;
}

// Matrix/scalar multiplication.
template<typename T>
Matrix<T> Matrix<T>::operator*(const T& rhs) const {
    Matrix<T> result(rows, cols, T());
    std::transform(EXECUTION_POLICY  mat.begin(), mat.end(), result.mat.begin(),
                   [rhs](T val) { return val * rhs; });
    return result;
}

// Cumulative scalar multiplication.
template<typename T>
Matrix<T>& Matrix<T>::operator*=(const T& rhs) {
    for (size_t i = 0; i < rows * cols; ++i) {
        mat[i] *= rhs;
    }
    return *this;
}

// Matrix/scalar division.
template<typename T>
Matrix<T> Matrix<T>::operator/(const T& rhs) const {
    Matrix<T> result(rows, cols, T());
    std::transform(EXECUTION_POLICY mat.begin(), mat.end(), result.mat.begin(),
                   [rhs](T val) { return val / rhs; });
    return result;
}

// Matrix-vector multiplication (vector size must equal number of columns).
template<typename T>
std::vector<T> Matrix<T>::operator*(const std::vector<T>& rhs) {
    assert(rhs.size() == cols);
    std::vector<T> result(rows, T());
    for (size_t i = 0; i < rows; ++i) {
        T sum = T();
        for (size_t j = 0; j < cols; ++j) {
            sum += (*this)(i, j) * rhs[j];
        }
        result[i] = sum;
    }
    return result;
}

// Return a vector containing the diagonal elements.
template<typename T>
std::vector<T> Matrix<T>::diag_vec() {
    size_t n = (rows < cols) ? rows : cols;
    std::vector<T> result(n, T());
    for (size_t i = 0; i < n; ++i) {
        result[i] = (*this)(i, i);
    }
    return result;
}


// Generic apply_transform
template<typename T>
template<typename Fn>
Matrix<T> Matrix<T>::apply_transform(Fn&& fn) const {
    Matrix<T> result(rows, cols, T());
    std::transform(EXECUTION_POLICY mat.begin(), mat.end(), result.mat.begin(), std::forward<Fn>(fn));
    return result;
}

// Generic apply_transform
template<typename T>
template<typename Fn>
void Matrix<T>::apply_inplace_transform(Fn&& fn) {
    Matrix<T> result(rows, cols, T());
    std::transform(EXECUTION_POLICY mat.begin(), mat.end(), mat.begin(), std::forward<Fn>(fn));
    return result;
}

// Overload for std::function
template<typename T>
Matrix<T> Matrix<T>::component_wise_transformation(const std::function<T(T)>& transformation) const {
    return apply_transform(transformation);
}
// In-place component-wise transformation.
template<typename T>
Matrix<T>& Matrix<T>::component_wise_transformation_in_place(const std::function<T(T)>& transformation) {
    apply_inplace_transform(transformation);
    return *this;
}

// Overload for function pointers
template<typename T>
Matrix<T> Matrix<T>::component_wise_transformation(T (*transformation)(T)) const {
    return apply_transform(transformation);
}
// In-place component-wise transformation.
template<typename T>
Matrix<T>& Matrix<T>::component_wise_transformation_in_place(T (*transformation)(T)) {
    apply_inplace_transform(transformation);
    return *this;
}


// Overloaded operator() for non-const element access.
template<typename T>
T& Matrix<T>::operator()(const unsigned& row, const unsigned& col) {
    assert(row < rows && col < cols);
    return mat[row * cols + col];
}

// Overloaded operator() for const element access.
template<typename T>
const T& Matrix<T>::operator()(const unsigned& row, const unsigned& col) const {
    assert(row < rows && col < cols);
    return mat[row * cols + col];
}

// Return the number of rows.
template<typename T>
unsigned Matrix<T>::get_rows() const {
    return rows;
}

// Return the number of columns.
template<typename T>
unsigned Matrix<T>::get_cols() const {
    return cols;
}

// Element-wise (Hadamard) multiplication.
template<typename T>
Matrix<T> Matrix<T>::hadamardMultiplication(const Matrix<T>& rhs) const {
    assert(rows == rhs.rows && cols == rhs.cols);
    Matrix<T> result(rows, cols, T());
    std::transform(EXECUTION_POLICY mat.begin(), mat.end(), rhs.mat.begin(), result.mat.begin(),
                   std::multiplies<T>());
    return result;
}

// In-place Hadamard multiplication.
template<typename T>
Matrix<T>& Matrix<T>::hadamardMultiplicationInPlace(const Matrix<T>& rhs) {
    assert(rows == rhs.rows && cols == rhs.cols);
    for (size_t i = 0; i < rows * cols; ++i) {
        mat[i] *= rhs.mat[i];
    }
    return *this;
}