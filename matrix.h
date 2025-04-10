#pragma once

#include <vector>
#include <functional>

template <typename T> 
class Matrix {
 private:
 
  size_t rows; 
  size_t cols;
  std::vector<T> mat; // row-major data
  
  template<typename Fn>
  Matrix<T> apply_transform(Fn&& fn) const;
 
  template<typename Fn>
  void apply_inplace_transform(Fn&& fn);

 public:
  Matrix(size_t _rows, size_t _cols, const T& _initial);
  Matrix(const Matrix<T>& rhs);
  Matrix(const std::vector<std::vector<T>>& _mat);
  Matrix(const std::vector<T>& _mat, size_t _rows, size_t _cols);
  Matrix(Matrix<T>&& rhs) noexcept;

  static Matrix<T> initRandomMatrix(size_t _rows, size_t _cols, const T& maxWeight);

  virtual ~Matrix();

  // Operator overloading, for "standard" mathematical matrix operations                                                                                                                                                          
  Matrix<T>& operator=(const Matrix<T>& rhs);
  Matrix<T>& operator=(Matrix<T>&& rhs) noexcept;

  // Matrix mathematical operations                                                                                                                                                                                               
  Matrix<T> operator+(const Matrix<T>& rhs) const;
  Matrix<T>& operator+=(const Matrix<T>& rhs);
  Matrix<T> operator-(const Matrix<T>& rhs) const;
  Matrix<T>& operator-=(const Matrix<T>& rhs);
  Matrix<T> operator*(const Matrix<T>& rhs) const;
  Matrix<T>& operator*=(const Matrix<T>& rhs);
  Matrix<T> transpose() const;
  Matrix<T>& transpose_in_place();

  // Matrix/scalar operations                                                                                                                                                                                                     
  Matrix<T> operator+(const T& rhs) const;
  Matrix<T> operator-(const T& rhs) const;
  Matrix<T> operator*(const T& rhs) const;
  Matrix<T>& operator*=(const T& rhs);
  Matrix<T> operator/(const T& rhs) const;

  Matrix<T> hadamardMultiplication(const Matrix<T>& rhs) const;
  Matrix<T>& hadamardMultiplicationInPlace(const Matrix<T>& rhs);

  // Overloads for std::function (for flexibility) and function pointers (for performance)
  Matrix<T> component_wise_transformation(const std::function<T(T)>& transformation) const;
  Matrix<T> component_wise_transformation(T (*transformation)(T)) const;

  Matrix<T>& component_wise_transformation_in_place(const std::function<T(T)>& transformation);
  Matrix<T>& component_wise_transformation_in_place(T (*transformation)(T));

  // Matrix/vector operations                                                                                                                                                                                                     
  std::vector<T> operator*(const std::vector<T>& rhs) const;
  std::vector<T> diag_vec() const;

  // Access the individual elements                                                                                                                                                                                               
  T& operator()(const size_t& row, const size_t& col);
  const T& operator()(const size_t& row, const size_t& col) const;

  // Access the row and column sizes                                                                                                                                                                                              
  size_t get_rows() const;
  size_t get_cols() const;

  // 0 indexed
  std::vector<T> get_row_copy(size_t row) const;
  std::vector<T> get_col_copy(size_t col) const;

};
#include "matrix.cpp"