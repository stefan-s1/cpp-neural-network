#pragma once

#include <vector>
#include <functional>


// link to original website: https://www.quantstart.com/articles/Matrix-Classes-in-C-The-Header-File/


template <typename T> class Matrix {
 private:
 
  unsigned rows; 
  unsigned cols;
  std::vector<T> mat; // row-major data
  
  template<typename Fn>
  Matrix<T> apply_transform(Fn&& fn) const;
 
  template<typename Fn>
  void apply_inplace_transform(Fn&& fn);

 public:
  Matrix(unsigned _rows, unsigned _cols, const T& _initial);
  Matrix(const Matrix<T>& rhs);
  Matrix(const std::vector<std::vector<T>>& _mat);
  Matrix(const std::vector<T>& _mat, size_t _rows, size_t _cols);
  Matrix(Matrix<T>&& rhs) noexcept;

  static Matrix<T> initRandomQSMatrix(size_t _rows, size_t _cols, const T& maxWeight);

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
  std::vector<T> operator*(const std::vector<T>& rhs);
  std::vector<T> diag_vec();

  // Access the individual elements                                                                                                                                                                                               
  T& operator()(const unsigned& row, const unsigned& col);
  const T& operator()(const unsigned& row, const unsigned& col) const;

  // Access the row and column sizes                                                                                                                                                                                              
  unsigned get_rows() const;
  unsigned get_cols() const;

};
#include "matrix.cpp"