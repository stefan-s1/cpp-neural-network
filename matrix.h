#ifndef __QS_MATRIX_H
#define __QS_MATRIX_H

#include <vector>
#include <functional>


// link to original website: https://www.quantstart.com/articles/Matrix-Classes-in-C-The-Header-File/


template <typename T> class QSMatrix {
 private:
 unsigned rows; 
 unsigned cols;
 std::vector<T> mat; // row-major data

 public:
  QSMatrix(unsigned _rows, unsigned _cols, const T& _initial);
  QSMatrix(const QSMatrix<T>& rhs);
  QSMatrix(const std::vector<std::vector<T>>& _mat);
  QSMatrix(const std::vector<T>& _mat, size_t _rows, size_t _cols);
  QSMatrix(QSMatrix<T>&& rhs) noexcept;

  static QSMatrix<T> initRandomQSMatrix(size_t _rows, size_t _cols, const T& maxWeight);

  virtual ~QSMatrix();

  // Operator overloading, for "standard" mathematical matrix operations                                                                                                                                                          
  QSMatrix<T>& operator=(const QSMatrix<T>& rhs);

  // Matrix mathematical operations                                                                                                                                                                                               
  QSMatrix<T> operator+(const QSMatrix<T>& rhs) const;
  QSMatrix<T>& operator+=(const QSMatrix<T>& rhs);
  QSMatrix<T> operator-(const QSMatrix<T>& rhs) const;
  QSMatrix<T>& operator-=(const QSMatrix<T>& rhs);
  QSMatrix<T> operator*(const QSMatrix<T>& rhs) const;
  QSMatrix<T>& operator*=(const QSMatrix<T>& rhs);
  QSMatrix<T> transpose() const;
  QSMatrix<T>& transpose_in_place();

  // Matrix/scalar operations                                                                                                                                                                                                     
  QSMatrix<T> operator+(const T& rhs) const;
  QSMatrix<T> operator-(const T& rhs) const;
  QSMatrix<T> operator*(const T& rhs) const;
  QSMatrix<T>& operator*=(const T& rhs);
  QSMatrix<T> operator/(const T& rhs) const;

  QSMatrix<T> hadamardMultiplication(const QSMatrix<T>& rhs) const;
  QSMatrix<T>& hadamardMultiplicationInPlace(const QSMatrix<T>& rhs);

  QSMatrix<T> component_wise_transformation(const std::function<T(T)>& transformation) const;
  QSMatrix<T>& component_wise_transformation_in_place(const std::function<T(T)>& transformation);

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

#endif