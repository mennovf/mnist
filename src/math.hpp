#pragma once
#include <vector>
#include <iostream>

struct Vec {
  std::vector<double> elements;

  Vec(size_t s): elements(s) {};
  Vec(std::vector<double> elems): elements{elems} {};

  double const& operator[](size_t idx) const {
    return this->elements[idx];
  }

  double& operator[](size_t idx) {
    return const_cast<double&>(static_cast<Vec const *>(this)->operator[](idx));
  }

  size_t size() const {
    return this->elements.size();
  }

  void apply(double (*f)(double)) {
    for (auto& val : this->elements) {
      val = f(val);
    }
  }
};

inline std::ostream& operator<<(std::ostream& out, Vec const& v) {
  out << "Vec[";
  for (auto val : v.elements) {
    out << val << ", ";
  }
  out << "]";
  return out;
}

struct Matrix {
  size_t rows;
  size_t columns;

  Matrix(size_t rows, size_t columns): elements(rows*columns) {};
  Matrix(size_t rows, size_t columns, std::vector<double> elements): rows{rows}, columns{columns}, elements{elements} {};

  std::vector<double> elements;
  double const& at(size_t row, size_t col) const {
    return this->elements[row*columns + col];
  }

  double& at(size_t row, size_t col) {
    return const_cast<double &>(static_cast<Matrix const&>(*this).at(row, col));
  }
};

inline std::ostream& operator<<(std::ostream& out, Matrix const& m) {
  out << "Matrix[";
  for (size_t ri = 0; ri < m.rows; ++ri) {
    out << "[";
    for (size_t ci = 0; ci < m.columns; ++ci) {
      out << m.at(ri, ci) << ", ";
    }
    out << "]";
  }
  out << "]";
  return out;
}

inline Vec operator+(Vec const& l, Vec const& r) {
  if (l.elements.size() != r.elements.size()) {
    std::cerr << "Invalid vector dimensions." << l.elements.size() << " + " << r.elements.size() << std::endl;
    std::exit(1);
  }

  Vec result(l.elements.size());

  for (size_t i = 0; i < l.elements.size(); ++i) {
    result[i] = l[i] + r[i];
  }

  return result;
}

inline Vec operator*(Matrix const& m, Vec const& v) {
  if (m.columns != v.elements.size()) {
    std::cerr << "Invalid matrix and vector dimensions." << m.rows << "x" << m.columns << " * " << v.elements.size() << std::endl;
    std::exit(1);
  }

  Vec result(m.rows);

  for (size_t oi = 0; oi < result.size(); ++oi) {
    result[oi] = 0;
    for (size_t ci = 0; ci < m.columns; ++ci) {
      result[oi] += m.at(oi, ci) * v[ci];
    }
  }

  return result;
}

