#pragma once
#include <vector>
#include <iostream>
#include <cmath>
#include <random>

struct Vec {
  std::vector<double> elements;

  Vec() = default;
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

  static double dot(Vec const& l, Vec const& r) {
    if (l.size() != r.size()) {
      std::cerr << "Invalid vector dimensions: " << l.size() << " . " << r.size() << std::endl;
      std::exit(-1);
    }

    return std::inner_product(l.elements.begin(), l.elements.end(), r.elements.begin(), 0.0);
  }

  void softmax() {
    this->apply(std::exp);
    double sum = 0;
    for (size_t i = 0; i < this->elements.size(); ++i) {
      sum += this->elements[i];
    }
    for (size_t i = 0; i < this->elements.size(); ++i) {
      this->elements[i] /= sum;
    }
  }

  Vec slice_n(size_t begin, size_t amount) const {
    Vec result(amount);
    for (size_t i = 0; i < amount; ++i) {
      result[i] = this->elements[begin + i];
    }
    return result;
  }

  template <class Random>
  void initialize(Random& r) {
    for (auto& el : this->elements) {
      el = r();
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
  std::vector<double> elements;

  Matrix(size_t rows, size_t columns): rows{rows}, columns{columns}, elements(rows*columns) {};
  Matrix(size_t rows, size_t columns, std::vector<double> elements): rows{rows}, columns{columns}, elements{elements} {};

  size_t size() const {
    return this->elements.size();
  }

  double const& at(size_t row, size_t col) const {
    return this->elements[row*columns + col];
  }

  double& at(size_t row, size_t col) {
    return const_cast<double &>(static_cast<Matrix const&>(*this).at(row, col));
  }

  void add_as_vec(Vec const& o) {
    if (o.size() != this->rows * this->columns) {
      std::cerr << "Trying to add vector of dimension " << o.size() << " to matrix of dimension " << this->rows << "x" << this->columns << std::endl;
      std::exit(-1);
    }

    for (size_t ri = 0; ri < this->rows; ++ri) {
      for (size_t ci = 0; ci < this->columns; ++ci) {
        this->at(ri, ci) += o[ri * this->columns + ci];
      }
    }
  }

  template <class Random>
  void initialize(Random& r) {
    for (auto& el : this->elements) {
      el = r();
    }
  }
};

inline Vec operator*(double p, Vec const& v) {
  Vec result(v.elements.size());
  for (size_t i = 0; i < v.elements.size(); ++i) {
    result[i] = p * v[i];
  }
  return result;
}
inline Vec operator*(Vec const& v, double p) {
  return p * v;
}


inline std::ostream& operator<<(std::ostream& out, Matrix const& m) {
  out << "[";
  for (size_t ri = 0; ri < m.rows; ++ri) {
    out << "[";
    for (size_t ci = 0; ci < m.columns; ++ci) {
      out << m.at(ri, ci) << ", ";
    }
    out << "],\n";
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

inline Vec operator-(Vec const& l, Vec const& r) {
  return l + (-1.*r);
}

inline Vec operator*(Matrix const& m, Vec const& v) {
  if (m.columns != v.elements.size()) {
    std::cerr << "Invalid matrix and vector dimensions: " << m.rows << "x" << m.columns << " * " << v.elements.size() << std::endl;
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

inline Vec grad_mat_mul(Vec const& v, Matrix const& m) {
  if (m.rows != v.elements.size()) {
    std::cerr << "Invalid matrix and vector dimensions: (" << m.rows << "x" << m.columns << ")^T * " << v.elements.size() << std::endl;
    std::exit(1);
  }

  Vec result(m.columns);

  for (size_t oi = 0; oi < result.size(); ++oi) {
    result[oi] = 0;
    for (size_t ri = 0; ri < m.rows; ++ri) {
      result[oi] += m.at(ri, oi) * v[ri];
    }
  }

  return result;
}

inline Vec hadamard_product(Vec const& l, Vec const& r) {
  if (l.elements.size() != r.elements.size()) {
    std::cerr << "Invalid vector dimensions." << l.elements.size() << " * " << r.elements.size() << std::endl;
    std::exit(1);
  }

  Vec result(l.elements.size());

  for (size_t i = 0; i < l.elements.size(); ++i) {
    result[i] = l[i] * r[i];
  }
  return result;
}



