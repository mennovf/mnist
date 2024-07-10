#pragma once
#include "layers/layer.hpp"
#include <cmath>

inline double sigmoid(double x) {
  return 1.0 / (1.0 + std::exp(-x));
}

inline double dsigmoid(double x) {
  double const emx = std::exp(-x);
  return emx / ((1.0 + emx) * (1.0 + emx));
}

struct FullyConnected : public Layer {
  size_t ninputs;
  size_t nneurons;

  Matrix weights;
  Vec biases;

  FullyConnected(size_t ninputs, size_t nneurons): ninputs{ninputs}, nneurons{nneurons}, weights{nneurons, ninputs}, biases{nneurons} {};



  virtual Gradient grad(Vec const& uppergrad) override {
    Vec dw(this->weights.size() + this->biases.size());
    size_t dwidx = 0;
    // Weights part
    for (size_t ri = 0; ri < this->weights.rows; ++ri) {
      for (size_t ci = 0; ci < this->weights.columns; ++ci) {
        dw[dwidx] = this->x[ci] * uppergrad[ci];
        ++dwidx;
      }
    }
    // Biases part
    for (size_t i = 0; i < this->biases.size(); ++i) {
      dw[dwidx] = uppergrad[i];
      ++dwidx;
    }

    Vec const dx = grad_mat_mul(uppergrad, this->weights);

    return {
      .dx = dx,
      .dw = dw
    };
  };

  virtual void adjust_weights(Vec const& wsandbs) override {
    this->weights.add_as_vec(wsandbs.slice_n(0, this->weights.size()));
    this->biases = this->biases + wsandbs.slice_n(this->weights.size(), this->biases.size());
  }

  private:
  virtual Vec eval(Vec const& x) override {
    Vec y =  this->weights * x + this->biases;
    y.apply(sigmoid);
    return y;
  }

};
