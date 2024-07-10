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


  virtual Vec eval(Vec const& x) override {
    Vec y =  this->weights * x + this->biases;
    y.apply(sigmoid);
    return y;
  }

  virtual Vec grad(Vec const& x) override {
    return x;
  }

  virtual void adjust_weights(Vec const& a) override {
    return ;
  }

};
