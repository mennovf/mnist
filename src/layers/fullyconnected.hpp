#pragma once
#include "layers/layer.hpp"
#include <cmath>

inline double sigmoid(double x) {
  return 1.0 / (1.0 + std::exp(-x));
}

struct FullyConnected : public Layer {
  size_t ninputs;
  size_t nneurons;

  Matrix weights;
  Vec biases;

  FullyConnected(size_t ninputs, size_t nneurons): ninputs{ninputs}, nneurons{nneurons}, weights{nneurons, ninputs}, biases{nneurons} {};


  Vec eval(Vec const& x) {
    Vec y =  this->weights * x + this->biases;
    y.apply(sigmoid);
    return y;
  }
};
