#pragma once
#include "math.hpp"
#include "layers/layer.hpp"

inline double sigmoid(double x) {
  return 1.0 / (1.0 + std::exp(-x));
}

inline double dsigmoid(double x) {
  double const emx = std::exp(-x);
  return emx / ((1.0 + emx) * (1.0 + emx));
}

struct Sigmoid : public Layer {
  virtual Gradient grad(Vec const& uppergrad) override {
    Vec dsx = this->x;
    dsx.apply(dsigmoid);
    
    return {
      .dx = hadamard_product(dsx, uppergrad),
      .dw = Vec()
    };
  };

  virtual void adjust_weights(Vec const& wsandbs) override {
    (void)wsandbs;
    return;
  }

  private:
  virtual Vec eval(Vec const& x) override {
    Vec y = x;
    y.apply(sigmoid);
    return y;
  }

};

