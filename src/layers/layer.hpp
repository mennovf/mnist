#pragma once
#include "math.hpp"
#include <functional>

struct Layer {
  struct Gradient {
    Vec dx;
    Vec dw;
  }; 

  Vec const& forward(Vec const& x) {
    this->x = x;
    this->fx = this->eval(x);
    return this->fx;
  }

  virtual Gradient grad(Vec const&) = 0;
  virtual void adjust_weights(Vec const&) = 0;
  virtual void dump_weights(std::ostream&) const = 0;
  virtual void load_weights(std::istream&) = 0;
  virtual void initialize(std::function<double(void)>&) = 0;

  Layer() : x{}, fx{} {};

  protected:
  Vec x;
  Vec fx;

  virtual Vec eval(Vec const&) = 0;
};
