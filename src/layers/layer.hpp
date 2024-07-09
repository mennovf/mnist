#pragma once
#include "math.hpp"

struct Layer {
  virtual Vec eval(Vec const&) = 0;
  virtual Vec grad(Vec const&) = 0;
};
