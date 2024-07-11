#pragma once

#include "math.hpp"
#include "layers/layer.hpp"

struct Convolution : public Layer {
  size_t iheight;
  size_t iwidth;
  size_t ichannels;

  size_t fheight;
  size_t fwidth;

  Convolution(size_t ih, size_t iw, size_t ic, size_t fh, size_t fw): iheight{ih}, iwidth{iw}, ichannels{ic}, fheight{fh}, fwidth{fw} {}

  virtual Gradient grad(Vec const& uppergrad) override {
    return {
      .dx = Vec(),
      .dw = Vec()
    };
  };

  virtual void adjust_weights(Vec const& wsandbs) override {
    (void)wsandbs;
    return;
  }

  private:
  virtual Vec eval(Vec const& x) override {
    return x;
  }

};

