#pragma once

#include "math.hpp"
#include "layers/layer.hpp"
#include <vector>
#include <stdint.h>
#include <stddef.h>

struct Convolution : public Layer {
  struct Channel {
    Vec weights;
    std::vector<size_t> input_channels;
  };

  size_t iheight;
  size_t iwidth;
  size_t ichannels;

  size_t fheight;
  size_t fwidth;

  std::vector<Channel> channels;

  Convolution(size_t ih, size_t iw, size_t ic, size_t fh, size_t fw, std::vector<Channel> cs): iheight{ih}, iwidth{iw}, ichannels{ic}, fheight{fh}, fwidth{fw}, channels{cs} {}

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

