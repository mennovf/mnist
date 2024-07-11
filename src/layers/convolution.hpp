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

  size_t padding;

  std::vector<Channel> channels;

  Convolution(size_t ih, size_t iw, size_t ic, size_t fh, size_t fw, size_t p, std::vector<Channel> cs): iheight{ih}, iwidth{iw}, ichannels{ic}, fheight{fh}, fwidth{fw}, padding{p}, channels{cs} {}
  Convolution(size_t ih, size_t iw, size_t ic, size_t fh, size_t fw, std::vector<Channel> cs): Convolution(ih, iw, ic, fh, fw, 0, cs) {}

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
    size_t oheight = iheight - fheight + 2*padding;
    size_t owidth = iwidth - fwidth + 2*padding;
    size_t osize = oheight * owidth;
    size_t isize = iwidth*iheight;
    Vec y(osize*this->channels.size());

    // For each output channel
    for (size_t ochannel = 0, ooutstart = 0; ochannel < this->channels.size(); ++ochannel, ooutstart += osize) {
      Channel& channel = this->channels[ochannel];

      // For each output "pixel"
      for (size_t orow = 0; orow < oheight; ++orow) {
        for (size_t ocol = 0; ocol < owidth; ++ocol) {

          double acc = 0;
          // For each input channel
          for (size_t ichannel : channel.input_channels) {

            // For each input pixel of the filter
            for (size_t frow = 0; frow < this->fheight; ++frow) {
              for (size_t fcol = 0; fcol < this->fwidth; ++fcol) {
                // Calculate the indices in the input channel
                size_t const iirow = orow + frow;
                size_t const iicol = ocol + fcol;

                // Check whether it is within the padding region
                if (iirow < padding || iirow >= iheight + padding || iicol < padding || iicol >= iwidth + padding) continue;

                // Correct to the real region
                size_t const irow = iirow - padding;
                size_t const icol = iicol - padding;
                acc += x[ichannel*isize + irow*iwidth + icol];
              }
            }
          }
          
          y[ooutstart + orow*owidth + ocol] = acc;
        }
      }
    }
    return x;
  }

};

