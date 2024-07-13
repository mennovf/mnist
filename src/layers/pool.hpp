#pragma once
#include "math.hpp"
#include "layers/layer.hpp"

struct AveragePooling : public Layer {
  size_t iheight;
  size_t iwidth;
  
  size_t pheight;
  size_t pwidth;

  AveragePooling(size_t ih, size_t iw, size_t ph, size_t pw): iheight{ih}, iwidth{iw}, pheight{ph}, pwidth{pw} {};

  virtual Gradient grad(Vec const& uppergrad) override {
    size_t const owidth = iwidth / pwidth;
    size_t const oheight = iheight / pheight;
    size_t const pchannels = uppergrad.size() / (owidth * oheight);

    size_t const psize = pwidth * pheight;
    size_t const isize = iwidth * iheight;
    size_t const osize = owidth*oheight;
    size_t const nin = isize*pchannels;
    double const Ninv = 1. / psize;
    
    Vec dx(nin);
    for (size_t ichannel = 0; ichannel < pchannels; ++ichannel) {
      for (size_t orow = 0; orow < oheight; ++orow) {
        for (size_t ocol = 0; ocol < owidth; ++ocol) {
          // Loop over the filter window
          for (size_t prow = 0; prow < pheight; ++prow) {
            for (size_t pcol = 0; pcol < pwidth; ++pcol) {
              size_t const iidx = (orow*pheight + prow)*iwidth + (ocol*pwidth + pcol); // Index within the channel
              dx[ichannel*isize + iidx] = Ninv * uppergrad[ichannel*osize + orow*owidth + ocol];
            }
          }
        }
      }
    }

    return {
      .dx = dx,
      .dw = Vec()
    };
  };

  virtual void dump_weights(std::ostream&) const override {}
  virtual void load_weights(std::istream&) override {}

  virtual void adjust_weights(Vec const& wsandbs) override {
    (void)wsandbs;
    return;
  }

  private:
  virtual Vec eval(Vec const& x) override {
    size_t const pchannels = x.size() / (iwidth * iheight);
    size_t const owidth = iwidth / pwidth;
    size_t const oheight = iheight / pheight;

    size_t const psize = pwidth * pheight;
    size_t const isize = iwidth * iheight;
    size_t const osize = owidth*oheight;
    size_t const nout = osize*pchannels;
    double const Ninv = 1. / psize;
    
    Vec y(nout);
    for (size_t ichannel = 0; ichannel < pchannels; ++ichannel) {
      for (size_t orow = 0; orow < oheight; ++orow) {
        for (size_t ocol = 0; ocol < owidth; ++ocol) {
          // Loop over the filter window
          double acc = 0;
          for (size_t prow = 0; prow < pheight; ++prow) {
            for (size_t pcol = 0; pcol < pwidth; ++pcol) {
              size_t const iidx = (orow*pheight + prow)*iwidth + (ocol*pwidth + pcol); // Index within the channel
              acc += x[ichannel*isize + iidx];
            }
          }
          y[ichannel*osize + orow*owidth + ocol] = Ninv * acc;
        }
      }
    }

    return y;
  }

  virtual void initialize(std::function<double(void)>&) override {};
};


