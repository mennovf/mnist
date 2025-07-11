#pragma once

#include "math.hpp"
#include "layers/layer.hpp"
#include <vector>
#include <stdint.h>
#include <stddef.h>

struct Convolution : public Layer {
  struct Channel {
    Vec weights;
    double bias;
    std::vector<size_t> input_channels;
    Channel(std::vector<size_t> ics): input_channels{ics} {}
  };

  size_t iheight;
  size_t iwidth;
  size_t ichannels;

  size_t fheight;
  size_t fwidth;

  size_t padding;

  std::vector<Channel> channels;
  std::vector<size_t> weights_start;

  size_t nweights;

  Convolution(size_t ih, size_t iw, size_t ic, size_t fh, size_t fw, size_t p, std::vector<Channel> cs): iheight{ih}, iwidth{iw}, ichannels{ic}, fheight{fh}, fwidth{fw}, padding{p}, channels{cs} {
    this->nweights = 0;
    for (Channel& c : this->channels) {
      c.weights.elements.resize(c.input_channels.size()*this->fheight*this->fwidth);
      this->weights_start.push_back(this->nweights);
      this->nweights += c.weights.size() + 1;
    }
    this->weights_start.push_back(this->nweights);
  }
  
  Convolution(size_t ih, size_t iw, size_t ic, size_t fh, size_t fw, std::vector<Channel> cs): Convolution(ih, iw, ic, fh, fw, 0, cs) {}

  virtual void dump_weights(std::ostream& out) const override {
    for (size_t channelidx = 0; channelidx < this->channels.size(); ++channelidx) {
      Channel const& channel = this->channels[channelidx];
      out.write((char const *)channel.weights.elements.data(), channel.weights.size()*(sizeof (decltype(channel.weights.elements)::value_type)));
      out.write((char const *)&channel.bias, sizeof(channel.bias));
    }
  }
  
  virtual void load_weights(std::istream& in) override {
    for (size_t channelidx = 0; channelidx < this->channels.size(); ++channelidx) {
      Channel const& channel = this->channels[channelidx];
      in.read((char *)channel.weights.elements.data(), channel.weights.size()*(sizeof (decltype(channel.weights.elements)::value_type)));
      in.read((char *)&channel.bias, sizeof(channel.bias));
    }
  }

  virtual void initialize(std::function<double(void)>& d) override {
    for (Channel& channel : this->channels) {
      channel.weights.initialize(d);
      channel.bias = d();
    }
  }

  virtual Gradient grad(Vec const& uppergrad) override {
    size_t const oheight = 1 + iheight - fheight + 2*padding;
    size_t const owidth = 1 + iwidth - fwidth + 2*padding;
    size_t const osize = oheight * owidth;
    size_t const isize = iwidth*iheight;
    size_t const fsize = this->fwidth*this->fheight;

    Vec dx(isize * this->ichannels);
    Vec dw(this->nweights);

    /*********** Adjusted eval code ********************/
    for (size_t ochannel = 0, ooutstart = 0; ochannel < this->channels.size(); ++ochannel, ooutstart += osize) {
      Channel& channel = this->channels[ochannel];

      // For each output "pixel"
      for (size_t orow = 0; orow < oheight; ++orow) {
        for (size_t ocol = 0; ocol < owidth; ++ocol) {

          double const y = uppergrad[ooutstart + orow*owidth + ocol];
          // For each input channel
          for (size_t ichannelidx = 0; ichannelidx < channel.input_channels.size(); ++ichannelidx) {
            size_t const ichannel = channel.input_channels[ichannelidx];

            // For each input pixel of the filter
            for (size_t frow = 0; frow < this->fheight; ++frow) {
              for (size_t fcol = 0; fcol < this->fwidth; ++fcol) {
                // Calculate the indices in the input channel (with "imaginary" padding)
                size_t const iirow = orow + frow;
                size_t const iicol = ocol + fcol;

                // Check whether it is within the padding region
                if (iirow < padding || iirow >= iheight + padding || iicol < padding || iicol >= iwidth + padding) continue;

                // Correct to the non-padding region
                size_t const irow = iirow - padding;
                size_t const icol = iicol - padding;

                // Modify dx and dw
                dw[this->weights_start[ochannel] + ichannelidx*fsize + frow*this->fwidth + fcol] += y*this->x[ichannel*isize + irow*iwidth + icol];
                dx[ichannel*isize + irow*iwidth + icol] += y*channel.weights[ichannelidx * fsize + frow*this->fwidth + fcol];
              }
            }
          }

          // Bias
          dw[this->weights_start[ochannel] + channel.input_channels.size()*fsize] += y;
        }
      }
    }


    return {
      .dx = dx,
      .dw = dw
    };
  };

  virtual void adjust_weights(Vec const& weights) override {
    for (size_t channelidx = 0; channelidx < this->channels.size(); ++channelidx) {
      this->channels[channelidx].weights = this->channels[channelidx].weights + weights.slice_n(this->weights_start[channelidx], this->channels[channelidx].weights.size());
      this->channels[channelidx].bias = this->channels[channelidx].bias + weights[this->weights_start[channelidx + 1] - 1];
    }
  }

  private:
  virtual Vec eval(Vec const& x) override {
    size_t const oheight = 1 + iheight - fheight + 2*padding;
    size_t const owidth = 1 + iwidth - fwidth + 2*padding;
    size_t const osize = oheight * owidth;
    size_t const isize = iwidth*iheight;
    size_t const fsize = this->fwidth * this->fheight;
    Vec y(osize*this->channels.size());

    // For each output channel
    for (size_t ochannel = 0, ooutstart = 0; ochannel < this->channels.size(); ++ochannel, ooutstart += osize) {
      Channel& channel = this->channels[ochannel];

      // For each output "pixel"
      for (size_t orow = 0; orow < oheight; ++orow) {
        for (size_t ocol = 0; ocol < owidth; ++ocol) {

          //std::cout << "y(" << ochannel << "," << orow << "," << ocol << ") <-\n";
          double acc = 0;
          // For each input channel
          for (size_t ichannelidx = 0; ichannelidx < channel.input_channels.size(); ++ichannelidx) {
            size_t const ichannel = channel.input_channels[ichannelidx];

            // For each input pixel of the filter
            for (size_t frow = 0; frow < this->fheight; ++frow) {
              for (size_t fcol = 0; fcol < this->fwidth; ++fcol) {
                // Calculate the indices in the input channel (with "imaginary" padding)
                size_t const iirow = orow + frow;
                size_t const iicol = ocol + fcol;

                // Check whether it is within the padding region
                if (iirow < padding || iirow >= iheight + padding || iicol < padding || iicol >= iwidth + padding) continue;

                // Correct to the non-padding region
                size_t const irow = iirow - padding;
                size_t const icol = iicol - padding;
                //std::cout << channel.weights.size() << "Reading weight at: " << ichannelidx * fsize + frow*this->fwidth + fcol << std::endl;;
                const double weight = channel.weights[ichannelidx * fsize + frow*this->fwidth + fcol];
                //std::cout << x.size() << "Reading x at: " << ichannel*isize + irow*iwidth + icol << std::endl;;
                const double xv = x[ichannel*isize + irow*iwidth + icol];
                acc += weight * xv;
                //std::cout << "    w(" << ichannelidx << ","<<frow << "," << fcol << ") * x(" <<  ichannel << "," << irow << "," << icol << ")\n";
              }
            }
          }
          
          y[ooutstart + orow*owidth + ocol] = acc + channel.bias;
        }
      }
    }
    return y;
  }

};

