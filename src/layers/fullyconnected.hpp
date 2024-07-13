#pragma once
#include "layers/layer.hpp"
#include <cmath>

struct FullyConnected : public Layer {
  size_t ninputs;
  size_t nneurons;

  Matrix weights;
  Vec biases;

  FullyConnected(size_t ninputs, size_t nneurons): ninputs{ninputs}, nneurons{nneurons}, weights{nneurons, ninputs}, biases{nneurons} { };


  virtual void dump_weights(std::ostream& out) const override {
      out.write((char const*)this->weights.elements.data(), this->weights.elements.size()*(sizeof (decltype(this->weights.elements)::value_type)));
      out.write((char const*)this->biases.elements.data(), this->biases.elements.size()*(sizeof (decltype(this->biases.elements)::value_type)));
  }
  
  virtual void load_weights(std::istream& in) override {
      in.read((char *)this->weights.elements.data(), this->weights.elements.size()*(sizeof (decltype(this->weights.elements)::value_type)));
      in.read((char *)this->biases.elements.data(), this->biases.elements.size()*(sizeof (decltype(this->biases.elements)::value_type)));
  }

  virtual void initialize(std::function<double(void)>& d) override {
    this->weights.initialize(d);
    this->biases.initialize(d);
  }

  virtual Gradient grad(Vec const& uppergrad) override {
    Vec dw(this->weights.size() + this->biases.size());
    size_t dwidx = 0;
    // Weights part
    for (size_t ri = 0; ri < this->weights.rows; ++ri) {
      for (size_t ci = 0; ci < this->weights.columns; ++ci) {
        dw[dwidx] = this->x[ci] * uppergrad[ri];
        ++dwidx;
      }
    }
    // Biases part
    for (size_t i = 0; i < this->biases.size(); ++i) {
      dw[dwidx] = uppergrad[i];
      ++dwidx;
    }

    Vec const dx = grad_mat_mul(uppergrad, this->weights);

    return {
      .dx = dx,
      .dw = dw
    };
  };

  virtual void adjust_weights(Vec const& wsandbs) override {
    this->weights.add_as_vec(wsandbs.slice_n(0, this->weights.size()));
    this->biases = this->biases + wsandbs.slice_n(this->weights.size(), this->biases.size());
  }

  private:
  virtual Vec eval(Vec const& x) override {
    return this->weights * x + this->biases;
  }

};
