#pragma once
#include "layers/layer.hpp"
#include "layers/fullyconnected.hpp"
#include "layers/function.hpp"
#include <memory>
#include <vector>
#include "math.hpp"

struct NeuralNetwork {
  std::vector<std::unique_ptr<Layer>> layers;
  std::vector<Vec> gradients; // In reverse order

  NeuralNetwork(std::initializer_list<Layer*> init): layers{init.begin(), init.end()} {};

  void reset() {
    this->gradients.clear();
  }

  Vec forward(Vec x) {
    for (auto& layer : this->layers) {
      x = layer->forward(x);
    }
    return x;
  }

  void train(Vec const& x, Vec const& y) {
     Vec output = this->forward(x);
     output.softmax();
     Vec const error = y - output;
     
     Layer::Gradient grad = {.dx = error, .dw = Vec()};
     size_t idx = 0;
     for (auto ilayer = std::rbegin(this->layers); ilayer != std::rend(this->layers); ++ilayer) {
         grad = (*ilayer)->grad(grad.dx);
         if (gradients.size() != layers.size()) {
             gradients.push_back(grad.dw);
         } else {
             gradients[idx] = gradients[idx] + grad.dw;
         }
         ++idx;
     }
  };

  void descent_gradient(double const rate) {
    for (size_t i = 0; i < this->layers.size(); ++i) {
      this->layers[i]->adjust_weights(rate * this->gradients[this->gradients.size() - 1 - i]);
    }
  }

  void dump_weights(std::ostream& out) const {
    for (auto const& layer : this->layers) {
      layer->dump_weights(out);
    }
  }

  void load_weights(std::istream& in) {
    for (auto const& layer : this->layers) {
      layer->load_weights(in);
    }
  }
};

