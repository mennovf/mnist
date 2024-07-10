#pragma once
#include "layers/layer.hpp"
#include <memory>
#include <vector>
#include "math.hpp"

struct NeuralNetwork {
  std::vector<std::unique_ptr<Layer>> layers;
  std::vector<Vec> gradients; // In reverse order

  NeuralNetwork(std::vector<std::unique_ptr<Layer>> layers): layers{std::move(layers)} {};

  void reset() {
    this->gradients.clear();
  }

  void train(Vec const& x, Vec const& y) {
    /*
     * Vec const output = this->forward(x);
     * Vec const error = y - output;
     * 
     * Layer::Gradient grad = this->grad(x);
     * size_t idx = 0;
     * for (auto begin = std::rbegin(this->layers); begin != std::rend(this->layers); ++begin) {
     *     grad = begin->grad(grad.dx, rate);
     *     if (gradients.size() != layers.size()) {
     *         gradients.push_back(grad.dw);
     *     } else {
     *         gradients[idx] = gradients[idx] + grad.dw;
     *     }
     *     ++idx;
     * }
     */
  };

  void descent_gradient(double const rate) {
    for (size_t i = 0; i < this->layers.size(); ++i) {
      layer[i]->adjust_weights(rate * this->gradients[this->gradients.size() - 1 - i]);
    }
  }
};

