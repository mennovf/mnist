#pragma once
#include "layers/layer.hpp"
#include <memory>
#include <vector>
#include "math.hpp"

struct NeuralNetwork {
  std::vector<std::unique_ptr<Layer>> layers;
  NeuralNetwork(std::vector<std::unique_ptr<Layer>> layers): layers{std::move(layers)} {};

  void reset() {
    for (auto& layer : this->layers) {
      layer->reset();
    }
  }

  void train(Vec const& x, Vec const& y) {
    /*
     * Vec const output = this->eval(x);
     * Vec const error = y - output;
     * 
     * Vec grad = this->grad(x);
     * for (auto begin = std::rbegin(this->layers); begin != std::rend(this->layers); ++begin) {
     *     grad = begin->grad(grad, rate);
     * }
     */
  };

  void descent_gradient(double const rate) {
    for (auto& layer : this->layers) {
      layer->descent_gradient(rate);
    }
  }
};

