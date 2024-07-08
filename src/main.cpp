#include <iostream>
#include "data.hpp"

int main (int argc, char *argv[]) {
  auto const DATA = data("./data");
  std::cout << "Test ->\t" << DATA.test << std::endl;
  std::cout << "Training ->\t" << DATA.train << std::endl;
  return 0;
}
