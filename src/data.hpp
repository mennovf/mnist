#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>
#include <stdint.h>
#include <vector>

namespace fs = std::filesystem;

using Label = std::vector<double>;
using Image = std::vector<double>;
struct Images {
    std::vector<Label> labels;
    std::vector<Image> images;
};

struct Data {
    Images test;
    Images train;
};

inline uint32_t read32be(std::ifstream& ifile) {
    uint8_t bytes[4];
    ifile.read((char*)bytes, sizeof(bytes));
    return ((uint32_t)bytes[0] << 24) | ((uint32_t)bytes[1] << 16) | ((uint32_t)bytes[2] << 8) | ((uint32_t)bytes[3]);
}

inline Data data(fs::path directory) {
    Images train, test;
    
    {
        std::ifstream training_labels(directory / "train-labels-idx1-ubyte", std::fstream::binary);
        if (uint32_t mn = read32be(training_labels); mn != 0x801) {
            std::cerr << "Invalid magic number: " << mn << std::endl;
            std::exit(1);
        }
        uint32_t const amount = read32be(training_labels);
        std::vector<uint8_t> raw_labels(amount);
        training_labels.read((char*)raw_labels.data(), amount);

        train.labels.resize(amount);
        for (size_t i = 0; i < amount; ++i) {
            train.labels[i].resize(10);
            train.labels[i][raw_labels[i]] = 1.0;
        }
    }

    {
        std::ifstream training_images(directory / "train-images-idx3-ubyte", std::fstream::binary);
        if (uint32_t mn = read32be(training_images); mn != 0x803) {
            std::cerr << "Invalid magic number: " << mn << std::endl;
            std::exit(1);
        }
        uint32_t const amount = read32be(training_images);
        uint32_t const rows = read32be(training_images);
        uint32_t const columns = read32be(training_images);
        uint32_t const n = rows * columns * amount;
        std::vector<uint8_t> raw_pixels(n);
        training_images.read((char*)raw_pixels.data(), n);

        train.images.resize(amount);
        for (size_t i = 0; i < amount; ++i) {
            train.images[i].resize(rows * columns);
            for (size_t j = 0; j < rows * columns; ++j) {
                train.images[i][j] = raw_pixels[i * rows * columns + j] / 255.0;
            }
        }
    }
    
    {
        std::ifstream test_labels(directory / "t10k-labels-idx1-ubyte", std::fstream::binary);
        if (uint32_t mn = read32be(test_labels); mn != 0x801) {
            std::cerr << "Invalid magic number: " << mn << std::endl;
            std::exit(1);
        }
        uint32_t const amount = read32be(test_labels);
        std::vector<uint8_t> raw_labels(amount);
        test_labels.read((char*)raw_labels.data(), amount);

        test.labels.resize(amount);
        for (size_t i = 0; i < amount; ++i) {
            test.labels[i].resize(10);
            test.labels[i][raw_labels[i]] = 1.0;
        }
    }

    {
        std::ifstream test_images(directory / "t10k-images-idx3-ubyte", std::fstream::binary);
        if (uint32_t mn = read32be(test_images); mn != 0x803) {
            std::cerr << "Invalid magic number: " << mn << std::endl;
            std::exit(1);
        }
        uint32_t const amount = read32be(test_images);
        uint32_t const rows = read32be(test_images);
        uint32_t const columns = read32be(test_images);
        uint32_t const n = rows * columns * amount;
        std::vector<uint8_t> raw_pixels(n);
        test_images.read((char*)raw_pixels.data(), n);

        test.images.resize(amount);
        for (size_t i = 0; i < amount; ++i) {
            test.images[i].resize(rows * columns);
            for (size_t j = 0; j < rows * columns; ++j) {
                test.images[i][j] = raw_pixels[i * rows * columns + j] / 255.0;
            }
        }
    }

    return {.test=test, .train=train};
}

namespace std {
inline ostream& operator<<(ostream& os, Images const& images) {
    os << "Nlabels: " << images.labels.size() << " Nimages: " << images.images.size() << std::endl;
    return os;
}
}

