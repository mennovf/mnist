#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>
#include <stdint.h>
#include <vector>

namespace fs = std::filesystem;

struct Images {
    std::vector<uint8_t> labels;
    std::vector<uint8_t> pixels;
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
        train.labels.resize(amount);
        training_labels.read((char*)train.labels.data(), amount);
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
        train.pixels.resize(amount*rows*columns);
        training_images.read((char*)train.pixels.data(), n);
    }
    
    {
        std::ifstream test_labels(directory / "t10k-labels-idx1-ubyte", std::fstream::binary);
        if (uint32_t mn = read32be(test_labels); mn != 0x801) {
            std::cerr << "Invalid magic number: " << mn << std::endl;
            std::exit(1);
        }
        uint32_t const amount = read32be(test_labels);
        test.labels.resize(amount);
        test_labels.read((char*)test.labels.data(), amount);
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
        test.pixels.resize(amount*rows*columns);
        test_images.read((char*)train.pixels.data(), n);
    }

    return {.test=test, .train=train};
}

namespace std {
inline ostream& operator<<(ostream& os, Images const& images) {
    os << "Nlabels: " << images.labels.size() << " Npixels: " << images.pixels.size() << std::endl;
    return os;
}
}

