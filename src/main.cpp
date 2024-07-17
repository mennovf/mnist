#include <iostream>
#include <GLFW/glfw3.h>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "data.hpp"
#include "math.hpp"
#include "layers/fullyconnected.hpp"
#include "layers/function.hpp"
#include "layers/pool.hpp"
#include "layers/convolution.hpp"
#include "neuralnetwork.hpp"
#include <memory>
#include <random>
#include <algorithm>
#include <fstream>

GLuint create_texture_from_pixels(uint8_t const * const pixels, int rows, int columns) {
    GLuint textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, columns, rows, 0, GL_RED, GL_UNSIGNED_BYTE, pixels);

    return textureID;
}

struct CLIOptions {
    char const * from_weights;
    char const * weights_out;
    char const * sgd_seed;
    char const * w_seed;
};

#define PS(p) ((p) ? (p) : "-")

std::ostream& operator<<(std::ostream& os, CLIOptions const& opts) {
    os << "from_weights: " << PS(opts.from_weights) << ", weights_out: " << PS(opts.weights_out) << ", sgd_seed: " << PS(opts.sgd_seed) << ", w_seed: " << PS(opts.w_seed);
    return os;
}

char * next_or_error(char ** argv, char const * const emsg) {
    char ** next = ++argv;
    if (*next == nullptr) {
        std::cerr << emsg << std::endl;
        std::exit(-1);
    }

    return *next;
}

int main(int argc, char ** argv) {
    auto const DATA = data("./data");

    CLIOptions opts = {};

    for (char ** arg = &argv[1]; arg != &argv[argc]; ++arg) {
        if (strcmp(*arg, "--from-weights") == 0) {
            opts.from_weights = next_or_error(arg, "Missing --from-weights argument"); 
        } else if (strcmp(*arg, "--weights-out") == 0) {
            opts.weights_out = next_or_error(arg, "Missing --weights-out argument");
        } else if (strcmp(*arg, "--seed-weights") == 0) {
            opts.w_seed = next_or_error(arg, "Missing --seed-weights argument");
        } else if (strcmp(*arg, "--seed-sgd") == 0) {
            opts.sgd_seed = next_or_error(arg, "Missing --seed-sgd argument");
        }
    }

    std::cout << "Running with options=" << opts << std::endl;

    /*
    auto TEST = Convolution(3, 3, 2, 3, 3, 1, std::vector<Convolution::Channel>{
            {{0, 1}},
            {{1}},
            });
    TEST.forward(Vec());
    std::exit(0);
    */

    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    // Create a windowed mode window and its OpenGL context
    GLFWwindow* window = glfwCreateWindow(1280, 720, "ImGui Example", NULL, NULL);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    // Make the window's context current
    glfwMakeContextCurrent(window);

    // Setup ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;

    // Setup ImGui style
    ImGui::StyleColorsDark();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    /*
#define NIMAGES 10
    GLuint textureIds[NIMAGES];
    for (size_t i = 0; i < NIMAGES; ++i) {
        std::vector<uint8_t> pixels(28*28);
        for (size_t j = 0; j < 28*28; ++j) {
            pixels[j] = static_cast<uint8_t>(DATA.train.images[i][j] * 255.0);
        }
        textureIds[i] = create_texture_from_pixels(pixels.data(), 28, 28);
    }
    */

    auto C1  = Convolution(28, 28, 1, 5, 5, 2, std::vector<Convolution::Channel>{
            {{0}},
            {{0}},
            {{0}},
            {{0}},
            {{0}},
            {{0}},
            });
    auto S2  = Sigmoid();
    auto P3  = AveragePooling(28, 28, 2, 2);
    auto C4  = Convolution(14, 14, 6, 5, 5, 0, std::vector<Convolution::Channel>{
            {{0, 1, 2}},
            {{1, 2, 3}},
            {{2, 3, 4}},
            {{3, 4, 5}},
            {{0, 4, 5}},
            {{0, 1, 5}},
            {{0, 1, 2, 3}},
            {{1, 2, 3, 4}},
            {{2, 3, 4, 5}},
            {{0, 3, 4, 5}},
            {{0, 1, 4, 5}},
            {{0, 1, 2, 5}},
            {{0, 1, 3, 4}},
            {{1, 2, 4, 5}},
            {{0, 2, 3, 5}},
            {{0, 1, 2, 3, 4, 5}}
            });
    auto S5  = Sigmoid();
    auto P6  = AveragePooling(10, 10, 2, 2);
    auto F7  = FullyConnected(5*5*16, 120);
    auto S8  = Sigmoid();
    auto F9  = FullyConnected(120, 84);
    auto S10 = Sigmoid();
    auto F11 = FullyConnected(84, 10);

    NeuralNetwork lenet5{
        &C1,
        &S2,
        &P3,
        &C4,
        &S5,
        &P6,
        &F7,
        &S8,
        &F9,
        &S10,
        &F11
    };

    if (opts.from_weights) {
        std::ifstream in(opts.from_weights, std::fstream::binary);
        lenet5.load_weights(in);
    } else {
        uint32_t seed;
        if (opts.w_seed) {
            seed = atol(opts.w_seed);
        } else {
            seed = std::random_device{}();
        }
        std::mt19937 w_rng(seed);
        std::uniform_real_distribution<double> rweights(-1.0, 1.0);
        std::function<double()> gen = [&](){ return rweights(w_rng); };
        lenet5.initialize(gen);

        std::cout << "Weights seed: " << seed << std::endl;
    }

    uint32_t sgd_seed;
    if (opts.sgd_seed) {
        sgd_seed = atol(opts.sgd_seed);
    } else {
        sgd_seed = std::random_device{}();
    }
    std::mt19937 sgd_rng(sgd_seed);
    std::cout << "SGD Seed: " << sgd_seed << std::endl;

    size_t const BATCH_SIZE = 100;
    size_t const NBATCHES = DATA.train.labels.size() / BATCH_SIZE;
    size_t const EVAL_SIZE = 100;
    size_t epoch = 0;
    double LEARNING_RATE = 0.1;

    std::vector<float> loss_train;
    std::vector<float> loss_eval;

    // Main loop
    bool close = false;
    while (!close) {
        /**************************************************************************************************/
        std::vector<size_t> indices(DATA.train.labels.size());
        std::iota(indices.begin(), indices.end(), 0);

        std::cout << "Epoch:" << epoch << std::endl;
        std::shuffle(std::begin(indices), std::end(indices), sgd_rng);

        for (size_t batch = 0; batch < NBATCHES && !close; ++batch) {
            std::cout << "Batch: " << batch << "/" << NBATCHES << std::endl;
            double tloss = 0.0;
            for (size_t i = 0; i < BATCH_SIZE; ++i) {
                size_t const idx = indices[batch * BATCH_SIZE + i];
                double const loss = lenet5.train(DATA.train.images[idx], DATA.train.labels[idx]);
                tloss += loss;
            }
            loss_train.push_back(std::log(tloss / BATCH_SIZE));
            lenet5.descend_gradient(LEARNING_RATE / BATCH_SIZE);

            /*
            std::ofstream after("after", std::fstream::binary);
            lenet5.dump_weights(after);
            std::exit(0);
            */
            

            double eloss = 0;
            for (size_t i = 0; i < EVAL_SIZE; ++i) {
                double const loss = lenet5.train(DATA.test.images[i], DATA.test.labels[i]);
                eloss += loss;
            }
            loss_eval.push_back(std::log(eloss / EVAL_SIZE));

            /************* ImGui stuff *********************/
            // Start the ImGui frame
            glfwPollEvents();
            if (glfwWindowShouldClose(window)) {
                close = true;
                break;
            }
            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            // Create a simple window
            ImGui::Begin("Hello, ImGui!");

            if (loss_eval.size() && loss_train.size()) {
                float const ymin = std::min(*std::min_element(std::begin(loss_train), std::end(loss_train)),
                        *std::min_element(std::begin(loss_eval), std::end(loss_eval)));
                float const ymax = std::max(*std::max_element(std::begin(loss_train), std::end(loss_train)),
                        *std::max_element(std::begin(loss_eval), std::end(loss_eval)));

                ImGui::PlotLines("Train", loss_train.data(), loss_train.size(), 0, nullptr, ymin, ymax, ImVec2(0, 240), sizeof(float));
                ImGui::PlotLines("Eval", loss_eval.data(), loss_eval.size(), 0, nullptr, ymin, ymax, ImVec2(0, 240), sizeof(float));
            }

            /*
               ImGui::Text("Train image");
               for (size_t i = 0; i < NIMAGES; ++i) ImGui::Image((void*)(intptr_t)textureIds[i], ImVec2(28, 28));
               */
            ImGui::End();
            // Rendering
            ImGui::Render();
            int display_w, display_h;
            glfwGetFramebufferSize(window, &display_w, &display_h);
            glViewport(0, 0, display_w, display_h);
            glClearColor(0.45f, 0.55f, 0.60f, 1.00f);
            glClear(GL_COLOR_BUFFER_BIT);
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

            glfwSwapBuffers(window);
            /************* ImGui stuff *********************/
        }

        ++epoch;
        /**************************************************************************************************/
    }

    if (opts.weights_out) {
        std::ofstream weights(opts.weights_out, std::fstream::binary);
        lenet5.dump_weights(weights);
    }
    std::cout << "Last log(training loss) was: " << loss_train.back() << std::endl;

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
