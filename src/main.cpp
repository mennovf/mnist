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

int main() {
    auto const DATA = data("./data");

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
            {{1}},
            {{1}},
            {{1}},
            {{1}},
            {{1}},
            {{1}},
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

    size_t const SEED = 0;
    std::mt19937 rng(SEED);

    std::uniform_real_distribution<double> rweights(-1.0, 1.0);
    std::function<double()> gen = [&](){ return rweights(rng); };
    lenet5.initialize(gen);

    size_t const BATCH_SIZE = 32;
    size_t const NBATCHES = DATA.train.labels.size() / BATCH_SIZE;
    size_t epoch = 0;
    size_t learning_rate = 0.01;

    std::vector<float> log_loss;

    // Main loop
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // Start the ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        float const ymin = *std::min_element(std::begin(log_loss), std::end(log_loss));
        float const ymax = *std::max_element(std::begin(log_loss), std::end(log_loss));

        // Create a simple window
        ImGui::Begin("Hello, ImGui!");
        ImGui::PlotLines("Loss", log_loss.data(), log_loss.size(), 0, nullptr, ymin, ymax, ImVec2(0, 240), sizeof(float));

        /*
        ImGui::Text("Train image");
        for (size_t i = 0; i < NIMAGES; ++i) ImGui::Image((void*)(intptr_t)textureIds[i], ImVec2(28, 28));
        */
        ImGui::End();

        /**************************************************************************************************
        std::vector<size_t> indices(DATA.train.labels.size());
        std::iota(indices.begin(), indices.end(), 0);

        while (true) {
            std::cout << "Epoch:" << epoch << std::endl;
            std::shuffle(std::begin(indices), std::end(indices), rng);
            
            for (size_t batch = 0; batch < NBATCHES; ++batch) {
                for (size_t i = 0; i < BATCH_SIZE; ++i) {
                    size_t const idx = indices[batch * BATCH_SIZE + i];
                    lenet.train(DATA.train.images[idx], DATA.train.labels[idx]);
                }

                lenet.descent_gradient(learning_rate / BATCH_SIZE);
            }

            ++epoch;
        }
        **************************************************************************************************/

        // Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.45f, 0.55f, 0.60f, 1.00f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
