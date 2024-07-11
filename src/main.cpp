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

#define NIMAGES 10

    GLuint textureIds[NIMAGES];
    for (size_t i = 0; i < NIMAGES; ++i) {
        std::vector<uint8_t> pixels(28*28);
        for (size_t j = 0; j < 28*28; ++j) {
            pixels[j] = static_cast<uint8_t>(DATA.train.images[i][j] * 255.0);
        }
        textureIds[i] = create_texture_from_pixels(pixels.data(), 28, 28);
    }

    // Main loop
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // Start the ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Create a simple window
        ImGui::Begin("Hello, ImGui!");
        ImGui::Text("Train image");
        for (size_t i = 0; i < NIMAGES; ++i) ImGui::Image((void*)(intptr_t)textureIds[i], ImVec2(28, 28));
        ImGui::End();

        /**************************************************************************************************/
        NeuralNetwork lenet{
                new Convolution(28, 28, 1, 3, 3),
                new Sigmoid(),
                new AveragePooling(28, 28, 2, 2),
                new Convolution(14, 14, 6, 5, 5),
                new Sigmoid(),
                new AveragePooling(10, 10, 2, 2),
                new FullyConnected(5*5*16, 120),
                new Sigmoid(),
                new FullyConnected(120, 84),
                new Sigmoid(),
                new FullyConnected(84, 10),
        };

        size_t const SEED = 0;
        std::mt19937 rng(SEED);

        size_t const BATCH_SIZE = 32;
        size_t const NBATCHES = DATA.train.labels.size() / BATCH_SIZE;
        size_t epoch = 0;
        size_t learning_rate = 0.01;

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
        /**************************************************************************************************/

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
