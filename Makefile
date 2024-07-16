CC = gcc
CXX = g++

# Compiler flags
CXXFLAGS = -Wall -Wextra -std=c++17

LIBS = -lglfw -lGL -ldl

INCLUDES = -Iinclude -Isrc -Iimgui -Iimgui/backends

# Name of the executable
TARGET = build/mnist

# Source files
CPP_SRCS = src/main.cpp $(wildcard imgui/imgui*.cpp) imgui/backends/imgui_impl_glfw.cpp imgui/backends/imgui_impl_opengl3.cpp

# Object files
CPP_OBJS = $(CPP_SRCS:.cpp=.o)
OBJS = $(C_OBJS) $(CPP_OBJS)

# Default target
all: $(TARGET)

# Rule to create the executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^ $(LIBS)

# Rule to compile .c files into .o files
%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@ $(LIBS)

# Rule to compile .cpp files into .o files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@ $(LIBS)

# Clean up
clean:
	rm -f $(OBJS) $(TARGET)

# Phony targets
.PHONY: all clean
