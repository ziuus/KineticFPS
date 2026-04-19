cmake_minimum_required(VERSION 3.10)
project(KineticFPS VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Optimize for speed
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math -march=native")

# Find OpenCV
find_package(OpenCV REQUIRED core videoio highgui imgproc dnn video)
include_directories(${OpenCV_INCLUDE_DIRS})

# Find X11 for Mouse Control
find_package(X11 REQUIRED)
include_directories(${X11_INCLUDE_DIR})

# Add executables
add_executable(kinetic_fps src/main.cpp)
target_link_libraries(kinetic_fps ${OpenCV_LIBS} ${X11_LIBRARIES} Xtst)

add_executable(inspect_models src/inspect.cpp)
target_link_libraries(inspect_models ${OpenCV_LIBS})
