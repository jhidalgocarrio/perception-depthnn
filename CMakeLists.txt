# CMakeLists.txt has to be located in the project folder and cmake has to be
# executed from 'project/build' with 'cmake ../'.
cmake_minimum_required(VERSION 3.9)
project(depthnn
        VERSION 0.1
        DESCRIPTION "Test libtorch C++ depth inference using MiDAS")
find_package(Rock)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(ROCK_TEST_ENABLED ON)

rock_init()
rock_standard_layout()
