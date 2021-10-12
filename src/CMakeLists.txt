rock_library(depthnn
    SOURCES Dummy.cpp
    HEADERS Dummy.hpp)

rock_executable(depthnn_bin Main.cpp
    DEPS depthnn)

