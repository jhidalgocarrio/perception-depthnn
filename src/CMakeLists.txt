find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

rock_library(depthnn
    SOURCES DepthNN.cpp
    HEADERS DepthNN.hpp
    DEPS_TARGET ${TORCH_LIBRARIES}
    DEPS ${OpenCV_LIBS}
    DEPS_PKGCONFIG base-logging
    )

rock_executable(depthnn_bin Main.cpp DEPS depthnn)

#add_executable(depthnn_bin Main.cpp)
#target_link_libraries(depthnn_bin ${TORCH_LIBRARIES} ${OpenCV_LIBS})
