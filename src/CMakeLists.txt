
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

#rock_library(depthnn
#    SOURCES Dummy.cpp
#    HEADERS Dummy.hpp)

rock_executable(depthnn_bin Main.cpp DEPS_PLAIN ${TORCH_LIBRARIES} DEPS ${OpenCV_LIBS})

#add_executable(depthnn_bin Main.cpp)
#target_link_libraries(depthnn_bin ${TORCH_LIBRARIES} ${OpenCV_LIBS})
