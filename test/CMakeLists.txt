find_package(Torch REQUIRED)

rock_testsuite(test_depthnn
    SOURCES suite.cpp test_depthnn.cpp
    DEPS depthnn
    DEPS_TARGET ${TORCH_LIBRARIES}
    )
