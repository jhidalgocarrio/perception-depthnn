
rock_testsuite(test_depthnn
    SOURCES suite.cpp test_depthnn.cpp
    DEPS depthnn
    #DEPS_PLAIN ${TORCH_LIBRARIES}
    )
