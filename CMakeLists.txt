cmake_minimum_required(VERSION 3.16)
project("fdml")
set(CMAKE_CXX_STANDARD 17)

set(MODULE_SOURCE "${PROJECT_SOURCE_DIR}/bindings")
set(HEADERS "${PROJECT_SOURCE_DIR}/include")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/fdml/src)


include_directories(${HEADERS}/fdml)
include_directories(${HEADERS}/external/Eigen)
include_directories(${HEADERS}/external/ThreadPool)
include_directories(${HEADERS}/external/CppNumericalSolvers)
include_directories(${HEADERS}/external/optim/include)
add_subdirectory(${HEADERS}/external/pybind11)

# add_executable(executable ${MODULE_SOURCE}/executable.cpp)

pybind11_add_module (_parameters ${MODULE_SOURCE}/parameters.cpp)
target_include_directories (_parameters PUBLIC ${MODULE_SOURCE})

pybind11_add_module (_utilities ${MODULE_SOURCE}/utilities.cpp)
target_include_directories (_utilities PUBLIC ${MODULE_SOURCE})

pybind11_add_module (_kernels ${MODULE_SOURCE}/kernels.cpp)
target_include_directories (_kernels PUBLIC ${MODULE_SOURCE})

pybind11_add_module (_base_models ${MODULE_SOURCE}/base_models.cpp)
target_include_directories (_base_models PUBLIC ${MODULE_SOURCE})

pybind11_add_module (_base_models2 ${MODULE_SOURCE}/base_models2.cpp)
target_include_directories (_base_models2 PUBLIC ${MODULE_SOURCE})

pybind11_add_module (_deep_models ${MODULE_SOURCE}/deep_models.cpp)
target_include_directories (_deep_models PUBLIC ${MODULE_SOURCE})

pybind11_add_module (_deep_models2 ${MODULE_SOURCE}/deep_models2.cpp)
target_include_directories (_deep_models2 PUBLIC ${MODULE_SOURCE})