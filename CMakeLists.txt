cmake_minimum_required(VERSION 3.16)
project("fdml")
set(CMAKE_CXX_STANDARD 17)
set(HEADERS "${PROJECT_SOURCE_DIR}/include")
set(EXESRC "${PROJECT_SOURCE_DIR}/C++/src")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/C++)

include_directories(${HEADERS})
include_directories(${HEADERS}/external/Eigen)
include_directories(${HEADERS}/external/EigenRand)
include_directories(${HEADERS}/external/ThreadPool)
include_directories(${HEADERS}/external/optim/include)
include_directories(${HEADERS}/external/rapidcsv/src)
include_directories(${HEADERS}/external/LBFGSB-F/include)

enable_language(Fortran)
set(LBFGSB "${PROJECT_SOURCE_DIR}/include/external/LBFGSB-F")
file(GLOB LBFGSB-SRC ${LBFGSB}/Lbfgsb.3.0/*.f )
file(GLOB LBFGSB-H ${LBFGSB}/include/*.h)
set(SOURCE_FILES ${LBFGSB-SRC})
add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES})
# Install library
install(TARGETS ${PROJECT_NAME} DESTINATION lib/${PROJECT_NAME})
# Install library headers
install(FILES ${LBFGSB-H} DESTINATION include/TEST)

add_subdirectory(${EXESRC})