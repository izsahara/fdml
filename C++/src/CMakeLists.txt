cmake_minimum_required(VERSION 3.16)
add_executable(engine engine.cpp)
target_link_libraries(engine ${PROJECT_NAME})