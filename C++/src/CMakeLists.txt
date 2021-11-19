cmake_minimum_required(VERSION 3.16)
add_executable(engine engine.cpp)
add_executable(analytic2 analytic2.cpp)
# add_executable(nrel nrel.cpp)

target_link_libraries(engine ${PROJECT_NAME})
target_link_libraries(analytic2 ${PROJECT_NAME})
# target_link_libraries(nrel ${PROJECT_NAME})