cmake_minimum_required(VERSION 3.16)
# add_executable(new_structure new_structure.cpp)
# target_link_libraries(new_structure ${PROJECT_NAME})

# add_executable(calibration calibration.cpp)
# target_link_libraries(calibration ${PROJECT_NAME})

# add_executable(airfoil airfoil.cpp)
# target_link_libraries(airfoil ${PROJECT_NAME})

# add_executable(engine engine.cpp)
# target_link_libraries(engine ${PROJECT_NAME})

add_executable(nrel-20 nrel-20.cpp)
target_link_libraries(nrel-20 ${PROJECT_NAME})

# add_executable(nrel-40 nrel-40.cpp)
# target_link_libraries(nrel-40 ${PROJECT_NAME})

# add_executable(heteroskedastic heteroskedastic.cpp)
# target_link_libraries(heteroskedastic ${PROJECT_NAME})