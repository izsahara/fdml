cmake_minimum_required(VERSION 3.16)
# add_executable(new_structure new_structure.cpp)
# target_link_libraries(new_structure ${PROJECT_NAME})

#add_executable(training training.cpp)
#target_link_libraries(training ${PROJECT_NAME})

# add_executable(prediction prediction.cpp)
# target_link_libraries(prediction ${PROJECT_NAME})

# add_executable(airfoil airfoil.cpp)
# target_link_libraries(airfoil ${PROJECT_NAME})

# add_executable(engine engine.cpp)
# target_link_libraries(engine ${PROJECT_NAME})

add_executable(nrel nrel.cpp)
target_link_libraries(nrel ${PROJECT_NAME})

# add_executable(heteroskedastic heteroskedastic.cpp)
# target_link_libraries(heteroskedastic ${PROJECT_NAME})