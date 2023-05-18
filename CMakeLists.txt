cmake_minimum_required(VERSION 3.16)
project(ml_assignment VERSION 0.1.0)

add_executable(${PROJECT_NAME} src/main.cpp)

add_subdirectory(lib/fmt)
add_subdirectory(lib/eigen)

target_include_directories(${PROJECT_NAME}
    PRIVATE include
    PUBLIC lib/fmt/include
    PUBLIC ${EIGEN_INCLUDE_DIR}
)

target_link_directories(${PROJECT_NAME}
    PRIVATE src
    PRIVATE lib/fmt/src
)

target_link_libraries(${PROJECT_NAME}
    fmt
)