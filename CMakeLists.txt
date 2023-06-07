cmake_minimum_required(VERSION 3.20)
project(ml_assignment VERSION 0.1.0)

add_executable(${PROJECT_NAME} src/main.cpp src/spec.cpp src/network.cpp src/hyperparams.cpp src/preprocessing.cpp)

set(CMAKE_CXX_STANDARD 20)
set(CXX_STANDARD_REQUIRED 20)

find_package(OpenMP)
add_subdirectory(lib/fmt)
add_subdirectory(lib/eigen)

target_include_directories(${PROJECT_NAME}
    PRIVATE ${PROJECT_SOURCE_DIR}/include
    PUBLIC lib/toml/include
    PUBLIC lib/fmt/include
    PUBLIC lib/eigen
)

target_link_directories(${PROJECT_NAME}
    PRIVATE src
    PRIVATE lib/fmt/src
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} fmt OpenMP::OpenMP_CXX )
else()
    target_link_libraries(${PROJECT_NAME}
        fmt
    )
endif()