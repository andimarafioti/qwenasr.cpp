cmake_minimum_required(VERSION 3.18)
project(qwenasr-cpp-native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python3 REQUIRED COMPONENTS Development.Embed)

add_library(qwenasr_bridge STATIC
    src/qwenasr.cpp
)
target_include_directories(qwenasr_bridge PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(qwenasr_bridge PUBLIC Python3::Python)

if(MSVC)
    target_compile_options(qwenasr_bridge PRIVATE /W4)
else()
    target_compile_options(qwenasr_bridge PRIVATE -Wall -Wextra -Wshadow -Wconversion -Wno-unused-parameter)
endif()

add_executable(qwen-asr tools/qwen-asr.cpp)
target_link_libraries(qwen-asr PRIVATE qwenasr_bridge)

if(MSVC)
    target_compile_options(qwen-asr PRIVATE /W4)
else()
    target_compile_options(qwen-asr PRIVATE -Wall -Wextra -Wshadow -Wconversion -Wno-unused-parameter)
endif()
