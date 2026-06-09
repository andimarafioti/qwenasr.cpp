cmake_minimum_required(VERSION 3.18)
project(qwenasr-cpp-native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

find_package(Python3 REQUIRED COMPONENTS Development.Embed)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build static libraries" FORCE)
add_subdirectory(ggml)

function(qwenasr_compile_options target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wshadow -Wconversion -Wno-unused-parameter)
    endif()
endfunction()

add_library(qwenasr_bridge STATIC
    src/qwenasr.cpp
)
target_include_directories(qwenasr_bridge PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(qwenasr_bridge PUBLIC Python3::Python)
qwenasr_compile_options(qwenasr_bridge)

add_library(qwenasr_native STATIC
    src/native-model.cpp
    src/gguf-model.cpp
    src/audio-features.cpp
    src/audio-conv.cpp
    src/tokenizer.cpp
)
target_include_directories(qwenasr_native PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/ggml/include
)
target_link_libraries(qwenasr_native PUBLIC ggml)
qwenasr_compile_options(qwenasr_native)

add_executable(qwen-asr tools/qwen-asr.cpp)
target_link_libraries(qwen-asr PRIVATE qwenasr_bridge)
qwenasr_compile_options(qwen-asr)

add_executable(qwen-asr-gguf-info tools/qwen-asr-gguf-info.cpp)
target_link_libraries(qwen-asr-gguf-info PRIVATE qwenasr_native)
qwenasr_compile_options(qwen-asr-gguf-info)

add_executable(qwen-asr-features tools/qwen-asr-features.cpp)
target_link_libraries(qwen-asr-features PRIVATE qwenasr_native)
qwenasr_compile_options(qwen-asr-features)

add_executable(qwen-asr-tokenize tools/qwen-asr-tokenize.cpp)
target_link_libraries(qwen-asr-tokenize PRIVATE qwenasr_native)
qwenasr_compile_options(qwen-asr-tokenize)

add_executable(qwen-asr-weights tools/qwen-asr-weights.cpp)
target_link_libraries(qwen-asr-weights PRIVATE qwenasr_native)
qwenasr_compile_options(qwen-asr-weights)

add_executable(qwen-asr-audio-conv tools/qwen-asr-audio-conv.cpp)
target_link_libraries(qwen-asr-audio-conv PRIVATE qwenasr_native)
qwenasr_compile_options(qwen-asr-audio-conv)

add_executable(qwen-asr-audio-cnn tools/qwen-asr-audio-cnn.cpp)
target_link_libraries(qwen-asr-audio-cnn PRIVATE qwenasr_native)
qwenasr_compile_options(qwen-asr-audio-cnn)
