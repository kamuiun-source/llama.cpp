add_executable(llama-expert-placement expert-placement.cpp)
target_link_libraries(llama-expert-placement PRIVATE llama ggml)
target_include_directories(llama-expert-placement PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_compile_features(llama-expert-placement PRIVATE cxx_std_17)
