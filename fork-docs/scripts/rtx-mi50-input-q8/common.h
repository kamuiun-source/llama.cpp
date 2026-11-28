#pragma once

#include <cstddef>

inline constexpr int width = 2560;
inline constexpr int rows = 3;
inline constexpr int elements = width * rows;
inline constexpr size_t f32_bytes = elements * sizeof(float);
inline constexpr size_t q8_bytes = 8640;

extern "C" void rtx_input_q8(const void * source, void * mapped_q8, void * stream);
extern "C" void mi50_input_setup(int device, void * host, size_t bytes, size_t q8_offset);
extern "C" void mi50_input_submit(unsigned arm);
extern "C" void mi50_input_sync();
extern "C" void mi50_input_read(void * output);
extern "C" void mi50_input_cleanup(void * host);
