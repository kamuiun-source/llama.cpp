#pragma once

#include "common.h"
#include "quantize.cuh"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>

static_assert(q8_bytes == elements / QK8_1 * sizeof(block_q8_1), "fixed MTP3 Q8_1 shape");

[[noreturn]] void ggml_cuda_error(const char * statement, const char * function, const char * file, int line, const char * message) {
    std::fprintf(stderr, "%s:%d: %s: %s: %s\n", file, line, function, statement, message);
    std::exit(2);
}

[[noreturn]] void ggml_abort(const char * file, int line, const char * format, ...) {
    std::fprintf(stderr, "%s:%d: ", file, line);
    va_list args;
    va_start(args, format);
    std::vfprintf(stderr, format, args);
    va_end(args);
    std::fputc('\n', stderr);
    std::exit(2);
}

static void quantize_fixed(const void * source, void * output, cudaStream_t stream) {
    quantize_row_q8_1_cuda(static_cast<const float *>(source), output, width, width, width, width * rows,
            width, 1, rows, 1, stream);
}
