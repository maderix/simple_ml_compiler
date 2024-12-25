#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>

namespace ml_compiler {

// Helper function declarations
void conv2d_relu(const float* input, const float* weight, const float* bias,
                float* output, int H, int W, int C_in, int C_out, 
                int kernel_size, int stride, int padding);

void maxpool2d(const float* input, float* output,
               int H, int W, int C, int kernel_size, int stride);

void linear_relu(const float* input, const float* weight, const float* bias,
                float* output, int batch_size, int in_features, int out_features);

void linear(const float* input, const float* weight, const float* bias,
           float* output, int batch_size, int in_features, int out_features);

void log_softmax(const float* input, float* output, 
                int batch_size, int features, int dim);

// Utility functions
void reshape(const float* input, float* output, 
            const int* input_shape, const int* output_shape,
            int input_dims, int output_dims);

} // namespace ml_compiler
