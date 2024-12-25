#include "kernels.hpp"
#include <cstring>
#include <numeric>

namespace ml_compiler {

void conv2d_relu(const float* input, const float* weight, const float* bias,
                float* output, int H, int W, int C_in, int C_out, 
                int kernel_size, int stride, int padding) {
    int output_H = (H + 2 * padding - kernel_size) / stride + 1;
    int output_W = (W + 2 * padding - kernel_size) / stride + 1;
    
    // Initialize output with zeros
    std::fill(output, output + output_H * output_W * C_out, 0.0f);
    
    // Create padded input if needed
    std::vector<float> padded_input;
    const float* input_ptr;
    int padded_H = H, padded_W = W;
    
    if (padding > 0) {
        padded_H = H + 2 * padding;
        padded_W = W + 2 * padding;
        padded_input.resize(padded_H * padded_W * C_in, 0.0f);
        
        // Copy input to padded array
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                for (int c = 0; c < C_in; c++) {
                    padded_input[((h + padding) * padded_W + w + padding) * C_in + c] = 
                        input[h * W * C_in + w * C_in + c];
                }
            }
        }
        input_ptr = padded_input.data();
    } else {
        input_ptr = input;
    }
    
    // Perform convolution with ReLU
    #pragma omp parallel for collapse(3)
    for (int h = 0; h < output_H; h++) {
        for (int w = 0; w < output_W; w++) {
            for (int cout = 0; cout < C_out; cout++) {
                float sum = bias[cout];
                for (int kh = 0; kh < kernel_size; kh++) {
                    for (int kw = 0; kw < kernel_size; kw++) {
                        for (int cin = 0; cin < C_in; cin++) {
                            int in_h = h * stride + kh;
                            int in_w = w * stride + kw;
                            sum += input_ptr[(in_h * padded_W + in_w) * C_in + cin] *
                                  weight[(cout * C_in + cin) * kernel_size * kernel_size +
                                        kh * kernel_size + kw];
                        }
                    }
                }
                output[h * output_W * C_out + w * C_out + cout] = std::max(0.0f, sum);
            }
        }
    }
}

void maxpool2d(const float* input, float* output,
               int H, int W, int C, int kernel_size, int stride) {
    int output_H = (H - kernel_size) / stride + 1;
    int output_W = (W - kernel_size) / stride + 1;
    
    #pragma omp parallel for collapse(3)
    for (int h = 0; h < output_H; h++) {
        for (int w = 0; w < output_W; w++) {
            for (int c = 0; c < C; c++) {
                float maxval = -std::numeric_limits<float>::infinity();
                for (int kh = 0; kh < kernel_size; kh++) {
                    for (int kw = 0; kw < kernel_size; kw++) {
                        int in_h = h * stride + kh;
                        int in_w = w * stride + kw;
                        maxval = std::max(maxval, 
                            input[(in_h * W + in_w) * C + c]);
                    }
                }
                output[(h * output_W + w) * C + c] = maxval;
            }
        }
    }
}

void linear_relu(const float* input, const float* weight, const float* bias,
                float* output, int batch_size, int in_features, int out_features) {
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch_size; b++) {
        for (int out = 0; out < out_features; out++) {
            float sum = bias[out];
            for (int in = 0; in < in_features; in++) {
                sum += input[b * in_features + in] * 
                       weight[out * in_features + in];
            }
            output[b * out_features + out] = std::max(0.0f, sum);
        }
    }
}

void linear(const float* input, const float* weight, const float* bias,
           float* output, int batch_size, int in_features, int out_features) {
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch_size; b++) {
        for (int out = 0; out < out_features; out++) {
            float sum = bias[out];
            for (int in = 0; in < in_features; in++) {
                sum += input[b * in_features + in] * 
                       weight[out * in_features + in];
            }
            output[b * out_features + out] = sum;
        }
    }
}

void log_softmax(const float* input, float* output, 
                int batch_size, int features, int dim) {
    #pragma omp parallel for
    for (int b = 0; b < batch_size; b++) {
        const float* batch_input = input + b * features;
        float* batch_output = output + b * features;
        
        // Find max for numerical stability
        float max_val = *std::max_element(batch_input, batch_input + features);
        
        // Compute exp sum
        float exp_sum = 0.0f;
        for (int i = 0; i < features; i++) {
            exp_sum += std::exp(batch_input[i] - max_val);
        }
        float log_sum = std::log(exp_sum);
        
        // Compute log softmax
        for (int i = 0; i < features; i++) {
            batch_output[i] = batch_input[i] - max_val - log_sum;
        }
    }
}

void reshape(const float* input, float* output, 
            const int* input_shape, const int* output_shape,
            int input_dims, int output_dims) {
    // Calculate total size
    int size = 1;
    for (int i = 0; i < input_dims; i++) {
        size *= input_shape[i];
    }
    
    // For reshape, we can just copy the memory since it's just a view change
    std::memcpy(output, input, size * sizeof(float));
}

} // namespace ml_compiler
