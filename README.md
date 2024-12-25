# Simple ML Model Compiler

A compiler pipeline that converts ML models to optimized executables. The project includes both a basic compiler and a complete pipeline with optimization passes.

## Project Structure

```
simple_ml_compiler/
├── basic_compiler/
│   └── ml_compiler.py      # Basic implementation of model compilation
├── code_generator.py       # Generates C++ implementation from IR
├── compiler_driver.py      # Main compiler driver
├── converter.py           # Converts PyTorch models to IR
├── kernels/
│   ├── kernels.cpp        # Kernel implementations
│   └── kernels.hpp        # Kernel interface declarations
├── model.py              # MNIST model definition
├── optimizer.py          # IR optimization passes
└── verifier.py          # Verifies compilation correctness
```

## Pipeline Overview

The compilation process consists of several stages:

1. **Model Definition** (`model.py`)
   - Defines PyTorch MNIST model
   - Exports model to TorchScript format

2. **Model Conversion** (`converter.py`)
   - Converts TorchScript model to IR
   - Extracts weights and network structure

3. **IR Optimization** (`optimizer.py`)
   - Applies optimization passes:
     - Operator fusion (Conv+ReLU, Linear+ReLU)
     - Dead code elimination
     - Memory layout optimization

4. **Code Generation** (`code_generator.py`)
   - Generates C++ implementation
   - Manages memory layout
   - Creates weight definitions

5. **Compilation** (`compiler_driver.py`)
   - Compiles generated code to binary
   - Links with kernel implementations
   - Handles build configurations

6. **Verification** (`verifier.py`)
   - Verifies correctness of compilation
   - Compares original and compiled model outputs

## Quick Start

1. **Setup Environment**
```bash
# Requirements
- Python 3.8+
- PyTorch
- NumPy
- G++ compiler with C++17 support
- OpenMP
```

2. **Export Model**
```bash
python model.py
# Creates mnist_cnn.pt
```

3. **Convert and Optimize**
```bash
# Convert to IR
python converter.py mnist_cnn.pt mnist_cnn_ir

# Optimize IR
python optimizer.py mnist_cnn_ir.graph.json mnist_cnn_ir.opt.json
```

4. **Compile**
```bash
python compiler_driver.py mnist_cnn_ir.opt.json mnist_cnn_ir.weights.bin mnist_model
```

5. **Test**
```python
import numpy as np

# Create test input
test_input = np.random.randn(28, 28).astype(np.float32)
test_input.tofile('test_input.bin')

# Run model
!./mnist_model test_input.bin output.bin

# Check output
output = np.fromfile('output.bin', dtype=np.float32)
print("Predicted digit:", output.argmax())
```

6. **Verify** (Optional)
```bash
python verifier.py mnist_cnn_ir.graph.json mnist_cnn_ir.opt.json
```

## Basic Compiler Version

A simplified version is available in `basic_compiler/ml_compiler.py`. This version:
- Directly converts PyTorch model to C++
- No optimization passes
- Single file implementation
- MNIST-specific

Usage:
```bash
python basic_compiler/ml_compiler.py mnist_cnn.pt mnist_model
```

## Development

### Adding New Operations

1. Add kernel declaration in `kernels/kernels.hpp`:
```cpp
void new_op(const float* input, float* output, ...);
```

2. Implement kernel in `kernels/kernels.cpp`:
```cpp
void new_op(const float* input, float* output, ...) {
    // Implementation
}
```

3. Add IR node handling in `converter.py`:
```python
def convert_new_op(self, node):
    # Convert op to IR
```

4. Add optimization handling in `optimizer.py`:
```python
def optimize_new_op(self, node):
    # Apply optimizations
```

5. Add code generation in `code_generator.py`:
```python
def _gen_new_op(self, node: Dict) -> str:
    # Generate C++ code
```

### Build Options

Compiler supports various build configurations:
```bash
# Debug build
python compiler_driver.py --debug input.json weights.bin output

# Custom kernel directory
python compiler_driver.py --kernel-dir ./my_kernels input.json weights.bin output

# Keep build files
python compiler_driver.py --keep-build input.json weights.bin output
```

## Future Work

- Support for more model formats (ONNX, TFLite)
- Additional optimization passes
- Auto-tuning for kernel parameters
- GPU/accelerator support
- Quantization support
- More comprehensive verification tools

## Contributing

1. Create an issue describing the feature/bug
2. Fork the repository
3. Create a feature branch
4. Submit a pull request

## License

MIT License
