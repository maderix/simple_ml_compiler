import logging
from typing import Dict, List, Any
from dataclasses import dataclass
import numpy as np
import json
import os

logging.basicConfig(level=logging.INFO)

@dataclass
class TensorInfo:
    name: str
    shape: List[int]
    size: int
    dtype: str = 'float32'

class CodeGenerationError(Exception):
    """Base class for code generation errors"""
    pass

class CodeGenerator:
    def __init__(self):
        self.logger = logging.getLogger("CodeGenerator")
        self.op_generators = {
            "conv2d": self._gen_conv2d,
            "conv2d_relu": self._gen_conv2d_relu,
            "maxpool2d": self._gen_maxpool2d,
            "reshape": self._gen_reshape,
            "linear": self._gen_linear,
            "linear_relu": self._gen_linear_relu,
            "log_softmax": self._gen_log_softmax
        }
        self.tensors: Dict[str, TensorInfo] = {}

    def generate(self, graph_path: str, weights_path: str, output_dir: str):
        """Generate C++ implementation from optimized graph"""
        self.logger.info(f"Loading graph from {graph_path}")
        with open(graph_path, 'r') as f:
            graph = json.load(f)

        self.logger.info(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

        # Generate source files
        self.logger.info("Generating model header")
        model_header = self._generate_model_header(graph)
        
        self.logger.info("Generating model implementation")
        model_impl = self._generate_model_impl(graph)
        
        self.logger.info("Generating weights file")
        weights_file = self._generate_weights_file(graph, weights_path)
        
        self.logger.info("Generating main file")
        main_file = self._generate_main_file()

        # Write files
        header_path = os.path.join(output_dir, "model.hpp")
        self.logger.info(f"Writing header to {header_path}")
        with open(header_path, 'w') as f:
            f.write(model_header)

        impl_path = os.path.join(output_dir, "model.cpp")
        self.logger.info(f"Writing implementation to {impl_path}")
        with open(impl_path, 'w') as f:
            f.write(model_impl)

        weights_path = os.path.join(output_dir, "weights.cpp")
        self.logger.info(f"Writing weights to {weights_path}")
        with open(weights_path, 'w') as f:
            f.write(weights_file)
            
        main_path = os.path.join(output_dir, "main.cpp")
        self.logger.info(f"Writing main to {main_path}")
        with open(main_path, 'w') as f:
            f.write(main_file)

        self.logger.info("Code generation complete")
    def _generate_model_header(self, graph: Dict) -> str:
        """Generate model header file"""
        self._analyze_tensors(graph['nodes'])

        code = """#pragma once
#include "kernels.hpp"
#include <vector>
#include <cstdint>

namespace ml_compiler {

// Forward declare weights
"""
        # Declare external weights
        for name in graph['weights'].keys():
            code += f"extern const float {name}[];\n"

        code += """
class Model {
public:
    Model();
    
    // Run inference
    void forward(const float* input, float* output);
    
    // Get input/output information
    static constexpr int input_size = 784;  // 28x28 MNIST image
    static constexpr int output_size = 10;  // 10 classes

private:
    // Internal buffers for intermediate tensors
"""
        # Declare tensor buffers (skip input/output)
        for tensor in self.tensors.values():
            if tensor.name not in ['input', 'output']:
                code += f"    std::vector<float> {tensor.name};\n"

        code += "};\n\n}  // namespace ml_compiler\n"
        return code

    def _generate_model_impl(self, graph: Dict) -> str:
        """Generate model implementation file"""
        code = """#include "model.hpp"
#include <algorithm>
#include <cstring>

namespace ml_compiler {

Model::Model() {
    // Initialize intermediate buffers
"""
        # Allocate tensors (skip input/output)
        for tensor in self.tensors.values():
            if tensor.name not in ['input', 'output']:
                code += f"    {tensor.name}.resize({tensor.size});\n"

        code += "}\n\n"

        # Generate forward function
        code += "void Model::forward(const float* input, float* output) {\n"
        
        # Generate computation for each node
        for i, node in enumerate(graph['nodes']):
            if node['op_type'] in self.op_generators:
                self.logger.debug(f"Generating code for node {i}: {node['op_type']}")
                code += self.op_generators[node['op_type']](node)
            else:
                self.logger.warning(f"Unknown operation type: {node['op_type']}")

        code += "}\n\n}  // namespace ml_compiler\n"
        return code

    def _generate_weights_file(self, graph: Dict, weights_path: str) -> str:
        """Generate weights definitions file with actual values"""
        self.logger.info(f"Loading weights from {weights_path}")
        
        code = """#include "model.hpp"

namespace ml_compiler {

// Model weights
"""
        try:
            # Read weights from binary file
            weights_data = {}
            weights_info = graph['weights']
            
            with open(weights_path, 'rb') as f:
                for name, info in weights_info.items():
                    shape = info['shape']
                    size = np.prod(shape)
                    offset = info['offset']
                    
                    self.logger.debug(f"Reading weight {name}: shape {shape}, offset {offset}")
                    
                    # Seek to weight's position and read
                    f.seek(offset)
                    data = np.fromfile(f, dtype=np.float32, count=size)
                    weights_data[name] = data.reshape(shape)
            
            # Generate weight arrays
            for name, info in weights_info.items():
                shape = info['shape']
                data = weights_data[name]
                
                code += f"const float {name}[] = {{\n"
                
                # Format weight values with proper indentation
                flat_data = data.flatten()
                values_per_line = 8
                for i in range(0, len(flat_data), values_per_line):
                    values = [f"{x:g}f" for x in flat_data[i:i+values_per_line]]
                    code += "    " + ", ".join(values) + ",\n"
                
                # Add array size comment
                total_size = np.prod(shape)
                code += f"}}; // shape: {shape}, size: {total_size}\n\n"

        except Exception as e:
            raise CodeGenerationError(f"Failed to generate weights: {e}")

        code += "}  // namespace ml_compiler\n"
        return code

    def _analyze_tensors(self, nodes: List[Dict]):
        """Analyze tensor shapes through the network"""
        self.logger.info("Analyzing tensor shapes")
        
        # Start with input
        self.tensors['input'] = TensorInfo(
            name='input',
            shape=[1, 1, 28, 28],  # MNIST specific
            size=784
        )
        
        # Process each node
        for node in nodes:
            op_type = node['op_type']
            attrs = node['attributes']
            inputs = node['inputs']
            outputs = node['outputs']

            self.logger.debug(f"Analyzing node: {op_type}")

            if op_type in ['conv2d', 'conv2d_relu']:
                in_shape = self.tensors[inputs[0]].shape
                out_channels = attrs['out_channels']
                kernel_size = attrs['kernel_size']
                stride = attrs.get('stride', 1)
                padding = attrs.get('padding', 0)
                
                H = in_shape[2]
                W = in_shape[3]
                out_h = (H + 2*padding - kernel_size) // stride + 1
                out_w = (W + 2*padding - kernel_size) // stride + 1
                
                for output in outputs:
                    self.tensors[output] = TensorInfo(
                        name=output,
                        shape=[1, out_channels, out_h, out_w],
                        size=out_channels * out_h * out_w
                    )

            elif op_type == 'maxpool2d':
                in_shape = self.tensors[inputs[0]].shape
                kernel_size = attrs['kernel_size']
                stride = attrs['stride']
                
                out_h = (in_shape[2] - kernel_size) // stride + 1
                out_w = (in_shape[3] - kernel_size) // stride + 1
                
                for output in outputs:
                    self.tensors[output] = TensorInfo(
                        name=output,
                        shape=[in_shape[0], in_shape[1], out_h, out_w],
                        size=in_shape[1] * out_h * out_w
                    )

            elif op_type == 'reshape':
                in_shape = self.tensors[inputs[0]].shape
                new_shape = attrs['shape']
                if new_shape[0] == -1:  # Handle dynamic batch size
                    new_shape[0] = in_shape[0]
                
                for output in outputs:
                    self.tensors[output] = TensorInfo(
                        name=output,
                        shape=new_shape,
                        size=abs(np.prod(new_shape))
                    )

            elif op_type in ['linear', 'linear_relu']:
                in_shape = self.tensors[inputs[0]].shape
                out_features = attrs['out_features']
                
                for output in outputs:
                    self.tensors[output] = TensorInfo(
                        name=output,
                        shape=[in_shape[0], out_features],
                        size=out_features
                    )

            elif op_type == 'log_softmax':
                in_shape = self.tensors[inputs[0]].shape
                for output in outputs:
                    self.tensors[output] = TensorInfo(
                        name=output,
                        shape=in_shape,
                        size=np.prod(in_shape)
                    )

    def _gen_conv2d(self, node: Dict) -> str:
        attrs = node['attributes']
        in_tensor = self.tensors[node['inputs'][0]]
        
        # Special handling for input tensor
        input_access = "input" if node['inputs'][0] == "input" else f"{node['inputs'][0]}.data()"
        
        return f"""    // Conv2D: {node['inputs'][0]} -> {node['outputs'][0]}
    conv2d_relu(
        {input_access},
        {attrs['weights_key']},
        {attrs['bias_key']},
        {node['outputs'][0]}.data(),
        {in_tensor.shape[2]},  // H
        {in_tensor.shape[3]},  // W
        {attrs['in_channels']},
        {attrs['out_channels']},
        {attrs['kernel_size']},
        {attrs.get('stride', 1)},
        {attrs.get('padding', 0)}
    );\n\n"""

    def _gen_conv2d_relu(self, node: Dict) -> str:
        return self._gen_conv2d(node)  # Same implementation since ReLU is fused

    def _gen_maxpool2d(self, node: Dict) -> str:
        attrs = node['attributes']
        in_tensor = self.tensors[node['inputs'][0]]
        
        # Special handling for input tensor
        input_access = "input" if node['inputs'][0] == "input" else f"{node['inputs'][0]}.data()"
        
        return f"""    // MaxPool2D: {node['inputs'][0]} -> {node['outputs'][0]}
    maxpool2d(
        {input_access},
        {node['outputs'][0]}.data(),
        {in_tensor.shape[2]},  // H
        {in_tensor.shape[3]},  // W
        {in_tensor.shape[1]},  // C
        {attrs['kernel_size']},
        {attrs['stride']}
    );\n\n"""

    def _gen_reshape(self, node: Dict) -> str:
        attrs = node['attributes']
        in_tensor = self.tensors[node['inputs'][0]]
        out_tensor = self.tensors[node['outputs'][0]]
        
        # Special handling for input tensor
        input_access = "input" if node['inputs'][0] == "input" else f"{node['inputs'][0]}.data()"
        
        # Create arrays for input and output shapes
        in_shape_str = "{" + ", ".join(map(str, in_tensor.shape)) + "}"
        out_shape_str = "{" + ", ".join(map(str, out_tensor.shape)) + "}"
        
        return f"""    // Reshape: {node['inputs'][0]} -> {node['outputs'][0]}
    const int in_shape[] = {in_shape_str};
    const int out_shape[] = {out_shape_str};
    reshape(
        {input_access},
        {node['outputs'][0]}.data(),
        in_shape,
        out_shape,
        {len(in_tensor.shape)},
        {len(out_tensor.shape)}
    );\n\n"""

    def _gen_linear(self, node: Dict) -> str:
        attrs = node['attributes']
        in_tensor = self.tensors[node['inputs'][0]]
        
        # Special handling for input tensor
        input_access = "input" if node['inputs'][0] == "input" else f"{node['inputs'][0]}.data()"
        
        return f"""    // Linear: {node['inputs'][0]} -> {node['outputs'][0]}
    linear(
        {input_access},
        {attrs['weights_key']},
        {attrs['bias_key']},
        {node['outputs'][0]}.data(),
        {in_tensor.shape[0]},  // batch_size
        {attrs['in_features']},
        {attrs['out_features']}
    );\n\n"""

    def _gen_linear_relu(self, node: Dict) -> str:
        attrs = node['attributes']
        in_tensor = self.tensors[node['inputs'][0]]
        
        # Special handling for input tensor
        input_access = "input" if node['inputs'][0] == "input" else f"{node['inputs'][0]}.data()"
        
        return f"""    // Linear+ReLU: {node['inputs'][0]} -> {node['outputs'][0]}
    linear_relu(
        {input_access},
        {attrs['weights_key']},
        {attrs['bias_key']},
        {node['outputs'][0]}.data(),
        {in_tensor.shape[0]},  // batch_size
        {attrs['in_features']},
        {attrs['out_features']}
    );\n\n"""
    def _gen_log_softmax(self, node: Dict) -> str:
        in_tensor = self.tensors[node['inputs'][0]]
        
        # Special handling for input tensor
        input_access = "input" if node['inputs'][0] == "input" else f"{node['inputs'][0]}.data()"
        
        return f"""    // LogSoftmax: {node['inputs'][0]} -> output
        log_softmax(
            {input_access},
            output,  // Final output pointer
            1,      // batch_size
            10,     // num_classes
            {node['attributes']['dim']}
        );\n\n"""

    def _generate_main_file(self) -> str:
        """Generate main.cpp with program entry point"""
        return """#include "model.hpp"
        #include <iostream>
        #include <fstream>
        #include <vector>
        
        int main(int argc, char** argv) {
            if (argc != 3) {
                std::cerr << "Usage: " << argv[0] << " <input_file> <output_file>" << std::endl;
                return 1;
            }
        
            // Load input
            std::ifstream input_file(argv[1], std::ios::binary);
            if (!input_file) {
                std::cerr << "Could not open input file: " << argv[1] << std::endl;
                return 1;
            }
        
            // Read input data
            std::vector<float> input(ml_compiler::Model::input_size);
            input_file.read(reinterpret_cast<char*>(input.data()), input.size() * sizeof(float));
        
            if (!input_file) {
                std::cerr << "Error reading input file" << std::endl;
                return 1;
            }
        
            // Create and run model
            ml_compiler::Model model;
            std::vector<float> output(ml_compiler::Model::output_size);  // Should be 10
            model.forward(input.data(), output.data());
        
            // Print outputs for debugging
            std::cout << "Predictions: ";
            for (int i = 0; i < ml_compiler::Model::output_size; i++) {
                std::cout << output[i] << " ";
            }
            std::cout << std::endl;
        
            // Find predicted digit
            int predicted_digit = 0;
            float max_val = output[0];
            for (int i = 1; i < ml_compiler::Model::output_size; i++) {
                if (output[i] > max_val) {
                    max_val = output[i];
                    predicted_digit = i;
                }
            }
            std::cout << "Predicted digit: " << predicted_digit << std::endl;
        
            // Write only the final output (10 values)
            std::ofstream output_file(argv[2], std::ios::binary);
            if (!output_file) {
                std::cerr << "Could not open output file: " << argv[2] << std::endl;
                return 1;
            }
        
            output_file.write(reinterpret_cast<char*>(output.data()), output.size() * sizeof(float));
        
            if (!output_file) {
                std::cerr << "Error writing output file" << std::endl;
                return 1;
            }
        
            return 0;
        }
        """

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate C++ code from optimized graph')
    parser.add_argument('graph_path', help='Path to optimized graph JSON')
    parser.add_argument('weights_path', help='Path to weights binary')
    parser.add_argument('output_dir', help='Output directory for generated files')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    generator = CodeGenerator()
    try:
        generator.generate(args.graph_path, args.weights_path, args.output_dir)
    except Exception as e:
        logging.error(f"Code generation failed: {e}")
        if args.debug:
            import traceback
            logging.error(traceback.format_exc())
        sys.exit(1)

if __name__ == '__main__':
    main()
