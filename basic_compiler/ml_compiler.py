import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any
import subprocess
import os

# Define the MNIST CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.fc1 = nn.Linear(20 * 5 * 5, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 20 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def save_model(model_path="mnist_cnn.pt"):
    # Create model
    model = SimpleCNN()
    model.eval()  # Set to evaluation mode
    
    # Create dummy input
    dummy_input = torch.randn(1, 1, 28, 28)
    
    # Export model
    try:
        # First trace the model
        traced_model = torch.jit.trace(model, dummy_input)
        
        # Save the traced model
        traced_model.save(model_path)
        
        print("Model saved successfully to", model_path)
        
        # Test loading
        loaded_model = torch.jit.load(model_path)
        print("Model loaded successfully for verification")
        
        # Print model graph
        print("\nModel graph:")
        print(loaded_model.graph)
        
    except Exception as e:
        print(f"Error saving model: {e}")
    
    return model_path

# Compiler related classes and functions
@dataclass
class IRNode:
    op_type: str
    inputs: List[str]
    outputs: List[str]
    attributes: Dict[str, Any]
    weights: Dict[str, np.ndarray] = None

class SimpleIR:
    def __init__(self):
        self.nodes = []
        self.weights = {}
        self.input_name = "input"
        self.output_name = "output"
    
    def add_node(self, node: IRNode):
        self.nodes.append(node)

class MinimalCompiler:
    def __init__(self):
        self.ir = SimpleIR()
    
    def parse_model(self, model_path: str) -> SimpleIR:
        # Load TorchScript model
        model = torch.jit.load(model_path)
        
        # Extract graph
        graph = model.graph
        
        print("Graph structure:")
        print(graph)
        
        # Access model parameters directly
        for name, param in model.named_parameters():
            print(f"Found parameter: {name} with shape {param.shape}")
            if 'conv1.weight' in name:
                self.ir.weights['conv1_weight'] = param.detach().numpy()
            elif 'conv1.bias' in name:
                self.ir.weights['conv1_bias'] = param.detach().numpy()
        
        # Create IR nodes for the first convolution layer
        conv1_node = IRNode(
            op_type="conv2d",
            inputs=["input"],
            outputs=["conv1_output"],
            attributes={
                "kernel_size": (3, 3),
                "in_channels": 1,
                "out_channels": 10
            }
        )
        self.ir.add_node(conv1_node)
        
        # Print debug information
        print("\nExtracted weights:")
        for name, weight in self.ir.weights.items():
            print(f"{name}: shape {weight.shape}")
        
        print("\nExtracted nodes:")
        for node in self.ir.nodes:
            print(f"Op type: {node.op_type}")
            print(f"Inputs: {node.inputs}")
            print(f"Outputs: {node.outputs}")
            print(f"Attributes: {node.attributes}")
            print("---")
        
        return self.ir
    
    def optimize_ir(self, ir: SimpleIR) -> SimpleIR:
        # Simple optimization: fuse Conv+ReLU
        optimized_nodes = []
        i = 0
        while i < len(ir.nodes):
            current_node = ir.nodes[i]
            if (i < len(ir.nodes) - 1 and 
                current_node.op_type == "conv2d" and 
                ir.nodes[i + 1].op_type == "relu"):
                # Fuse Conv+ReLU
                fused_node = IRNode(
                    op_type="conv2d_relu",
                    inputs=current_node.inputs,
                    outputs=ir.nodes[i + 1].outputs,
                    attributes=current_node.attributes
                )
                optimized_nodes.append(fused_node)
                i += 2
            else:
                optimized_nodes.append(current_node)
                i += 1
        
        ir.nodes = optimized_nodes
        return ir

    def generate_cpp_code(self, ir: SimpleIR) -> str:
        # Generate C++ implementation
        code = """
#include <vector>
#include <cmath>
#include <fstream>
#include <iostream>

// Helper functions
void conv2d_relu(const float* input, const float* weight, const float* bias,
                float* output, int H, int W, int C_in, int C_out, 
                int kernel_size) {
    // Initialize output with zeros
    int output_H = H - kernel_size + 1;
    int output_W = W - kernel_size + 1;
    for (int i = 0; i < output_H * output_W * C_out; ++i) {
        output[i] = 0.0f;
    }

    for (int cout = 0; cout < C_out; cout++) {
        for (int h = 0; h < H-kernel_size+1; h++) {
            for (int w = 0; w < W-kernel_size+1; w++) {
                float sum = bias[cout];
                for (int cin = 0; cin < C_in; cin++) {
                    for (int kh = 0; kh < kernel_size; kh++) {
                        for (int kw = 0; kw < kernel_size; kw++) {
                            int input_idx = (h+kh)*W*C_in + (w+kw)*C_in + cin;
                            int weight_idx = cout*C_in*kernel_size*kernel_size + 
                                        cin*kernel_size*kernel_size +
                                        kh*kernel_size + kw;
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
                int out_idx = h*(W-kernel_size+1)*C_out + w*C_out + cout;
                output[out_idx] = sum > 0 ? sum : 0;  // ReLU
            }
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file>" << std::endl;
        return 1;
    }
    
    // Load input image (28x28)
    std::ifstream input_file(argv[1], std::ios::binary);
    if (!input_file) {
        std::cerr << "Could not open input file" << std::endl;
        return 1;
    }
    
    std::vector<float> input(28*28);
    input_file.read(reinterpret_cast<char*>(input.data()), 
                  input.size() * sizeof(float));
    
    // Define weights
"""
        # Add weights as const arrays
        if 'conv1_weight' in ir.weights:
            weight = ir.weights['conv1_weight']
            code += f"\n    const float conv1_weight[] = {{"
            code += ", ".join(map(lambda x: f"{x:.6f}f", weight.flatten()))
            code += "};\n"
        
        if 'conv1_bias' in ir.weights:
            bias = ir.weights['conv1_bias']
            code += f"\n    const float conv1_bias[] = {{"
            code += ", ".join(map(lambda x: f"{x:.6f}f", bias.flatten()))
            code += "};\n"
        
        # Complete the main function
        code += """
    // Network inference
    std::vector<float> output(10);
    std::vector<float> intermediate(26*26*10);  // Output size after first conv
    
    // Layer 1: Conv + ReLU
    conv2d_relu(input.data(), conv1_weight, conv1_bias,
               intermediate.data(), 28, 28, 1, 10, 3);
    
    // Write output
    std::ofstream output_file("output.bin", std::ios::binary);
    if (!output_file) {
        std::cerr << "Could not create output file" << std::endl;
        return 1;
    }
    
    output_file.write(reinterpret_cast<char*>(intermediate.data()),
                    intermediate.size() * sizeof(float));
    
    std::cout << "Inference completed successfully" << std::endl;
    return 0;
}
"""
        return code

    def compile_to_binary(self, cpp_code: str) -> str:
        # Write C++ code to file
        with open("model.cpp", "w") as f:
            f.write(cpp_code)
        
        print("Generated C++ code written to model.cpp")
        
        # Compile with g++
        try:
            subprocess.run(["g++", "-O3", "model.cpp", "-o", "model"], check=True)
            print("Successfully compiled to binary")
        except subprocess.CalledProcessError as e:
            print(f"Compilation failed: {e}")
            raise
        
        return "./model"

    def compile(self, model_path: str) -> str:
        print(f"Starting compilation of {model_path}")
        
        # Parse model to IR
        ir = self.parse_model(model_path)
        print("Model parsed to IR successfully")
        
        # Optimize IR
        ir = self.optimize_ir(ir)
        print("IR optimization completed")
        
        # Generate C++ code
        cpp_code = self.generate_cpp_code(ir)
        print("C++ code generated")
        
        # Compile to binary
        binary_path = self.compile_to_binary(cpp_code)
        print(f"Binary compiled to {binary_path}")
        
        return binary_path

def create_test_input(filename="test_input.bin"):
    # Create a test input (28x28 array of random values)
    test_input = np.random.randn(28, 28).astype(np.float32)
    
    # Save to binary file
    test_input.tofile(filename)
    print(f"Created test input file: {filename}")
    return filename

def test_binary(binary_path, input_path="test_input.bin"):
    # Run the binary
    try:
        subprocess.run([binary_path, input_path], check=True)
        print("Binary executed successfully")
        
        # Read the output
        output = np.fromfile('output.bin', dtype=np.float32)
        print(f"Output shape: {output.shape}")
        print("First few values:", output[:10])
        
    except subprocess.CalledProcessError as e:
        print(f"Error running binary: {e}")

if __name__ == "__main__":
    # 1. Save the model
    model_path = save_model()
    
    # 2. Compile the model
    compiler = MinimalCompiler()
    try:
        binary_path = compiler.compile(model_path)
        print(f"Compiled model binary: {binary_path}")
        
        # 3. Create test input
        input_path = create_test_input()
        
        # 4. Test the binary
        test_binary(binary_path, input_path)
        
    except Exception as e:
        print(f"Error during compilation or testing: {e}")
