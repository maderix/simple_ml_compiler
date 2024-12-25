import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

@dataclass
class IRNode:
    op_type: str
    inputs: List[str]
    outputs: List[str]
    attributes: Dict[str, Any]
    weights: Optional[Dict[str, np.ndarray]] = None

class IR:
    def __init__(self):
        self.nodes: List[IRNode] = []
        self.weights: Dict[str, np.ndarray] = {}
        self.input_name: str = "input"
        self.output_name: str = "output"
    
    def add_node(self, node: IRNode):
        self.nodes.append(node)
        
    def print_graph(self):
        print("\nIR Graph Structure:")
        for i, node in enumerate(self.nodes):
            print(f"\nNode {i}:")
            print(f"  Type: {node.op_type}")
            print(f"  Inputs: {node.inputs}")
            print(f"  Outputs: {node.outputs}")
            print(f"  Attributes: {node.attributes}")
    
    def save(self, prefix: str):
        # Save weights to binary file
        weight_metadata = {}
        weight_offset = 0
        
        with open(f"{prefix}.weights.bin", "wb") as f:
            for name, weight in self.weights.items():
                weight.tofile(f)
                weight_metadata[name] = {
                    "offset": weight_offset,
                    "shape": list(weight.shape),
                    "dtype": str(weight.dtype)
                }
                weight_offset += weight.nbytes
        
        # Save graph structure to JSON
        graph_structure = {
            "nodes": [
                {
                    "op_type": node.op_type,
                    "inputs": node.inputs,
                    "outputs": node.outputs,
                    "attributes": node.attributes
                } for node in self.nodes
            ],
            "weights": weight_metadata,
            "input_name": self.input_name,
            "output_name": self.output_name,
            "input_shape": [1, 1, 28, 28],  # MNIST specific
            "output_shape": [1, 10]         # MNIST specific
        }
        
        with open(f"{prefix}.graph.json", "w") as f:
            json.dump(graph_structure, f, indent=2)
        print(f"\nSaved IR to {prefix}.graph.json and {prefix}.weights.bin")

class ModelConverter:
    def __init__(self):
        self.ir = IR()
        self.logger = logging.getLogger("ModelConverter")
    
    def _get_tensor_name(self, node_type: str, idx: int) -> str:
        """Generate a unique tensor name"""
        return f"{node_type}_{idx}"
    
    def convert(self, model_path: str, output_prefix: str):
        """Convert PyTorch model to IR format"""
        self.logger.info(f"Loading model from {model_path}")
        model = torch.jit.load(model_path)
        model.eval()  # Important for inference
        
        # Print the model graph for debugging
        self.logger.info("TorchScript Graph:")
        self.logger.info(model.graph)
        
        # Extract model parameters
        self.logger.info("\nExtracting parameters:")
        for name, param in model.named_parameters():
            self.logger.info(f"Found parameter: {name} with shape {param.shape}")
            clean_name = name.replace('.', '_')
            self.ir.weights[clean_name] = param.detach().numpy()
        
        # Create IR nodes
        self.logger.info("\nCreating IR nodes...")
        self._create_nodes()
        
        # Print final IR structure
        self.ir.print_graph()
        
        # Save IR
        self.ir.save(output_prefix)
        return self.ir
    
    def _create_nodes(self):
        """Create properly connected IR nodes"""
        current_tensor = "input"
        
        # Conv1 + ReLU
        conv1_out = self._get_tensor_name("conv1", 0)
        self.ir.add_node(IRNode(
            op_type="conv2d",
            inputs=[current_tensor],
            outputs=[conv1_out],
            attributes={
                "in_channels": 1,
                "out_channels": 10,
                "kernel_size": 3,
                "stride": 1,
                "padding": 0,
                "weights_key": "conv1_weight",
                "bias_key": "conv1_bias"
            }
        ))
        
        relu1_out = self._get_tensor_name("relu1", 0)
        self.ir.add_node(IRNode(
            op_type="relu",
            inputs=[conv1_out],
            outputs=[relu1_out],
            attributes={}
        ))
        current_tensor = relu1_out
        
        # MaxPool1
        pool1_out = self._get_tensor_name("pool1", 0)
        self.ir.add_node(IRNode(
            op_type="maxpool2d",
            inputs=[current_tensor],
            outputs=[pool1_out],
            attributes={
                "kernel_size": 2,
                "stride": 2,
                "padding": 0
            }
        ))
        current_tensor = pool1_out
        
        # Conv2 + ReLU
        conv2_out = self._get_tensor_name("conv2", 0)
        self.ir.add_node(IRNode(
            op_type="conv2d",
            inputs=[current_tensor],
            outputs=[conv2_out],
            attributes={
                "in_channels": 10,
                "out_channels": 20,
                "kernel_size": 3,
                "stride": 1,
                "padding": 0,
                "weights_key": "conv2_weight",
                "bias_key": "conv2_bias"
            }
        ))
        
        relu2_out = self._get_tensor_name("relu2", 0)
        self.ir.add_node(IRNode(
            op_type="relu",
            inputs=[conv2_out],
            outputs=[relu2_out],
            attributes={}
        ))
        current_tensor = relu2_out
        
        # MaxPool2
        pool2_out = self._get_tensor_name("pool2", 0)
        self.ir.add_node(IRNode(
            op_type="maxpool2d",
            inputs=[current_tensor],
            outputs=[pool2_out],
            attributes={
                "kernel_size": 2,
                "stride": 2,
                "padding": 0
            }
        ))
        current_tensor = pool2_out
        
        # Flatten
        flatten_out = self._get_tensor_name("flatten", 0)
        self.ir.add_node(IRNode(
            op_type="reshape",
            inputs=[current_tensor],
            outputs=[flatten_out],
            attributes={
                "shape": [-1, 20 * 5 * 5]
            }
        ))
        current_tensor = flatten_out
        
        # FC1 + ReLU
        fc1_out = self._get_tensor_name("fc1", 0)
        self.ir.add_node(IRNode(
            op_type="linear",
            inputs=[current_tensor],
            outputs=[fc1_out],
            attributes={
                "in_features": 20 * 5 * 5,
                "out_features": 50,
                "weights_key": "fc1_weight",
                "bias_key": "fc1_bias"
            }
        ))
        
        relu3_out = self._get_tensor_name("relu3", 0)
        self.ir.add_node(IRNode(
            op_type="relu",
            inputs=[fc1_out],
            outputs=[relu3_out],
            attributes={}
        ))
        current_tensor = relu3_out
        
        # FC2
        fc2_out = self._get_tensor_name("fc2", 0)
        self.ir.add_node(IRNode(
            op_type="linear",
            inputs=[current_tensor],
            outputs=[fc2_out],
            attributes={
                "in_features": 50,
                "out_features": 10,
                "weights_key": "fc2_weight",
                "bias_key": "fc2_bias"
            }
        ))
        current_tensor = fc2_out
        
        # LogSoftmax
        self.ir.add_node(IRNode(
            op_type="log_softmax",
            inputs=[current_tensor],
            outputs=["output"],
            attributes={
                "dim": 1
            }
        ))

def create_sample_model():
    """Create and save a sample MNIST model"""
    class SimpleCNN(torch.nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=3)
            self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=3)
            self.fc1 = torch.nn.Linear(20 * 5 * 5, 50)
            self.fc2 = torch.nn.Linear(50, 10)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.max_pool2d(x, 2)
            x = torch.relu(self.conv2(x))
            x = torch.max_pool2d(x, 2)
            x = x.view(-1, 20 * 5 * 5)
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return torch.log_softmax(x, dim=1)

    model = SimpleCNN()
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 1, 28, 28)
    model_path = "mnist_cnn.pt"
    
    # Export model
    traced_model = torch.jit.trace(model, dummy_input)
    traced_model.save(model_path)
    
    print(f"Created sample model: {model_path}")
    return model_path

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        # If no arguments provided, create a sample model
        if len(sys.argv) == 1:
            model_path = create_sample_model()
            output_prefix = "mnist_cnn_ir"
        else:
            print("Usage: python model_converter.py <input_model.pt> <output_prefix>")
            sys.exit(1)
    else:
        model_path = sys.argv[1]
        output_prefix = sys.argv[2]
    
    converter = ModelConverter()
    converter.convert(model_path, output_prefix)
