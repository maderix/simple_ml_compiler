import torch
import numpy as np
import json
from typing import Dict, List, Any
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class GraphVerifier:
    def __init__(self):
        self.logger = logging.getLogger("GraphVerifier")
        self.op_implementations = {
            "conv2d": self._run_conv2d,
            "conv2d_relu": self._run_conv2d_relu,
            "relu": self._run_relu,
            "maxpool2d": self._run_maxpool2d,
            "reshape": self._run_reshape,
            "linear": self._run_linear,
            "linear_relu": self._run_linear_relu,
            "log_softmax": self._run_log_softmax
        }
        self.tensors = {}
    
    def _load_graph(self, graph_path: str) -> tuple[List[Dict], Dict]:
        """Load graph and weights info from JSON"""
        with open(graph_path, 'r') as f:
            data = json.load(f)
            self.logger.info(f"Graph structure loaded from {graph_path}")
            self.logger.info(f"Number of nodes: {len(data['nodes'])}")
            self.logger.info(f"Number of weights: {len(data['weights'])}")
        return data['nodes'], data['weights']
    
    def _load_weights(self, weights_path: str, weights_info: Dict) -> Dict[str, np.ndarray]:
        """Load weights from binary file using metadata"""
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        
        self.logger.info(f"Loading weights from {weights_path}")
        self.logger.info(f"Expected weights: {list(weights_info.keys())}")
        
        file_size = os.path.getsize(weights_path)
        self.logger.info(f"Weights file size: {file_size} bytes")
        
        weights = {}
        with open(weights_path, 'rb') as f:
            for name, info in weights_info.items():
                try:
                    shape = info['shape']
                    weight_size = np.prod(shape) * 4  # float32 = 4 bytes
                    
                    self.logger.info(f"Loading {name}: shape {shape}, size {weight_size} bytes")
                    
                    if 'offset' in info:
                        offset = info['offset']
                        f.seek(offset)
                    
                    weight_data = np.fromfile(f, dtype=np.float32, count=np.prod(shape))
                    
                    if weight_data.size != np.prod(shape):
                        raise ValueError(
                            f"Failed to read weight {name}: "
                            f"expected {np.prod(shape)} elements, got {weight_data.size}"
                        )
                    
                    weights[name] = weight_data.reshape(shape)
                    self.logger.info(f"Successfully loaded {name}")
                
                except Exception as e:
                    self.logger.error(f"Error loading weight {name}: {e}")
                    raise
        
        return weights
    
    def _run_conv2d(self, node: Dict, weights: Dict) -> np.ndarray:
        x = self.tensors[node['inputs'][0]]
        w = weights[node['attributes']['weights_key']]
        b = weights[node['attributes']['bias_key']]
        
        # Convert to PyTorch tensors
        x_torch = torch.from_numpy(x).float()
        w_torch = torch.from_numpy(w).float()
        b_torch = torch.from_numpy(b).float()
        
        # Run convolution
        out = torch.nn.functional.conv2d(x_torch, w_torch, b_torch,
                                       stride=node['attributes'].get('stride', 1),
                                       padding=node['attributes'].get('padding', 0))
        return out.numpy()
    
    def _run_conv2d_relu(self, node: Dict, weights: Dict) -> np.ndarray:
        conv_out = self._run_conv2d(node, weights)
        return np.maximum(conv_out, 0)
    
    def _run_relu(self, node: Dict, weights: Dict) -> np.ndarray:
        x = self.tensors[node['inputs'][0]]
        return np.maximum(x, 0)
    
    def _run_maxpool2d(self, node: Dict, weights: Dict) -> np.ndarray:
        x = self.tensors[node['inputs'][0]]
        x_torch = torch.from_numpy(x).float()
        
        kernel_size = node['attributes']['kernel_size']
        stride = node['attributes']['stride']
        
        out = torch.nn.functional.max_pool2d(x_torch, kernel_size=kernel_size, stride=stride)
        return out.numpy()
    
    def _run_reshape(self, node: Dict, weights: Dict) -> np.ndarray:
        x = self.tensors[node['inputs'][0]]
        new_shape = list(x.shape[:1]) + node['attributes']['shape'][1:]
        return x.reshape(new_shape)
    
    def _run_linear(self, node: Dict, weights: Dict) -> np.ndarray:
        x = self.tensors[node['inputs'][0]]
        w = weights[node['attributes']['weights_key']]
        b = weights[node['attributes']['bias_key']]
        
        # Ensure x is 2D
        if len(x.shape) > 2:
            x = x.reshape(x.shape[0], -1)
        
        # Convert to PyTorch tensors
        x_torch = torch.from_numpy(x).float()
        w_torch = torch.from_numpy(w).float()
        b_torch = torch.from_numpy(b).float()
        
        # Run linear layer
        out = torch.nn.functional.linear(x_torch, w_torch, b_torch)
        return out.numpy()
    
    def _run_linear_relu(self, node: Dict, weights: Dict) -> np.ndarray:
        linear_out = self._run_linear(node, weights)
        return np.maximum(linear_out, 0)
    
    def _run_log_softmax(self, node: Dict, weights: Dict) -> np.ndarray:
        x = self.tensors[node['inputs'][0]]
        x_torch = torch.from_numpy(x).float()
        out = torch.nn.functional.log_softmax(x_torch, dim=node['attributes']['dim'])
        return out.numpy()
    
    def _execute_graph(self, nodes: List[Dict], weights: Dict, input_data: np.ndarray) -> Dict[str, np.ndarray]:
        """Execute graph with given input"""
        self.tensors = {'input': input_data}
        
        for i, node in enumerate(nodes):
            self.logger.info(f"Executing node {i}: {node['op_type']}")
            try:
                out = self.op_implementations[node['op_type']](node, weights)
                for output_name in node['outputs']:
                    self.tensors[output_name] = out
                    self.logger.debug(f"Output shape: {out.shape}")
            except Exception as e:
                self.logger.error(f"Error executing node {i} ({node['op_type']}): {e}")
                self.logger.error(f"Node details: {node}")
                raise
        
        return self.tensors
    
    def verify_graphs(self, original_graph_path: str, optimized_graph_path: str,
                     weights_path: str, num_tests: int = 10, rtol: float = 1e-5) -> bool:
        """Verify that original and optimized graphs produce same outputs"""
        self.logger.info("Loading graphs...")
        orig_nodes, weights_info = self._load_graph(original_graph_path)
        opt_nodes, _ = self._load_graph(optimized_graph_path)
        
        self.logger.info("Loading weights...")
        weights = self._load_weights(weights_path, weights_info)
        
        self.logger.info(f"Running {num_tests} verification tests...")
        all_passed = True
        
        for i in range(num_tests):
            # Generate random input (using appropriate scale)
            input_data = np.random.randn(1, 1, 28, 28).astype(np.float32) * 0.1
            
            # Run both graphs
            self.logger.info(f"\nTest {i+1}:")
            
            self.logger.info("Running original graph...")
            orig_outputs = self._execute_graph(orig_nodes, weights, input_data)
            
            self.logger.info("Running optimized graph...")
            opt_outputs = self._execute_graph(opt_nodes, weights, input_data)
            
            # Compare outputs
            orig_final = orig_outputs['output']
            opt_final = opt_outputs['output']
            
            # Check if outputs are close
            try:
                max_diff = np.max(np.abs(orig_final - opt_final))
                if max_diff > rtol:
                    raise AssertionError(f"Max difference: {max_diff}")
                self.logger.info(f"Test {i+1} PASSED (max diff: {max_diff:e})")
            except AssertionError as e:
                self.logger.error(f"Test {i+1} FAILED: {e}")
                self.logger.error("Original output:")
                self.logger.error(orig_final)
                self.logger.error("Optimized output:")
                self.logger.error(opt_final)
                all_passed = False
        
        if all_passed:
            self.logger.info("\nAll verification tests PASSED!")
        else:
            self.logger.error("\nSome verification tests FAILED!")
        
        return all_passed

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Verify graph optimization correctness')
    parser.add_argument('original_graph', help='Path to original graph JSON')
    parser.add_argument('optimized_graph', help='Path to optimized graph JSON')
    parser.add_argument('weights', help='Path to weights binary file')
    parser.add_argument('--num-tests', type=int, default=10, help='Number of tests to run')
    parser.add_argument('--rtol', type=float, default=1e-5, help='Relative tolerance for comparison')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    verifier = GraphVerifier()
    success = verifier.verify_graphs(
        args.original_graph,
        args.optimized_graph,
        args.weights,
        num_tests=args.num_tests,
        rtol=args.rtol
    )
    
    # Return non-zero exit code if verification failed
    import sys
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
