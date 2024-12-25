import json
from typing import List, Dict, Any, Set
from dataclasses import dataclass
import numpy as np
import logging
from copy import deepcopy

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

class OptimizationPass:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.logger = logging.getLogger(f"opt.{name}")
        self.stats = {"applied": 0, "skipped": 0}
        self.tensor_counter = {}

    def _get_tensor_name(self, prefix: str) -> str:
        """Generate unique tensor names with consistent format"""
        if prefix not in self.tensor_counter:
            self.tensor_counter[prefix] = 0
        count = self.tensor_counter[prefix]
        self.tensor_counter[prefix] += 1
        return f"{prefix}_{count}"

    def reset_stats(self):
        self.stats = {"applied": 0, "skipped": 0}

    def log_application(self, node_info: str):
        self.stats["applied"] += 1
        self.logger.info(f"Applied optimization on {node_info}")

    def log_skip(self, node_info: str, reason: str):
        self.stats["skipped"] += 1
        self.logger.info(f"Skipped {node_info}: {reason}")

class IROptimizer:
    def __init__(self):
        self.logger = logging.getLogger("IROptimizer")
        self.passes = {
            "conv_relu_fusion": OptimizationPass(
                "conv_relu_fusion",
                "Fuses Conv2D + ReLU into a single operation"
            ),
            "linear_relu_fusion": OptimizationPass(
                "linear_relu_fusion",
                "Fuses Linear + ReLU into a single operation"
            ),
            "dead_code_elimination": OptimizationPass(
                "dead_code_elimination",
                "Removes unused operations and tensors"
            ),
            "memory_layout": OptimizationPass(
                "memory_layout",
                "Optimizes memory access patterns"
            )
        }

    def _can_fuse_nodes(self, node1: IRNode, node2: IRNode, type1: str, type2: str) -> bool:
        """Check if two nodes can be fused"""
        if node1.op_type != type1 or node2.op_type != type2:
            return False
        return (len(node2.inputs) == 1 and 
                node2.inputs[0] in node1.outputs)

    def find_tensor_uses(self, nodes: List[IRNode]) -> Dict[str, int]:
        """Count how many times each tensor is used"""
        uses = {}
        # Initialize uses count for all outputs
        for node in nodes:
            for output in node.outputs:
                uses[output] = 0
        
        # Count uses as inputs
        for node in nodes:
            for inp in node.inputs:
                uses[inp] = uses.get(inp, 0) + 1
                
        self.logger.debug(f"Tensor uses: {uses}")
        return uses

    def optimize(self, graph_path: str, weights_path: str) -> tuple[List[IRNode], Dict]:
        """Run all optimization passes on the IR"""
        self.logger.info("Starting optimization process")
        
        # Load IR
        with open(graph_path, 'r') as f:
            graph_data = json.load(f)
        
        nodes = [IRNode(**node) for node in graph_data['nodes']]
        weights_info = graph_data['weights']
        
        # Reset statistics
        for opt_pass in self.passes.values():
            opt_pass.reset_stats()
        
        # Print initial graph
        self.logger.info("\nInitial graph structure:")
        self._print_graph(nodes)
        
        # Run optimization passes
        optimized_nodes = self._fuse_conv_relu(nodes)
        optimized_nodes = self._fuse_linear_relu(optimized_nodes)
        optimized_nodes = self._eliminate_dead_code(optimized_nodes)
        optimized_nodes = self._optimize_memory_layout(optimized_nodes)
        
        # Print optimized graph
        self.logger.info("\nOptimized graph structure:")
        self._print_graph(optimized_nodes)
        
        # Print optimization statistics
        self._print_optimization_stats()
        
        return optimized_nodes, weights_info

    def _print_graph(self, nodes: List[IRNode]):
        """Print the current graph structure"""
        for i, node in enumerate(nodes):
            self.logger.info(f"\nNode {i}:")
            self.logger.info(f"  Type: {node.op_type}")
            self.logger.info(f"  Inputs: {node.inputs}")
            self.logger.info(f"  Outputs: {node.outputs}")
            self.logger.info(f"  Attributes: {node.attributes}")

    def _fuse_conv_relu(self, nodes: List[IRNode]) -> List[IRNode]:
        """Fuse Conv2D + ReLU into Conv2D_ReLU"""
        opt_pass = self.passes["conv_relu_fusion"]
        self.logger.info("Starting Conv-ReLU fusion pass")
        
        optimized = []
        i = 0
        while i < len(nodes) - 1:
            current_node = nodes[i]
            next_node = nodes[i + 1]
            
            if self._can_fuse_nodes(current_node, next_node, "conv2d", "relu"):
                fused_node = IRNode(
                    op_type="conv2d_relu",
                    inputs=current_node.inputs,
                    outputs=next_node.outputs,
                    attributes={
                        **current_node.attributes,
                        "fused": True,
                        "original_nodes": [current_node.op_type, next_node.op_type]
                    }
                )
                
                opt_pass.log_application(
                    f"Fused {current_node.op_type}({current_node.outputs[0]}) + "
                    f"{next_node.op_type}({next_node.outputs[0]}) -> "
                    f"{fused_node.op_type}({fused_node.outputs[0]})"
                )
                optimized.append(fused_node)
                i += 2
            else:
                if current_node.op_type == "conv2d":
                    opt_pass.log_skip(
                        f"Conv2D({current_node.outputs[0]})",
                        "No compatible following ReLU found"
                    )
                optimized.append(current_node)
                i += 1
        
        # Append any remaining node
        if i < len(nodes):
            optimized.append(nodes[i])
            
        return optimized

    def _fuse_linear_relu(self, nodes: List[IRNode]) -> List[IRNode]:
        """Fuse Linear + ReLU into Linear_ReLU"""
        opt_pass = self.passes["linear_relu_fusion"]
        self.logger.info("Starting Linear-ReLU fusion pass")
        
        optimized = []
        i = 0
        while i < len(nodes) - 1:
            current_node = nodes[i]
            next_node = nodes[i + 1]
            
            if self._can_fuse_nodes(current_node, next_node, "linear", "relu"):
                fused_node = IRNode(
                    op_type="linear_relu",
                    inputs=current_node.inputs,
                    outputs=next_node.outputs,
                    attributes={
                        **current_node.attributes,
                        "fused": True,
                        "original_nodes": [current_node.op_type, next_node.op_type]
                    }
                )
                
                opt_pass.log_application(
                    f"Fused {current_node.op_type}({current_node.outputs[0]}) + "
                    f"{next_node.op_type}({next_node.outputs[0]}) -> "
                    f"{fused_node.op_type}({fused_node.outputs[0]})"
                )
                optimized.append(fused_node)
                i += 2
            else:
                if current_node.op_type == "linear":
                    opt_pass.log_skip(
                        f"Linear({current_node.outputs[0]})",
                        "No compatible following ReLU found"
                    )
                optimized.append(current_node)
                i += 1
        
        # Append any remaining node
        if i < len(nodes):
            optimized.append(nodes[i])
            
        return optimized

    def _eliminate_dead_code(self, nodes: List[IRNode]) -> List[IRNode]:
        """Remove unused operations and tensors"""
        opt_pass = self.passes["dead_code_elimination"]
        self.logger.info("Starting dead code elimination pass")
        
        # Find all tensor uses and outputs
        uses = self.find_tensor_uses(nodes)
        output_tensors = {node.outputs[0] for node in nodes 
                         if node.op_type.endswith("softmax")}
        
        # Find live tensors starting from outputs
        live_tensors: Set[str] = output_tensors.copy()
        self.logger.info(f"Starting from output tensors: {live_tensors}")
        
        # Propagate backwards until no new tensors are added
        changed = True
        while changed:
            changed = False
            for node in nodes:
                if any(out in live_tensors for out in node.outputs):
                    for inp in node.inputs:
                        if inp not in live_tensors:
                            live_tensors.add(inp)
                            changed = True
                            self.logger.debug(f"Added {inp} to live tensors")
        
        # Keep only nodes that produce live tensors
        optimized = []
        for node in nodes:
            if any(output in live_tensors for output in node.outputs):
                opt_pass.log_application(f"Kept live node {node.op_type}({node.outputs[0]})")
                optimized.append(node)
            else:
                opt_pass.log_application(
                    f"Removed dead node {node.op_type}({node.outputs[0]})"
                )
        
        return optimized

    def _optimize_memory_layout(self, nodes: List[IRNode]) -> List[IRNode]:
        """Optimize memory layout for better cache utilization"""
        opt_pass = self.passes["memory_layout"]
        self.logger.info("Starting memory layout optimization pass")
        
        optimized = []
        for node in nodes:
            node_copy = deepcopy(node)
            
            if node.op_type in ["conv2d", "conv2d_relu"]:
                opt_pass.log_application(
                    f"Set NCHW layout for {node.op_type}({node.outputs[0]})"
                )
                node_copy.attributes["memory_layout"] = "NCHW"
                optimized.append(node_copy)
                
            elif node.op_type in ["linear", "linear_relu"]:
                opt_pass.log_application(
                    f"Set NC layout for {node.op_type}({node.outputs[0]})"
                )
                node_copy.attributes["memory_layout"] = "NC"
                optimized.append(node_copy)
                
            else:
                opt_pass.log_skip(
                    f"{node.op_type}({node.outputs[0]})",
                    "No specific layout optimization available"
                )
                optimized.append(node_copy)
        
        return optimized

    def _print_optimization_stats(self):
        """Print statistics for all optimization passes"""
        self.logger.info("\nOptimization Statistics:")
        for name, opt_pass in self.passes.items():
            self.logger.info(f"\n{name}:")
            self.logger.info(f"  Description: {opt_pass.description}")
            self.logger.info(f"  Applied: {opt_pass.stats['applied']}")
            self.logger.info(f"  Skipped: {opt_pass.stats['skipped']}")

    def save_optimized_ir(self, nodes: List[IRNode], weights_info: Dict,
                         output_prefix: str):
        """Save optimized IR to files"""
        self.logger.info(f"Saving optimized IR to {output_prefix}")
        
        graph_structure = {
            "nodes": [
                {
                    "op_type": node.op_type,
                    "inputs": node.inputs,
                    "outputs": node.outputs,
                    "attributes": node.attributes
                } for node in nodes
            ],
            "weights": weights_info,
            "optimization_stats": {
                name: pass_obj.stats 
                for name, pass_obj in self.passes.items()
            }
        }
        
        output_path = f"{output_prefix}.opt.json"
        with open(output_path, "w") as f:
            json.dump(graph_structure, f, indent=2)
        
        self.logger.info(f"Saved optimized IR to {output_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python optimizer.py <input_graph.json> <output_prefix>")
        sys.exit(1)
    
    optimizer = IROptimizer()
    optimized_nodes, weights_info = optimizer.optimize(sys.argv[1], sys.argv[2])
    optimizer.save_optimized_ir(optimized_nodes, weights_info, sys.argv[2])
