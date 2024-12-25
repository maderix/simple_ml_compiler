#!/usr/bin/env python3
import os
import sys
import logging
import argparse
import subprocess
import shutil
from pathlib import Path
from typing import List, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class CompilerError(Exception):
    """Base class for compiler errors"""
    pass

class CompilerDriver:
    def __init__(self):
        self.logger = logging.getLogger("CompilerDriver")
        self.build_dir = None
        self.kernel_dir = None

    def compile(self, 
                opt_graph: str, 
                weights: str, 
                output: str,
                kernel_dir: Optional[str] = None,
                debug: bool = False,
                extra_flags: Optional[List[str]] = None) -> bool:
        """
        Compile optimized graph to binary
        
        Args:
            opt_graph: Path to optimized IR JSON
            weights: Path to weights binary
            output: Path for output binary
            kernel_dir: Directory containing kernel implementations
            debug: Whether to include debug info
            extra_flags: Additional compiler flags
        """
        try:
            # Validate inputs
            self._validate_inputs(opt_graph, weights)

            # Setup build environment
            self._setup_build_env(kernel_dir)

            # Generate code
            self._generate_code(opt_graph, weights)

            # Compile to binary
            self._compile_code(output, debug, extra_flags)

            self.logger.info(f"Successfully compiled to {output}")
            return True

        except Exception as e:
            self.logger.error(f"Compilation failed: {e}")
            if debug:
                import traceback
                self.logger.error(traceback.format_exc())
            return False

        finally:
            # Cleanup if not in debug mode
            if not debug and self.build_dir:
                self._cleanup()

    def _validate_inputs(self, opt_graph: str, weights: str):
        """Validate input files"""
        if not os.path.exists(opt_graph):
            raise CompilerError(f"Optimized graph file not found: {opt_graph}")
        if not os.path.exists(weights):
            raise CompilerError(f"Weights file not found: {weights}")

        # Check file sizes
        if os.path.getsize(opt_graph) == 0:
            raise CompilerError(f"Empty optimized graph file: {opt_graph}")
        if os.path.getsize(weights) == 0:
            raise CompilerError(f"Empty weights file: {weights}")

    def _setup_build_env(self, kernel_dir: Optional[str] = None):
        """Setup build environment"""
        self.logger.info("Setting up build environment")
        
        # Create build directory
        self.build_dir = Path("./build")
        self.build_dir.mkdir(exist_ok=True)
        
        # Set kernel directory
        if kernel_dir:
            self.kernel_dir = Path(kernel_dir)
        else:
            # Use default kernels from our package
            self.kernel_dir = Path(__file__).parent / "kernels"
        
        if not self.kernel_dir.exists():
            raise CompilerError(f"Kernel directory not found: {self.kernel_dir}")
        
        # Copy kernel files
        kernel_files = list(self.kernel_dir.glob("*.?pp"))  # both .cpp and .hpp
        if not kernel_files:
            raise CompilerError(f"No kernel files found in {self.kernel_dir}")
        
        for kernel_file in kernel_files:
            dst = self.build_dir / kernel_file.name
            self.logger.debug(f"Copying {kernel_file} to {dst}")
            shutil.copy2(kernel_file, dst)

    def _generate_code(self, opt_graph: str, weights: str):
        """Generate C++ code using code generator"""
        self.logger.info("Generating C++ code")
        
        try:
            # Import code generator
            code_gen_path = Path(__file__).parent
            if str(code_gen_path) not in sys.path:
                sys.path.append(str(code_gen_path))
            
            from code_generator import CodeGenerator
            generator = CodeGenerator()
            generator.generate(opt_graph, weights, str(self.build_dir))
            
        except ImportError as e:
            raise CompilerError(f"Failed to import code generator: {e}")
        except Exception as e:
            raise CompilerError(f"Code generation failed: {e}")

    def _get_compiler_flags(self, debug: bool, extra_flags: Optional[List[str]] = None) -> List[str]:
        """Get compiler flags"""
        flags = ["-std=c++17", "-fopenmp"]
        
        if debug:
            flags.extend(["-g", "-O0", "-DDEBUG"])
        else:
            flags.extend(["-O3", "-march=native", "-DNDEBUG"])
            
        # Warning flags
        flags.extend([
            "-Wall", "-Wextra", "-Wpedantic",
            "-Wno-unused-parameter"  # Allow unused parameters
        ])
        
        # Add extra flags if provided
        if extra_flags:
            flags.extend(extra_flags)
        
        return flags

    def _compile_code(self, output: str, debug: bool, extra_flags: Optional[List[str]] = None):
       """Compile generated code to binary"""
       self.logger.info("Compiling to binary")
       
       # Collect source files in specific order (main.cpp last)
       sources = []
       for file in ['weights.cpp', 'model.cpp', 'kernels.cpp', 'main.cpp']:
           source = self.build_dir / file
           if source.exists():
               sources.append(str(source))
       
       if not sources:
           raise CompilerError("No source files found in build directory")
       
       # Setup compiler command
       compiler = os.environ.get("CXX", "g++")
       flags = self._get_compiler_flags(debug, extra_flags)
       
       # Include paths
       include_paths = ["-I", str(self.build_dir)]
       
       # Build command
       cmd = [compiler, *flags, *include_paths, "-o", output, *sources]
       
       self.logger.debug(f"Compilation command: {' '.join(cmd)}")
       
       # Run compilation
       try:
           result = subprocess.run(
               cmd,
               check=True,
               stdout=subprocess.PIPE,
               stderr=subprocess.PIPE,
               universal_newlines=True
           )
           
           # Log compiler output
           if result.stdout:
               self.logger.debug(f"Compiler stdout:\n{result.stdout}")
           if result.stderr:
               self.logger.warning(f"Compiler stderr:\n{result.stderr}")
               
       except subprocess.CalledProcessError as e:
           self.logger.error(f"Compilation failed with return code {e.returncode}")
           if e.stdout:
               self.logger.error(f"Compiler stdout:\n{e.stdout}")
           if e.stderr:
               self.logger.error(f"Compiler stderr:\n{e.stderr}")
           raise CompilerError("Compilation failed")
    def _cleanup(self):
        """Cleanup build directory"""
        try:
            if self.build_dir and self.build_dir.exists():
                shutil.rmtree(self.build_dir)
        except Exception as e:
            self.logger.warning(f"Failed to cleanup build directory: {e}")

def main():
    parser = argparse.ArgumentParser(
        description='Compile optimized ML model to binary',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('graph', help='Path to optimized graph JSON')
    parser.add_argument('weights', help='Path to weights binary')
    parser.add_argument('output', help='Path for output binary')
    
    parser.add_argument('--kernel-dir', 
                       help='Directory containing kernel implementations')
    parser.add_argument('--keep-build', action='store_true',
                       help='Keep build directory (for debugging)')
    parser.add_argument('--debug', action='store_true',
                       help='Build with debug information')
    parser.add_argument('--extra-flags', nargs='+',
                       help='Additional compiler flags')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()

    # Setup logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Run compiler
    driver = CompilerDriver()
    success = driver.compile(
        args.graph,
        args.weights,
        args.output,
        kernel_dir=args.kernel_dir,
        debug=args.debug or args.keep_build,
        extra_flags=args.extra_flags
    )
    
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
