"""
Profiling and benchmarking utilities for EmptyDrops algorithm.
"""

import cProfile
import pstats
import time
import numpy as np
from typing import Callable, Any, Dict, List
import matplotlib.pyplot as plt
from memory_profiler import profile as memory_profile
import psutil
import os
from functools import wraps
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Profiler:
    """
    Comprehensive profiling for EmptyDrops algorithm.
    """
    def __init__(self, output_dir: str = "profiling_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize profilers
        self.cpu_profiler = cProfile.Profile()
        self.stats = {}
        
    def profile_function(self, func: Callable) -> Callable:
        """Decorator for profiling individual functions."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Start profiling
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss
            
            # Run function with CPU profiling
            self.cpu_profiler.enable()
            result = func(*args, **kwargs)
            self.cpu_profiler.disable()
            
            # Collect metrics
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss
            
            # Save statistics
            self.stats[func.__name__] = {
                'execution_time': end_time - start_time,
                'memory_usage': (end_memory - start_memory) / 1024 / 1024,  # MB
                'timestamp': self.timestamp
            }
            
            return result
        return wrapper
    
    def save_results(self):
        """Save profiling results to files."""
        # Save CPU profiling stats
        stats_file = os.path.join(self.output_dir, f"cpu_stats_{self.timestamp}.prof")
        self.cpu_profiler.dump_stats(stats_file)
        
        # Generate readable stats
        readable_stats = os.path.join(self.output_dir, f"readable_stats_{self.timestamp}.txt")
        with open(readable_stats, 'w') as f:
            stats = pstats.Stats(self.cpu_profiler, stream=f)
            stats.sort_stats('cumulative')
            stats.print_stats()
        
        # Save function-specific stats
        import json
        stats_json = os.path.join(self.output_dir, f"function_stats_{self.timestamp}.json")
        with open(stats_json, 'w') as f:
            json.dump(self.stats, f, indent=4)
        
        logger.info(f"Profiling results saved to {self.output_dir}")

class Benchmark:
    """
    Benchmarking utilities for EmptyDrops performance testing.
    """
    def __init__(self):
        self.results = []
        
    def benchmark_function(
        self,
        func: Callable,
        args: tuple,
        kwargs: dict,
        n_runs: int = 5,
        warmup_runs: int = 1
    ) -> Dict[str, float]:
        """
        Benchmark a function with multiple runs.
        """
        # Warmup runs
        for _ in range(warmup_runs):
            func(*args, **kwargs)
        
        # Actual benchmark runs
        times = []
        memory_usage = []
        
        for i in range(n_runs):
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss
            
            # Run function
            result = func(*args, **kwargs)
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss
            
            times.append(end_time - start_time)
            memory_usage.append((end_memory - start_memory) / 1024 / 1024)  # MB
        
        # Calculate statistics
        stats = {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'mean_memory': np.mean(memory_usage),
            'std_memory': np.std(memory_usage)
        }
        
        self.results.append({
            'function': func.__name__,
            'stats': stats,
            'timestamp': datetime.now().isoformat()
        })
        
        return stats
    
    def plot_results(self, output_dir: str = "benchmark_results"):
        """
        Generate plots from benchmark results.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Time comparison plot
        plt.figure(figsize=(10, 6))
        functions = [r['function'] for r in self.results]
        times = [r['stats']['mean_time'] for r in self.results]
        errors = [r['stats']['std_time'] for r in self.results]
        
        plt.bar(functions, times, yerr=errors)
        plt.title('Execution Time Comparison')
        plt.xlabel('Function')
        plt.ylabel('Time (seconds)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'time_comparison.png'))
        plt.close()
        
        # Memory usage plot
        plt.figure(figsize=(10, 6))
        memory = [r['stats']['mean_memory'] for r in self.results]
        memory_errors = [r['stats']['std_memory'] for r in self.results]
        
        plt.bar(functions, memory, yerr=memory_errors)
        plt.title('Memory Usage Comparison')
        plt.xlabel('Function')
        plt.ylabel('Memory (MB)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'memory_comparison.png'))
        plt.close()

def compare_implementations(
    implementations: List[Callable],
    test_data: tuple,
    n_runs: int = 5
) -> Dict[str, Dict[str, float]]:
    """
    Compare different implementations of the same functionality.
    """
    benchmark = Benchmark()
    results = {}
    
    for impl in implementations:
        stats = benchmark.benchmark_function(
            impl,
            test_data,
            {},
            n_runs=n_runs
        )
        results[impl.__name__] = stats
    
    benchmark.plot_results()
    return results

# Example usage
if __name__ == "__main__":
    # Profile EmptyDrops
    from empty_drops import empty_drops
    from optimizations import parallel_monte_carlo, gpu_monte_carlo
    
    profiler = Profiler()
    
    @profiler.profile_function
    def run_empty_drops(data):
        return empty_drops(data)
    
    # Run profiling
    import scanpy as sc
    data = sc.read_h5ad("test_data.h5ad")
    result = run_empty_drops(data)
    
    # Save profiling results
    profiler.save_results()
    
    # Compare implementations
    implementations = [
        parallel_monte_carlo,
        gpu_monte_carlo
    ]
    
    # Generate test data
    test_data = (
        np.random.randint(0, 100, size=1000),  # totals
        np.random.random(100),                  # ambient_props
        1000                                    # n_iter
    )
    
    # Run comparison
    results = compare_implementations(implementations, test_data)
    print("Comparison results:", results) 