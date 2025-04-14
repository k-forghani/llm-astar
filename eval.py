import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import List, Dict, Tuple, Union, Optional, Any
from pathlib import Path
import json
import time
from scipy.stats import gmean
from matplotlib.gridspec import GridSpec
from llmastar.pather import LLMAStar, AStar


class MetricsEvaluator:
    """Class for evaluating metrics for pathfinding algorithms as described in LLM-A* paper."""
    
    def __init__(self):
        """Initialize the evaluator."""
        self.results = {}
        self.metric_names = ["operations", "storage", "path_length", "valid_path"]
        self.scale_results = {}
        
    def calculate_operation_ratio(self, llm_a_star_ops: List[int], a_star_ops: List[int]) -> float:
        """
        Calculate operation ratio using geometric mean.
        
        Args:
            llm_a_star_ops: Number of operations performed by LLM-A*
            a_star_ops: Number of operations performed by A*
            
        Returns:
            Geometric mean of operation ratios (LLM-A*/A*)
        """
        ratios = [llm / a for llm, a in zip(llm_a_star_ops, a_star_ops)]
        return gmean(ratios) * 100  # Convert to percentage
    
    def calculate_storage_ratio(self, llm_a_star_storage: List[int], a_star_storage: List[int]) -> float:
        """
        Calculate storage ratio using geometric mean.
        
        Args:
            llm_a_star_storage: Storage used by LLM-A*
            a_star_storage: Storage used by A*
            
        Returns:
            Geometric mean of storage ratios (LLM-A*/A*)
        """
        ratios = [llm / a for llm, a in zip(llm_a_star_storage, a_star_storage)]
        return gmean(ratios) * 100  # Convert to percentage
    
    def calculate_relative_path_length(self, 
                                      algorithm_paths: List[float], 
                                      optimal_paths: List[float]) -> float:
        """
        Calculate relative path length using geometric mean.
        
        Args:
            algorithm_paths: Path lengths from the algorithm being evaluated
            optimal_paths: Optimal path lengths (usually from A*)
            
        Returns:
            Geometric mean of path length ratios (algorithm/optimal)
        """
        ratios = [alg / opt for alg, opt in zip(algorithm_paths, optimal_paths)]
        return gmean(ratios) * 100  # Convert to percentage
    
    def calculate_valid_path_ratio(self, valid_paths: List[bool]) -> float:
        """
        Calculate the proportion of valid paths.
        
        Args:
            valid_paths: List of booleans indicating if each path is valid
            
        Returns:
            Percentage of valid paths
        """
        return (sum(valid_paths) / len(valid_paths)) * 100
    
    def calculate_growth_factor(self, 
                               base_operations: int, 
                               base_storage: int,
                               scaled_operations: List[int], 
                               scaled_storage: List[int],
                               scales: List[int]) -> Dict[str, List[float]]:
        """
        Calculate growth factors for operations and storage across different scales.
        
        Args:
            base_operations: Base number of operations at scale 1
            base_storage: Base storage at scale 1
            scaled_operations: Operations at different scales
            scaled_storage: Storage at different scales
            scales: List of scale factors
            
        Returns:
            Dictionary with growth factors for operations and storage
        """
        operation_growth = [op / base_operations for op in scaled_operations]
        storage_growth = [st / base_storage for st in scaled_storage]
        
        return {
            "scales": scales,
            "operation_growth": operation_growth,
            "storage_growth": storage_growth
        }
    
    def evaluate(self, 
                llm_a_star_data: Dict[str, List], 
                a_star_data: Dict[str, List],
                llm_only_data: Optional[Dict[str, List]] = None) -> Dict[str, float]:
        """
        Evaluate all metrics for LLM-A* against A*.
        
        Args:
            llm_a_star_data: Dictionary containing LLM-A* metrics
            a_star_data: Dictionary containing A* metrics
            llm_only_data: Optional dictionary containing LLM-only metrics
            
        Returns:
            Dictionary of computed metrics
        """
        results = {}
        
        # Calculate operation ratio
        results["operation_ratio"] = self.calculate_operation_ratio(
            llm_a_star_data["operations"], 
            a_star_data["operations"]
        )
        
        # Calculate storage ratio
        results["storage_ratio"] = self.calculate_storage_ratio(
            llm_a_star_data["storage"], 
            a_star_data["storage"]
        )
        
        # Calculate relative path length for LLM-A*
        results["llm_a_star_path_length"] = self.calculate_relative_path_length(
            llm_a_star_data["path_length"], 
            a_star_data["path_length"]
        )
        
        # Calculate valid path ratio for LLM-A*
        results["llm_a_star_valid_path"] = self.calculate_valid_path_ratio(
            llm_a_star_data["valid_path"]
        )
        
        # If LLM-only data is provided, calculate its metrics
        if llm_only_data:
            results["llm_only_path_length"] = self.calculate_relative_path_length(
                llm_only_data["path_length"], 
                a_star_data["path_length"]
            )
            
            results["llm_only_valid_path"] = self.calculate_valid_path_ratio(
                llm_only_data["valid_path"]
            )
        
        self.results = results
        return results
    
    def evaluate_scaling(self,
                        scales: List[int],
                        llm_a_star_scaled_data: Dict[str, List],
                        a_star_scaled_data: Dict[str, List]) -> Dict[str, Dict]:
        """
        Evaluate scaling performance across different environment sizes.
        
        Args:
            scales: List of scale factors
            llm_a_star_scaled_data: Dictionary with LLM-A* data at different scales
            a_star_scaled_data: Dictionary with A* data at different scales
            
        Returns:
            Dictionary with growth factors for both algorithms
        """
        llm_a_star_growth = self.calculate_growth_factor(
            llm_a_star_scaled_data["operations"][0],
            llm_a_star_scaled_data["storage"][0],
            llm_a_star_scaled_data["operations"],
            llm_a_star_scaled_data["storage"],
            scales
        )
        
        a_star_growth = self.calculate_growth_factor(
            a_star_scaled_data["operations"][0],
            a_star_scaled_data["storage"][0],
            a_star_scaled_data["operations"],
            a_star_scaled_data["storage"],
            scales
        )
        
        self.scale_results = {
            "llm_a_star": llm_a_star_growth,
            "a_star": a_star_growth,
            "scales": scales
        }
        
        return self.scale_results
    
    def save_results(self, output_path: str = "results"):
        """
        Save results to JSON files.
        
        Args:
            output_path: Directory to save results in
        """
        output_dir = Path(output_path)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Save main results
        with open(output_dir / "evaluation_results.json", "w") as f:
            json.dump(self.results, f, indent=4)
        
        # Save scaling results if available
        if self.scale_results:
            with open(output_dir / "scaling_results.json", "w") as f:
                # Convert numpy arrays to lists for JSON serialization
                serializable_results = {
                    k: {
                        sk: sv.tolist() if isinstance(sv, np.ndarray) else sv
                        for sk, sv in v.items()
                    } if isinstance(v, dict) else v
                    for k, v in self.scale_results.items()
                }
                json.dump(serializable_results, f, indent=4)


class MetricsVisualizer:
    """Class for visualizing metrics for pathfinding algorithms as described in LLM-A* paper."""
    
    def __init__(self, output_dir: str = "figures"):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save figures
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Set style
        sns.set_theme(style="whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
        
    def plot_efficiency_metrics(self, results: Dict[str, float], 
                               prompt_methods: Optional[List[str]] = None,
                               models: Optional[List[str]] = None) -> None:
        """
        Plot efficiency metrics (operation ratio, storage ratio, path length).
        
        Args:
            results: Dictionary of results from different models/methods
            prompt_methods: Optional list of prompt methods used
            models: Optional list of models used
        """
        if not prompt_methods or not models:
            # Simple bar chart for overall results
            metrics = ['operation_ratio', 'storage_ratio', 'llm_a_star_path_length']
            labels = ['Operation Ratio', 'Storage Ratio', 'Relative Path Length']
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(labels, [results[m] for m in metrics], color=['#2C7FB8', '#7FCDBB', '#FD8D3C'])
            
            plt.axhline(y=100, color='r', linestyle='--', alpha=0.7, label='A* Baseline')
            plt.ylabel('Percentage (%)')
            plt.title('LLM-A* Performance Metrics Relative to A*')
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.2f}%', ha='center', va='bottom')
            
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.output_dir / 'efficiency_metrics.png', dpi=300)
            plt.close()
            
        else:
            # Create DataFrame for grouped bar chart
            data = []
            for model in models:
                for method in prompt_methods:
                    key = f"{model}_{method}"
                    if key in results:
                        data.append({
                            'Model': model,
                            'Method': method,
                            'Operation Ratio': results[key]['operation_ratio'],
                            'Storage Ratio': results[key]['storage_ratio'],
                            'Path Length': results[key]['llm_a_star_path_length']
                        })
            
            df = pd.DataFrame(data)
            
            # Plot grouped bar chart
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            metrics = ['Operation Ratio', 'Storage Ratio', 'Path Length']
            titles = ['Operation Ratio (%)', 'Storage Ratio (%)', 'Relative Path Length (%)']
            
            for i, (metric, title) in enumerate(zip(metrics, titles)):
                ax = axes[i]
                sns.barplot(x='Model', y=metric, hue='Method', data=df, ax=ax)
                ax.axhline(y=100, color='r', linestyle='--', alpha=0.7)
                ax.set_title(title)
                ax.set_ylim(0, max(df[metric].max() * 1.1, 110))
                
                # Add value labels
                for p in ax.patches:
                    ax.annotate(f'{p.get_height():.1f}', 
                              (p.get_x() + p.get_width() / 2., p.get_height()),
                              ha = 'center', va = 'bottom', xytext = (0, 5),
                              textcoords = 'offset points')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'efficiency_metrics_by_model_method.png', dpi=300)
            plt.close()
    
    def plot_valid_path_ratio(self, results: Dict[str, Dict[str, float]]) -> None:
        """
        Plot valid path ratio for different approaches.
        
        Args:
            results: Dictionary of results containing valid path ratios
        """
        # Extract data
        approaches = []
        valid_ratios = []
        
        # Add A* as baseline
        approaches.append('A*')
        valid_ratios.append(100)  # A* always finds a valid path if one exists
        
        # Add LLM-A*
        if 'llm_a_star_valid_path' in results:
            approaches.append('LLM-A*')
            valid_ratios.append(results['llm_a_star_valid_path'])
        
        # Add LLM-only if available
        if 'llm_only_valid_path' in results:
            approaches.append('LLM-only')
            valid_ratios.append(results['llm_only_valid_path'])
        
        # Create plot
        plt.figure(figsize=(10, 6))
        bars = plt.bar(approaches, valid_ratios, color=['#1B9E77', '#D95F02', '#7570B3'])
        
        plt.ylabel('Valid Path Ratio (%)')
        plt.title('Valid Path Ratio by Approach')
        plt.ylim(0, 105)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.2f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'valid_path_ratio.png', dpi=300)
        plt.close()
    
    def plot_growth_factors(self, scale_results: Dict[str, Dict]) -> None:
        """
        Plot growth factors for operations and storage across different scales.
        
        Args:
            scale_results: Dictionary with growth factors across scales
        """
        scales = scale_results['scales']
        
        # Create figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Plot operation growth factor
        axes[0].plot(scales, scale_results['a_star']['operation_growth'], 
                   'o-', color='#D95F02', linewidth=2, label='A*')
        axes[0].plot(scales, scale_results['llm_a_star']['operation_growth'], 
                   'o-', color='#1B9E77', linewidth=2, label='LLM-A*')
        axes[0].set_title('Growth Factor of Operations')
        axes[0].set_xlabel('Environment Scale')
        axes[0].set_ylabel('Growth Factor')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot storage growth factor
        axes[1].plot(scales, scale_results['a_star']['storage_growth'], 
                   'o-', color='#D95F02', linewidth=2, label='A*')
        axes[1].plot(scales, scale_results['llm_a_star']['storage_growth'], 
                   'o-', color='#1B9E77', linewidth=2, label='LLM-A*')
        axes[1].set_title('Growth Factor of Storage')
        axes[1].set_xlabel('Environment Scale')
        axes[1].set_ylabel('Growth Factor')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'growth_factors.png', dpi=300)
        plt.close()
        
        # Create log scale version for better visualization of exponential vs linear growth
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Plot operation growth factor with log scale
        axes[0].plot(scales, scale_results['a_star']['operation_growth'], 
                   'o-', color='#D95F02', linewidth=2, label='A*')
        axes[0].plot(scales, scale_results['llm_a_star']['operation_growth'], 
                   'o-', color='#1B9E77', linewidth=2, label='LLM-A*')
        axes[0].set_title('Growth Factor of Operations (Log Scale)')
        axes[0].set_xlabel('Environment Scale')
        axes[0].set_ylabel('Growth Factor (log scale)')
        axes[0].set_yscale('log')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot storage growth factor with log scale
        axes[1].plot(scales, scale_results['a_star']['storage_growth'], 
                   'o-', color='#D95F02', linewidth=2, label='A*')
        axes[1].plot(scales, scale_results['llm_a_star']['storage_growth'], 
                   'o-', color='#1B9E77', linewidth=2, label='LLM-A*')
        axes[1].set_title('Growth Factor of Storage (Log Scale)')
        axes[1].set_xlabel('Environment Scale')
        axes[1].set_ylabel('Growth Factor (log scale)')
        axes[1].set_yscale('log')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'growth_factors_log_scale.png', dpi=300)
        plt.close()
    
    def create_summary_dashboard(self, results: Dict[str, float], scale_results: Dict[str, Dict]) -> None:
        """
        Create a comprehensive dashboard of all metrics.
        
        Args:
            results: Dictionary of evaluation results
            scale_results: Dictionary with growth factors across scales
        """
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(2, 3, figure=fig)
        
        # 1. Efficiency metrics
        ax1 = fig.add_subplot(gs[0, 0])
        metrics = ['operation_ratio', 'storage_ratio', 'llm_a_star_path_length']
        labels = ['Operation\nRatio', 'Storage\nRatio', 'Relative\nPath Length']
        colors = ['#2C7FB8', '#7FCDBB', '#FD8D3C']
        
        bars = ax1.bar(labels, [results[m] for m in metrics], color=colors)
        ax1.axhline(y=100, color='r', linestyle='--', alpha=0.7, label='A* Baseline')
        ax1.set_ylabel('Percentage (%)')
        ax1.set_title('Efficiency Metrics')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.1f}%', ha='center', va='bottom')
        
        # 2. Valid path ratio
        ax2 = fig.add_subplot(gs[0, 1])
        approaches = ['A*', 'LLM-A*']
        valid_ratios = [100, results.get('llm_a_star_valid_path', 0)]
        
        if 'llm_only_valid_path' in results:
            approaches.append('LLM-only')
            valid_ratios.append(results['llm_only_valid_path'])
        
        bars = ax2.bar(approaches, valid_ratios, color=['#1B9E77', '#D95F02', '#7570B3'][:len(approaches)])
        ax2.set_ylabel('Valid Path Ratio (%)')
        ax2.set_title('Valid Path Ratio')
        ax2.set_ylim(0, 105)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.1f}%', ha='center', va='bottom')
        
        # 3. Operations growth factor
        ax3 = fig.add_subplot(gs[0, 2])
        scales = scale_results['scales']
        ax3.plot(scales, scale_results['a_star']['operation_growth'], 
               'o-', color='#D95F02', linewidth=2, label='A*')
        ax3.plot(scales, scale_results['llm_a_star']['operation_growth'], 
               'o-', color='#1B9E77', linewidth=2, label='LLM-A*')
        ax3.set_title('Growth Factor of Operations')
        ax3.set_xlabel('Environment Scale')
        ax3.set_ylabel('Growth Factor')
        ax3.legend()
        ax3.grid(True)
        
        # 4. Storage growth factor
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.plot(scales, scale_results['a_star']['storage_growth'], 
               'o-', color='#D95F02', linewidth=2, label='A*')
        ax4.plot(scales, scale_results['llm_a_star']['storage_growth'], 
               'o-', color='#1B9E77', linewidth=2, label='LLM-A*')
        ax4.set_title('Growth Factor of Storage')
        ax4.set_xlabel('Environment Scale')
        ax4.set_ylabel('Growth Factor')
        ax4.legend()
        ax4.grid(True)
        
        # 5. Log scale operations growth
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.plot(scales, scale_results['a_star']['operation_growth'], 
               'o-', color='#D95F02', linewidth=2, label='A*')
        ax5.plot(scales, scale_results['llm_a_star']['operation_growth'], 
               'o-', color='#1B9E77', linewidth=2, label='LLM-A*')
        ax5.set_title('Growth Factor of Operations (Log Scale)')
        ax5.set_xlabel('Environment Scale')
        ax5.set_ylabel('Growth Factor')
        ax5.set_yscale('log')
        ax5.legend()
        ax5.grid(True)
        
        # 6. Log scale storage growth
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.plot(scales, scale_results['a_star']['storage_growth'], 
               'o-', color='#D95F02', linewidth=2, label='A*')
        ax6.plot(scales, scale_results['llm_a_star']['storage_growth'], 
               'o-', color='#1B9E77', linewidth=2, label='LLM-A*')
        ax6.set_title('Growth Factor of Storage (Log Scale)')
        ax6.set_xlabel('Environment Scale')
        ax6.set_ylabel('Growth Factor')
        ax6.set_yscale('log')
        ax6.legend()
        ax6.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'metrics_dashboard.png', dpi=300)
        plt.close()
        


def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, 'r') as f:
        return json.load(f)

def run_single_evaluation(data_entry):
    results = {
        "a_star": {"operations": [], "storage": [], "path_length": [], "valid_path": []},
        "llm_a_star": {"operations": [], "storage": [], "path_length": [], "valid_path": []}
    }
    
    
    range_x = data_entry["range_x"]
    range_y = data_entry["range_y"]
    horizontal_barriers = data_entry["horizontal_barriers"]
    vertical_barriers = data_entry["vertical_barriers"]
    

    i = 0
    for start_goal in data_entry["start_goal"]:
        if i == 9 :
            print("i = 9")
            pass
        start, goal = start_goal[0], start_goal[1]
        
        query = {
            "start": start, 
            "goal": goal, 
            "size": [range_x[1], range_y[1]],
            "horizontal_barriers": horizontal_barriers,
            "vertical_barriers": vertical_barriers,
            "range_x": range_x, 
            "range_y": range_y
        }
        
        astar_start_time = time.time()
        astar_result = astar.searching(query=query, filepath=f'evaluation_results/results/astar/{data_entry["id"]}_{i}.png')
        astar_time = time.time() - astar_start_time

        try :
            llm_start_time = time.time()
            llm_result = llm_astar.searching(query=query, filepath=f'evaluation_results/results/llm_astar/{data_entry["id"]}_{i}.png')
            llm_time = time.time() - llm_start_time
            
            # Extract metrics from results
            results["a_star"]["operations"].append(astar_result.get("operations", 0))
            results["a_star"]["storage"].append(astar_result.get("storage", 0))
            results["a_star"]["path_length"].append(astar_result.get("path_length", 0))
            results["a_star"]["valid_path"].append(astar_result.get("valid_path", False))
            
            results["llm_a_star"]["operations"].append(llm_result.get("operations", 0))
            results["llm_a_star"]["storage"].append(llm_result.get("storage", 0))
            results["llm_a_star"]["path_length"].append(llm_result.get("path_length", 0))
            results["llm_a_star"]["valid_path"].append(llm_result.get("valid_path", False))
        
            print(f"Processed {start} -> {goal} (ID: {data_entry['id']}_{i})")
        except :
            print(f"path not found on below query with {data_entry['id']}_{i} ID :")
            print(query)
            
        i+=1
    
    return results

def run_scaling_evaluation(data_entry, scales):
    scale_results = {
        "a_star": {"operations": [], "storage": []},
        "llm_a_star": {"operations": [], "storage": []}
    }
    
    start, goal = data_entry["start_goal"][0][0], data_entry["start_goal"][0][1]
    horizontal_barriers = data_entry["horizontal_barriers"]
    vertical_barriers = data_entry["vertical_barriers"]

    for scale in scales:

        range_x = [0, data_entry["range_x"][1] * scale]
        range_y = [0, data_entry["range_y"][1] * scale]
        
        scaled_start = [s * scale for s in start]
        scaled_goal = [g * scale for g in goal]
        
        scaled_horizontal_barriers = [[h[0] * scale, h[1] * scale, h[2] * scale] for h in horizontal_barriers]
        scaled_vertical_barriers = [[v[0] * scale, v[1] * scale, v[2] * scale] for v in vertical_barriers]

        query = {
            "start": scaled_start, 
            "goal": scaled_goal, 
            "size": [range_x[1], range_y[1]],
            "horizontal_barriers": scaled_horizontal_barriers,
            "vertical_barriers": scaled_vertical_barriers,
            "range_x": range_x, 
            "range_y": range_y
        }
        
        astar_result = astar.searching(query=query, filepath=f'evaluation_results/results/scaling/astar_scale_{scale}.png')

        llm_result = llm_astar.searching(query=query, filepath=f'evaluation_results/results/scaling/llm_astar_scale_{scale}.png')
 
        scale_results["a_star"]["operations"].append(astar_result.get("operations", 0))
        scale_results["a_star"]["storage"].append(astar_result.get("storage", 0))
        
        scale_results["llm_a_star"]["operations"].append(llm_result.get("operations", 0))
        scale_results["llm_a_star"]["storage"].append(llm_result.get("storage", 0))
        
        print(f"Processed scale {scale}")
    
    return scale_results

def run_evaluation(dataset_path, output_dir="evaluation_results"):
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    Path(f"{output_dir}/results").mkdir(exist_ok=True, parents=True)
    Path(f"{output_dir}/results/astar").mkdir(exist_ok=True, parents=True)
    Path(f"{output_dir}/results/llm_astar").mkdir(exist_ok=True, parents=True)
    Path(f"{output_dir}/results/scaling").mkdir(exist_ok=True, parents=True)
    Path(f"{output_dir}/figures").mkdir(exist_ok=True, parents=True)
    
    dataset = load_dataset(dataset_path)
    
    all_a_star = {"operations": [], "storage": [], "path_length": [], "valid_path": []}
    all_llm_a_star = {"operations": [], "storage": [], "path_length": [], "valid_path": []}
    
    for data_entry in dataset:
        print(f"Processing environment {data_entry['id']} ...")
        
        results = run_single_evaluation(data_entry)
        
        for metric in ["operations", "storage", "path_length", "valid_path"]:
            all_a_star[metric].extend(results["a_star"][metric])
            all_llm_a_star[metric].extend(results["llm_a_star"][metric])
    
    scales = [1, 2, 4, 6, 8, 10]
    scale_results = run_scaling_evaluation(dataset[0], scales)
    
    evaluator = MetricsEvaluator()
    
    evaluation_results = evaluator.evaluate(
        llm_a_star_data=all_llm_a_star,
        a_star_data=all_a_star
    )
    
    scaling_results = evaluator.evaluate_scaling(
        scales=scales,
        llm_a_star_scaled_data=scale_results["llm_a_star"],
        a_star_scaled_data=scale_results["a_star"]
    )
    
    evaluator.save_results(f"{output_dir}/metrics")
    
    visualizer = MetricsVisualizer(f"{output_dir}/figures")
    visualizer.plot_efficiency_metrics(evaluation_results)
    visualizer.plot_valid_path_ratio(evaluation_results)
    visualizer.plot_growth_factors(scaling_results)
    visualizer.create_summary_dashboard(evaluation_results, scaling_results)
    
    print(f"Evaluation complete. Results saved to {output_dir}")
    
    # Return results summary
    return {
        "evaluation_results": evaluation_results,
        "scaling_results": scaling_results
    }
  


astar = AStar()
llm_astar = LLMAStar(llm='qwen', variant="Qwen2.5-7B-Instruct")

results = run_evaluation("dataset/environment_50_30.json")

print("\nEvaluation Results Summary:")
for metric, value in results["evaluation_results"].items():
    print(f"{metric}: {value:.2f}%")
