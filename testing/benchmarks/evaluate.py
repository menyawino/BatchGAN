"""
Benchmark and evaluation module for valve histology batch correction.
Provides comprehensive testing for comparing different batch correction methods.
"""
import os
import sys
import argparse
import logging
import yaml
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import shutil

import numpy as np
import torch
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add project root to path to allow imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.metrics import calculate_ssim, calculate_psnr, calculate_fid, get_tissue_feature_correlation
from src.models.trainer import CycleGANTrainer
import src.utils.metrics as metrics


def setup_logging(log_level=logging.INFO):
    """Set up logging configuration."""
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


class BatchCorrectionEvaluator:
    """Class for evaluating and benchmarking different batch correction methods."""
    
    def __init__(self, output_dir: str = "results/benchmarks"):
        """
        Initialize the evaluator.
        
        Args:
            output_dir: Directory to save evaluation results
        """
        self.logger = logging.getLogger(__name__)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Dictionary to store results for all methods
        self.results = {}
        
        # Default metrics to calculate
        self.metrics = ["ssim", "psnr", "tissue_features"]
        
        # Add FID if available
        if metrics.HAS_FID:
            self.metrics.append("fid")
        
    def evaluate_cyclegan(
        self, 
        config_path: str, 
        checkpoint_path: str, 
        test_data_dir: str,
        method_name: str = "cyclegan_default"
    ) -> Dict[str, float]:
        """
        Evaluate a trained CycleGAN model.
        
        Args:
            config_path: Path to the model configuration file
            checkpoint_path: Path to the model checkpoint
            test_data_dir: Directory containing test data (with 'healthy' and 'diseased' subfolders)
            method_name: Name of the method for results
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.logger.info(f"Evaluating CycleGAN model: {method_name}")
        
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Update config with test data paths
        test_healthy_dir = os.path.join(test_data_dir, "healthy")
        test_diseased_dir = os.path.join(test_data_dir, "diseased")
        
        if not os.path.exists(test_healthy_dir) or not os.path.exists(test_diseased_dir):
            self.logger.error(f"Test data directories not found: {test_healthy_dir}, {test_diseased_dir}")
            return {}
            
        config["data"]["healthy_dir"] = test_healthy_dir
        config["data"]["diseased_dir"] = test_diseased_dir
        
        # Load trainer and model
        trainer = CycleGANTrainer(config_path)
        trainer.load_checkpoint(checkpoint_path)
        trainer.model.eval()
        
        # Create data loaders for test data
        from src.data.dataset import get_data_loaders
        _, _, test_loader = get_data_loaders(config)
        
        # Run inference on test data
        real_x_list, fake_y_list = [], []
        real_y_list, fake_x_list = [], []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Evaluating {method_name}"):
                real_x = batch["healthy"].to(trainer.device)
                real_y = batch["diseased"].to(trainer.device)
                
                # Forward pass
                outputs = trainer.model(real_x, real_y)
                
                # Collect images
                real_x_list.append(real_x.cpu())
                fake_y_list.append(outputs["fake_y"].cpu())
                real_y_list.append(real_y.cpu())
                fake_x_list.append(outputs["fake_x"].cpu())
        
        # Concatenate all batches
        real_x_all = torch.cat(real_x_list, dim=0)
        fake_y_all = torch.cat(fake_y_list, dim=0)
        real_y_all = torch.cat(real_y_list, dim=0)
        fake_x_all = torch.cat(fake_x_list, dim=0)
        
        # Calculate metrics
        results = {}
        
        # SSIM for structure preservation
        if "ssim" in self.metrics:
            ssim_x2y = calculate_ssim(real_x_all, fake_y_all)
            ssim_y2x = calculate_ssim(real_y_all, fake_x_all)
            results["ssim_x2y"] = ssim_x2y
            results["ssim_y2x"] = ssim_y2x
            results["ssim_avg"] = (ssim_x2y + ssim_y2x) / 2
            
        # PSNR for image quality
        if "psnr" in self.metrics:
            psnr_x2y = calculate_psnr(real_x_all, fake_y_all)
            psnr_y2x = calculate_psnr(real_y_all, fake_x_all)
            results["psnr_x2y"] = psnr_x2y
            results["psnr_y2x"] = psnr_y2x
            results["psnr_avg"] = (psnr_x2y + psnr_y2x) / 2
            
        # FID for distribution similarity
        if "fid" in self.metrics and metrics.HAS_FID:
            fid_x2y = calculate_fid(real_y_all, fake_y_all)
            fid_y2x = calculate_fid(real_x_all, fake_x_all)
            results["fid_x2y"] = fid_x2y
            results["fid_y2x"] = fid_y2x
            results["fid_avg"] = (fid_x2y + fid_y2x) / 2
            
        # Tissue feature correlation
        if "tissue_features" in self.metrics:
            tf_x2y = get_tissue_feature_correlation(real_x_all, fake_y_all)
            tf_y2x = get_tissue_feature_correlation(real_y_all, fake_x_all)
            results["tissue_features_x2y"] = tf_x2y
            results["tissue_features_y2x"] = tf_y2x
            results["tissue_features_avg"] = (tf_x2y + tf_y2x) / 2
            
        # Save results
        self.results[method_name] = results
        
        # Save sample images
        self._save_sample_images(
            real_x_all[:8], fake_y_all[:8], 
            real_y_all[:8], fake_x_all[:8], 
            method_name
        )
        
        return results
    
    def evaluate_baseline_histogram_matching(
        self, 
        test_data_dir: str,
        method_name: str = "histogram_matching"
    ) -> Dict[str, float]:
        """
        Evaluate a simple histogram matching baseline.
        
        Args:
            test_data_dir: Directory containing test data
            method_name: Name of the method for results
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.logger.info(f"Evaluating baseline method: {method_name}")
        
        # Load test images
        test_healthy_dir = os.path.join(test_data_dir, "healthy")
        test_diseased_dir = os.path.join(test_data_dir, "diseased")
        
        if not os.path.exists(test_healthy_dir) or not os.path.exists(test_diseased_dir):
            self.logger.error(f"Test data directories not found: {test_healthy_dir}, {test_diseased_dir}")
            return {}
        
        # Get image files
        healthy_files = self._get_image_files(test_healthy_dir)
        diseased_files = self._get_image_files(test_diseased_dir)
        
        # Load images
        healthy_images = []
        diseased_images = []
        
        for f in healthy_files[:50]:  # Limit to 50 images for speed
            img = cv2.imread(f)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                healthy_images.append(img)
                
        for f in diseased_files[:50]:  # Limit to 50 images for speed
            img = cv2.imread(f)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                diseased_images.append(img)
        
        self.logger.info(f"Loaded {len(healthy_images)} healthy images and {len(diseased_images)} diseased images")
        
        # Apply histogram matching
        fake_diseased = []  # healthy → diseased
        fake_healthy = []   # diseased → healthy
        
        # Reference images for matching (use the mean of each domain)
        healthy_ref = np.mean([img.astype(np.float32) for img in healthy_images], axis=0).astype(np.uint8)
        diseased_ref = np.mean([img.astype(np.float32) for img in diseased_images], axis=0).astype(np.uint8)
        
        # Histogram matching functions
        def hist_match(source, reference):
            result = np.zeros_like(source)
            for c in range(3):  # RGB channels
                # Calculate histograms
                src_hist, src_bins = np.histogram(source[:,:,c].flatten(), 256, [0,256], density=True)
                ref_hist, ref_bins = np.histogram(reference[:,:,c].flatten(), 256, [0,256], density=True)
                
                # Calculate CDFs
                src_cdf = src_hist.cumsum()
                src_cdf /= src_cdf[-1]
                
                ref_cdf = ref_hist.cumsum()
                ref_cdf /= ref_cdf[-1]
                
                # Map source to reference histogram
                interp_values = np.interp(src_cdf, ref_cdf, np.arange(256))
                
                # Apply mapping
                result[:,:,c] = interp_values[source[:,:,c]]
                
            return result.astype(np.uint8)
        
        # Apply histogram matching to each image
        for img in tqdm(healthy_images, desc="Processing healthy → diseased"):
            fake_diseased.append(hist_match(img, diseased_ref))
            
        for img in tqdm(diseased_images, desc="Processing diseased → healthy"):
            fake_healthy.append(hist_match(img, healthy_ref))
            
        # Convert to PyTorch tensors for metric calculation
        def img_to_tensor(img_list):
            tensors = []
            for img in img_list:
                # Normalize to [-1, 1]
                img_float = img.astype(np.float32) / 127.5 - 1.0
                # Convert to tensor [C, H, W]
                tensor = torch.from_numpy(img_float.transpose(2, 0, 1))
                tensors.append(tensor)
            return torch.stack(tensors)
            
        real_x_all = img_to_tensor(healthy_images)
        fake_y_all = img_to_tensor(fake_diseased)
        real_y_all = img_to_tensor(diseased_images)
        fake_x_all = img_to_tensor(fake_healthy)
        
        # Calculate metrics
        results = {}
        
        # SSIM for structure preservation
        if "ssim" in self.metrics:
            ssim_x2y = calculate_ssim(real_x_all, fake_y_all)
            ssim_y2x = calculate_ssim(real_y_all, fake_x_all)
            results["ssim_x2y"] = ssim_x2y
            results["ssim_y2x"] = ssim_y2x
            results["ssim_avg"] = (ssim_x2y + ssim_y2x) / 2
            
        # PSNR for image quality
        if "psnr" in self.metrics:
            psnr_x2y = calculate_psnr(real_x_all, fake_y_all)
            psnr_y2x = calculate_psnr(real_y_all, fake_x_all)
            results["psnr_x2y"] = psnr_x2y
            results["psnr_y2x"] = psnr_y2x
            results["psnr_avg"] = (psnr_x2y + psnr_y2x) / 2
            
        # FID for distribution similarity
        if "fid" in self.metrics and metrics.HAS_FID:
            fid_x2y = calculate_fid(real_y_all, fake_y_all)
            fid_y2x = calculate_fid(real_x_all, fake_x_all)
            results["fid_x2y"] = fid_x2y
            results["fid_y2x"] = fid_y2x
            results["fid_avg"] = (fid_x2y + fid_y2x) / 2
            
        # Tissue feature correlation
        if "tissue_features" in self.metrics:
            tf_x2y = get_tissue_feature_correlation(real_x_all, fake_y_all)
            tf_y2x = get_tissue_feature_correlation(real_y_all, fake_x_all)
            results["tissue_features_x2y"] = tf_x2y
            results["tissue_features_y2x"] = tf_y2x
            results["tissue_features_avg"] = (tf_x2y + tf_y2x) / 2
            
        # Save results
        self.results[method_name] = results
        
        # Save sample images
        self._save_sample_images(
            real_x_all[:8], fake_y_all[:8], 
            real_y_all[:8], fake_x_all[:8], 
            method_name
        )
        
        return results
    
    def compare_methods(self) -> pd.DataFrame:
        """
        Compare all evaluated methods and generate comparison plots.
        
        Returns:
            DataFrame with comparative results
        """
        if not self.results:
            self.logger.error("No methods have been evaluated yet")
            return pd.DataFrame()
        
        # Create a DataFrame with all results
        results_df = pd.DataFrame()
        
        for method, metrics_dict in self.results.items():
            # Create a row for this method
            row = {'method': method}
            row.update(metrics_dict)
            results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)
        
        # Save the results
        results_csv_path = self.output_dir / "benchmark_results.csv"
        results_df.to_csv(results_csv_path, index=False)
        self.logger.info(f"Saved benchmark results to {results_csv_path}")
        
        # Generate comparison plots
        self._generate_comparison_plots(results_df)
        
        return results_df
    
    def _get_image_files(self, directory: str) -> List[str]:
        """Get all image files in a directory."""
        image_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
        image_files = []
        
        for root, _, files in os.walk(directory):
            for file in files:
                if os.path.splitext(file)[1].lower() in image_extensions:
                    image_files.append(os.path.join(root, file))
                    
        return image_files
    
    def _save_sample_images(
        self, 
        real_x: torch.Tensor, 
        fake_y: torch.Tensor, 
        real_y: torch.Tensor,
        fake_x: torch.Tensor,
        method_name: str
    ):
        """Save sample images for visual comparison."""
        # Create directory for sample images
        samples_dir = self.output_dir / "samples" / method_name
        samples_dir.mkdir(parents=True, exist_ok=True)
        
        # Function to convert tensor to numpy image
        def tensor_to_image(tensor):
            if isinstance(tensor, torch.Tensor):
                tensor = tensor.clone().detach().cpu()
                tensor = tensor.numpy().transpose(1, 2, 0)
                # Scale from [-1, 1] to [0, 1]
                tensor = tensor * 0.5 + 0.5
                tensor = np.clip(tensor, 0, 1)
                # Scale to [0, 255]
                tensor = (tensor * 255).astype(np.uint8)
            return tensor
        
        # Save individual samples
        num_samples = min(len(real_x), 8)
        for i in range(num_samples):
            # Create a figure comparing original and generated images
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            
            # Set titles
            axes[0, 0].set_title("Real Healthy")
            axes[0, 1].set_title("Generated Diseased")
            axes[1, 0].set_title("Real Diseased")
            axes[1, 1].set_title("Generated Healthy")
            
            # Plot images
            axes[0, 0].imshow(tensor_to_image(real_x[i]))
            axes[0, 1].imshow(tensor_to_image(fake_y[i]))
            axes[1, 0].imshow(tensor_to_image(real_y[i]))
            axes[1, 1].imshow(tensor_to_image(fake_x[i]))
            
            # Remove ticks
            for ax in axes.flatten():
                ax.axis('off')
                
            # Save the figure
            sample_path = samples_dir / f"sample_{i}.png"
            plt.tight_layout()
            plt.savefig(sample_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
        # Create a montage with all samples
        fig, axes = plt.subplots(4, num_samples, figsize=(2*num_samples, 8))
        
        row_titles = ["Real Healthy", "Generated Diseased", "Real Diseased", "Generated Healthy"]
        
        for j, title in enumerate(row_titles):
            axes[j, 0].set_ylabel(title, fontsize=12, rotation=90, labelpad=15)
        
        # Plot all samples
        for i in range(num_samples):
            axes[0, i].imshow(tensor_to_image(real_x[i]))
            axes[1, i].imshow(tensor_to_image(fake_y[i]))
            axes[2, i].imshow(tensor_to_image(real_y[i]))
            axes[3, i].imshow(tensor_to_image(fake_x[i]))
            
            # Remove ticks
            for j in range(4):
                axes[j, i].axis('off')
                
        plt.tight_layout()
        montage_path = self.output_dir / "samples" / f"{method_name}_montage.png"
        plt.savefig(montage_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        
    def _generate_comparison_plots(self, results_df: pd.DataFrame):
        """Generate comparison plots for all evaluated methods."""
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up the style
        sns.set_theme(style="whitegrid")
        plt.rcParams.update({'font.size': 12})
        
        # Get average metrics that we want to compare
        metrics_to_compare = []
        for metric in ["ssim_avg", "psnr_avg", "fid_avg", "tissue_features_avg"]:
            if metric in results_df.columns:
                metrics_to_compare.append(metric)
        
        # 1. Bar plot for each metric
        for metric in metrics_to_compare:
            plt.figure(figsize=(10, 6))
            
            # Sort by metric value (higher is better except for FID)
            should_reverse = "fid" not in metric
            sorted_df = results_df.sort_values(by=metric, ascending=not should_reverse)
            
            # Create bar plot
            ax = sns.barplot(x="method", y=metric, data=sorted_df)
            
            # Add value labels
            for i, v in enumerate(sorted_df[metric]):
                ax.text(i, v + (0.01 * (v if v > 0 else 1)), 
                        f"{v:.3f}", ha='center', fontsize=10)
            
            plt.title(f"Comparison of {metric}")
            plt.xlabel("Method")
            plt.ylabel(metric)
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            
            metric_plot_path = plots_dir / f"comparison_{metric}.png"
            plt.savefig(metric_plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
        # 2. Combined metrics plot (radar chart)
        if len(metrics_to_compare) >= 3:
            self._generate_radar_chart(results_df, metrics_to_compare, plots_dir)
            
        # 3. Heatmap of all metrics for all methods
        plt.figure(figsize=(12, 8))
        
        # Get metrics columns only (exclude 'method')
        metrics_cols = [col for col in results_df.columns if col != 'method']
        
        # Set method as index
        plot_df = results_df.set_index('method')
        
        # Generate heatmap
        sns.heatmap(plot_df[metrics_cols], annot=True, fmt=".3f", cmap="YlGnBu")
        plt.title("All Metrics Comparison")
        plt.tight_layout()
        
        heatmap_path = plots_dir / "metrics_heatmap.png"
        plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    def _generate_radar_chart(self, results_df: pd.DataFrame, metrics: List[str], output_dir: Path):
        """Generate a radar chart comparing methods across metrics."""
        # Normalize metrics to [0, 1] range for fair comparison
        normalized_df = results_df.copy()
        
        for metric in metrics:
            if "fid" in metric:
                # For FID, lower is better, so invert the normalization
                min_val = results_df[metric].min()
                max_val = results_df[metric].max()
                if min_val < max_val:
                    normalized_df[metric] = 1 - ((results_df[metric] - min_val) / (max_val - min_val))
            else:
                # For other metrics, higher is better
                min_val = results_df[metric].min()
                max_val = results_df[metric].max()
                if min_val < max_val:
                    normalized_df[metric] = (results_df[metric] - min_val) / (max_val - min_val)
        
        # Create radar chart
        plt.figure(figsize=(10, 8))
        
        # Number of metrics
        N = len(metrics)
        
        # Angles for each metric
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        # Plot each method
        ax = plt.subplot(111, polar=True)
        
        for idx, method in enumerate(normalized_df['method']):
            # Get values for this method
            values = normalized_df.loc[normalized_df['method'] == method, metrics].values.flatten().tolist()
            values += values[:1]  # Close the loop
            
            # Plot values
            ax.plot(angles, values, linewidth=2, label=method)
            ax.fill(angles, values, alpha=0.1)
        
        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.title("Method Comparison (Normalized Metrics)")
        
        radar_path = output_dir / "radar_chart.png"
        plt.savefig(radar_path, dpi=150, bbox_inches='tight')
        plt.close()


def main():
    """Main entry point for benchmarking different batch correction methods."""
    parser = argparse.ArgumentParser(description="Benchmark and evaluate histology batch correction methods")
    
    parser.add_argument("--test_data_dir", type=str, required=True,
                       help="Directory containing test data")
    parser.add_argument("--output_dir", type=str, default="results/benchmarks",
                       help="Directory to save evaluation results")
    parser.add_argument("--methods", type=str, nargs="+", 
                       choices=["cyclegan", "histogram_matching", "all"],
                       default=["all"], 
                       help="Methods to evaluate")
    parser.add_argument("--cyclegan_config", type=str,
                       help="Path to CycleGAN configuration file")
    parser.add_argument("--cyclegan_checkpoint", type=str,
                       help="Path to CycleGAN checkpoint file")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    logger.info("Starting batch correction benchmarking")
    
    # Initialize evaluator
    evaluator = BatchCorrectionEvaluator(output_dir=args.output_dir)
    
    # Determine which methods to evaluate
    methods_to_evaluate = args.methods
    if "all" in methods_to_evaluate:
        methods_to_evaluate = ["cyclegan", "histogram_matching"]
    
    # Evaluate each method
    for method in methods_to_evaluate:
        if method == "cyclegan":
            if not args.cyclegan_config or not args.cyclegan_checkpoint:
                logger.error("CycleGAN evaluation requires both config and checkpoint paths")
                continue
                
            evaluator.evaluate_cyclegan(
                args.cyclegan_config,
                args.cyclegan_checkpoint,
                args.test_data_dir,
                method_name="cyclegan"
            )
        elif method == "histogram_matching":
            evaluator.evaluate_baseline_histogram_matching(
                args.test_data_dir,
                method_name="histogram_matching"
            )
    
    # Compare methods
    results_df = evaluator.compare_methods()
    logger.info(f"Benchmark results:\n{results_df}")
    
    logger.info(f"Benchmarking complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()