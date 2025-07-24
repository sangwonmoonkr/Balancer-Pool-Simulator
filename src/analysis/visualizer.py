import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns
from typing import Dict, List, Optional, Any, Union, Tuple

from ..core.utils import Visualizer
from ..config import PLOT_DPI, PLOT_FIGSIZE, PLOT_STYLE


class FeeModelVisualizer:
    """
    Generates visualizations for dynamic fee model analysis results.
    """
    
    def __init__(self, analysis_results: Dict[str, Any], output_dir: str):
        """
        Initialize visualizer.
        
        Args:
            analysis_results: Analysis results from FeeAnalyzer
            output_dir: Directory to save visualizations
        """
        self.analysis_results = analysis_results
        self.output_dir = output_dir
        
        # Set plot style
        Visualizer.set_plot_style(PLOT_STYLE, PLOT_FIGSIZE, PLOT_DPI)
    
    def plot_deltaD_vs_fees(self, df: pd.DataFrame) -> None:
        """
        Create scatter plots of deltaD vs actual and dynamic fees.
        
        Args:
            df: DataFrame with fee and deltaD data
        """
        fig = plt.figure(figsize=(12, 6))
        
        # Plot actual fees
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.scatter(df['deltaD'], df['actual_fee_percent'], alpha=0.5, s=5, label='Actual Fee')
        ax1.set_xlabel('ΔD (postdiff - curdiff)')
        ax1.set_ylabel('Fee Ratio')
        ax1.set_title('ΔD vs Actual Fee')
        ax1.grid(True, alpha=0.3)
        
        # Plot dynamic fees with theoretical curve
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.scatter(df['deltaD'], df['dynamic_fee_percent'], alpha=0.5, s=5, label='Dynamic Fee')
        
        ax2.set_xlabel('ΔD (postdiff - curdiff)')
        ax2.set_ylabel('Fee Ratio')
        ax2.set_title('ΔD vs Dynamic Fee')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        fig.tight_layout()
        Visualizer.save_figure(fig, 'deltaD_vs_fees.png', PLOT_DPI, self.output_dir)
    
    def plot_actual_vs_dynamic_fees(self, df: pd.DataFrame) -> None:
        """
        Create scatter plot of actual vs dynamic fees.
        
        Args:
            df: DataFrame with fee data
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(df['actual_fee_percent'], df['dynamic_fee_percent'], alpha=0.5, s=5)
        
        # Add 1:1 line
        if not df.empty:
            min_fee = min(df['actual_fee_percent'].min(), df['dynamic_fee_percent'].min())
            max_fee = max(df['actual_fee_percent'].max(), df['dynamic_fee_percent'].max())
            ax.plot([min_fee, max_fee], [min_fee, max_fee], 'r--', label='1:1 Line')
        
        ax.set_xlabel('Actual Fee Ratio')
        ax.set_ylabel('Dynamic Fee Ratio')
        ax.set_title('Actual vs Dynamic Fee Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        Visualizer.save_figure(fig, 'actual_vs_dynamic_fees.png', PLOT_DPI, self.output_dir)
    
    def plot_error_distribution(self, df: pd.DataFrame) -> None:
        """
        Create histograms of fee errors.
        
        Args:
            df: DataFrame with error data
        """
        fig = plt.figure(figsize=(12, 5))
        
        # Absolute error
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.hist(df['fee_percent_error'], bins=50, alpha=0.7)
        ax1.axvline(x=0, color='r', linestyle='--')
        ax1.set_xlabel('Error (Dynamic - Actual)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Fee Error Distribution')
        ax1.grid(True, alpha=0.3)
        
        # Relative error
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.hist(df['relative_fee_percent_error'], bins=50, alpha=0.7)
        ax2.axvline(x=0, color='r', linestyle='--')
        ax2.set_xlabel('Relative Error (%)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Fee Relative Error Distribution')
        ax2.grid(True, alpha=0.3)
        
        fig.tight_layout()
        Visualizer.save_figure(fig, 'fee_error_distribution.png', PLOT_DPI, self.output_dir)
    
    def plot_time_series(self) -> None:
        """
        Create time series plot of actual and dynamic fees.
        """
        daily_stats = self.analysis_results['daily_stats']
        
        if not daily_stats.empty:
            fig = plt.figure(figsize=(12, 6))
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(daily_stats['timestamp'], daily_stats['actual_fee_percent'], 
                    label='Actual Fee Rate', marker='o')
            ax.plot(daily_stats['timestamp'], daily_stats['dynamic_fee_percent'], 
                    label='Dynamic Fee Rate', marker='x')
            ax.plot(daily_stats['timestamp'], daily_stats['base_fee_percent'], 
                    label='Base Fee Rate', marker='^', linestyle='--')
            ax.set_xlabel('Date')
            ax.set_ylabel('Average Fee Ratio')
            ax.set_title('Fee Trends Over Time')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            Visualizer.save_figure(fig, 'fee_time_series.png', PLOT_DPI, self.output_dir)
    
    def plot_fee_bin_comparison(self) -> None:
        """
        Create bar chart comparing actual and dynamic fees by fee bin.
        """
        fee_stats = self.analysis_results['fee_bin_stats']
        
        if not fee_stats.empty:
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(1, 1, 1)
            fee_bins = fee_stats.index.tolist()
            
            x = np.arange(len(fee_bins))
            width = 0.35
            
            ax.bar(x - width/2, fee_stats[('actual_fee_percent', 'count')], 
                   width, label='Actual Fee Count', alpha=0.7)
            ax.bar(x + width/2, fee_stats[('dynamic_fee_percent', 'count')], 
                   width, label='Dynamic Fee Count', alpha=0.7)
            
            ax.set_xlabel('Fee Bin')
            ax.set_ylabel('Count')
            ax.set_title('Actual vs Dynamic Fee by Fee Bin')
            ax.set_xticks(x)
            ax.set_xticklabels(fee_bins)
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            Visualizer.save_figure(fig, 'fee_bin_comparison.png', PLOT_DPI, self.output_dir)
    
    def plot_deltaD_fee_comparison(self) -> None:
        """
        Create bar chart comparing fees for positive and negative deltaD.
        """
        stats = self.analysis_results['stats']
        
        if 'positive_deltaD_actual_fee_percent_mean' in stats:
            fig = plt.figure(figsize=(10, 6))
            
            # Prepare data
            categories = ['ΔD > 0', 'ΔD < 0', 'All']
            actual_fees = [
                stats.get('positive_deltaD_actual_fee_percent_mean', 0),
                stats.get('negative_deltaD_actual_fee_percent_mean', 0),
                stats['actual_fee_percent_mean']
            ]
            dynamic_fees = [
                stats.get('positive_deltaD_dynamic_fee_percent_mean', 0),
                stats.get('negative_deltaD_dynamic_fee_percent_mean', 0),
                stats['dynamic_fee_percent_mean']
            ]
            
            x = np.arange(len(categories))
            width = 0.35
            
            ax = fig.add_subplot(1, 1, 1)
            ax.bar(x - width/2, actual_fees, width, label='Actual Fee Rate', alpha=0.7)
            ax.bar(x + width/2, dynamic_fees, width, label='Dynamic Fee Rate', alpha=0.7)
            
            ax.set_xlabel('ΔD Range')
            ax.set_ylabel('Average Fee Ratio')
            ax.set_title('Fee Comparison by ΔD Range')
            ax.set_xticks(x)
            ax.set_xticklabels(categories)
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            Visualizer.save_figure(fig, 'deltaD_fee_comparison.png', PLOT_DPI, self.output_dir)
    
    def plot_fee_change_distribution(self, df: pd.DataFrame) -> None:
        """
        Create bar chart showing distribution of fee changes.
        
        Args:
            df: DataFrame with fee data
        """
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(1, 1, 1)
        
        # Prepare data
        fee_change_categories = ['Dynamic > Actual', 'Dynamic < Actual', 'Dynamic = Actual']
        fee_change_counts = [
            (df['fee_percent_error'] > 0.00001).sum(),
            (df['fee_percent_error'] < -0.00001).sum(),
            (abs(df['fee_percent_error']) <= 0.00001).sum()
        ]
        
        ax.bar(fee_change_categories, fee_change_counts, alpha=0.7)
        
        ax.set_xlabel('Fee Change')
        ax.set_ylabel('Transaction Count')
        ax.set_title('Fee Change Distribution')
        ax.grid(axis='y', alpha=0.3)
        
        # Add percentage labels
        total = sum(fee_change_counts)
        for i, count in enumerate(fee_change_counts):
            percentage = count / total * 100
            ax.text(i, count + 5, f'{percentage:.1f}%', ha='center')
        
        Visualizer.save_figure(fig, 'fee_change_distribution.png', PLOT_DPI, self.output_dir)
    
    def plot_deltaD_fee_error(self, df: pd.DataFrame) -> None:
        """
        Create scatter plot of deltaD vs fee error.
        
        Args:
            df: DataFrame with deltaD and error data
        """
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(1, 1, 1)
        
        # Color by fee change direction
        colors = ['red' if err > 0 else 'blue' for err in df['fee_percent_error']]
        
        ax.scatter(df['deltaD'], df['fee_percent_error'], alpha=0.5, s=5, c=colors)
        ax.axhline(y=0, color='black', linestyle='--')
        ax.axvline(x=0, color='black', linestyle='--')
        
        ax.set_xlabel('ΔD (postdiff - curdiff)')
        ax.set_ylabel('Fee Rate Error (Dynamic - Actual)')
        ax.set_title('Relationship Between ΔD and Fee Rate Error')
        ax.grid(True, alpha=0.3)
        
        Visualizer.save_figure(fig, 'deltaD_fee_error.png', PLOT_DPI, self.output_dir)
    
    def plot_slope_sensitivity(self, sensitivity_df: pd.DataFrame) -> None:
        """
        Create line plots showing sensitivity to slope parameter.
        
        Args:
            sensitivity_df: DataFrame with sensitivity analysis results
        """
        fig = plt.figure(figsize=(14, 10))
        
        metrics = ['mse', 'mae', 'r2', 'correlation']
        titles = [
            'MSE (Mean Squared Error)', 
            'MAE (Mean Absolute Error)', 
            'R² (Coefficient of Determination)', 
            'Correlation Coefficient'
        ]
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = fig.add_subplot(2, 2, i+1)
            ax.plot(sensitivity_df['slope'], sensitivity_df[metric], marker='o')
            ax.set_xlabel('Slope Value')
            ax.set_ylabel(title)
            ax.set_title(f'Change in {title} with Slope')
            ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        Visualizer.save_figure(fig, 'slope_sensitivity.png', PLOT_DPI, self.output_dir)
    
    def plot_revenue_analysis(self, revenue_results: Dict[str, Any]) -> None:
        """
        Create visualizations for revenue analysis.
        
        Args:
            revenue_results: Revenue analysis results
        """
        if not revenue_results:
            return
        
        fig = plt.figure(figsize=(12, 10))
        
        # Daily revenue visualization
        daily_revenue = revenue_results.get('daily_revenue')
        if daily_revenue is not None and not daily_revenue.empty:
            ax1 = fig.add_subplot(2, 2, 1)
            ax1.plot(daily_revenue['timestamp'], daily_revenue['actual_fee_amount'], 
                    label='Actual Fee', marker='o')
            ax1.plot(daily_revenue['timestamp'], daily_revenue['dynamic_fee_amount'], 
                    label='Dynamic Fee', marker='x')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Daily Fee Revenue')
            ax1.set_title('Fee Revenue Over Time')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            ax2 = fig.add_subplot(2, 2, 2)
            ax2.bar(daily_revenue['timestamp'], daily_revenue['revenue_change_percent'])
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Revenue Change (%)')
            ax2.set_title('Daily Revenue Change')
            ax2.grid(axis='y', alpha=0.3)
        
        # Pool-level revenue visualization
        pool_revenue = revenue_results.get('pool_revenue')
        if pool_revenue is not None and not pool_revenue.empty:
            top_pools = pool_revenue.nlargest(10, 'actual_fee_amount')
            
            ax3 = fig.add_subplot(2, 2, 3)
            ax3.bar(range(len(top_pools)), top_pools['actual_fee_amount'], label='Actual Fee')
            ax3.bar(range(len(top_pools)), top_pools['dynamic_fee_amount'], 
                   label='Dynamic Fee', alpha=0.5)
            ax3.set_xticks(range(len(top_pools)))
            ax3.set_xticklabels(
                [f"{chain}:{addr[:6]}..." for chain, addr in 
                 zip(top_pools['chain'], top_pools['pool_address'])], 
                rotation=45, ha='right')
            ax3.set_xlabel('Pool')
            ax3.set_ylabel('Total Fee Revenue')
            ax3.set_title('Fee Revenue Comparison for Top 10 Pools')
            ax3.legend()
            ax3.grid(axis='y', alpha=0.3)
            
            ax4 = fig.add_subplot(2, 2, 4)
            ax4.bar(range(len(top_pools)), top_pools['revenue_change_percent'])
            ax4.set_xticks(range(len(top_pools)))
            ax4.set_xticklabels(
                [f"{chain}:{addr[:6]}..." for chain, addr in 
                 zip(top_pools['chain'], top_pools['pool_address'])], 
                rotation=45, ha='right')
            ax4.set_xlabel('Pool')
            ax4.set_ylabel('Revenue Change (%)')
            ax4.set_title('Revenue Change for Top 10 Pools')
            ax4.grid(axis='y', alpha=0.3)
        
        fig.tight_layout()
        Visualizer.save_figure(fig, 'revenue_analysis.png', PLOT_DPI, self.output_dir)
    
    def plot_all(self) -> None:
        """
        Generate all visualizations.
        """
        # Check if we have result data
        if not self.analysis_results:
            print("No analysis results available for visualization")
            return
        
        # Get processed data
        processed_data_path = os.path.join(self.output_dir, 'processed_data.csv')
        if not os.path.exists(processed_data_path):
            print(f"Processed data file not found: {processed_data_path}")
            return
        
        processed_df = pd.read_csv(processed_data_path)
        
        # Generate all plots
        print("Generating deltaD vs fees plot...")
        self.plot_deltaD_vs_fees(processed_df)
        
        print("Generating actual vs dynamic fees plot...")
        self.plot_actual_vs_dynamic_fees(processed_df)
        
        print("Generating error distribution plot...")
        self.plot_error_distribution(processed_df)
        
        print("Generating time series plot...")
        self.plot_time_series()
        
        print("Generating fee bin comparison plot...")
        self.plot_fee_bin_comparison()
        
        print("Generating deltaD fee comparison plot...")
        self.plot_deltaD_fee_comparison()
        
        print("Generating fee change distribution plot...")
        self.plot_fee_change_distribution(processed_df)
        
        print("Generating deltaD fee error plot...")
        self.plot_deltaD_fee_error(processed_df)
        
        # These plots need additional data that might not be available
        sensitivity_path = os.path.join(self.output_dir, 'parameter_sensitivity.csv')
        if os.path.exists(sensitivity_path):
            print("Generating slope sensitivity plot...")
            sensitivity_df = pd.read_csv(sensitivity_path)
            self.plot_slope_sensitivity(sensitivity_df)
        
        # Try to get revenue results from previously saved data
        # In a real implementation, this should be passed directly to the method
        try:
            revenue_path = os.path.join(self.output_dir, 'revenue_analysis.json')
            if os.path.exists(revenue_path):
                import json
                with open(revenue_path, 'r') as f:
                    revenue_results = json.load(f)
                print("Generating revenue analysis plot...")
                self.plot_revenue_analysis(revenue_results)
        except Exception as e:
            print(f"Error generating revenue analysis plot: {e}")
        
        print("All visualizations generated successfully") 