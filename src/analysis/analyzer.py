import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import ast

from ..core.models import DynamicFeeModel
from ..core.utils import DataIO, JSONHandler
from ..config import DATA_DIR, RESULTS_DIR, ANALYSIS_DIR, DEFAULT_MAX_FEE


class DataProcessor:
    """
    Processes data for analysis.
    """
    
    @staticmethod
    def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for analysis.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Processed DataFrame
        """
        # Create a copy
        processed_df = df.copy()
        
        # Ensure chain column exists
        if 'chain' not in processed_df.columns:
            print("Warning: 'chain' column not found in data. Adding default value 'unknown'.")
            processed_df['chain'] = 'unknown'
        
        # Ensure base_fee_percent column exists
        if 'base_fee_percent' not in processed_df.columns:
            print("Warning: 'base_fee_percent' column not found in data.")
            if 'base_fee' in processed_df.columns and 'amount_in' in processed_df.columns:
                # Calculate base_fee_percent from base_fee and amount_in
                print("Calculating base_fee_percent from base_fee and amount_in.")
                processed_df['base_fee_percent'] = processed_df.apply(
                    lambda row: row['base_fee'] / row['amount_in'] if row['amount_in'] > 0 else 0.001, 
                    axis=1
                )
            elif 'fee_percent' in processed_df.columns:
                # Use fee_percent as base_fee_percent
                print("Using fee_percent as base_fee_percent.")
                processed_df['base_fee_percent'] = processed_df['fee_percent']
            else:
                # Use default value
                print("Using default value 0.001 for base_fee_percent.")
                processed_df['base_fee_percent'] = 0.001
        
        # Parse JSON string columns
        for col in ['pre_event_weights', 'post_event_weights', 'pre_event_balances', 'post_event_balances']:
            if col in processed_df.columns:
                try:
                    processed_df[col] = processed_df[col].apply(
                        lambda x: json.loads(x) if isinstance(x, str) else x
                    )
                except:
                    try:
                        processed_df[col] = processed_df[col].apply(
                            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
                        )
                    except:
                        print(f"Failed to parse column {col}")
        
        # Process pools
        all_processed = []
        
        for (chain, pool_address), group in processed_df.groupby(['chain', 'pool_address']):
            print(f"Processing pool {chain}:{pool_address} ({len(group)} transactions)...")
            
            # Calculate optimal weights (equal weights)
            first_row = group.iloc[0]
            if 'pre_event_weights' in group.columns:
                n_tokens = len(first_row['pre_event_weights'])
            else:
                # Estimate from balances if weights not available
                n_tokens = len(first_row['pre_event_balances']) if 'pre_event_balances' in group.columns else 2
            
            optimal_weights = np.array([1.0 / n_tokens] * n_tokens)
            
            # Define deviation calculation function
            def deviation(weights):
                return np.sum(np.abs(np.array(weights) - optimal_weights))
            
            # Calculate current and post-action deviations
            if 'curdiff' not in group.columns and 'pre_event_weights' in group.columns:
                group['curdiff'] = group['pre_event_weights'].apply(deviation)
            
            if 'postdiff' not in group.columns and 'post_event_weights' in group.columns:
                group['postdiff'] = group['post_event_weights'].apply(deviation)
            
            # Calculate delta D (postdiff - curdiff)
            if 'curdiff' in group.columns and 'postdiff' in group.columns:
                group['deltaD'] = group['postdiff'] - group['curdiff']
            
            all_processed.append(group)
        
        if not all_processed:
            return processed_df
        
        # Combine all processed data
        return pd.concat(all_processed, ignore_index=True)


class FeeAnalyzer:
    """
    Analyzes dynamic fee models.
    """
    
    def __init__(self, df: pd.DataFrame, output_dir: str = ANALYSIS_DIR):
        """
        Initialize fee analyzer.
        
        Args:
            df: DataFrame with pool data
            output_dir: Output directory for analysis results
        """
        self.df = df
        self.output_dir = output_dir
        self.model = None
        self.result_df = None
        self.analysis_results = None
        
        # Ensure output directory exists
        DataIO.ensure_directory_exists(output_dir)
    
    def calculate_dynamic_fees(self, model: DynamicFeeModel) -> pd.DataFrame:
        """
        Apply dynamic fee model to calculate fees.
        
        Args:
            model: Dynamic fee model
            
        Returns:
            DataFrame with dynamic fees added
        """
        # Store the model for later use
        self.model = model
        
        # Create a copy of the dataframe
        result_df = self.df.copy()
        
        # Calculate dynamic fees using the model
        result_df['dynamic_fee_percent'] = result_df.apply(
            lambda row: model.calculate_fee(row['curdiff'], row['postdiff'], row['base_fee_percent']), 
            axis=1
        )
        
        # Extract actual fees
        result_df['actual_fee_percent'] = result_df['fee_percent']
        
        # Calculate fee errors
        result_df['fee_percent_error'] = result_df['dynamic_fee_percent'] - result_df['actual_fee_percent']
        
        # Calculate relative errors (%)
        result_df['relative_fee_percent_error'] = result_df.apply(
            lambda row: row['fee_percent_error'] / row['actual_fee_percent'] * 100 
                       if row['actual_fee_percent'] > 0 else 0, 
            axis=1
        )
        
        # Store the result
        self.result_df = result_df
        
        return result_df
    
    def analyze_fee_model(self, slope: float, max_fee: float) -> Dict[str, Any]:
        """
        Run comprehensive analysis of a dynamic fee model.
        
        Args:
            slope: Slope parameter for dynamic fee model
            max_fee: Maximum fee cap
            
        Returns:
            Dictionary with analysis results
        """
        # Create model
        model = DynamicFeeModel(slope=slope, max_fee=max_fee)
        
        # Process data
        processed_df = DataProcessor.prepare_data(self.df)
        
        # Calculate dynamic fees
        result_df = self.calculate_dynamic_fees(model)
        
        # Basic statistics
        stats = {
            'actual_fee_percent_mean': result_df['actual_fee_percent'].mean(),
            'dynamic_fee_percent_mean': result_df['dynamic_fee_percent'].mean(),
            'actual_fee_percent_median': result_df['actual_fee_percent'].median(),
            'dynamic_fee_percent_median': result_df['dynamic_fee_percent'].median(),
            'actual_fee_percent_std': result_df['actual_fee_percent'].std(),
            'dynamic_fee_percent_std': result_df['dynamic_fee_percent'].std(),
            'mse': mean_squared_error(result_df['actual_fee_percent'], result_df['dynamic_fee_percent']),
            'mae': mean_absolute_error(result_df['actual_fee_percent'], result_df['dynamic_fee_percent']),
            'r2': r2_score(result_df['actual_fee_percent'], result_df['dynamic_fee_percent']),
            'correlation': result_df['actual_fee_percent'].corr(result_df['dynamic_fee_percent']),
            'model_params': model.to_dict(),
            'base_fee_percent_mean': result_df['base_fee_percent'].mean(),
            'base_fee_percent_median': result_df['base_fee_percent'].median(),
            'base_fee_percent_std': result_df['base_fee_percent'].std()
        }
        
        # Error statistics
        stats.update({
            'fee_percent_error_mean': result_df['fee_percent_error'].mean(),
            'fee_percent_error_median': result_df['fee_percent_error'].median(),
            'fee_percent_error_std': result_df['fee_percent_error'].std(),
            'fee_percent_error_abs_mean': result_df['fee_percent_error'].abs().mean(),
            'relative_fee_percent_error_mean': result_df['relative_fee_percent_error'].mean(),
            'relative_fee_percent_error_median': result_df['relative_fee_percent_error'].median()
        })
        
        # Delta D statistics (positive)
        positive_delta = result_df[result_df['deltaD'] > 0]
        if not positive_delta.empty:
            stats.update({
                'positive_deltaD_count': len(positive_delta),
                'positive_deltaD_percent': len(positive_delta) / len(result_df) * 100,
                'positive_deltaD_actual_fee_percent_mean': positive_delta['actual_fee_percent'].mean(),
                'positive_deltaD_dynamic_fee_percent_mean': positive_delta['dynamic_fee_percent'].mean(),
                'positive_deltaD_fee_percent_error_mean': positive_delta['fee_percent_error'].mean(),
                'positive_deltaD_relative_fee_percent_error_mean': positive_delta['relative_fee_percent_error'].mean()
            })
        
        # Delta D statistics (negative)
        negative_delta = result_df[result_df['deltaD'] < 0]
        if not negative_delta.empty:
            stats.update({
                'negative_deltaD_count': len(negative_delta),
                'negative_deltaD_percent': len(negative_delta) / len(result_df) * 100,
                'negative_deltaD_actual_fee_percent_mean': negative_delta['actual_fee_percent'].mean(),
                'negative_deltaD_dynamic_fee_percent_mean': negative_delta['dynamic_fee_percent'].mean(),
                'negative_deltaD_fee_percent_error_mean': negative_delta['fee_percent_error'].mean(),
                'negative_deltaD_relative_fee_percent_error_mean': negative_delta['relative_fee_percent_error'].mean()
            })
        
        # Fee bin statistics
        fee_bins = [0, 0.005, 0.01, 0.02, 0.05, 0.1, 1.0]
        fee_labels = ['0-0.05', '0.05-0.1', '0.1-0.2', '0.2-0.5', '0.5-1', '>1']
        
        result_df['fee_bin'] = pd.cut(result_df['actual_fee_percent'], bins=fee_bins, labels=fee_labels)
        fee_stats = result_df.groupby('fee_bin').agg({
            'actual_fee_percent': ['mean', 'count'],
            'dynamic_fee_percent': ['mean', 'count'],
            'fee_percent_error': ['mean', 'std'],
            'relative_fee_percent_error': ['mean', 'median']
        })
        
        # Pool performance
        pool_performance = result_df.groupby(['chain', 'pool_address']).apply(
            lambda x: pd.Series({
                'count': len(x),
                'base_fee_percent': x['base_fee_percent'].iloc[0],
                'actual_fee_percent_mean': x['actual_fee_percent'].mean(),
                'dynamic_fee_percent_mean': x['dynamic_fee_percent'].mean(),
                'mse': mean_squared_error(x['actual_fee_percent'], x['dynamic_fee_percent']),
                'r2': r2_score(x['actual_fee_percent'], x['dynamic_fee_percent']) if len(x) > 1 else 0,
                'correlation': x['actual_fee_percent'].corr(x['dynamic_fee_percent']) if len(x) > 1 else 0
            })
        ).reset_index()
        
        # Time-based statistics
        daily_stats = pd.DataFrame()
        weekly_stats = pd.DataFrame()
        
        if 'timestamp' in result_df.columns:
            # Daily aggregation
            daily_stats = result_df.groupby(result_df['timestamp'].dt.date).agg({
                'actual_fee_percent': 'mean',
                'dynamic_fee_percent': 'mean',
                'fee_percent_error': 'mean',
                'relative_fee_percent_error': 'mean',
                'deltaD': 'mean',
                'base_fee_percent': 'mean'
            }).reset_index()
            
            # Weekly aggregation
            result_df['week'] = result_df['timestamp'].dt.isocalendar().week
            result_df['year'] = result_df['timestamp'].dt.isocalendar().year
            weekly_stats = result_df.groupby(['year', 'week']).agg({
                'actual_fee_percent': 'mean',
                'dynamic_fee_percent': 'mean',
                'fee_percent_error': 'mean',
                'relative_fee_percent_error': 'mean',
                'deltaD': 'mean',
                'base_fee_percent': 'mean'
            }).reset_index()
        
        # Save processed data
        DataIO.save_dataframe(result_df, 'processed_data.csv', self.output_dir)
        
        # Compile results
        analysis_results = {
            'stats': stats,
            'fee_bin_stats': fee_stats,
            'daily_stats': daily_stats,
            'weekly_stats': weekly_stats,
            'pool_performance': pool_performance
        }
        
        # Store results for later use
        self.analysis_results = analysis_results
        
        return analysis_results
    
    def analyze_parameter_sensitivity(self, slope_values: Optional[List[float]] = None) -> pd.DataFrame:
        """
        Analyze sensitivity of model to different slope values.
        
        Args:
            slope_values: List of slope values to test
            
        Returns:
            DataFrame with sensitivity analysis results
        """
        if slope_values is None:
            slope_values = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
        
        # Store results
        sensitivity_results = []
        
        # Analyze each slope value
        for slope in slope_values:
            model = DynamicFeeModel(slope=slope, max_fee=DEFAULT_MAX_FEE)
            result_df = self.calculate_dynamic_fees(model)
            analysis = self.analyze_fee_model(slope, DEFAULT_MAX_FEE)
            
            # Extract key metrics
            sensitivity_results.append({
                'slope': slope,
                'mse': analysis['stats']['mse'],
                'mae': analysis['stats']['mae'],
                'r2': analysis['stats']['r2'],
                'correlation': analysis['stats']['correlation'],
                'actual_fee_percent_mean': analysis['stats']['actual_fee_percent_mean'],
                'dynamic_fee_percent_mean': analysis['stats']['dynamic_fee_percent_mean'],
                'fee_percent_error_mean': analysis['stats']['fee_percent_error_mean'],
                'relative_fee_percent_error_mean': analysis['stats']['relative_fee_percent_error_mean']
            })
        
        # Create DataFrame
        sensitivity_df = pd.DataFrame(sensitivity_results)
        
        # Save results
        DataIO.save_dataframe(sensitivity_df, 'parameter_sensitivity.csv', self.output_dir)
        
        # Find optimal parameters
        best_r2 = sensitivity_df.loc[sensitivity_df['r2'].idxmax()]
        best_mse = sensitivity_df.loc[sensitivity_df['mse'].idxmin()]
        
        print(f"\nOptimal Slope (based on R²):")
        print(f"  Slope: {best_r2['slope']}")
        print(f"  R²: {best_r2['r2']:.4f}")
        
        print(f"\nOptimal Slope (based on MSE):")
        print(f"  Slope: {best_mse['slope']}")
        print(f"  MSE: {best_mse['mse']:.8f}")
        
        return sensitivity_df
    
    def calculate_expected_revenue(self) -> Dict[str, Any]:
        """
        Calculate expected revenue when dynamic fee model is applied.
        
        Returns:
            Dictionary with revenue analysis results
        """
        if self.result_df is None:
            print("No result data available. Run analyze_fee_model first.")
            return {}
        
        result_df = self.result_df
        
        # Calculate original fee amount
        amount_col = "amount_in"
        if amount_col and 'actual_fee_percent' in result_df.columns:
            result_df['actual_fee_amount'] = result_df[amount_col] * result_df['actual_fee_percent']
            
            # Calculate dynamic fee amount
            result_df['dynamic_fee_amount'] = result_df[amount_col] * result_df['dynamic_fee_percent']
            
            # Calculate fee amount difference
            result_df['fee_amount_diff'] = result_df['dynamic_fee_amount'] - result_df['actual_fee_amount']
            
            # Calculate totals
            total_actual_fee = result_df['actual_fee_amount'].sum()
            total_dynamic_fee = result_df['dynamic_fee_amount'].sum()
            total_fee_diff = result_df['fee_amount_diff'].sum()
            
            # Calculate revenue change percentage
            revenue_change_percent = (total_dynamic_fee / total_actual_fee - 1) * 100 if total_actual_fee > 0 else 0
            
            # Calculate pool-level revenue
            pool_revenue = result_df.groupby(['chain', 'pool_address']).agg({
                'actual_fee_amount': 'sum',
                'dynamic_fee_amount': 'sum',
                'fee_amount_diff': 'sum',
            }).reset_index()
            
            pool_revenue['revenue_change_percent'] = (
                pool_revenue['dynamic_fee_amount'] / pool_revenue['actual_fee_amount'] - 1
            ) * 100
            
            # Calculate daily revenue
            daily_revenue = pd.DataFrame()
            if 'timestamp' in result_df.columns:
                daily_revenue = result_df.groupby(result_df['timestamp'].dt.date).agg({
                    'actual_fee_amount': 'sum',
                    'dynamic_fee_amount': 'sum',
                    'fee_amount_diff': 'sum'
                }).reset_index()
                
                daily_revenue['revenue_change_percent'] = (
                    daily_revenue['dynamic_fee_amount'] / daily_revenue['actual_fee_amount'] - 1
                ) * 100
            
            return {
                'total_actual_fee': total_actual_fee,
                'total_dynamic_fee': total_dynamic_fee,
                'total_fee_diff': total_fee_diff,
                'revenue_change_percent': revenue_change_percent,
                'daily_revenue': daily_revenue,
                'pool_revenue': pool_revenue
            }
        else:
            print("Amount column or fee information is missing. Cannot calculate revenue.")
            return {}
    
    def generate_summary_report(self, 
                              sensitivity_results: Optional[pd.DataFrame] = None, 
                              revenue_results: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a summary report of analysis results.
        
        Args:
            sensitivity_results: Sensitivity analysis results
            revenue_results: Revenue analysis results
            
        Returns:
            Summary report as string
        """
        if self.analysis_results is None:
            print("No analysis results available. Run analyze_fee_model first.")
            return ""
        
        analysis_results = self.analysis_results
        stats = analysis_results['stats']
        
        report = f"""
# Dynamic Fee Model Analysis Report

## 1. Dataset Overview
- Total Transactions: {len(self.result_df)}
- Number of Pools: {self.result_df['pool_address'].nunique()}
- Number of Chains: {self.result_df['chain'].nunique()}
- Pools per Chain:
{self.result_df.groupby('chain')['pool_address'].nunique().reset_index().rename(columns={'pool_address': 'Pool Count'}).to_string(index=False)}

## 2. Model Parameters
- Slope: {stats['model_params']['slope']}
- Max Fee: {stats['model_params']['max_fee']}
- Average Base Fee Rate: {stats['base_fee_percent_mean']:.6f} ({stats['base_fee_percent_mean']*100:.4f}%)
- Median Base Fee Rate: {stats['base_fee_percent_median']:.6f} ({stats['base_fee_percent_median']*100:.4f}%)

## 3. Performance Metrics
- MSE (Mean Squared Error): {stats['mse']:.8f}
- MAE (Mean Absolute Error): {stats['mae']:.8f}
- R² (Coefficient of Determination): {stats['r2']:.4f}
- Correlation Coefficient: {stats['correlation']:.4f}

## 4. Fee Statistics
- Average Actual Fee Rate: {stats['actual_fee_percent_mean']:.6f} ({stats['actual_fee_percent_mean']*100:.4f}%)
- Average Dynamic Fee Rate: {stats['dynamic_fee_percent_mean']:.6f} ({stats['dynamic_fee_percent_mean']*100:.4f}%)
- Median Actual Fee Rate: {stats['actual_fee_percent_median']:.6f} ({stats['actual_fee_percent_median']*100:.4f}%)
- Median Dynamic Fee Rate: {stats['dynamic_fee_percent_median']:.6f} ({stats['dynamic_fee_percent_median']*100:.4f}%)
- Actual Fee Rate Standard Deviation: {stats['actual_fee_percent_std']:.6f}
- Dynamic Fee Rate Standard Deviation: {stats['dynamic_fee_percent_std']:.6f}

## 5. Error Analysis
- Mean Error: {stats['fee_percent_error_mean']:.6f}
- Median Error: {stats['fee_percent_error_median']:.6f}
- Mean Absolute Error: {stats['fee_percent_error_abs_mean']:.6f}
- Mean Relative Error: {stats['relative_fee_percent_error_mean']:.2f}%
- Median Relative Error: {stats['relative_fee_percent_error_median']:.2f}%
"""
        
        # ΔD Analysis
        if 'positive_deltaD_count' in stats:
            report += f"""
## 6. ΔD Analysis
- ΔD > 0 (Increased Imbalance) Ratio: {stats['positive_deltaD_percent']:.2f}% ({stats['positive_deltaD_count']} transactions)
- ΔD < 0 (Decreased Imbalance) Ratio: {stats['negative_deltaD_percent']:.2f}% ({stats['negative_deltaD_count']} transactions)
- ΔD > 0 Average Actual Fee Rate: {stats['positive_deltaD_actual_fee_percent_mean']:.6f} ({stats['positive_deltaD_actual_fee_percent_mean']*100:.4f}%)
- ΔD > 0 Average Dynamic Fee Rate: {stats['positive_deltaD_dynamic_fee_percent_mean']:.6f} ({stats['positive_deltaD_dynamic_fee_percent_mean']*100:.4f}%)
- ΔD < 0 Average Actual Fee Rate: {stats['negative_deltaD_actual_fee_percent_mean']:.6f} ({stats['negative_deltaD_actual_fee_percent_mean']*100:.4f}%)
- ΔD < 0 Average Dynamic Fee Rate: {stats['negative_deltaD_dynamic_fee_percent_mean']:.6f} ({stats['negative_deltaD_dynamic_fee_percent_mean']*100:.4f}%)
"""
        
        # Revenue Analysis
        if revenue_results:
            report += f"""
## 7. Revenue Analysis
- Total Actual Fee: {revenue_results['total_actual_fee']:.2f}
- Total Dynamic Fee: {revenue_results['total_dynamic_fee']:.2f}
- Total Fee Difference: {revenue_results['total_fee_diff']:.2f}
- Revenue Change Percentage: {revenue_results['revenue_change_percent']:.2f}%
"""
        
        # Pool-level performance analysis
        pool_performance = analysis_results['pool_performance']
        if not pool_performance.empty:
            top_pools = pool_performance.nlargest(5, 'count')
            
            report += f"""
## 8. Top Pool Performance Analysis
"""
            
            for _, row in top_pools.iterrows():
                report += f"""
### {row['chain']}:{row['pool_address']}
- Transaction Count: {row['count']}
- Base Fee Rate: {row['base_fee_percent']:.6f} ({row['base_fee_percent']*100:.4f}%)
- Average Actual Fee Rate: {row['actual_fee_percent_mean']:.6f} ({row['actual_fee_percent_mean']*100:.4f}%)
- Average Dynamic Fee Rate: {row['dynamic_fee_percent_mean']:.6f} ({row['dynamic_fee_percent_mean']*100:.4f}%)
- MSE: {row['mse']:.8f}
- R²: {row['r2']:.4f}
- Correlation Coefficient: {row['correlation']:.4f}
"""
        
        # Sensitivity analysis results
        if sensitivity_results is not None:
            best_r2 = sensitivity_results.loc[sensitivity_results['r2'].idxmax()]
            best_mse = sensitivity_results.loc[sensitivity_results['mse'].idxmin()]
            
            report += f"""
## 9. Sensitivity Analysis Results
- Optimal Slope (based on R²):
  - Slope: {best_r2['slope']}
  - R²: {best_r2['r2']:.4f}

- Optimal Slope (based on MSE):
  - Slope: {best_mse['slope']}
  - MSE: {best_mse['mse']:.8f}
"""
        
        # Save report
        report_path = os.path.join(self.output_dir, 'dynamic_fee_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"Report saved to {report_path}")
        
        return report 