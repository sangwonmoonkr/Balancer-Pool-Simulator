import argparse
import time
import os
from typing import Optional
import pandas as pd

from .core.reconstructor import StateReconstructor
from .loader import DataLoader
from .analysis.analyzer import FeeAnalyzer, DataProcessor
from .core.utils import DataIO
from .analysis.visualizer import FeeModelVisualizer
from .config import (
    RESULTS_DIR,
    ANALYSIS_DIR,
    GRAPH_API_KEY,
    DUNE_API_KEY,
    DEFAULT_SLOPE,
    DEFAULT_MAX_FEE
)


def main(chain: Optional[str] = None) -> None:
    """
    Main execution pipeline for the dynamic fee analysis.

    Args:
        chain: The blockchain network to analyze, or None to analyze all chains
    """
    # Ensure API key is available
    if not GRAPH_API_KEY:
        print("Error: GRAPH_API_KEY not found in environment variables. Please set it in .env file.")
        return
    
    # Create data loader
    data_loader = DataLoader(GRAPH_API_KEY, DUNE_API_KEY)
    
    # Track reconstructed data
    reconstructed_df = pd.DataFrame()
    output_name = ""

    # Process specific chain or all available chains
    if chain:
        print(f"===== Starting Dynamic Fee Analysis for {chain.upper()} =====")
        output_name = chain
        
        # Extract chain data first
        print(f"\nStep 1: Extracting data for {chain}...")
        chain_data = data_loader.extract_chain_data(chain_name=chain)
        if not chain_data or not chain_data.get('pool_addresses'):
            print(f"Failed to extract data for {chain}. Aborting analysis.")
            return
            
        # Reconstruct state
        print(f"\nStep 2: Reconstructing pool states for {chain}...")
        reconstructor = StateReconstructor(chain_name=chain)
        reconstructed_df = reconstructor.run()
    else:
        print("===== Starting Dynamic Fee Analysis for ALL Chains =====")
        output_name = "all"
        
        # Get available chains
        available_chains = DataLoader.get_available_chains()
        if not available_chains:
            print("No available chains found based on data files. Aborting.")
            return

        print(f"Found chains: {', '.join(available_chains)}")
        
        # Process each chain
        all_dfs = []
        for chain_name in available_chains:
            print(f"\n--- Processing chain: {chain_name} ---")
            
            # Extract chain data first
            print(f"Extracting data for {chain_name}...")
            chain_data = data_loader.extract_chain_data(chain_name=chain_name)
            if not chain_data or not chain_data.get('pool_addresses'):
                print(f"Failed to extract data for {chain_name}. Skipping.")
                continue
                
            # Reconstruct state
            print(f"Reconstructing pool states for {chain_name}...")
            reconstructor = StateReconstructor(chain_name=chain_name)
            df = reconstructor.run()
            all_dfs.append(df)
            
            # Perform analysis for this chain
            chain_output_dir = os.path.join(ANALYSIS_DIR, chain_name)
            DataIO.ensure_directory_exists(chain_output_dir)
            
            if df.empty:
                print(f"State reconstruction yielded no data for {chain_name}. Skipping analysis.")
                continue
            
            print(f"\nStep 3: Running dynamic fee model analysis for {chain_name}...")
            processed_df = DataProcessor.prepare_data(df)
            analyzer = FeeAnalyzer(processed_df, output_dir=chain_output_dir)
            analysis_results = analyzer.analyze_fee_model(slope=DEFAULT_SLOPE, max_fee=DEFAULT_MAX_FEE)
            
            if not analysis_results:
                print(f"Analysis produced no results for {chain_name}. Skipping visualization.")
                continue
            
            print(f"\nStep 5: Calculating expected revenue for {chain_name}...")
            revenue_results = analyzer.calculate_expected_revenue()
            
            print(f"\nStep 6: Generating summary report for {chain_name}...")
            analyzer.generate_summary_report(None, revenue_results)
            
            print(f"\nStep 7: Generating visualizations for {chain_name}...")
            visualizer = FeeModelVisualizer(analysis_results, chain_output_dir)
            visualizer.plot_all()
        
        # Combine all results
        if all_dfs:
            reconstructed_df = pd.concat(all_dfs, ignore_index=True)
        
    start_time = time.time()

    # Define and create output directory
    analysis_output_dir = os.path.join(ANALYSIS_DIR, output_name)
    DataIO.ensure_directory_exists(analysis_output_dir)
    
    if reconstructed_df.empty:
        print("State reconstruction yielded no data. Aborting analysis.")
        return

    # Process data and run dynamic fee model analysis
    print("\nStep 3: Running dynamic fee model analysis...")
    
    # Process data
    processed_df = DataProcessor.prepare_data(reconstructed_df)
    
    # Create analyzer
    analyzer = FeeAnalyzer(processed_df, output_dir=analysis_output_dir)
    
    # Run analysis
    analysis_results = analyzer.analyze_fee_model(
        slope=DEFAULT_SLOPE,
        max_fee=DEFAULT_MAX_FEE
    )

    if not analysis_results:
        print("Analysis produced no results. Aborting visualization.")
        return

    # Revenue analysis
    print("\nStep 5: Calculating expected revenue...")
    revenue_results = analyzer.calculate_expected_revenue()

    # Generate summary report
    print("\nStep 6: Generating summary report...")
    analyzer.generate_summary_report(None, revenue_results)

    # Step 7: Generating visualizations
    print("\nStep 7: Generating visualizations...")
    visualizer = FeeModelVisualizer(analysis_results, analysis_output_dir)
    visualizer.plot_all()

    end_time = time.time()
    print(f"\nAnalysis complete. Total time: {end_time - start_time:.2f} seconds")
    print(f"Results are saved in the '{analysis_output_dir}' directory.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the dynamic fee analysis pipeline for a specific blockchain or all available chains."
    )
    parser.add_argument(
        "--chain",
        type=str,
        default=None,
        help="The blockchain to analyze (e.g., ethereum, arbitrum). If omitted, all chains are analyzed.",
    )
    args = parser.parse_args()

    main(chain=args.chain) 