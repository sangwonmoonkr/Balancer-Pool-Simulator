# Dynamic Fee Mechanism Evaluation

This project evaluates the commercial viability of a proposed dynamic fee mechanism compared to existing protocols like Curve StableSwapNG and Balancer StableSurge.

## Overview

The dynamic fee mechanism uses the following formula:

```
fee = base_fee + (postdiff - curdiff) * slope + (1/2) * (postdiff - curdiff)^2 * slope^2
```

Where:
- `postdiff - curdiff`: Change in difference from optimal state (Delta D)
- `slope`: Scaling factor controlling fee increase rate
- `(1/2) * (postdiff - curdiff)^2 * slope^2`: Quadratic term for rapid increase on large deviations

## File Structure

- `src/main.py`: Main execution script
- `src/loader.py`: Data loading and extraction
- `src/core/`: Core models and utilities
  - `models.py`: Dynamic fee model
  - `pool.py`: Pool state and event classes
  - `reconstructor.py`: State reconstruction
  - `utils.py`: I/O and visualization helpers
- `src/analysis/`: Analysis modules
  - `analyzer.py`: Fee analysis and processing
  - `visualizer.py`: Visualization generation
- `config.py`: Configuration settings
- `requirements.txt`: Dependencies
- `report.md`: Analysis report template
- `data/`: Data files (gitignore'd)
- `results/`: Output results (gitignore'd)

## Installation

1. Install requirements:
```
pip install -r requirements.txt
```

2. Set API keys in .env file (copy from .envexample):
```
GRAPH_API_KEY=your_graph_api_key
DUNE_API_KEY=your_dune_api_key
```

## Usage

Run the main script:
```
python -m src.main --chain <chain_name>  # For specific chain (e.g., ethereum, arbitrum)
python -m src.main  # For all chains
```

When no chain is specified, it processes each chain individually and generates separate analyses, then combines for overall analysis.

## Results

Results are saved in `results/analysis/`:
- Per chain directories (e.g., ethereum/)
- Combined 'all/' directory

Each contains:
- processed_data.csv
- dynamic_fee_report.md
- Various visualization PNGs
