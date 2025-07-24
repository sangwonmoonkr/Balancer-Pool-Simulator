import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# API Keys from environment variables
GRAPH_API_KEY = os.getenv("GRAPH_API_KEY", "")
DUNE_API_KEY = os.getenv("DUNE_API_KEY", "")

# Data Path Settings
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
ANALYSIS_DIR = os.path.join(RESULTS_DIR, "analysis")

# Data File Name Settings
POOL_ADDRESSES_FILE = "stable_surge_pools.json"
POOL_SNAPSHOTS_FILE = "pool_snapshots.csv"
SWAPS_FILE = "swaps.csv"
ADD_REMOVES_FILE = "add_removes.csv"
UNBALANCE_LIQUIDITY_FILE = "unbalanced_liquidity_swaps.csv"
RECONSTRUCTED_STATE_FILE = "reconstructed_state.csv"

# Chain Settings
CHAINS: Dict[str, Dict[str, str]] = {
    "ethereum": {
        "pools_api": "https://gateway.thegraph.com/api/[api-key]/subgraphs/id/C4tijcwi6nThKJYBmT5JaYK2As2kJGADs89AoQaCnYz7",
        "events_api": "https://gateway.thegraph.com/api/[api-key]/subgraphs/id/4rixbLvpuBCwXTJSwyAzQgsLR8KprnyMfyCuXT8Fj5cd"
    },
    "arbitrum": {
        "pools_api": "https://gateway.thegraph.com/api/[api-key]/subgraphs/id/EjSsjATNpZexLhozmDTe9kBHpZUt1GKjWdpZ2P9xmhsv",
        "events_api": "https://gateway.thegraph.com/api/[api-key]/subgraphs/id/Ad1cgTzScNmiDPSCeGYxgMU3YdRPrQXGkCZgpmPauauk"
    },
    "base": {
        "pools_api": "https://gateway.thegraph.com/api/[api-key]/subgraphs/id/42QYdE4P8ZMKgPx4Mkw1Vnx3Zf6AEtWFVoeet1HZ4ntB",
        "events_api": "https://gateway.thegraph.com/api/[api-key]/subgraphs/id/9b7UBHq8DXxrfGsYhAzF3jZn5mNRgZb5Ag18UL9GJ3cV"
    },
    "avalanche": {
        "pools_api": "https://gateway.thegraph.com/api/[api-key]/deployments/id/QmchdxtRDQJxtt8VkV5MSmcUPvLmo1wgXD7Y7ZCNKNebN1",
        "events_api": "https://gateway.thegraph.com/api/[api-key]/deployments/id/QmSj437ejL2f1pMP2r5E2m5GjhqJa3rmbvFD5kyscmq7u2"
    },
    "gnosis": {
        "pools_api": "https://gateway.thegraph.com/api/[api-key]/subgraphs/id/yeZGqiwNf3Lqpeo8XNHih83bk5Tbu4KvFwWVy3Dbus6",
        "events_api": "https://gateway.thegraph.com/api/[api-key]/subgraphs/id/DDoABVc9xCRQwuXRq2QLZ6YLkjoFet74vnfncQDgJVo2"
    }
}

# Pagination Settings
ITEMS_PER_PAGE: int = 1000  # Maximum items per API call

# Plot Settings
PLOT_DPI: int = 300
PLOT_FIGSIZE: tuple = (12, 8)
PLOT_STYLE: str = "seaborn-v0_8-darkgrid"

# Data Extraction Settings
DATA_START_DATE: str = "2023-01-01"  # YYYY-MM-DD format
DATA_END_DATE: str = "2023-12-31"    # YYYY-MM-DD format

# Model Settings
DEFAULT_SLOPE: float = 0.5
DEFAULT_MAX_FEE: float = 0.95

CREATE_DUNE_QUERY = False
