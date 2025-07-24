import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict, Optional, Type

from .config import DATA_DIR, ANALYSIS_DIR, PLOT_DPI


def ensure_directory_exists(directory_path: str) -> None:
    """Ensures that a directory exists, creating it if it does not."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")


def save_dataframe(df: pd.DataFrame, filename: str, directory: str = DATA_DIR) -> str:
    """Saves a DataFrame to a CSV file."""
    ensure_directory_exists(directory)
    file_path = os.path.join(directory, filename)
    df.to_csv(file_path, index=False, encoding='utf-8')
    print(f"File saved to: {file_path}")
    return file_path


def load_dataframe(filename: str, directory: str = DATA_DIR) -> Optional[pd.DataFrame]:
    """Loads a DataFrame from a CSV file."""
    file_path = os.path.join(directory, filename)
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        print(f"File not found: {file_path}")
        return None


class CustomJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder to handle special data types like numpy arrays and pandas objects.
    """
    def default(self, obj: Any) -> Any:
        if isinstance(obj, (pd.DataFrame, pd.Series)):
            return obj.to_dict()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if pd.isna(obj):
            return None
        return super().default(obj)


def save_json(data: Dict, filename: str, directory: str = DATA_DIR, encoder: Type[json.JSONEncoder] = CustomJSONEncoder) -> str:
    """Saves data to a JSON file using a custom encoder."""
    ensure_directory_exists(directory)
    file_path = os.path.join(directory, filename)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, cls=encoder, indent=2, ensure_ascii=False)
    print(f"JSON file saved to: {file_path}")
    return file_path


def load_json(filename: str, directory: str = DATA_DIR) -> Optional[Dict]:
    """Loads data from a JSON file."""
    file_path = os.path.join(directory, filename)
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        print(f"File not found: {file_path}")
        return None


def save_figure(fig: plt.Figure, filename: str, dpi: int = PLOT_DPI, directory: str = ANALYSIS_DIR) -> str:
    """Saves a Matplotlib figure to a file and closes it."""
    ensure_directory_exists(directory)
    file_path = os.path.join(directory, filename)
    fig.savefig(file_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved to: {file_path}")
    return file_path 