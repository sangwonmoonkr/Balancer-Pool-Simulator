import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict, Optional, Type, List, Union, Tuple

class DataIO:
    """
    Utility class for data input/output operations.
    """
    
    @staticmethod
    def ensure_directory_exists(directory_path: str) -> None:
        """
        Ensure that a directory exists, creating it if it does not.
        
        Args:
            directory_path: Path to the directory
        """
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            print(f"Created directory: {directory_path}")

    @staticmethod
    def save_dataframe(df: pd.DataFrame, filename: str, directory: str) -> str:
        """
        Save a DataFrame to a CSV file.
        
        Args:
            df: DataFrame to save
            filename: Filename for the CSV file
            directory: Directory to save to
            
        Returns:
            Path to the saved file
        """
        DataIO.ensure_directory_exists(directory)
        file_path = os.path.join(directory, filename)
        df.to_csv(file_path, index=False, encoding='utf-8')
        print(f"File saved to: {file_path}")
        return file_path

    @staticmethod
    def load_dataframe(filename: str, directory: str, parse_dates: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
        """
        Load a DataFrame from a CSV file.
        
        Args:
            filename: Filename of the CSV file
            directory: Directory to load from
            parse_dates: List of columns to parse as dates
            
        Returns:
            DataFrame if file exists, None otherwise
        """
        file_path = os.path.join(directory, filename)
        if os.path.exists(file_path):
            return pd.read_csv(file_path, parse_dates=parse_dates)
        else:
            print(f"File not found: {file_path}")
            return None

class JSONHandler:
    """
    Utility class for JSON serialization and deserialization.
    """
    
    class CustomJSONEncoder(json.JSONEncoder):
        """
        Custom JSON encoder to handle special data types.
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

    @classmethod
    def save_json(cls, data: Dict[str, Any], filename: str, directory: str) -> str:
        """
        Save data to a JSON file using a custom encoder.
        
        Args:
            data: Data to save
            filename: Filename for the JSON file
            directory: Directory to save to
            
        Returns:
            Path to the saved file
        """
        DataIO.ensure_directory_exists(directory)
        file_path = os.path.join(directory, filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, cls=cls.CustomJSONEncoder, indent=2, ensure_ascii=False)
        print(f"JSON file saved to: {file_path}")
        return file_path

    @staticmethod
    def load_json(filename: str, directory: str) -> Optional[Dict[str, Any]]:
        """
        Load data from a JSON file.
        
        Args:
            filename: Filename of the JSON file
            directory: Directory to load from
            
        Returns:
            Data if file exists, None otherwise
        """
        file_path = os.path.join(directory, filename)
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            print(f"File not found: {file_path}")
            return None

class Visualizer:
    """
    Utility class for visualization.
    """
    
    @staticmethod
    def save_figure(fig: plt.Figure, filename: str, dpi: int, directory: str) -> str:
        """
        Save a Matplotlib figure to a file and close it.
        
        Args:
            fig: Figure to save
            filename: Filename for the figure
            dpi: DPI for the saved figure
            directory: Directory to save to
            
        Returns:
            Path to the saved figure
        """
        DataIO.ensure_directory_exists(directory)
        file_path = os.path.join(directory, filename)
        fig.savefig(file_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"Figure saved to: {file_path}")
        return file_path
    
    @staticmethod
    def set_plot_style(style: str = 'seaborn-v0_8-darkgrid', figsize: Tuple[int, int] = (12, 8), dpi: int = 300) -> None:
        """
        Set global plot style.
        
        Args:
            style: Matplotlib style name
            figsize: Figure size (width, height)
            dpi: DPI for figures
        """
        plt.style.use(style)
        plt.rcParams['figure.figsize'] = figsize
        plt.rcParams['figure.dpi'] = dpi 