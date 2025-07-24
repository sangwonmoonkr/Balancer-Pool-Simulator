from typing import Dict, Any, Optional, List, Tuple
import numpy as np


class DynamicFeeModel:
    """
    Dynamic Fee Model class that implements the fee calculation logic.
    
    The model uses the formula:
    fee = base_fee + (postdiff - curdiff) * slope + (1/2) * (postdiff - curdiff)^2 * slope^2
    
    Attributes:
        slope (float): Scaling factor that controls fee increase rate
        max_fee (float): Maximum fee cap
    """
    
    def __init__(self, slope: float, max_fee: float):
        """
        Initialize the dynamic fee model.
        
        Args:
            slope: Scaling factor that controls fee increase rate
            max_fee: Maximum fee cap
        """
        self.slope = slope
        self.max_fee = max_fee
    
    def calculate_fee(self, curdiff: float, postdiff: float, base_fee: float) -> float:
        """
        Calculate dynamic fee based on the deviation change.
        
        Formula: fee = base_fee + (postdiff - curdiff) * slope + (1/2) * (postdiff - curdiff)^2 * slope^2
        
        Args:
            curdiff: Current difference from optimal state
            postdiff: Post-action difference from optimal state
            base_fee: Base fee for the pool
            
        Returns:
            Calculated fee amount (capped at max_fee)
        """
        # Calculate delta_d (postdiff - curdiff)
        delta_d = postdiff - curdiff
        
        # Start with base fee
        fee = base_fee
        
        # Apply additional fee components only when difference increases (delta_d > 0)
        if delta_d > 0:
            # Linear term
            fee += delta_d * self.slope
            
            # Quadratic term
            fee += 0.5 * (delta_d ** 2) * (self.slope ** 2)
        
        # Apply maximum fee cap
        return min(fee, self.max_fee)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert model parameters to dictionary.
        
        Returns:
            Dictionary with model parameters
        """
        return {
            'slope': self.slope,
            'max_fee': self.max_fee
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DynamicFeeModel':
        """
        Create a model instance from dictionary.
        
        Args:
            data: Dictionary with model parameters
            
        Returns:
            New DynamicFeeModel instance
        """
        return cls(
            slope=data.get('slope', 0.01),
            max_fee=data.get('max_fee', 0.0095)
        ) 