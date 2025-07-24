from typing import Dict, List, Tuple, Optional, Any, Callable, Union
import json
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum


class Token:
    """
    Represents a token in a liquidity pool.
    """
    def __init__(self, address: str, symbol: str = "", decimals: int = 18):
        """
        Initialize a token.
        
        Args:
            address: Token contract address
            symbol: Token symbol (e.g., 'USDC', 'ETH')
            decimals: Token decimal precision
        """
        self.address = address.lower()  # Normalize to lowercase
        self.symbol = symbol
        self.decimals = decimals
    
    @property
    def scaling_factor(self) -> float:
        """
        Return the scaling factor based on token decimals.
        
        Returns:
            Scaling factor (e.g., 10^18 for 18 decimals)
        """
        return 10 ** self.decimals
    
    def __eq__(self, other: object) -> bool:
        """
        Check if two tokens are equal.
        
        Args:
            other: Another token object
            
        Returns:
            True if tokens have the same address
        """
        if not isinstance(other, Token):
            return False
        return self.address == other.address

    def __hash__(self) -> int:
        """
        Get hash of token.
        
        Returns:
            Hash value based on address
        """
        return hash(self.address)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert token to dictionary.
        
        Returns:
            Dictionary representation of the token
        """
        return {
            'address': self.address,
            'symbol': self.symbol,
            'decimals': self.decimals
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Token':
        """
        Create token from dictionary.
        
        Args:
            data: Dictionary with token data
            
        Returns:
            New Token instance
        """
        return cls(
            address=data['address'],
            symbol=data.get('symbol', ''),
            decimals=data.get('decimals', 18)
        )


class PoolState:
    """
    Represents the state of a liquidity pool at a specific point in time.
    """
    def __init__(self, 
                 pool_address: str, 
                 tokens: List[Token], 
                 balances: List[float], 
                 timestamp: Optional[pd.Timestamp] = None,
                 block_number: Optional[int] = None):
        """
        Initialize pool state.
        
        Args:
            pool_address: Address of the pool
            tokens: List of tokens in the pool
            balances: List of token balances (in same order as tokens)
            timestamp: Timestamp of this state
            block_number: Block number when this state was recorded
        """
        self.pool_address = pool_address.lower()  # Normalize to lowercase
        self.tokens = tokens
        self.balances = balances
        self.timestamp = timestamp
        self.block_number = block_number
        self._token_map = {token.address: idx for idx, token in enumerate(tokens)}
    
    def get_balance(self, token: Union[Token, str]) -> float:
        """
        Get balance of a specific token.
        
        Args:
            token: Token object or address
            
        Returns:
            Balance of the token
            
        Raises:
            ValueError: If token not in pool
        """
        token_address = token.address if isinstance(token, Token) else token.lower()
        
        if token_address not in self._token_map:
            raise ValueError(f"Token {token_address} not in pool")
        
        idx = self._token_map[token_address]
        return self.balances[idx]
    
    def get_weights(self) -> List[float]:
        """
        Calculate token weights based on balances.
        
        Returns:
            List of weight values (0-1)
        """
        total_balance = sum(self.balances)
        if total_balance == 0:
            return [0.0] * len(self.balances)
        
        return [balance / total_balance for balance in self.balances]
    
    def calculate_deviation(self, optimal_weights: List[float]) -> float:
        """
        Calculate deviation from optimal weights.
        
        Args:
            optimal_weights: List of optimal weight values
            
        Returns:
            Sum of absolute deviations from optimal weights
        """
        current_weights = self.get_weights()
        return sum(abs(cw - ow) for cw, ow in zip(current_weights, optimal_weights))
    
    def update_balance(self, token: Union[Token, str], amount: float, is_increase: bool = True) -> 'PoolState':
        """
        Update balance of a specific token.
        
        Args:
            token: Token object or address
            amount: Amount to add/subtract
            is_increase: True to add, False to subtract
            
        Returns:
            New PoolState with updated balance
            
        Raises:
            ValueError: If token not in pool
        """
        token_address = token.address if isinstance(token, Token) else token.lower()
        
        if token_address not in self._token_map:
            raise ValueError(f"Token {token_address} not in pool")
        
        idx = self._token_map[token_address]
        new_balances = self.balances.copy()
        
        if is_increase:
            new_balances[idx] += amount
        else:
            new_balances[idx] -= amount
            
        return PoolState(
            pool_address=self.pool_address,
            tokens=self.tokens,
            balances=new_balances,
            timestamp=self.timestamp,
            block_number=self.block_number
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert pool state to dictionary.
        
        Returns:
            Dictionary representation of the pool state
        """
        return {
            'pool_address': self.pool_address,
            'tokens': [token.to_dict() for token in self.tokens],
            'balances': self.balances,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'block_number': self.block_number
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PoolState':
        """
        Create pool state from dictionary.
        
        Args:
            data: Dictionary with pool state data
            
        Returns:
            New PoolState instance
        """
        tokens = [Token.from_dict(token_data) for token_data in data['tokens']]
        timestamp = pd.Timestamp(data['timestamp']) if data['timestamp'] else None
        
        return cls(
            pool_address=data['pool_address'],
            tokens=tokens,
            balances=data['balances'],
            timestamp=timestamp,
            block_number=data.get('block_number')
        )


class SwapEvent:
    """
    Represents a swap event in a liquidity pool.
    """
    def __init__(self, 
                 pool_address: str,
                 token_in: Union[Token, str],
                 token_out: Union[Token, str],
                 amount_in: float,
                 amount_out: float,
                 fee: float,
                 fee_percent: float,
                 timestamp: Optional[pd.Timestamp] = None,
                 block_number: Optional[int] = None,
                 transaction_hash: Optional[str] = None,
                 log_index: Optional[int] = None):
        """
        Initialize swap event.
        
        Args:
            pool_address: Address of the pool
            token_in: Token being swapped in
            token_out: Token being swapped out
            amount_in: Amount of token_in
            amount_out: Amount of token_out
            fee: Fee amount in token_in
            fee_percent: Fee percentage
            timestamp: Event timestamp
            block_number: Block number
            transaction_hash: Transaction hash
            log_index: Log index in the block
        """
        self.pool_address = pool_address.lower()
        self.token_in = token_in.address.lower() if isinstance(token_in, Token) else token_in.lower()
        self.token_out = token_out.address.lower() if isinstance(token_out, Token) else token_out.lower()
        self.amount_in = amount_in
        self.amount_out = amount_out
        self.fee = fee
        self.fee_percent = fee_percent
        self.timestamp = timestamp
        self.block_number = block_number
        self.transaction_hash = transaction_hash
        self.log_index = log_index
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert swap event to dictionary.
        
        Returns:
            Dictionary representation of the swap event
        """
        return {
            'pool_address': self.pool_address,
            'token_in': self.token_in,
            'token_out': self.token_out,
            'amount_in': self.amount_in,
            'amount_out': self.amount_out,
            'fee': self.fee,
            'fee_percent': self.fee_percent,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'block_number': self.block_number,
            'transaction_hash': self.transaction_hash,
            'log_index': self.log_index
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SwapEvent':
        """
        Create swap event from dictionary.
        
        Args:
            data: Dictionary with swap event data
            
        Returns:
            New SwapEvent instance
        """
        timestamp = pd.Timestamp(data['timestamp']) if data.get('timestamp') else None
        
        return cls(
            pool_address=data['pool_address'],
            token_in=data['token_in'],
            token_out=data['token_out'],
            amount_in=data['amount_in'],
            amount_out=data['amount_out'],
            fee=data['fee'],
            fee_percent=data['fee_percent'],
            timestamp=timestamp,
            block_number=data.get('block_number'),
            transaction_hash=data.get('transaction_hash'),
            log_index=data.get('log_index')
        )


class LiquidityEvent:
    """
    Represents a liquidity addition or removal event.
    """
    class EventType(Enum):
        """Type of liquidity event"""
        ADD = "Add"
        REMOVE = "Remove"
    
    def __init__(self,
                 pool_address: str,
                 event_type: EventType,
                 amounts: List[float],
                 timestamp: Optional[pd.Timestamp] = None,
                 block_number: Optional[int] = None,
                 transaction_hash: Optional[str] = None,
                 log_index: Optional[int] = None,
                 swap_fee_amounts: Optional[List[float]] = None):
        """
        Initialize liquidity event.
        
        Args:
            pool_address: Address of the pool
            event_type: Type of event (ADD or REMOVE)
            amounts: List of token amounts added/removed
            timestamp: Event timestamp
            block_number: Block number
            transaction_hash: Transaction hash
            log_index: Log index in the block
            swap_fee_amounts: Swap fee amounts applied to this event
        """
        self.pool_address = pool_address.lower()
        self.event_type = event_type
        self.amounts = amounts
        self.timestamp = timestamp
        self.block_number = block_number
        self.transaction_hash = transaction_hash
        self.log_index = log_index
        self.swap_fee_amounts = swap_fee_amounts
    
    def is_add(self) -> bool:
        """
        Check if this is an add liquidity event.
        
        Returns:
            True if event type is ADD
        """
        return self.event_type == self.EventType.ADD
    
    def is_remove(self) -> bool:
        """
        Check if this is a remove liquidity event.
        
        Returns:
            True if event type is REMOVE
        """
        return self.event_type == self.EventType.REMOVE
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert liquidity event to dictionary.
        
        Returns:
            Dictionary representation of the liquidity event
        """
        return {
            'pool_address': self.pool_address,
            'event_type': self.event_type.value,
            'amounts': self.amounts,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'block_number': self.block_number,
            'transaction_hash': self.transaction_hash,
            'log_index': self.log_index,
            'swap_fee_amounts': self.swap_fee_amounts
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LiquidityEvent':
        """
        Create liquidity event from dictionary.
        
        Args:
            data: Dictionary with liquidity event data
            
        Returns:
            New LiquidityEvent instance
        """
        event_type = cls.EventType(data['event_type'])
        timestamp = pd.Timestamp(data['timestamp']) if data.get('timestamp') else None
        
        return cls(
            pool_address=data['pool_address'],
            event_type=event_type,
            amounts=data['amounts'],
            timestamp=timestamp,
            block_number=data.get('block_number'),
            transaction_hash=data.get('transaction_hash'),
            log_index=data.get('log_index'),
            swap_fee_amounts=data.get('swap_fee_amounts')
        )


def mul_down(x: float, y: float) -> float:
    """
    Perform fixed-point multiplication (with truncation down).
    Mimics the mulDown function in smart contracts.
    
    Args:
        x: First operand
        y: Second operand (typically a ratio)
        
    Returns:
        Result of fixed-point multiplication
    """
    # 10^18 scaling (equivalent to 1e18 in Solidity)
    ONE = 10**18
    
    # Convert to integers and compute
    x_scaled = int(x * ONE)
    y_scaled = int(y * ONE)
    
    # Multiply and adjust scale (truncating down)
    result = (x_scaled * y_scaled) // ONE
    
    # Convert back to float
    return result / ONE 