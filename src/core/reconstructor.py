import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union, Set
from tqdm import tqdm

from .pool import Token, PoolState, SwapEvent, LiquidityEvent, mul_down
from .utils import DataIO, JSONHandler
from ..config import (
    DATA_DIR,
    RESULTS_DIR,
    RECONSTRUCTED_STATE_FILE
)


class StateReconstructor:
    """
    Reconstructs the historical state of liquidity pools based on events.
    """
    
    def __init__(self, chain_name: str = "ethereum"):
        """
        Initialize the state reconstructor.
        
        Args:
            chain_name: Chain name
        """
        self.chain_name = chain_name
        self.unbalanced_events_df = self._load_unbalanced_events()
    
    def _load_unbalanced_events(self) -> pd.DataFrame:
        """
        Load unbalanced liquidity events data.
        
        Returns:
            DataFrame with unbalanced liquidity events
        """
        file_path = os.path.join(DATA_DIR, f"{self.chain_name}_unbalanced_liquidity_events.csv")
        if os.path.exists(file_path):
            print(f"Loading unbalanced liquidity events: {file_path}")
            try:
                df = pd.read_csv(file_path)
                # Convert date columns if they exist
                date_columns = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
                for col in date_columns:
                    try:
                        df[col] = pd.to_datetime(df[col])
                    except:
                        pass
                
                print(f"Unbalanced events columns: {df.columns.tolist()}")
                return df
            except Exception as e:
                print(f"Error loading unbalanced events: {e}")
                return pd.DataFrame()
        else:
            print(f"Unbalanced liquidity events file not found: {file_path}")
            return pd.DataFrame()
    
    def parse_token_info(self, tokens_data: Union[str, List[Dict[str, Any]]]) -> List[Token]:
        """
        Parse token information from raw data.
        
        Args:
            tokens_data: Token data (string or list of dictionaries)
            
        Returns:
            List of Token objects
        """
        if isinstance(tokens_data, str):
            try:
                # Try standard JSON parsing
                tokens_data = json.loads(tokens_data)
            except json.JSONDecodeError:
                try:
                    # Try handling Python string representation
                    tokens_data = eval(tokens_data.replace("'", '"'))
                except (SyntaxError, NameError):
                    print(f"Failed to parse token information: {tokens_data}")
                    return []
        
        tokens = []
        for token_info in tokens_data:
            token = Token(
                address=token_info['address'],
                symbol=token_info.get('name', ''),  # Some APIs use 'name' instead of 'symbol'
                decimals=int(token_info['decimals'])
            )
            tokens.append(token)
        
        return tokens
    
    def get_token_map(self, snapshots_df: pd.DataFrame, pool_address: str) -> Dict[str, Token]:
        """
        Get a mapping of token addresses to Token objects for a specific pool.
        
        Args:
            snapshots_df: DataFrame with pool snapshots
            pool_address: Pool address
            
        Returns:
            Dictionary mapping token addresses to Token objects
        """
        # Find snapshots for this pool
        pool_snapshots = snapshots_df[snapshots_df['pool_address'].str.lower() == pool_address.lower()]
        
        if pool_snapshots.empty:
            print(f"No snapshots found for pool {pool_address}")
            return {}
        
        # Extract token information from first snapshot
        first_snapshot = pool_snapshots.iloc[0]
        tokens = self.parse_token_info(first_snapshot['tokens'])
        
        # Create mapping
        return {token.address: token for token in tokens}
    
    def parse_amounts(self, amounts_str: str) -> List[float]:
        """
        Parse amounts from a string representation.
        
        Args:
            amounts_str: String representation of amounts
            
        Returns:
            List of amounts as float values
        """
        try:
            # Parse JSON array
            return json.loads(amounts_str.replace("'", '"'))
        except (json.JSONDecodeError, AttributeError):
            return []
    
    def update_balances_for_swap(self, 
                               current_balances: List[float], 
                               token_in_idx: int, 
                               token_out_idx: int, 
                               amount_in: float, 
                               amount_out: float) -> List[float]:
        """
        Update balances after a swap event.
        
        Args:
            current_balances: Current balances
            token_in_idx: Index of input token
            token_out_idx: Index of output token
            amount_in: Amount of input token
            amount_out: Amount of output token
            
        Returns:
            Updated balances
        """
        new_balances = current_balances.copy()
        
        # Increase input token balance
        new_balances[token_in_idx] = float(new_balances[token_in_idx]) + float(amount_in)
        
        # Decrease output token balance
        new_balances[token_out_idx] = float(new_balances[token_out_idx]) - float(amount_out)
        
        return new_balances
    
    def parse_swap_fee_amounts(self, 
                             swap_fee_amounts_str: str, 
                             token_map: Dict[str, Token], 
                             pool_tokens: List[str]) -> List[float]:
        """
        Parse swap fee amounts and convert to decimal values.
        
        Args:
            swap_fee_amounts_str: String representation of fee amounts
            token_map: Mapping of token addresses to Token objects
            pool_tokens: List of token addresses in the pool
            
        Returns:
            List of fee amounts
        """
        try:
            # Parse JSON array
            raw_amounts = json.loads(swap_fee_amounts_str.replace("'", '"'))
            
            # Convert amounts using token decimals
            decimal_amounts = []
            for i, raw_amount in enumerate(raw_amounts):
                if i < len(pool_tokens):
                    token_address = pool_tokens[i].lower()
                    if token_address in token_map:
                        token = token_map[token_address]
                        decimal_amount = float(raw_amount) / (10 ** token.decimals)
                        decimal_amounts.append(decimal_amount)
                    else:
                        print(f"Token {token_address} not found in token map")
                        decimal_amounts.append(float(raw_amount))
                else:
                    decimal_amounts.append(float(raw_amount))
            
            return decimal_amounts
        except (json.JSONDecodeError, AttributeError) as e:
            print(f"Failed to parse swap fee amounts: {swap_fee_amounts_str}, error: {e}")
            return []
    
    def update_balances_for_liquidity_event(self, 
                                          current_balances: List[float], 
                                          amounts: List[float], 
                                          event_type: str, 
                                          tx_hash: Optional[str] = None,
                                          token_map: Optional[Dict[str, Token]] = None,
                                          pool_tokens: Optional[List[str]] = None) -> List[float]:
        """
        Update balances after a liquidity event (add or remove).
        
        Args:
            current_balances: Current balances
            amounts: Token amounts
            event_type: Event type ('Add' or 'Remove')
            tx_hash: Transaction hash
            token_map: Mapping of token addresses to Token objects
            pool_tokens: List of token addresses in the pool
            
        Returns:
            Updated balances
        """
        new_balances = current_balances.copy()
        
        # Find swap fee amounts if applicable
        swap_fee_amounts = None
        if not self.unbalanced_events_df.empty and tx_hash is not None:
            matching_events = self.unbalanced_events_df[
                (self.unbalanced_events_df['evt_tx_hash'] == tx_hash) &
                (self.unbalanced_events_df.get('event_type', '') == event_type)  # Add check for event_type if column exists
            ]
            
            if not matching_events.empty:
                if len(matching_events) > 1:
                    print(f"Multiple matching unbalanced events found: {tx_hash}")
                
                # Check if swapFeeAmountsRaw column exists
                if 'swapFeeAmountsRaw' in matching_events.columns:
                    # Get fee amounts from first matching event
                    swap_fee_amounts_raw = matching_events.iloc[0]['swapFeeAmountsRaw']
                    
                    # Parse fee amounts
                    if token_map and pool_tokens:
                        swap_fee_amounts = self.parse_swap_fee_amounts(
                            swap_fee_amounts_raw,
                            token_map,
                            pool_tokens
                        )
        
        # Update balances for each token
        for i, amount in enumerate(amounts):
            if i < len(new_balances):
                # Base amount
                amount_value = float(amount)
                
                # Apply swap fee if available
                if swap_fee_amounts and i < len(swap_fee_amounts):
                    # Apply 50% of fee (fixed-point multiplication)
                    fee_amount = mul_down(swap_fee_amounts[i], 0.5)
                    
                    if fee_amount > 0:
                        if event_type == 'Add':
                            # For liquidity addition, exclude fee
                            amount_value -= fee_amount
                        elif event_type == 'Remove':
                            # For liquidity removal, add fee
                            amount_value += fee_amount
                
                # Update balance
                if event_type == 'Add':
                    # Add token: increase balance
                    new_balances[i] = float(new_balances[i]) + amount_value
                elif event_type == 'Remove':
                    # Remove token: decrease balance
                    new_balances[i] = float(new_balances[i]) - amount_value
        
        return new_balances
    
    def calculate_weights(self, balances: List[float]) -> List[float]:
        """
        Calculate weights based on balances.
        
        Args:
            balances: Token balances
            
        Returns:
            List of weights
        """
        total_balance = sum(float(balance) for balance in balances)
        
        if total_balance == 0:
            return [0] * len(balances)
        
        return [float(balance) / total_balance for balance in balances]
    
    def find_token_index(self, pool_tokens: List[str], token_address: str) -> int:
        """
        Find the index of a token in the pool tokens list.
        
        Args:
            pool_tokens: List of token addresses
            token_address: Token address to find
            
        Returns:
            Token index, or -1 if not found
        """
        for i, token in enumerate(pool_tokens):
            if token.lower() == token_address.lower():
                return i
        return -1
    
    def adjust_snapshot_times(self, snapshots_df: pd.DataFrame) -> pd.DataFrame:
        """
        Adjust snapshot times to the end of each day (23:59:59).
        
        Args:
            snapshots_df: DataFrame with pool snapshots
            
        Returns:
            DataFrame with adjusted timestamps
        """
        adjusted_df = snapshots_df.copy()
        
        # Adjust each snapshot's time to 23:59:59 of the same day
        adjusted_df['timestamp'] = adjusted_df['timestamp'].apply(
            lambda x: pd.Timestamp(x.date()) + pd.Timedelta(hours=23, minutes=59, seconds=59)
        )
        
        return adjusted_df
    
    def run(self) -> pd.DataFrame:
        """
        Reconstruct pool states based on events.
        
        Returns:
            DataFrame with reconstructed states
        """
        print("Reconstructing pool states...")
        
        # Check for cached results
        cache_file = os.path.join(RESULTS_DIR, f"{self.chain_name}_{RECONSTRUCTED_STATE_FILE}")
        
        if os.path.exists(cache_file):
            print(f"Loading cached reconstruction data: {cache_file}")
            return pd.read_csv(cache_file, parse_dates=['timestamp'])
        
        # Load input data files
        snapshots_file = os.path.join(DATA_DIR, f"{self.chain_name}_pool_snapshots.csv")
        add_removes_file = os.path.join(DATA_DIR, f"{self.chain_name}_add_removes.csv")
        swaps_file = os.path.join(DATA_DIR, f"{self.chain_name}_swaps.csv")
        
        if not os.path.exists(snapshots_file):
            print(f"Snapshot file not found: {snapshots_file}")
            return pd.DataFrame()
        
        # Load data
        snapshots_df = pd.read_csv(snapshots_file, parse_dates=['timestamp'])
        
        add_removes_df = pd.DataFrame()
        if os.path.exists(add_removes_file):
            add_removes_df = pd.read_csv(add_removes_file, parse_dates=['blockTimestamp'])
        else:
            print(f"AddRemove event file not found: {add_removes_file}")
        
        swaps_df = pd.DataFrame()
        if os.path.exists(swaps_file):
            swaps_df = pd.read_csv(swaps_file, parse_dates=['blockTimestamp'])
        else:
            print(f"Swap event file not found: {swaps_file}")
        
        # Return empty DataFrame if both event types are missing
        if add_removes_df.empty and swaps_df.empty:
            print("No event data available for reconstruction")
            return pd.DataFrame()
        
        # Adjust snapshot times
        snapshots_df = self.adjust_snapshot_times(snapshots_df)
        
        # Get unique pool addresses
        pool_addresses = snapshots_df['pool_address'].unique()
        
        # Process each pool
        all_reconstructed_states = []
        
        for pool_address in tqdm(pool_addresses, desc="Processing pools"):
            print(f"\nProcessing pool {pool_address}...")
            
            # Filter snapshots for this pool
            pool_snapshots = snapshots_df[snapshots_df['pool_address'] == pool_address].sort_values('timestamp')
            
            if pool_snapshots.empty:
                print(f"No snapshots found for pool {pool_address}, skipping")
                continue
            
            # Get the last snapshot
            last_snapshot = pool_snapshots.iloc[-1]
            last_snapshot_time = last_snapshot['timestamp']
            last_snapshot_balances = self.parse_amounts(last_snapshot['balances'])
            
            # Get token addresses for this pool
            first_snapshot = pool_snapshots.iloc[0]
            tokens_data = first_snapshot['tokens']
            tokens = self.parse_token_info(tokens_data)
            pool_tokens = [token.address for token in tokens]
            
            if not pool_tokens:
                print(f"No token information available for pool {pool_address}, skipping")
                continue
            
            # Create token map
            token_map = {token.address: token for token in tokens}
            
            # Initialize balances
            current_balances = [0.0] * len(pool_tokens)
            
            # Filter events for this pool
            pool_add_removes = add_removes_df[add_removes_df['pool_address'] == pool_address].copy()
            pool_swaps = swaps_df[swaps_df['pool_address'] == pool_address].copy()
            
            # Combine events and sort by timestamp, block number, and log index
            all_events = []
            
            # Add AddRemove events
            for _, event in pool_add_removes.iterrows():
                all_events.append({
                    'timestamp': event['blockTimestamp'],
                    'block_number': event['blockNumber'],
                    'log_index': event['logIndex'],
                    'event_type': event['type'],
                    'data': event
                })
            
            # Add Swap events
            for _, event in pool_swaps.iterrows():
                all_events.append({
                    'timestamp': event['blockTimestamp'],
                    'block_number': event['blockNumber'],
                    'log_index': event['logIndex'],
                    'event_type': 'Swap',
                    'data': event
                })
            
            # Sort events
            all_events.sort(key=lambda x: (
                x['timestamp'], 
                x['block_number'], 
                x['log_index']
            ))
            
            # Process each event
            for event_data in all_events:
                event = event_data['data']
                event_type = event_data['event_type']
                
                # Save pre-event state
                pre_event_balances = current_balances.copy()
                pre_event_weights = self.calculate_weights(pre_event_balances)
                
                # Update balances based on event type
                if event_type == 'Swap':
                    # Handle swap event
                    token_in = event['tokenIn']
                    token_out = event['tokenOut']
                    amount_in = float(event['tokenAmountIn'])
                    amount_out = float(event['tokenAmountOut'])
                    fee = float(event['swapFeeAmount'])
                    base_fee = float(event['swapFeeBaseAmount'])
                    delta_fee = float(event['swapFeeDeltaAmount'])
                    fee_percent = float(event['swapFeePercentage'])
                    
                    # Find token indices
                    token_in_idx = self.find_token_index(pool_tokens, token_in)
                    token_out_idx = self.find_token_index(pool_tokens, token_out)
                    
                    if token_in_idx >= 0 and token_out_idx >= 0:
                        # Apply 50% fee using fixed-point multiplication
                        half_fee = mul_down(fee, 0.5)
                        
                        # Update balances
                        current_balances = self.update_balances_for_swap(
                            current_balances, 
                            token_in_idx, 
                            token_out_idx,
                            amount_in - half_fee,  # Subtract half of fee
                            amount_out
                        )
                    else:
                        print(f"Token index lookup failed: {token_in} {token_out}")
                        print(f"Pool tokens: {pool_tokens}")
                        continue
                
                elif event_type in ['Add', 'Remove']:
                    # Handle liquidity event
                    amounts = self.parse_amounts(event['amounts'])
                    tx_hash = event['transactionHash']
                    
                    # Update balances
                    current_balances = self.update_balances_for_liquidity_event(
                        current_balances,
                        amounts,
                        event_type,
                        tx_hash,
                        token_map,
                        pool_tokens
                    )
                
                # For swap events, create a reconstructed state record
                if event_type == 'Swap':
                    # Calculate post-event weights
                    post_event_balances = current_balances.copy()
                    post_event_weights = self.calculate_weights(post_event_balances)
                    
                    # Create record
                    reconstructed_state = {
                        'chain': self.chain_name,
                        'pool_address': pool_address,
                        'timestamp': event_data['timestamp'],
                        'block_number': event_data['block_number'],
                        'log_index': event_data['log_index'],
                        'token_in': token_in_idx,
                        'token_out': token_out_idx,
                        'amount_in': amount_in,
                        'amount_out': amount_out,
                        'fee': fee,
                        'fee_percent': fee_percent,
                        'base_fee': base_fee,
                        'delta_fee': delta_fee,
                        'pre_event_balances': json.dumps(pre_event_balances),
                        'post_event_balances': json.dumps(post_event_balances),
                        'pre_event_weights': json.dumps(pre_event_weights),
                        'post_event_weights': json.dumps(post_event_weights),
                    }
                    
                    all_reconstructed_states.append(reconstructed_state)
            
            # Compare reconstructed final state with last snapshot
            if all_events:
                # Calculate difference between reconstructed state and snapshot
                final_balances = current_balances
                
                balance_diff = []
                for i, (final, snapshot) in enumerate(zip(final_balances, last_snapshot_balances)):
                    diff = float(snapshot) - float(final)
                    balance_diff.append(diff)
                
                # Skip detailed comparison if differences are small
                if sum(abs(diff) for diff in balance_diff) < 1:
                    continue

                print(f"Reconstructed final balances: {final_balances}")
                print(f"Last snapshot balances: {last_snapshot_balances}")
                print(f"Balance differences: {balance_diff}")
                
                # Calculate relative errors
                relative_errors = []
                for final, snapshot in zip(final_balances, last_snapshot_balances):
                    if float(snapshot) != 0:
                        error = abs(float(final) - float(snapshot)) / float(snapshot)
                        relative_errors.append(error)
                
                if relative_errors:
                    avg_error = sum(relative_errors) / len(relative_errors)
                    print(f"Average relative error: {avg_error:.4f} ({avg_error*100:.2f}%)")
            else:
                print(f"No events found for pool {pool_address}")
        
        # Return empty DataFrame if no states were reconstructed
        if not all_reconstructed_states:
            print("No pool states were reconstructed")
            return pd.DataFrame()
        
        # Create DataFrame
        result_df = pd.DataFrame(all_reconstructed_states)
        
        # Save results
        DataIO.save_dataframe(result_df, f"{self.chain_name}_{RECONSTRUCTED_STATE_FILE}", RESULTS_DIR)
        
        return result_df
    
    def analyze_weight_changes(self, reconstructed_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze weight changes in the reconstructed pool states.
        
        Args:
            reconstructed_df: DataFrame with reconstructed states
            
        Returns:
            Dictionary with analysis results
        """
        print("Analyzing weight changes...")
        
        # Filter for swap events
        swap_events = reconstructed_df[reconstructed_df['event_type'] == 'SWAP'].copy()
        
        if swap_events.empty:
            print("No swap events found")
            return {}
        
        # Calculate weight changes for each swap event
        weight_changes = []
        
        for _, event in swap_events.iterrows():
            pre_weights = json.loads(event['pre_event_weights'])
            post_weights = json.loads(event['post_event_weights'])
            
            # Calculate maximum weight change
            max_weight_change = max(abs(post - pre) for pre, post in zip(pre_weights, post_weights))
            
            weight_changes.append({
                'pool_address': event['pool_address'],
                'timestamp': event['timestamp'],
                'block_number': event['block_number'],
                'log_index': event['log_index'],
                'pre_weights': pre_weights,
                'post_weights': post_weights,
                'max_weight_change': max_weight_change
            })
        
        # Create DataFrame
        weight_changes_df = pd.DataFrame(weight_changes)
        
        # Calculate statistics
        stats = {
            'mean_max_weight_change': weight_changes_df['max_weight_change'].mean(),
            'median_max_weight_change': weight_changes_df['max_weight_change'].median(),
            'max_weight_change': weight_changes_df['max_weight_change'].max(),
            'min_weight_change': weight_changes_df['max_weight_change'].min(),
            'std_weight_change': weight_changes_df['max_weight_change'].std()
        }
        
        return {
            'stats': stats,
            'weight_changes': weight_changes_df
        } 