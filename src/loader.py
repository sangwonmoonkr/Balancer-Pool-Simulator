import os
import json
import requests
import pandas as pd
import time
from typing import Dict, List, Optional, Union, Any, Tuple
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

from .core.utils import DataIO, JSONHandler
from .config import (
    DATA_DIR,
    RESULTS_DIR,
    CHAINS,
    ITEMS_PER_PAGE,
    POOL_ADDRESSES_FILE,
    POOL_SNAPSHOTS_FILE,
    SWAPS_FILE,
    ADD_REMOVES_FILE,
    UNBALANCE_LIQUIDITY_FILE,
    CREATE_DUNE_QUERY
)

import dune_client.client as dune
from dune_client.types import QueryParameter, Address
from dune_client.query import QueryBase


class APIClient:
    """
    Client for interacting with various APIs including GraphQL and Dune.
    """
    
    def __init__(self, graphql_api_key: str, dune_api_key: str):
        """
        Initialize API client.
        
        Args:
            graphql_api_key: API key for GraphQL
            dune_api_key: API key for Dune
        """
        self.graphql_api_key = graphql_api_key
        self.dune_api_key = dune_api_key
        self.client = dune.DuneClient(self.dune_api_key)
    
    def execute_graphql_query(self, url: str, query: str, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a GraphQL query.
        
        Args:
            url: GraphQL API URL
            query: GraphQL query string
            variables: Query variables
            
        Returns:
            Query result
            
        Raises:
            requests.exceptions.RequestException: If API request fails
        """
        if variables is None:
            variables = {}
        
        # Insert API key into URL
        api_url = url.replace("[api-key]", self.graphql_api_key)
        
        # Prepare request data
        request_data = {
            "query": query,
            "variables": variables
        }
        
        # Execute API request
        try:
            response = requests.post(api_url, json=request_data)
            response.raise_for_status()  # Raise exception for HTTP errors
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            return {}

    def execute_dune_query(self, query_sql: str, params: List[QueryParameter], query_type: str) -> pd.DataFrame:
        if CREATE_DUNE_QUERY:
            query = self.client.create_query(
                name=f"Temp Unbalanced Liquidity {query_type.capitalize()} Query",
                query_sql=query_sql,
                params=params,
                is_private=True
            )
            query_id = query.base.query_id
        else:
            env_var = f'DUNE_{query_type.upper()}_QUERY_ID'
            query_id = int(os.getenv(env_var))
            if not query_id:
                raise ValueError(f"{env_var} not set in environment")
        
        query_base = QueryBase(
            query_id=query_id,
            params=params
        )
        return self.client.run_query_dataframe(query_base)

class DataLoader:
    """
    Loads and processes data from various sources.
    """
    
    def __init__(self, graphql_api_key: str, dune_api_key: str):
        """
        Initialize data loader.
        
        Args:
            graphql_api_key: API key for GraphQL APIs
            dune_api_key: API key for Dune
        """
        self.api_client = APIClient(graphql_api_key, dune_api_key)
        
        # Ensure directories exist
        DataIO.ensure_directory_exists(DATA_DIR)
        DataIO.ensure_directory_exists(RESULTS_DIR)

    @staticmethod
    def get_available_chains() -> List[str]:
        """
        Get available chains based on data files.
        
        Returns:
            List of available chain names
        """
        chains = CHAINS.keys()
        return list(chains)
    
    def get_pool_addresses(self, chain_name: str = "ethereum") -> List[str]:
        """
        Get StableSurge pool addresses from a specific chain.
        
        Args:
            chain_name: Chain name (ethereum, arbitrum, etc.)
            
        Returns:
            List of pool addresses
        """
        print(f"Fetching StableSurge pool addresses from {chain_name} chain...")
        
        # Cached file path
        cache_file = os.path.join(DATA_DIR, f"{chain_name}_{POOL_ADDRESSES_FILE}")
        
        # Load cached file if exists
        if os.path.exists(cache_file):
            print(f"Loading cached pool addresses: {cache_file}")
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                
                # Extract addresses from dictionary
                if isinstance(data, dict) and 'addresses' in data:
                    return data['addresses']
                return data
            except Exception as e:
                print(f"Error loading cached pool addresses: {e}")
                # Continue to fetch new data if loading fails
        
        # Get chain API URL
        if chain_name not in CHAINS:
            print(f"Unsupported chain: {chain_name}")
            return []
        
        api_url = CHAINS[chain_name]["pools_api"]
        
        # GraphQL query
        query = """
        query StableSurgePools {
          pools(where: {stableSurgeParams_not: null}) {
            address
          }
        }
        """
        
        # Execute query
        result = self.api_client.execute_graphql_query(api_url, query)
        
        if not result or 'data' not in result or 'pools' not in result['data']:
            print(f"Failed to get pool addresses from {chain_name} chain")
            return []
        
        # Extract pool addresses
        pools = result['data']['pools']
        pool_addresses = [pool['address'] for pool in pools]
        
        print(f"Found {len(pool_addresses)} StableSurge pools on {chain_name} chain")
        
        # Save to file
        pool_data = {
            "addresses": pool_addresses
        }
        
        # Ensure directory exists
        DataIO.ensure_directory_exists(DATA_DIR)
        
        # Save directly to file
        try:
            with open(cache_file, 'w') as f:
                json.dump(pool_data, f, indent=2)
            print(f"Pool addresses saved to: {cache_file}")
        except Exception as e:
            print(f"Error saving pool addresses: {e}")
        
        return pool_addresses
    
    def get_pool_snapshots(self, chain_name: str, pool_addresses: List[str]) -> pd.DataFrame:
        """
        Get pool snapshot data from a specific chain.
        
        Args:
            chain_name: Chain name
            pool_addresses: List of pool addresses
            
        Returns:
            DataFrame with pool snapshot data
        """
        print(f"Extracting snapshot data for {len(pool_addresses)} pools on {chain_name} chain...")
        
        # Cached file path
        cache_file = os.path.join(DATA_DIR, f"{chain_name}_{POOL_SNAPSHOTS_FILE}")
        
        # Load cached file if exists
        if os.path.exists(cache_file):
            print(f"Loading cached snapshot data: {cache_file}")
            df = pd.read_csv(cache_file, parse_dates=['timestamp'])
            
            # Convert tokens field from string to JSON
            if 'tokens' in df.columns and isinstance(df['tokens'].iloc[0], str):
                df['tokens'] = df['tokens'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
            
            return df
        
        # Get chain API URL
        if chain_name not in CHAINS:
            print(f"Unsupported chain: {chain_name}")
            return pd.DataFrame()
        
        api_url = CHAINS[chain_name]["events_api"]
        
        # List to store all snapshot data
        all_snapshots = []
        
        # Create batches of addresses (due to API request size limits)
        max_addresses_per_query = 10
        address_batches = [pool_addresses[i:i+max_addresses_per_query] 
                          for i in range(0, len(pool_addresses), max_addresses_per_query)]
        
        # Process each address batch
        for batch_idx, address_batch in enumerate(address_batches):
            print(f"Processing address batch {batch_idx+1}/{len(address_batches)}...")
            
            # Convert address list to string
            addresses_str = '", "'.join(address_batch)
            addresses_str = f'["{addresses_str}"]'
            
            # Pagination variables
            skip = 0
            has_more_data = True
            
            while has_more_data:
                # GraphQL query
                query = f"""
                query PoolSnapshots {{
                  poolSnapshots(
                    first: {ITEMS_PER_PAGE}
                    skip: {skip}
                    where: {{pool_: {{address_in: {addresses_str}}}}}
                    orderBy: timestamp
                    orderDirection: asc
                  ) {{
                    pool {{
                      address
                      tokens {{
                        address
                        name
                        decimals
                      }}
                    }}
                    balances
                    timestamp
                  }}
                }}
                """
                
                # Execute query
                result = self.api_client.execute_graphql_query(api_url, query)
                
                if not result or 'data' not in result or 'poolSnapshots' not in result['data']:
                    print("Failed to get snapshot data")
                    break
                
                snapshots = result['data']['poolSnapshots']
                
                if not snapshots:
                    has_more_data = False
                    continue
                
                # Process and store data
                for snapshot in snapshots:
                    row = {
                        'pool_address': snapshot['pool']['address'],
                        'tokens': json.dumps(snapshot['pool']['tokens']),  # Store token info as JSON string
                        'balances': snapshot['balances'],
                        'timestamp': int(snapshot['timestamp'])
                    }
                    all_snapshots.append(row)
                
                # Move to next page
                if len(snapshots) < ITEMS_PER_PAGE:
                    has_more_data = False
                else:
                    skip += ITEMS_PER_PAGE
                    print(f"Fetching next {ITEMS_PER_PAGE} snapshot records...")
        
        # Return empty DataFrame if no data
        if not all_snapshots:
            print("No snapshot data found")
            return pd.DataFrame()
        
        # Create DataFrame and convert timestamp
        df = pd.DataFrame(all_snapshots)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # Save data
        DataIO.save_dataframe(df, f"{chain_name}_{POOL_SNAPSHOTS_FILE}", DATA_DIR)
        
        # Convert tokens field to JSON objects
        df['tokens'] = df['tokens'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
        
        return df
    
    def get_add_removes(self, chain_name: str, pool_addresses: List[str]) -> pd.DataFrame:
        """
        Get AddRemove event data from a specific chain.
        
        Args:
            chain_name: Chain name
            pool_addresses: List of pool addresses
            
        Returns:
            DataFrame with AddRemove event data
        """
        print(f"Extracting AddRemove event data from {chain_name} chain...")
        
        # Cached file path
        cache_file = os.path.join(DATA_DIR, f"{chain_name}_{ADD_REMOVES_FILE}")
        
        # Load cached file if exists
        if os.path.exists(cache_file):
            print(f"Loading cached AddRemove event data: {cache_file}")
            return pd.read_csv(cache_file, parse_dates=['blockTimestamp'])
        
        # Get chain API URL
        if chain_name not in CHAINS:
            print(f"Unsupported chain: {chain_name}")
            return pd.DataFrame()
        
        api_url = CHAINS[chain_name]["events_api"]
        
        # List to store all event data
        all_events = []
        
        # Create batches of addresses
        max_addresses_per_query = 10
        address_batches = [pool_addresses[i:i+max_addresses_per_query] 
                          for i in range(0, len(pool_addresses), max_addresses_per_query)]
        
        # Process each address batch
        for batch_idx, address_batch in enumerate(address_batches):
            print(f"Processing address batch {batch_idx+1}/{len(address_batches)}...")
            
            # Convert address list to string
            addresses_str = '", "'.join(address_batch)
            addresses_str = f'["{addresses_str}"]'
            
            # Pagination variables
            skip = 0
            has_more_data = True
            
            while has_more_data:
                # GraphQL query
                query = f"""
                query AddRemoveEvents {{
                  addRemoves(
                    first: {ITEMS_PER_PAGE}
                    skip: {skip}
                    where: {{pool_: {{address_in: {addresses_str}}}}}
                    orderBy: blockTimestamp
                    orderDirection: asc
                  ) {{
                    blockNumber
                    blockTimestamp
                    amounts
                    type
                    logIndex
                    transactionHash
                    pool {{
                      address
                    }}
                  }}
                }}
                """
                
                # Execute query
                result = self.api_client.execute_graphql_query(api_url, query)
                
                if not result or 'data' not in result or 'addRemoves' not in result['data']:
                    print("Failed to get AddRemove event data")
                    break
                
                events = result['data']['addRemoves']
                
                if not events:
                    has_more_data = False
                    continue
                
                # Process and store data
                for event in events:
                    row = {
                        'pool_address': event['pool']['address'],
                        'blockNumber': int(event['blockNumber']),
                        'blockTimestamp': int(event['blockTimestamp']),
                        'amounts': event['amounts'],
                        'type': event['type'],
                        'logIndex': int(event['logIndex']),
                        'transactionHash': event['transactionHash']
                    }
                    all_events.append(row)
                
                # Move to next page
                if len(events) < ITEMS_PER_PAGE:
                    has_more_data = False
                else:
                    skip += ITEMS_PER_PAGE
                    print(f"Fetching next {ITEMS_PER_PAGE} AddRemove event records...")
        
        # Return empty DataFrame if no data
        if not all_events:
            print("No AddRemove event data found")
            return pd.DataFrame()
        
        # Create DataFrame and convert timestamp
        df = pd.DataFrame(all_events)
        df['blockTimestamp'] = pd.to_datetime(df['blockTimestamp'], unit='s')
        
        # Save data
        DataIO.save_dataframe(df, f"{chain_name}_{ADD_REMOVES_FILE}", DATA_DIR)
        
        return df
    
    def get_swaps(self, chain_name: str, pool_addresses: List[str]) -> pd.DataFrame:
        """
        Get Swap event data from a specific chain.
        
        Args:
            chain_name: Chain name
            pool_addresses: List of pool addresses
            
        Returns:
            DataFrame with Swap event data
        """
        print(f"Extracting Swap event data from {chain_name} chain...")
        
        # Cached file path
        cache_file = os.path.join(DATA_DIR, f"{chain_name}_{SWAPS_FILE}")
        
        # Load cached file if exists
        if os.path.exists(cache_file):
            print(f"Loading cached Swap event data: {cache_file}")
            return pd.read_csv(cache_file, parse_dates=['blockTimestamp'])
        
        # Get chain API URL
        if chain_name not in CHAINS:
            print(f"Unsupported chain: {chain_name}")
            return pd.DataFrame()
        
        api_url = CHAINS[chain_name]["events_api"]
        
        # List to store all swap data
        all_swaps = []
        
        # Create batches of addresses
        max_addresses_per_query = 10
        address_batches = [pool_addresses[i:i+max_addresses_per_query] 
                          for i in range(0, len(pool_addresses), max_addresses_per_query)]
        
        # Process each address batch
        for batch_idx, address_batch in enumerate(address_batches):
            print(f"Processing address batch {batch_idx+1}/{len(address_batches)}...")
            
            # Convert address list to string
            addresses_str = '", "'.join(address_batch)
            addresses_str = f'["{addresses_str}"]'
            
            # Pagination variables
            skip = 0
            has_more_data = True
            
            while has_more_data:
                # GraphQL query
                query = f"""
                query SwapEvents {{
                  swaps(
                    first: {ITEMS_PER_PAGE}
                    skip: {skip}
                    where: {{pool_in: {addresses_str}}}
                    orderBy: blockTimestamp
                    orderDirection: asc
                  ) {{
                    blockTimestamp
                    blockNumber
                    hasDynamicSwapFee
                    swapFeePercentage
                    swapFeeAmount
                    swapFeeBaseAmount
                    swapFeeDeltaAmount
                    pool
                    tokenAmountIn
                    tokenAmountOut
                    tokenIn
                    tokenInSymbol
                    tokenOut
                    tokenOutSymbol
                    logIndex
                    transactionHash
                  }}
                }}
                """
                
                # Execute query
                result = self.api_client.execute_graphql_query(api_url, query)
                
                if not result or 'data' not in result or 'swaps' not in result['data']:
                    print("Failed to get Swap event data")
                    break
                
                swaps = result['data']['swaps']
                
                if not swaps:
                    has_more_data = False
                    continue
                
                # Process and store data
                for swap in swaps:
                    row = {
                        'pool_address': swap['pool'],
                        'blockNumber': int(swap['blockNumber']),
                        'blockTimestamp': int(swap['blockTimestamp']),
                        'hasDynamicSwapFee': swap['hasDynamicSwapFee'],
                        'swapFeePercentage': swap['swapFeePercentage'],
                        'swapFeeAmount': swap['swapFeeAmount'],
                        'swapFeeBaseAmount': swap['swapFeeBaseAmount'],
                        'swapFeeDeltaAmount': swap['swapFeeDeltaAmount'],
                        'tokenAmountIn': swap['tokenAmountIn'],
                        'tokenAmountOut': swap['tokenAmountOut'],
                        'tokenIn': swap['tokenIn'],
                        'tokenInSymbol': swap['tokenInSymbol'],
                        'tokenOut': swap['tokenOut'],
                        'tokenOutSymbol': swap['tokenOutSymbol'],
                        'logIndex': int(swap['logIndex']),
                        'transactionHash': swap['transactionHash']
                    }
                    all_swaps.append(row)
                
                # Move to next page
                if len(swaps) < ITEMS_PER_PAGE:
                    has_more_data = False
                else:
                    skip += ITEMS_PER_PAGE
                    print(f"Fetching next {ITEMS_PER_PAGE} Swap event records...")
        
        # Return empty DataFrame if no data
        if not all_swaps:
            print("No Swap event data found")
            return pd.DataFrame()
        
        # Create DataFrame and convert timestamp
        df = pd.DataFrame(all_swaps)
        df['blockTimestamp'] = pd.to_datetime(df['blockTimestamp'], unit='s')
        
        # Save data
        DataIO.save_dataframe(df, f"{chain_name}_{SWAPS_FILE}", DATA_DIR)
        
        return df

    def extract_chain_data(self, chain_name: str = "ethereum") -> Dict[str, Any]:
        """
        Extract all data for a specific chain.
        
        Args:
            chain_name: Chain name
            
        Returns:
            Dictionary containing all extracted data
        """
        # 1. Get StableSurge pool addresses
        pool_addresses = self.get_pool_addresses(chain_name)
        
        if not pool_addresses or len(pool_addresses) == 0:
            print(f"No pool addresses found for {chain_name} chain")
            return {}
        
        # Ensure pool_addresses is a list
        if not isinstance(pool_addresses, list):
            print(f"Pool addresses is not a list. Current type: {type(pool_addresses)}")
            try:
                # Try to extract addresses from dictionary
                if isinstance(pool_addresses, dict) and 'addresses' in pool_addresses:
                    pool_addresses = pool_addresses['addresses']
                else:
                    # Try to convert to list
                    pool_addresses = list(pool_addresses)
            except Exception as e:
                print(f"Error converting pool addresses format: {e}")
                return {}
        
        print(f"Processing {len(pool_addresses)} pool addresses")
        
        # 2. Extract pool snapshot data
        snapshots_df = self.get_pool_snapshots(chain_name, pool_addresses)
        
        # 3. Extract AddRemove event data
        add_removes_df = self.get_add_removes(chain_name, pool_addresses)
        
        # 4. Extract Swap event data
        swaps_df = self.get_swaps(chain_name, pool_addresses)
        
        # 5. Extract Unbalanced Liquidity events
        unbalanced_df = self.get_unbalance_liquidity_swaps(chain_name)
        
        return {
            'pool_addresses': pool_addresses,
            'snapshots': snapshots_df,
            'add_removes': add_removes_df,
            'swaps': swaps_df,
            'unbalanced_liquidity': unbalanced_df
        }

    def get_unbalance_liquidity_swaps(self, chain: str) -> pd.DataFrame:
        """
        Load unbalanced liquidity added events using Dune Client for multiple pools.
        
        Args:
            chain: Chain name
            
        Returns:
            DataFrame with unbalanced liquidity events
        """
        print(f"Fetching unbalanced liquidity events for all pools on {chain} chain...")
        # Get all pool addresses for the chain
        pool_addresses = self.get_pool_addresses(chain)
        
        if not pool_addresses:
            print(f"No pool addresses found for {chain}")
            return pd.DataFrame()

        print(f"Fetching unbalanced liquidity events for {chain} chain and pools {pool_addresses}...")
        
        # Cached file path (use a hash or something for multiple addresses, but for simplicity, use chain name)
        cache_file = os.path.join(DATA_DIR, f"{chain}_{UNBALANCE_LIQUIDITY_FILE}")
        
        # Load cached file if exists
        if os.path.exists(cache_file):
            print(f"Loading cached unbalanced liquidity data: {cache_file}")
            return pd.read_csv(cache_file)
        
        # Define queries for add and remove
        add_query = """
        WITH pool_addresses AS (
            SELECT from_hex(lower(trim(addr_txt))) AS address
            FROM unnest(
                    split(
                        regexp_replace('{{pool_addresses}}', '\\s+', ''), 
                        ','
                    )
                ) AS t(addr_txt)
            WHERE length(trim(addr_txt)) > 0
        )
        SELECT
            evt_tx_hash,
            chain,
            pool,
            swapFeeAmountsRaw
        FROM balancer_v3_multichain.vault_evt_liquidityadded
        WHERE chain = '{{chain}}'                             
        AND pool IN (SELECT address FROM pool_addresses)
        AND swapFeeAmountsRaw IS NOT NULL
        AND cardinality(filter(swapFeeAmountsRaw, x -> x > 0)) > 0
        ORDER BY evt_block_time DESC;
        """
        
        remove_query = """
        WITH pool_addresses AS (
            SELECT from_hex(lower(trim(addr_txt))) AS address
            FROM unnest(
                    split(
                        regexp_replace('{{pool_addresses}}', '\\s+', ''), 
                        ','
                    )
                ) AS t(addr_txt)
            WHERE length(trim(addr_txt)) > 0
        )
        SELECT
            evt_tx_hash,
            chain,
            pool,
            swapFeeAmountsRaw
        FROM balancer_v3_multichain.vault_evt_liquidityremoved
        WHERE chain = '{{chain}}'                             
        AND pool IN (SELECT address FROM pool_addresses)
        AND swapFeeAmountsRaw IS NOT NULL
        AND cardinality(filter(swapFeeAmountsRaw, x -> x > 0)) > 0
        ORDER BY evt_block_time DESC;
        """
        
        pool_addresses_str = ', '.join(f"{addr}" for addr in pool_addresses)

        print("pool_addresses_str", pool_addresses_str)
        
        # Prepare parameters
        dune_params = [
            QueryParameter.text_type(name="chain", value=chain),
            QueryParameter.enum_type(name="pool_addresses", value=pool_addresses_str)
        ]
        
        # Execute add query
        add_df = self.api_client.execute_dune_query(add_query, dune_params, 'add')
        if not add_df.empty:
            add_df['event_type'] = 'Add'
        
        # Execute remove query
        remove_df = self.api_client.execute_dune_query(remove_query, dune_params, 'remove')
        if not remove_df.empty:
            remove_df['event_type'] = 'Remove'
        
        # Combine
        df = pd.concat([add_df, remove_df], ignore_index=True)
        
        if not df.empty:
            df.to_csv(cache_file, index=False)
            print(f"Unbalanced liquidity (add/remove) data saved to: {cache_file}")
            return df
        else:
            print("No unbalanced liquidity data found")
            return pd.DataFrame()