import os
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from tqdm import tqdm
from dune_client.client import DuneClient

from .config import (
    DATA_DIR,
    RESULTS_DIR,
    GRAPH_API_KEY,
    CHAINS,
    ITEMS_PER_PAGE,
    POOL_ADDRESSES_FILE,
    POOL_SNAPSHOTS_FILE,
    SWAPS_FILE,
    ADD_REMOVES_FILE,
    DUNE_API_KEY
)
from .utils import ensure_directory_exists, save_dataframe, save_json, load_json

dune = DuneClient(DUNE_API_KEY)

def save_all_stable_surge_pools():
    chain_names = CHAINS.keys()
    df = pd.DataFrame(columns=["chain", "pool"])
    for chain_name in chain_names:
        pool_addresses = get_stable_pool_addresses(chain_name)
        print(f"{chain_name} 체인에서 {len(pool_addresses)}개의 StableSurge 풀을 찾았습니다.")
        for pool_address in pool_addresses:
            df = pd.concat([df, pd.DataFrame([{"chain": chain_name, "pool": pool_address}])], ignore_index=True)
    save_dataframe(df, f"stable_surge_pools.csv")

def get_unbalanced_liquidity_events():
    query_result = dune.get_latest_result(5271197)
    df = pd.DataFrame(query_result.result.rows)
    save_dataframe(df, f"unbalanced_liquidity_events.csv")


def run_graph_query(url, query, variables=None):
    """
    The Graph API에 GraphQL 쿼리를 실행합니다.
    
    Args:
        url (str): The Graph API URL
        query (str): GraphQL 쿼리 문자열
        variables (dict, optional): 쿼리 변수
        
    Returns:
        dict: 쿼리 결과
    """
    if variables is None:
        variables = {}
    
    # API 키 설정
    api_url = url.replace("[api-key]", GRAPH_API_KEY)
    
    # 요청 데이터 준비
    request_data = {
        "query": query,
        "variables": variables
    }
    
    # API 요청
    try:
        response = requests.post(api_url, json=request_data)
        response.raise_for_status()  # HTTP 오류 발생 시 예외 발생
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"API 요청 실패: {e}")
        return {}

def get_stable_pool_addresses(chain_name="ethereum"):
    """
    특정 체인에서 StableSurge 풀 주소를 추출합니다.
    
    Args:
        chain_name (str): 체인 이름 (ethereum, arbitrum 등)
        
    Returns:
        list: 풀 주소 목록
    """
    print(f"{chain_name} 체인에서 StableSurge 풀 주소를 검색 중...")
    
    # 캐시된 파일 경로
    cache_file = os.path.join(DATA_DIR, f"{chain_name}_{POOL_ADDRESSES_FILE}")
    
    # 캐시된 파일이 있으면 로드
    if os.path.exists(cache_file):
        print(f"캐시된 풀 주소를 로드합니다: {cache_file}")
        data = load_json(cache_file)
        # 딕셔너리에서 주소 리스트 추출
        if isinstance(data, dict) and 'addresses' in data:
            return data['addresses']
        return data
    
    # 체인 API URL 가져오기
    if chain_name not in CHAINS:
        print(f"지원되지 않는 체인: {chain_name}")
        return []
    
    api_url = CHAINS[chain_name]["pools_api"]
    
    # GraphQL 쿼리 작성 - name 필드 제거
    query = """
    query StableSurgePools {
      pools(where: {stableSurgeParams_not: null}) {
        address
      }
    }
    """
    
    # 쿼리 실행
    result = run_graph_query(api_url, query)
    
    if not result or 'data' not in result or 'pools' not in result['data']:
        print(f"{chain_name} 체인에서 풀 주소를 가져오지 못했습니다.")
        return []
    
    # 풀 주소 추출
    pools = result['data']['pools']
    pool_addresses = [pool['address'] for pool in pools]
    
    print(f"{chain_name} 체인에서 {len(pool_addresses)}개의 StableSurge 풀을 찾았습니다.")
    
    # 파일로 저장 - name 필드 없이 주소만 저장
    pool_data = {
        "addresses": pool_addresses
    }
    save_json(pool_data, f"{chain_name}_{POOL_ADDRESSES_FILE}")
    
    return pool_addresses

def get_pool_snapshots(chain_name, pool_addresses):
    """
    특정 체인에서 풀 스냅샷 데이터를 가져옵니다.
    
    Args:
        chain_name (str): 체인 이름
        pool_addresses (list): 풀 주소 목록
        
    Returns:
        pandas.DataFrame: 풀 스냅샷 데이터
    """
    print(f"{chain_name} 체인에서 {len(pool_addresses)}개 풀의 스냅샷 데이터를 추출 중...")
    
    # 캐시된 파일 경로
    cache_file = os.path.join(DATA_DIR, f"{chain_name}_{POOL_SNAPSHOTS_FILE}")
    
    # 캐시된 파일이 있으면 로드
    if os.path.exists(cache_file):
        print(f"캐시된 풀 스냅샷 데이터를 로드합니다: {cache_file}")
        df = pd.read_csv(cache_file, parse_dates=['timestamp'])
        
        # tokens 필드가 문자열로 저장되어 있으면 JSON으로 변환
        if 'tokens' in df.columns and isinstance(df['tokens'].iloc[0], str):
            df['tokens'] = df['tokens'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
        
        return df
    
    # 체인 API URL 가져오기
    if chain_name not in CHAINS:
        print(f"지원되지 않는 체인: {chain_name}")
        return pd.DataFrame()
    
    api_url = CHAINS[chain_name]["events_api"]
    
    # 모든 스냅샷 데이터를 저장할 리스트
    all_snapshots = []
    
    # 풀 주소 묶음 생성 (API 요청 크기 제한 때문에)
    max_addresses_per_query = 10  # 한 번에 처리할 최대 풀 주소 수
    address_batches = [pool_addresses[i:i+max_addresses_per_query] 
                      for i in range(0, len(pool_addresses), max_addresses_per_query)]
    
    # 각 주소 묶음에 대해 쿼리 실행
    for batch_idx, address_batch in enumerate(address_batches):
        print(f"주소 묶음 {batch_idx+1}/{len(address_batches)} 처리 중...")
        
        # 주소 목록을 문자열로 변환
        addresses_str = '", "'.join(address_batch)
        addresses_str = f'["{addresses_str}"]'
        
        # 페이지네이션을 위한 변수
        skip = 0
        has_more_data = True
        
        while has_more_data:
            # GraphQL 쿼리 작성
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
            
            # 쿼리 실행
            result = run_graph_query(api_url, query)
            
            if not result or 'data' not in result or 'poolSnapshots' not in result['data']:
                print(f"스냅샷 데이터를 가져오지 못했습니다.")
                break
            
            snapshots = result['data']['poolSnapshots']
            
            if not snapshots:
                has_more_data = False
                continue
            
            # 데이터 변환 및 저장
            for snapshot in snapshots:
                row = {
                    'pool_address': snapshot['pool']['address'],
                    'tokens': json.dumps(snapshot['pool']['tokens']),  # 토큰 정보를 JSON 문자열로 저장
                    'balances': snapshot['balances'],
                    'timestamp': int(snapshot['timestamp'])
                }
                all_snapshots.append(row)
            
            # 다음 페이지로 이동
            if len(snapshots) < ITEMS_PER_PAGE:
                has_more_data = False
            else:
                skip += ITEMS_PER_PAGE
                print(f"다음 {ITEMS_PER_PAGE}개 스냅샷 데이터를 가져오는 중...")
    
    # 결과가 없으면 빈 DataFrame 반환
    if not all_snapshots:
        print("스냅샷 데이터가 없습니다.")
        return pd.DataFrame()
    
    # DataFrame 생성 및 시간 데이터 변환
    df = pd.DataFrame(all_snapshots)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    
    # 데이터 저장
    save_dataframe(df, f"{chain_name}_{POOL_SNAPSHOTS_FILE}")
    
    # tokens 필드를 다시 JSON 객체로 변환
    df['tokens'] = df['tokens'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
    
    return df

def get_add_removes(chain_name, pool_addresses):
    """
    특정 체인에서 AddRemove 이벤트 데이터를 가져옵니다.
    
    Args:
        chain_name (str): 체인 이름
        pool_addresses (list): 풀 주소 목록
        
    Returns:
        pandas.DataFrame: AddRemove 이벤트 데이터
    """
    print(f"{chain_name} 체인에서 AddRemove 이벤트 데이터를 추출 중...")
    
    # 캐시된 파일 경로
    cache_file = os.path.join(DATA_DIR, f"{chain_name}_{ADD_REMOVES_FILE}")
    
    # 캐시된 파일이 있으면 로드
    if os.path.exists(cache_file):
        print(f"캐시된 AddRemove 이벤트 데이터를 로드합니다: {cache_file}")
        return pd.read_csv(cache_file, parse_dates=['blockTimestamp'])
    
    # 체인 API URL 가져오기
    if chain_name not in CHAINS:
        print(f"지원되지 않는 체인: {chain_name}")
        return pd.DataFrame()
    
    api_url = CHAINS[chain_name]["events_api"]
    
    # 모든 AddRemove 이벤트 데이터를 저장할 리스트
    all_events = []
    
    # 풀 주소 묶음 생성
    max_addresses_per_query = 10
    address_batches = [pool_addresses[i:i+max_addresses_per_query] 
                      for i in range(0, len(pool_addresses), max_addresses_per_query)]
    
    # 각 주소 묶음에 대해 쿼리 실행
    for batch_idx, address_batch in enumerate(address_batches):
        print(f"주소 묶음 {batch_idx+1}/{len(address_batches)} 처리 중...")
        
        # 주소 목록을 문자열로 변환
        addresses_str = '", "'.join(address_batch)
        addresses_str = f'["{addresses_str}"]'
        
        # 페이지네이션을 위한 변수
        skip = 0
        has_more_data = True
        
        while has_more_data:
            # GraphQL 쿼리 작성 - name 필드 제거
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
            
            # 쿼리 실행
            result = run_graph_query(api_url, query)
            
            if not result or 'data' not in result or 'addRemoves' not in result['data']:
                print(f"AddRemove 이벤트 데이터를 가져오지 못했습니다.")
                break
            
            events = result['data']['addRemoves']
            
            if not events:
                has_more_data = False
                continue
            
            # 데이터 변환 및 저장 - name 필드 제거
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
            
            # 다음 페이지로 이동
            if len(events) < ITEMS_PER_PAGE:
                has_more_data = False
            else:
                skip += ITEMS_PER_PAGE
                print(f"다음 {ITEMS_PER_PAGE}개 AddRemove 이벤트 데이터를 가져오는 중...")
    
    # 결과가 없으면 빈 DataFrame 반환
    if not all_events:
        print("AddRemove 이벤트 데이터가 없습니다.")
        return pd.DataFrame()
    
    # DataFrame 생성 및 시간 데이터 변환
    df = pd.DataFrame(all_events)
    df['blockTimestamp'] = pd.to_datetime(df['blockTimestamp'], unit='s')
    
    # 데이터 저장
    save_dataframe(df, f"{chain_name}_{ADD_REMOVES_FILE}")
    
    return df

def get_swaps(chain_name, pool_addresses):
    """
    특정 체인에서 Swap 이벤트 데이터를 가져옵니다.
    
    Args:
        chain_name (str): 체인 이름
        pool_addresses (list): 풀 주소 목록
        
    Returns:
        pandas.DataFrame: Swap 이벤트 데이터
    """
    print(f"{chain_name} 체인에서 Swap 이벤트 데이터를 추출 중...")
    
    # 캐시된 파일 경로
    cache_file = os.path.join(DATA_DIR, f"{chain_name}_{SWAPS_FILE}")
    
    # 캐시된 파일이 있으면 로드
    if os.path.exists(cache_file):
        print(f"캐시된 Swap 이벤트 데이터를 로드합니다: {cache_file}")
        return pd.read_csv(cache_file, parse_dates=['blockTimestamp'])
    
    # 체인 API URL 가져오기
    if chain_name not in CHAINS:
        print(f"지원되지 않는 체인: {chain_name}")
        return pd.DataFrame()
    
    api_url = CHAINS[chain_name]["events_api"]
    
    # 모든 Swap 이벤트 데이터를 저장할 리스트
    all_swaps = []
    
    # 풀 주소 묶음 생성
    max_addresses_per_query = 10
    address_batches = [pool_addresses[i:i+max_addresses_per_query] 
                      for i in range(0, len(pool_addresses), max_addresses_per_query)]
    
    # 각 주소 묶음에 대해 쿼리 실행
    for batch_idx, address_batch in enumerate(address_batches):
        print(f"주소 묶음 {batch_idx+1}/{len(address_batches)} 처리 중...")
        
        # 주소 목록을 문자열로 변환
        addresses_str = '", "'.join(address_batch)
        addresses_str = f'["{addresses_str}"]'
        
        # 페이지네이션을 위한 변수
        skip = 0
        has_more_data = True
        
        while has_more_data:
            # GraphQL 쿼리 작성
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
            
            # 쿼리 실행
            result = run_graph_query(api_url, query)
            
            if not result or 'data' not in result or 'swaps' not in result['data']:
                print(f"Swap 이벤트 데이터를 가져오지 못했습니다.")
                break
            
            swaps = result['data']['swaps']
            
            if not swaps:
                has_more_data = False
                continue
            
            # 데이터 변환 및 저장
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
            
            # 다음 페이지로 이동
            if len(swaps) < ITEMS_PER_PAGE:
                has_more_data = False
            else:
                skip += ITEMS_PER_PAGE
                print(f"다음 {ITEMS_PER_PAGE}개 Swap 이벤트 데이터를 가져오는 중...")
    
    # 결과가 없으면 빈 DataFrame 반환
    if not all_swaps:
        print("Swap 이벤트 데이터가 없습니다.")
        return pd.DataFrame()
    
    # DataFrame 생성 및 시간 데이터 변환
    df = pd.DataFrame(all_swaps)
    df['blockTimestamp'] = pd.to_datetime(df['blockTimestamp'], unit='s')
    
    # 데이터 저장
    save_dataframe(df, f"{chain_name}_{SWAPS_FILE}")
    
    return df

def extract_all_data(chain_name="ethereum"):
    """
    특정 체인에서 모든 데이터를 추출합니다.
    
    Args:
        chain_name (str): 체인 이름
        
    Returns:
        dict: 모든 추출 데이터
    """
    ensure_directory_exists(DATA_DIR)
    ensure_directory_exists(RESULTS_DIR)
    
    # 1. StableSurge 풀 주소 추출
    pool_addresses = get_stable_pool_addresses(chain_name)
    
    if not pool_addresses or len(pool_addresses) == 0:
        print(f"{chain_name} 체인에서 풀 주소를 찾지 못했습니다.")
        return {}
    
    # 주소가 리스트인지 확인
    if not isinstance(pool_addresses, list):
        print(f"풀 주소가 리스트 형식이 아닙니다. 현재 타입: {type(pool_addresses)}")
        try:
            # 딕셔너리에서 주소 리스트 추출 시도
            if isinstance(pool_addresses, dict) and 'addresses' in pool_addresses:
                pool_addresses = pool_addresses['addresses']
            else:
                # 타입 변환 시도
                pool_addresses = list(pool_addresses)
        except Exception as e:
            print(f"풀 주소 형식 변환 중 오류 발생: {e}")
            return {}
    
    print(f"처리할 풀 주소 수: {len(pool_addresses)}")
    
    # 2. 풀 스냅샷 데이터 추출
    snapshots_df = get_pool_snapshots(chain_name, pool_addresses)
    
    # 3. AddRemove 이벤트 데이터 추출
    add_removes_df = get_add_removes(chain_name, pool_addresses)
    
    # 4. Swap 이벤트 데이터 추출
    swaps_df = get_swaps(chain_name, pool_addresses)
    
    return {
        'pool_addresses': pool_addresses,
        'snapshots': snapshots_df,
        'add_removes': add_removes_df,
        'swaps': swaps_df
    }

if __name__ == "__main__":
    # 데이터 추출 실행
    # for chain_name in CHAINS.keys():
    #     print(f"체인: {chain_name}")
    #     data = extract_all_data(chain_name)
        
    #     # 결과 요약
    #     print("\n데이터 추출 요약:")
    #     print(f"풀 주소: {len(data.get('pool_addresses', []))}개")
    #     print(f"스냅샷: {len(data.get('snapshots', pd.DataFrame()))}개")
    #     print(f"AddRemove 이벤트: {len(data.get('add_removes', pd.DataFrame()))}개")
    #     print(f"Swap 이벤트: {len(data.get('swaps', pd.DataFrame()))}개") 

    # save_all_stable_surge_pools()
    get_unbalanced_liquidity_events()