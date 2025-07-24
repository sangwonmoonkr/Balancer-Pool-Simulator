import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
from config import (
    DATA_DIR,
    RESULTS_DIR,
    RECONSTRUCTED_STATE_FILE,
    CHAINS
)
from utils import ensure_directory_exists, save_dataframe, load_json

def parse_balances(balances_str):
    """
    문자열 형태의 밸런스를 파싱합니다.
    
    Args:
        balances_str (str): 밸런스 문자열
        
    Returns:
        list: 밸런스 리스트
    """
    try:
        # 문자열에서 JSON 배열 구문 분석
        return json.loads(balances_str.replace("'", '"'))
    except (json.JSONDecodeError, AttributeError):
        return []

def parse_amounts(amounts_str):
    """
    문자열 형태의 금액을 파싱합니다.
    
    Args:
        amounts_str (str): 금액 문자열
        
    Returns:
        list: 금액 리스트
    """
    try:
        # 문자열에서 JSON 배열 구문 분석
        return json.loads(amounts_str.replace("'", '"'))
    except (json.JSONDecodeError, AttributeError):
        return []

def update_balances_for_swap(current_balances, token_in_idx, token_out_idx, amount_in, amount_out):
    """
    스왑 이벤트에 대해 밸런스를 업데이트합니다.
    
    Args:
        current_balances (list): 현재 밸런스 리스트
        token_in_idx (int): 입력 토큰 인덱스
        token_out_idx (int): 출력 토큰 인덱스
        amount_in (float): 입력 토큰 양
        amount_out (float): 출력 토큰 양
        
    Returns:
        list: 업데이트된 밸런스 리스트
    """
    new_balances = current_balances.copy()
    
    # 입력 토큰 밸런스 증가
    new_balances[token_in_idx] = float(new_balances[token_in_idx]) + float(amount_in)
    
    # 출력 토큰 밸런스 감소
    new_balances[token_out_idx] = float(new_balances[token_out_idx]) - float(amount_out)
    
    return new_balances

def load_unbalanced_liquidity_events():
    """
    불균형한 유동성 이벤트 데이터를 로드합니다.
    
    Returns:
        pandas.DataFrame: 불균형한 유동성 이벤트 데이터
    """
    file_path = os.path.join(DATA_DIR, "unbalanced_liquidity_events.csv")
    if os.path.exists(file_path):
        print(f"불균형한 유동성 이벤트 데이터를 로드합니다: {file_path}")
        df = pd.read_csv(file_path, parse_dates=['evt_block_time'])
        return df
    else:
        print(f"불균형한 유동성 이벤트 파일이 없습니다: {file_path}")
        return pd.DataFrame()

def get_token_decimals(snapshots_df, pool_address):
    """
    특정 풀의 토큰 소수점(decimal) 정보를 가져옵니다.
    
    Args:
        snapshots_df (pandas.DataFrame): 스냅샷 데이터프레임
        pool_address (str): 풀 주소
        
    Returns:
        dict: 토큰 주소를 키로, 소수점(decimal)을 값으로 하는 딕셔너리
    """
    # 해당 풀의 스냅샷에서 토큰 정보 찾기
    pool_snapshots = snapshots_df[snapshots_df['pool_address'].str.lower() == pool_address.lower()]
    
    if pool_snapshots.empty:
        print(f"풀 {pool_address}의 스냅샷이 없습니다.")
        return {}
    
    # 첫 번째 스냅샷에서 토큰 정보 추출
    first_snapshot = pool_snapshots.iloc[0]
    
    # tokens 필드 가져오기
    tokens = first_snapshot['tokens']
    
    # tokens 필드가 문자열인 경우 처리
    if isinstance(tokens, str):
        try:
            # 표준 JSON 형식 시도
            tokens = json.loads(tokens)
        except json.JSONDecodeError:
            try:
                # Python 문자열 표현을 처리 (작은따옴표를 큰따옴표로 변환)
                tokens = eval(tokens.replace("'", '"'))
            except (SyntaxError, NameError):
                print(f"토큰 정보를 파싱할 수 없습니다: {tokens}")
                return {}
    
    # 주소와 소수점 정보 추출
    token_decimals = {}
    for token in tokens:
        # 각 토큰 객체에서 주소와 소수점 추출
        address = token['address'].lower()
        decimals = int(token['decimals'])
        token_decimals[address] = decimals
    
    return token_decimals

def convert_raw_amount_to_decimal(raw_amount, decimals):
    """
    원시 금액을 소수점(decimal)을 고려한 값으로 변환합니다.
    
    Args:
        raw_amount (str): 원시 금액
        decimals (int): 토큰의 소수점
        
    Returns:
        float: 변환된 금액
    """
    try:
        raw_value = float(raw_amount)
        decimal_value = raw_value / (10 ** decimals)
        return decimal_value
    except (ValueError, TypeError) as e:
        print(f"금액 변환 오류: {e}, raw_amount: {raw_amount}, decimals: {decimals}")
        return 0.0

def mul_down(x, y):
    """
    고정 소수점 곱셈을 수행합니다 (내림).
    스마트 컨트랙트의 mulDown 함수와 동일한 동작을 구현합니다.
    
    Args:
        x (float): 첫 번째 피연산자
        y (float): 두 번째 피연산자 (일반적으로 비율)
        
    Returns:
        float: 결과
    """
    # 10^18 스케일링 (Solidity의 1e18과 동일)
    ONE = 10**18
    
    # 정수로 변환하여 계산
    x_scaled = int(x * ONE)
    y_scaled = int(y * ONE)
    
    # 곱셈 후 스케일 조정 (내림)
    result = (x_scaled * y_scaled) // ONE
    
    # float으로 변환하여 반환
    return result / ONE

def parse_swap_fee_amounts(swap_fee_amounts_str, token_decimals, pool_tokens):
    """
    문자열 형태의 스왑 수수료 금액을 파싱하고 소수점(decimal)을 고려하여 변환합니다.
    
    Args:
        swap_fee_amounts_str (str): 스왑 수수료 금액 문자열
        token_decimals (dict): 토큰 주소를 키로, 소수점(decimal)을 값으로 하는 딕셔너리
        pool_tokens (list): 풀의 토큰 주소 목록
        
    Returns:
        list: 소수점이 적용된 스왑 수수료 금액 리스트
    """
    try:
        # 문자열에서 JSON 배열 구문 분석
        raw_amounts = json.loads(swap_fee_amounts_str.replace("'", '"'))
        
        # 소수점을 고려하여 금액 변환
        decimal_amounts = []
        for i, raw_amount in enumerate(raw_amounts):
            if i < len(pool_tokens):
                token_address = pool_tokens[i].lower()
                if token_address in token_decimals:
                    decimals = token_decimals[token_address]
                    decimal_amount = convert_raw_amount_to_decimal(raw_amount, decimals)
                    decimal_amounts.append(decimal_amount)
                else:
                    print(f"토큰 {token_address}의 소수점 정보를 찾을 수 없습니다.")
                    decimal_amounts.append(float(raw_amount))
            else:
                decimal_amounts.append(float(raw_amount))
        
        return decimal_amounts
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"스왑 수수료 금액을 파싱할 수 없습니다: {swap_fee_amounts_str}, 오류: {e}")
        return []

def update_balances_for_add_remove(current_balances, amounts, event_type, chain=None, tx_hash=None, unbalanced_events_df=None, token_decimals=None, pool_tokens=None):
    """
    AddRemove 이벤트에 대해 밸런스를 업데이트합니다.
    스왑 수수료를 고려하여 밸런스를 조정합니다.
    
    Args:
        current_balances (list): 현재 밸런스 리스트
        amounts (list): 금액 리스트
        event_type (str): 이벤트 타입 ('ADD' 또는 'REMOVE')
        chain (str): 체인 이름
        tx_hash (str): 트랜잭션 해시
        unbalanced_events_df (pandas.DataFrame): 불균형한 유동성 이벤트 데이터
        token_decimals (dict): 토큰 주소를 키로, 소수점(decimal)을 값으로 하는 딕셔너리
        pool_tokens (list): 풀의 토큰 주소 목록
        
    Returns:
        list: 업데이트된 밸런스 리스트
    """
    new_balances = current_balances.copy()
    
    # 스왑 수수료 찾기
    swap_fee_amounts = None
    if unbalanced_events_df is not None and chain is not None and tx_hash is not None:
        # 체인과 트랜잭션 해시가 일치하는 이벤트 찾기
        matching_events = unbalanced_events_df[
            (unbalanced_events_df['chain'] == chain) & 
            (unbalanced_events_df['evt_tx_hash'] == tx_hash)
        ]
        
        if not matching_events.empty:
            # 첫 번째 일치하는 이벤트의 스왑 수수료 가져오기
            if len(matching_events) > 1:
                print(f"일치하는 불균형 이벤트가 여러 개 있습니다: {tx_hash}")
                print(matching_events)
                raise Exception(f"일치하는 불균형 이벤트가 여러 개 있습니다: {tx_hash}")
            swap_fee_amounts_raw = matching_events.iloc[0]['swapFeeAmountsRaw']
            
            # 소수점을 고려하여 스왑 수수료 파싱
            swap_fee_amounts = parse_swap_fee_amounts(
                swap_fee_amounts_raw,
                token_decimals,
                pool_tokens
            )
            # print(f"일치하는 불균형 이벤트 발견: {tx_hash}, 스왑 수수료: {swap_fee_amounts_raw} {swap_fee_amounts}")
    
    # 각 토큰에 대해 밸런스 업데이트
    for i, amount in enumerate(amounts):
        if i < len(new_balances):
            # 기본 금액
            amount_value = float(amount)
            
            # 스왑 수수료 적용 (있는 경우)
            if swap_fee_amounts and i < len(swap_fee_amounts):
                # mulDown(500000000000000000) 적용 (50%를 고정 소수점 산술로 곱함)
                fee_amount = mul_down(swap_fee_amounts[i], 0.5)  # 0.5 = 500000000000000000/10^18
                
                if fee_amount > 0:
                    # print(f"토큰 {i}에 스왑 수수료 {fee_amount} 적용")
                    if event_type == 'Add':
                        # 유동성 추가 시 수수료 제외
                        amount_value -= fee_amount
                    elif event_type == 'Remove':
                        # 유동성 제거 시 수수료 추가
                        amount_value += fee_amount
            
            # 밸런스 업데이트
            if event_type == 'Add':
                # 토큰 추가: 밸런스 증가
                new_balances[i] = float(new_balances[i]) + amount_value
            elif event_type == 'Remove':
                # 토큰 제거: 밸런스 감소
                new_balances[i] = float(new_balances[i]) - amount_value
    
    return new_balances

def calculate_balance_weights(balances):
    """
    밸런스 리스트에서 가중치를 계산합니다.
    
    Args:
        balances (list): 밸런스 리스트
        
    Returns:
        list: 가중치 리스트
    """
    total_balance = sum(float(balance) for balance in balances)
    
    if total_balance == 0:
        return [0] * len(balances)
    
    return [float(balance) / total_balance for balance in balances]

def find_token_index(pool_tokens, token_address):
    """
    풀 토큰 목록에서 특정 토큰의 인덱스를 찾습니다.
    
    Args:
        pool_tokens (list): 풀 토큰 주소 목록
        token_address (str): 찾을 토큰 주소
        
    Returns:
        int: 토큰 인덱스 (찾지 못하면 -1)
    """
    for i, token in enumerate(pool_tokens):
        if token.lower() == token_address.lower():
            return i
    return -1

def get_pool_tokens(snapshots_df, pool_address):
    """
    특정 풀의 토큰 목록을 가져옵니다.
    
    Args:
        snapshots_df (pandas.DataFrame): 스냅샷 데이터프레임
        pool_address (str): 풀 주소
        
    Returns:
        list: 토큰 주소 목록
    """
    # 해당 풀의 스냅샷에서 토큰 정보 찾기
    pool_snapshots = snapshots_df[snapshots_df['pool_address'].str.lower() == pool_address.lower()]
    
    if pool_snapshots.empty:
        print(f"풀 {pool_address}의 스냅샷이 없습니다.")
        return []
    
    # 첫 번째 스냅샷에서 토큰 정보 추출
    first_snapshot = pool_snapshots.iloc[0]
    
    # tokens 필드 가져오기
    tokens = first_snapshot['tokens']
    
    # tokens 필드가 문자열인 경우 처리
    if isinstance(tokens, str):
        try:
            # 표준 JSON 형식 시도
            tokens = json.loads(tokens)
        except json.JSONDecodeError:
            try:
                # Python 문자열 표현을 처리 (작은따옴표를 큰따옴표로 변환)
                tokens = eval(tokens)
            except (SyntaxError, NameError):
                print(f"토큰 정보를 파싱할 수 없습니다: {tokens}")
                return []
    
    # 주소 목록 추출
    address_list = []

    for token in tokens:
        # 각 토큰 객체에서 주소 추출
        address_list.append(token['address'])
    return address_list

def adjust_snapshot_time(snapshots_df):
    """
    스냅샷 시간을 해당 일자의 23:59:59로 조정합니다.
    
    Args:
        snapshots_df (pandas.DataFrame): 스냅샷 데이터프레임
        
    Returns:
        pandas.DataFrame: 시간이 조정된 스냅샷 데이터프레임
    """
    adjusted_df = snapshots_df.copy()
    
    # 각 스냅샷의 시간을 해당 일자의 23:59:59로 조정
    adjusted_df['timestamp'] = adjusted_df['timestamp'].apply(
        lambda x: pd.Timestamp(x.date()) + pd.Timedelta(hours=23, minutes=59, seconds=59)
    )
    
    return adjusted_df

def reconstruct_pool_state(snapshots_df, add_removes_df, swaps_df, chain_name="ethereum"):
    """
    풀 상태를 재구성합니다. 초기 유동성을 0으로 가정하고 모든 이벤트를 시간순으로 처리합니다.
    
    Args:
        snapshots_df (pandas.DataFrame): 풀 스냅샷 데이터
        add_removes_df (pandas.DataFrame): AddRemove 이벤트 데이터
        swaps_df (pandas.DataFrame): Swap 이벤트 데이터
        chain_name (str): 체인 이름
        
    Returns:
        pandas.DataFrame: 재구성된 풀 상태 데이터
    """
    print("풀 상태 재구성 중...")
    
    # 캐시된 파일 경로
    cache_file = os.path.join(RESULTS_DIR, f"{chain_name}_{RECONSTRUCTED_STATE_FILE}")
    
    # 캐시된 파일이 있으면 로드
    if os.path.exists(cache_file):
        print(f"캐시된 재구성 데이터를 로드합니다: {cache_file}")
        return pd.read_csv(cache_file, parse_dates=['timestamp'])
    
    # 스냅샷, AddRemove, Swap 데이터가 없으면 빈 DataFrame 반환
    if snapshots_df.empty or (add_removes_df.empty and swaps_df.empty):
        print("재구성에 필요한 데이터가 없습니다.")
        return pd.DataFrame()
    
     # 불균형한 유동성 이벤트 데이터 로드
    unbalanced_events_df = load_unbalanced_liquidity_events()
    
    # 스냅샷 시간 조정
    snapshots_df = adjust_snapshot_time(snapshots_df)

    
    # 유니크한 풀 주소 리스트 생성
    pool_addresses = snapshots_df['pool_address'].unique()
    
    # 각 풀에 대해 처리
    all_reconstructed_states = []
    
    for pool_address in tqdm(pool_addresses, desc="풀 처리 중"):
        print(f"\n풀 {pool_address} 처리 중...")
        
        # 해당 풀의 스냅샷 필터링 (시간순 정렬)
        pool_snapshots = snapshots_df[snapshots_df['pool_address'] == pool_address].sort_values('timestamp')
        
        if pool_snapshots.empty:
            print(f"풀 {pool_address}의 스냅샷이 없습니다. 건너뜁니다.")
            continue
        
        # 마지막 스냅샷 가져오기
        last_snapshot = pool_snapshots.iloc[-1]
        last_snapshot_time = last_snapshot['timestamp']
        last_snapshot_balances = parse_balances(last_snapshot['balances'])
        
        # 풀 토큰 정보 가져오기
        pool_tokens = get_pool_tokens(snapshots_df, pool_address)
        
        if not pool_tokens:
            print(f"풀 {pool_address}의 토큰 정보를 가져올 수 없습니다. 건너뜁니다.")
            continue

         
        # 토큰 소수점(decimal) 정보 가져오기
        token_decimals = get_token_decimals(snapshots_df, pool_address)
        
        # 토큰 수에 맞게 초기 밸런스를 0으로 설정
        current_balances = [0.0] * len(pool_tokens)
        
        # 해당 풀의 AddRemove 이벤트 필터링
        pool_add_removes = add_removes_df[add_removes_df['pool_address'] == pool_address].copy()
        
        # 해당 풀의 Swap 이벤트 필터링
        pool_swaps = swaps_df[swaps_df['pool_address'] == pool_address].copy()
        
        # 모든 이벤트를 시간순으로 정렬
        all_events = []
        
        # AddRemove 이벤트 추가
        for _, event in pool_add_removes.iterrows():
            all_events.append({
                'timestamp': event['blockTimestamp'],
                'block_number': event['blockNumber'],
                'log_index': event['logIndex'],
                'event_type': event['type'],
                'data': event
            })
        
        # Swap 이벤트 추가
        for _, event in pool_swaps.iterrows():
            all_events.append({
                'timestamp': event['blockTimestamp'],
                'block_number': event['blockNumber'],
                'log_index': event['logIndex'],
                'event_type': 'Swap',
                'data': event
            })
        
        # 시간, 블록, 로그 인덱스 순으로 정렬
        all_events.sort(key=lambda x: (
            x['timestamp'], 
            x['block_number'], 
            x['log_index']
        ))
        
        # 각 이벤트에 대해 처리
        for event_data in all_events:
            event = event_data['data']
            event_type = event_data['event_type']
            
            # 이벤트 전 밸런스 및 가중치
            pre_event_balances = current_balances.copy()
            pre_event_weights = calculate_balance_weights(pre_event_balances)
            
            # 이벤트에 따라 밸런스 업데이트
            if event_type == 'Swap':
                # 스왑 이벤트 처리
                token_in = event['tokenIn']
                token_out = event['tokenOut']
                amount_in = float(event['tokenAmountIn'])
                amount_out = float(event['tokenAmountOut'])
                fee = float(event['swapFeeAmount'])
                base_fee = float(event['swapFeeBaseAmount'])
                delta_fee = float(event['swapFeeDeltaAmount'])
                fee_percent = float(event['swapFeePercentage'])

                # 토큰 인덱스 찾기
                token_in_idx = find_token_index(pool_tokens, token_in)
                token_out_idx = find_token_index(pool_tokens, token_out)
                
                if token_in_idx >= 0 and token_out_idx >= 0:
                    # mulDown(500000000000000000) 적용
                    half_fee = mul_down(fee, 0.5)
                    
                    current_balances = update_balances_for_swap(
                        current_balances, 
                        token_in_idx, 
                        token_out_idx,
                        amount_in - half_fee,
                        amount_out,
                    )
                else:
                    print(f"토큰 인덱스 찾기 실패: {token_in} {token_out}")
                    print(f"풀 토큰 목록: {pool_tokens}")
                    continue
            
            elif event_type in ['Add', 'Remove']:
                # AddRemove 이벤트 처리
                amounts = parse_amounts(event['amounts'])
                add_remove_type = event['type']
                tx_hash = event['transactionHash']
                
                current_balances = update_balances_for_add_remove(
                    current_balances,
                    amounts,
                    add_remove_type,
                    chain_name,
                    tx_hash,
                    unbalanced_events_df,
                    token_decimals,
                    pool_tokens
                )
            

            if event_type == 'Swap':
                # 이벤트 후 밸런스 및 가중치
                post_event_balances = current_balances.copy()
                post_event_weights = calculate_balance_weights(post_event_balances)
                
                # 재구성된 상태 저장
                reconstructed_state = {
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
        
        # 마지막 재구성 상태와 마지막 스냅샷 비교
        if all_events:
            # 마지막 재구성된 밸런스
            final_balances = current_balances
            
            # 차이 계산
            balance_diff = []
            for i, (final, snapshot) in enumerate(zip(final_balances, last_snapshot_balances)):
                diff = float(snapshot) - float(final)
                balance_diff.append(diff)
            # 리스트 요소 각각에 abs 함수 적용 후 합계 계산
            if sum([abs(diff) for diff in balance_diff]) < 1:
                continue

            print(f"재구성된 최종 밸런스: {final_balances}")
            print(f"마지막 스냅샷 밸런스: {last_snapshot_balances}")
            print(f"밸런스 차이: {balance_diff}")
            
            # 재구성 정확도 계산 (상대 오차의 평균)
            relative_errors = []
            for final, snapshot in zip(final_balances, last_snapshot_balances):
                if float(snapshot) != 0:
                    error = abs(float(final) - float(snapshot)) / float(snapshot)
                    relative_errors.append(error)
            
            if relative_errors:
                avg_error = sum(relative_errors) / len(relative_errors)
                print(f"평균 상대 오차: {avg_error:.4f} ({avg_error*100:.2f}%)")
        else:
            print(f"풀 {pool_address}에 이벤트가 없습니다.")
    
    # 결과가 없으면 빈 DataFrame 반환
    if not all_reconstructed_states:
        print("재구성된 풀 상태가 없습니다.")
        return pd.DataFrame()
    
    # DataFrame 생성
    result_df = pd.DataFrame(all_reconstructed_states)
    
    # 데이터 저장
    save_dataframe(result_df, f"{chain_name}_{RECONSTRUCTED_STATE_FILE}", directory=RESULTS_DIR)
    
    return result_df


def analyze_weight_changes(reconstructed_df):
    """
    재구성된 풀 상태에서 가중치 변화를 분석합니다.
    
    Args:
        reconstructed_df (pandas.DataFrame): 재구성된 풀 상태 데이터
        
    Returns:
        dict: 분석 결과
    """
    print("가중치 변화 분석 중...")
    
    # 스왑 이벤트만 필터링
    swap_events = reconstructed_df[reconstructed_df['event_type'] == 'SWAP'].copy()
    
    if swap_events.empty:
        print("스왑 이벤트가 없습니다.")
        return {}
    
    # 각 스왑 이벤트에 대한 가중치 변화 계산
    weight_changes = []
    
    for _, event in swap_events.iterrows():
        pre_weights = json.loads(event['pre_event_weights'])
        post_weights = json.loads(event['post_event_weights'])
        
        # 최대 가중치 변화 계산
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
    
    # 결과 DataFrame 생성
    weight_changes_df = pd.DataFrame(weight_changes)
    
    # 통계 계산
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

def main(chain_name="ethereum"):
    """
    메인 실행 함수
    
    Args:
        chain_name (str): 체인 이름
    """
    # 디렉토리 생성
    ensure_directory_exists(DATA_DIR)
    ensure_directory_exists(RESULTS_DIR)
    
    # 캐시된 데이터 파일 경로
    snapshots_file = os.path.join(DATA_DIR, f"{chain_name}_balancer_pool_snapshots.csv")
    add_removes_file = os.path.join(DATA_DIR, f"{chain_name}_balancer_add_removes.csv")
    swaps_file = os.path.join(DATA_DIR, f"{chain_name}_balancer_swaps.csv")
    
    # 데이터 로드
    print("데이터 로드 중...")
    
    if os.path.exists(snapshots_file):
        snapshots_df = pd.read_csv(snapshots_file, parse_dates=['timestamp'])
    else:
        print(f"스냅샷 파일을 찾을 수 없습니다: {snapshots_file}")
        return
    
    if os.path.exists(add_removes_file):
        add_removes_df = pd.read_csv(add_removes_file, parse_dates=['blockTimestamp'])
    else:
        print(f"AddRemove 이벤트 파일을 찾을 수 없습니다: {add_removes_file}")
        add_removes_df = pd.DataFrame()
    
    if os.path.exists(swaps_file):
        swaps_df = pd.read_csv(swaps_file, parse_dates=['blockTimestamp'])
    else:
        print(f"Swap 이벤트 파일을 찾을 수 없습니다: {swaps_file}")
        swaps_df = pd.DataFrame()
    
    # 풀 상태 재구성
    reconstructed_df = reconstruct_pool_state(snapshots_df, add_removes_df, swaps_df, chain_name)
    
    # # 가중치 변화 분석
    # if not reconstructed_df.empty:
    #     analysis_results = analyze_weight_changes(reconstructed_df)
        
    #     if analysis_results:
    #         print("\n가중치 변화 분석 결과:")
    #         for key, value in analysis_results['stats'].items():
    #             print(f"{key}: {value}")

if __name__ == "__main__":
    # main("ethereum")
    for chain in CHAINS:
        main(chain)