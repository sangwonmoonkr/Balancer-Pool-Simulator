import os
import pandas as pd
import numpy as np
import json
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats

from .config import (
    DATA_DIR,
    RESULTS_DIR,
    ANALYSIS_DIR,
    PLOT_DPI,
    PLOT_FIGSIZE,
    PLOT_STYLE
)
from .utils import ensure_directory_exists, save_dataframe, save_figure

# 시각화 설정
plt.style.use(PLOT_STYLE)
plt.rcParams['figure.figsize'] = PLOT_FIGSIZE
plt.rcParams['figure.dpi'] = PLOT_DPI

class DynamicFeeModel:
    """
    동적 수수료 모델 클래스.
    
    Attributes:
        slope (float): 수수료 증가 속도를 제어하는 스케일링 요소
        max_fee (float): 최대 수수료 상한 (예: 1%)
    """
    
    def __init__(self, slope=0.01, max_fee=0.0095):
        """
        동적 수수료 모델 초기화.
        
        Args:
            slope (float): 수수료 증가 속도를 제어하는 스케일링 요소
            max_fee (float): 최대 수수료 상한
        """
        self.slope = slope
        self.max_fee = max_fee
    
    def calculate_fee(self, curdiff, postdiff, base_fee):
        """
        동적 수수료를 계산합니다.
        
        공식: fee = base_fee + (postdiff - curdiff) * slope + (1/2) * (postdiff - curdiff)^2 * slope^2
        
        Args:
            curdiff (float): 현재 차이
            postdiff (float): 행동 후 차이
            base_fee (float): 기본 수수료 (풀별로 다름)
            
        Returns:
            float: 계산된 수수료
        """
        # deltaD 계산 (postdiff - curdiff)
        delta_d = postdiff - curdiff
        
        # 기본 수수료 설정
        fee = base_fee
        
        # 차이가 증가하는 경우 (delta_d > 0)에만 추가 수수료 적용
        if delta_d > 0:
            # 선형 항
            fee += delta_d * self.slope
            
            # 이차 항
            fee += 0.5 * (delta_d ** 2) * (self.slope ** 2)
        
        # 최대 수수료 적용
        return min(fee, self.max_fee)

def create_stable_surge_swaps_dataset():
    """
    여러 체인의 reconstructed_state 파일들을 결합하여 분석용 데이터셋을 생성합니다.
    
    Returns:
        pandas.DataFrame: 분석용 데이터셋
    """
    print("안정적인 풀 거래 데이터셋 생성 중...")
    
    # 결과 디렉토리 내의 모든 파일 검색
    all_files = []
    for filename in os.listdir(RESULTS_DIR):
        if filename.endswith("_balancer_reconstructed_state.csv"):
            chain_name = filename.split('_')[0]
            file_path = os.path.join(RESULTS_DIR, filename)
            all_files.append((chain_name, file_path))
    
    if not all_files:
        print("reconstructed_state 파일을 찾을 수 없습니다.")
        return pd.DataFrame()
    
    # 모든 파일을 읽어서 결합
    all_data = []
    for chain_name, file_path in all_files:
        print(f"{chain_name} 체인 데이터 로드 중: {file_path}")
        try:
            df = pd.read_csv(file_path, parse_dates=['timestamp'])
            df['chain'] = chain_name  # 체인 정보 추가
            all_data.append(df)
        except Exception as e:
            print(f"파일 로드 중 오류 발생: {e}")
            continue
    
    if not all_data:
        print("로드된 데이터가 없습니다.")
        return pd.DataFrame()
    
    # 모든 데이터 결합
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # 풀별 거래 수 집계
    pool_counts = combined_df.groupby(['chain', 'pool_address']).size().reset_index(name='swap_count')
    
    # 통계적으로 유의미한 데이터셋 식별
    # 스왑 수가 최소 30개 이상인 풀만 선택 (중심극한정리 기준)
    min_swaps = 30
    significant_pools = pool_counts[pool_counts['swap_count'] >= min_swaps]
    
    if significant_pools.empty:
        print(f"최소 {min_swaps}개 이상의 스왑이 있는 풀이 없습니다.")
        return pd.DataFrame()
    
    print(f"충분한 데이터가 있는 풀 개수: {len(significant_pools)}")
    
    # 유의미한 풀에 대한 데이터만 필터링
    filtered_swaps = pd.merge(
        combined_df,
        significant_pools[['chain', 'pool_address']],
        on=['chain', 'pool_address'],
        how='inner'
    )
    
    # 각 풀의 기본 수수료 추출
    # 첫 번째 방법: 풀의 초기 스왑 수수료를 기본 수수료로 설정
    pool_base_fees = filtered_swaps.groupby(['chain', 'pool_address']).apply(
        lambda x: x.iloc[0]['fee'] / x.iloc[0]['amount_in']
    ).reset_index(name='base_fee_percent')
    
    # 데이터셋에 기본 수수료 정보 추가
    final_dataset = pd.merge(
        filtered_swaps,
        pool_base_fees,
        on=['chain', 'pool_address'],
        how='inner'
    )
    
    # 데이터 저장
    output_path =  'stable_surge_swaps.csv'
    save_dataframe(final_dataset, output_path, directory=DATA_DIR)
    
    print(f"분석용 데이터셋이 생성되었습니다: {output_path}")
    print(f"총 {len(final_dataset)}개의 거래 데이터")
    
    # 데이터셋 통계
    pool_stats = final_dataset.groupby(['chain', 'pool_address']).size().reset_index(name='count')
    print(f"풀별 거래 수 통계:")
    print(f"- 최소: {pool_stats['count'].min()}")
    print(f"- 최대: {pool_stats['count'].max()}")
    print(f"- 평균: {pool_stats['count'].mean():.1f}")
    print(f"- 중앙값: {pool_stats['count'].median()}")
    
    return final_dataset

def prepare_data(df):
    """
    데이터를 분석을 위해 준비합니다.
    
    Args:
        df (pandas.DataFrame): 원본 데이터프레임
        
    Returns:
        pandas.DataFrame: 처리된 데이터프레임
    """
    # 복사본 생성
    processed_df = df.copy()
    
    # JSON 문자열 컬럼 파싱
    for col in ['pre_event_weights', 'post_event_weights', 'pre_event_balances', 'post_event_balances']:
        if col in processed_df.columns:
            try:
                processed_df[col] = processed_df[col].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
            except:
                try:
                    processed_df[col] = processed_df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
                except:
                    print(f"컬럼 {col}을 파싱할 수 없습니다.")
    
    # 풀별로 처리
    all_processed = []
    
    for (chain, pool_address), group in processed_df.groupby(['chain', 'pool_address']):
        print(f"풀 {chain}:{pool_address} 처리 중 ({len(group)}개 거래)...")
        
        # 최적 가중치 벡터 계산 (균등 가중치)
        first_row = group.iloc[0]
        if 'pre_event_weights' in group.columns:
            n_tokens = len(first_row['pre_event_weights'])
        else:
            # 가중치가 없으면 밸런스에서 토큰 수 추정
            n_tokens = len(first_row['pre_event_balances']) if 'pre_event_balances' in group.columns else 2
        
        optimal_weights = np.array([1.0 / n_tokens] * n_tokens)
        
        # 편차 계산 함수
        def deviation(weights):
            return np.sum(np.abs(np.array(weights) - optimal_weights))
        
        # 현재 및 행동 후 편차 계산
        if 'curdiff' not in group.columns and 'pre_event_weights' in group.columns:
            group['curdiff'] = group['pre_event_weights'].apply(deviation)
        
        if 'postdiff' not in group.columns and 'post_event_weights' in group.columns:
            group['postdiff'] = group['post_event_weights'].apply(deviation)
        
        # deltaD 계산
        if 'curdiff' in group.columns and 'postdiff' in group.columns:
            group['deltaD'] = group['postdiff'] - group['curdiff']
        
        all_processed.append(group)
    
    if not all_processed:
        return processed_df
    
    # 모든 처리된 데이터 결합
    return pd.concat(all_processed, ignore_index=True)

def calculate_dynamic_fees(df, model):
    """
    동적 수수료 모델을 적용하여 수수료를 계산합니다.
    
    Args:
        df (pandas.DataFrame): 처리된 데이터프레임
        model (DynamicFeeModel): 동적 수수료 모델
        
    Returns:
        pandas.DataFrame: 동적 수수료가 추가된 데이터프레임
    """
    # 복사본 생성
    result_df = df.copy()
    
    # 각 거래에 대해 기본 수수료를 사용하여 동적 수수료 계산
    result_df['dynamic_fee_percent'] = result_df.apply(
        lambda row: model.calculate_fee(row['curdiff'], row['postdiff'], row['base_fee_percent']), 
        axis=1
    )
    
    # 실제 수수료 추출
    result_df['actual_fee_percent'] = result_df['fee_percent']
    # 오차 계산
    result_df['fee_percent_error'] = result_df['dynamic_fee_percent'] - result_df['actual_fee_percent']
    
    # 상대 오차 계산 (%)
    result_df['relative_fee_percent_error'] = result_df.apply(
        lambda row: row['fee_percent_error'] / row['actual_fee_percent'] * 100 if row['actual_fee_percent'] > 0 else 0, 
        axis=1
    )
    
    return result_df

def analyze_fee_model(df, model):
    """
    동적 수수료 모델의 성능을 분석합니다.
    
    Args:
        df (pandas.DataFrame): 동적 수수료가 추가된 데이터프레임
        model (DynamicFeeModel): 사용된 동적 수수료 모델
        
    Returns:
        dict: 분석 결과
    """
    # 기본 통계
    stats = {
        'actual_fee_percent_mean': df['actual_fee_percent'].mean(),
        'dynamic_fee_percent_mean': df['dynamic_fee_percent'].mean(),
        'actual_fee_percent_median': df['actual_fee_percent'].median(),
        'dynamic_fee_percent_median': df['dynamic_fee_percent'].median(),
        'actual_fee_percent_std': df['actual_fee_percent'].std(),
        'dynamic_fee_percent_std': df['dynamic_fee_percent'].std(),
        'mse': mean_squared_error(df['actual_fee_percent'], df['dynamic_fee_percent']),
        'mae': mean_absolute_error(df['actual_fee_percent'], df['dynamic_fee_percent']),
        'r2': r2_score(df['actual_fee_percent'], df['dynamic_fee_percent']),
        'correlation': df['actual_fee_percent'].corr(df['dynamic_fee_percent']),
        'model_params': {
            'slope': model.slope,
            'max_fee': model.max_fee
        },
        'base_fee_percent_mean': df['base_fee_percent'].mean(),
        'base_fee_percent_median': df['base_fee_percent'].median(),
        'base_fee_percent_std': df['base_fee_percent'].std()
    }
    
    # 추가 통계: 오차 분포
    stats.update({
        'fee_percent_error_mean': df['fee_percent_error'].mean(),
        'fee_percent_error_median': df['fee_percent_error'].median(),
        'fee_percent_error_std': df['fee_percent_error'].std(),
        'fee_percent_error_abs_mean': df['fee_percent_error'].abs().mean(),
        'relative_fee_percent_error_mean': df['relative_fee_percent_error'].mean(),
        'relative_fee_percent_error_median': df['relative_fee_percent_error'].median()
    })
    
    # 추가 통계: deltaD가 양수인 경우 (차이가 증가하는 경우)
    positive_delta = df[df['deltaD'] > 0]
    if not positive_delta.empty:
        stats.update({
            'positive_deltaD_count': len(positive_delta),
            'positive_deltaD_percent': len(positive_delta) / len(df) * 100,
            'positive_deltaD_actual_fee_percent_mean': positive_delta['actual_fee_percent'].mean(),
            'positive_deltaD_dynamic_fee_percent_mean': positive_delta['dynamic_fee_percent'].mean(),
            'positive_deltaD_fee_percent_error_mean': positive_delta['fee_percent_error'].mean(),
            'positive_deltaD_relative_fee_percent_error_mean': positive_delta['relative_fee_percent_error'].mean()
        })
    
    # 추가 통계: deltaD가 음수인 경우 (차이가 감소하는 경우)
    negative_delta = df[df['deltaD'] < 0]
    if not negative_delta.empty:
        stats.update({
            'negative_deltaD_count': len(negative_delta),
            'negative_deltaD_percent': len(negative_delta) / len(df) * 100,
            'negative_deltaD_actual_fee_percent_mean': negative_delta['actual_fee_percent'].mean(),
            'negative_deltaD_dynamic_fee_percent_mean': negative_delta['dynamic_fee_percent'].mean(),
            'negative_deltaD_fee_percent_error_mean': negative_delta['fee_percent_error'].mean(),
            'negative_deltaD_relative_fee_percent_error_mean': negative_delta['relative_fee_percent_error'].mean()
        })
    
    # 수수료 구간별 통계
    fee_bins = [0, 0.005, 0.01, 0.02, 0.05, 0.1, 1.0]
    fee_labels = ['0-0.05', '0.05-0.1', '0.1-0.2', '0.2-0.5', '0.5-1', '>1']
    
    df['fee_bin'] = pd.cut(df['actual_fee_percent'], bins=fee_bins, labels=fee_labels)
    fee_stats = df.groupby('fee_bin').agg({
        'actual_fee_percent': ['mean', 'count'],
        'dynamic_fee_percent': ['mean', 'count'],
        'fee_percent_error': ['mean', 'std'],
        'relative_fee_percent_error': ['mean', 'median']
    })
    
    # 풀별 성능 분석
    pool_performance = df.groupby(['chain', 'pool_address']).apply(
        lambda x: pd.Series({
            'count': len(x),
            'base_fee_percent': x['base_fee_percent'].iloc[0],
            'actual_fee_percent_mean': x['actual_fee_percent'].mean(),
            'dynamic_fee_percent_mean': x['dynamic_fee_percent'].mean(),
            'mse': mean_squared_error(x['actual_fee_percent'], x['dynamic_fee_percent']),
            'r2': r2_score(x['actual_fee_percent'], x['dynamic_fee_percent']) if len(x) > 1 else 0,
            'correlation': x['actual_fee_percent'].corr(x['dynamic_fee_percent']) if len(x) > 1 else 0
        })
    ).reset_index()
    
    # 시간 기반 통계
    if 'timestamp' in df.columns:
        # 일별 집계
        daily_stats = df.groupby(df['timestamp'].dt.date).agg({
            'actual_fee_percent': 'mean',
            'dynamic_fee_percent': 'mean',
            'fee_percent_error': 'mean',
            'relative_fee_percent_error': 'mean',
            'deltaD': 'mean',
            'base_fee_percent': 'mean'
        }).reset_index()
        
        # 주별 집계
        df['week'] = df['timestamp'].dt.isocalendar().week
        df['year'] = df['timestamp'].dt.isocalendar().year
        weekly_stats = df.groupby(['year', 'week']).agg({
            'actual_fee_percent': 'mean',
            'dynamic_fee_percent': 'mean',
            'fee_percent_error': 'mean',
            'relative_fee_percent_error': 'mean',
            'deltaD': 'mean',
            'base_fee_percent': 'mean'
        }).reset_index()
    else:
        daily_stats = pd.DataFrame()
        weekly_stats = pd.DataFrame()
    
    return {
        'stats': stats,
        'fee_bin_stats': fee_stats,
        'daily_stats': daily_stats,
        'weekly_stats': weekly_stats,
        'pool_performance': pool_performance
    }

def visualize_fee_model(df, analysis_results):
    """
    Visualize the performance of the dynamic fee model.
    
    Args:
        df (pandas.DataFrame): DataFrame with dynamic fees added
        analysis_results (dict): Analysis results
        
    Returns:
        None
    """
    
    # 1. deltaD vs actual/dynamic fees
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(df['deltaD'], df['actual_fee_percent'], alpha=0.5, s=5, label='Actual Fee')
    plt.xlabel('ΔD (postdiff - curdiff)')
    plt.ylabel('Fee Ratio')
    plt.title('ΔD vs Actual Fee')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.scatter(df['deltaD'], df['dynamic_fee_percent'], alpha=0.5, s=5, label='Dynamic Fee')
    
    # Add theoretical dynamic fee curve
    delta_range = np.linspace(0, df['deltaD'].max(), 100)
    model_params = analysis_results['stats']['model_params']
    base_fee_percent_mean = analysis_results['stats']['base_fee_percent_mean']
    model = DynamicFeeModel(slope=model_params['slope'], max_fee=model_params['max_fee'])
    
    # Add curve with average base_fee
    fees = [model.calculate_fee(0, delta, base_fee_percent_mean) for delta in delta_range]
    plt.plot(delta_range, fees, 'r-')
    
    plt.xlabel('ΔD (postdiff - curdiff)')
    plt.ylabel('Fee Ratio')
    plt.title('ΔD vs Dynamic Fee')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(plt, 'deltaD_vs_fees.png', directory=ANALYSIS_DIR)
    
    # 2. Actual vs dynamic fee scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(df['actual_fee_percent'], df['dynamic_fee_percent'], alpha=0.5, s=5)
    
    # Add 1:1 line
    min_fee = min(df['actual_fee_percent'].min(), df['dynamic_fee_percent'].min())
    max_fee = max(df['actual_fee_percent'].max(), df['dynamic_fee_percent'].max())
    plt.plot([min_fee, max_fee], [min_fee, max_fee], 'r--', label='1:1 Line')
    
    plt.xlabel('Actual Fee Ratio')
    plt.ylabel('Dynamic Fee Ratio')
    plt.title('Actual vs Dynamic Fee Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_figure(plt, 'actual_vs_dynamic_fees.png', directory=ANALYSIS_DIR)
    
    # 3. Error histogram
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(df['fee_percent_error'], bins=50, alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('Error (Dynamic - Actual)')
    plt.ylabel('Frequency')
    plt.title('Fee Error Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(df['relative_fee_percent_error'], bins=50, alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('Relative Error (%)')
    plt.ylabel('Frequency')
    plt.title('Fee Relative Error Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(plt, 'fee_error_distribution.png', directory=ANALYSIS_DIR)
    
    # 4. Time series of actual and dynamic fees
    if not analysis_results['daily_stats'].empty:
        daily_stats = analysis_results['daily_stats']
        
        plt.figure(figsize=(12, 6))
        plt.plot(daily_stats['timestamp'], daily_stats['actual_fee_percent'], label='Actual Fee Rate', marker='o')
        plt.plot(daily_stats['timestamp'], daily_stats['dynamic_fee_percent'], label='Dynamic Fee Rate', marker='x')
        plt.plot(daily_stats['timestamp'], daily_stats['base_fee_percent'], label='Base Fee Rate', marker='^', linestyle='--')
        plt.xlabel('Date')
        plt.ylabel('Average Fee Ratio')
        plt.title('Fee Trends Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        save_figure(plt, 'fee_time_series.png', directory=ANALYSIS_DIR)
    
    # 5. Fee bin comparison
    fee_stats = analysis_results['fee_bin_stats']
    
    plt.figure(figsize=(10, 6))
    fee_bins = fee_stats.index
    
    x = np.arange(len(fee_bins))
    width = 0.35
    
    plt.bar(x - width/2, fee_stats[('actual_fee_percent', 'count')], width, label='Actual Fee Rate', alpha=0.7)
    plt.bar(x + width/2, fee_stats[('dynamic_fee_percent', 'count')], width, label='Dynamic Fee Rate', alpha=0.7)
    
    plt.xlabel('Fee Bin')
    plt.ylabel('Average Fee Ratio')
    plt.title('Actual vs Dynamic Fee by Fee Bin')
    plt.xticks(x, fee_bins)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    save_figure(plt, 'fee_bin_comparison.png', directory=ANALYSIS_DIR)
    
    # 6. Fee comparison by deltaD
    plt.figure(figsize=(10, 6))
    
    # Prepare data
    categories = ['ΔD > 0', 'ΔD < 0', 'All']
    actual_fees = [
        analysis_results['stats'].get('positive_deltaD_actual_fee_percent_mean', 0),
        analysis_results['stats'].get('negative_deltaD_actual_fee_percent_mean', 0),
        analysis_results['stats']['actual_fee_percent_mean']
    ]
    dynamic_fees = [
        analysis_results['stats'].get('positive_deltaD_dynamic_fee_percent_mean', 0),
        analysis_results['stats'].get('negative_deltaD_dynamic_fee_percent_mean', 0),
        analysis_results['stats']['dynamic_fee_percent_mean']
    ]
    
    x = np.arange(len(categories))
    width = 0.35
    
    plt.bar(x - width/2, actual_fees, width, label='Actual Fee Rate', alpha=0.7)
    plt.bar(x + width/2, dynamic_fees, width, label='Dynamic Fee Rate', alpha=0.7)
    
    plt.xlabel('ΔD Range')
    plt.ylabel('Average Fee Ratio')
    plt.title('Fee Comparison by ΔD Range')
    plt.xticks(x, categories)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    save_figure(plt, 'deltaD_fee_comparison.png', directory=ANALYSIS_DIR)
    
    # 7. Fee change distribution
    plt.figure(figsize=(10, 6))
    
    # Prepare data
    fee_change_categories = ['Dynamic > Actual', 'Dynamic < Actual', 'Dynamic = Actual']
    fee_change_counts = [
        (df['fee_percent_error'] > 0.00001).sum(),
        (df['fee_percent_error'] < -0.00001).sum(),
        (abs(df['fee_percent_error']) <= 0.00001).sum()
    ]
    
    plt.bar(fee_change_categories, fee_change_counts, alpha=0.7)
    
    plt.xlabel('Fee Change')
    plt.ylabel('Transaction Count')
    plt.title('Fee Change Distribution')
    plt.grid(axis='y', alpha=0.3)
    
    # Add percentage labels
    total = sum(fee_change_counts)
    for i, count in enumerate(fee_change_counts):
        percentage = count / total * 100
        plt.text(i, count + 5, f'{percentage:.1f}%', ha='center')
    
    save_figure(plt, 'fee_change_distribution.png', directory=ANALYSIS_DIR)
    
    # 8. Relationship between fee change and deltaD
    plt.figure(figsize=(10, 6))
    
    # Color by fee change
    colors = ['red' if err > 0 else 'blue' for err in df['fee_percent_error']]
    
    plt.scatter(df['deltaD'], df['fee_percent_error'], alpha=0.5, s=5, c=colors)
    plt.axhline(y=0, color='black', linestyle='--')
    plt.axvline(x=0, color='black', linestyle='--')
    
    plt.xlabel('ΔD (postdiff - curdiff)')
    plt.ylabel('Fee Rate Error (Dynamic - Actual)')
    plt.title('Relationship Between ΔD and Fee Rate Error')
    plt.grid(True, alpha=0.3)
    
    save_figure(plt, 'deltaD_fee_error.png', directory=ANALYSIS_DIR)

def analyze_parameter_sensitivity(df, slope_values=None):
    """
    Analyze the sensitivity of the model to different slope values.
    
    Args:
        df (pandas.DataFrame): Processed DataFrame
        slope_values (list): List of slope values to test
        output_dir (str): Output directory
        
    Returns:
        pandas.DataFrame: Sensitivity analysis results
    """
    if slope_values is None:
        slope_values = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
    
    # List to store results
    sensitivity_results = []
    
    # Analyze for each slope value
    for slope in slope_values:
        model = DynamicFeeModel(slope=slope)
        result_df = calculate_dynamic_fees(df, model)
        analysis = analyze_fee_model(result_df, model)
        
        # Extract key metrics
        sensitivity_results.append({
            'slope': slope,
            'mse': analysis['stats']['mse'],
            'mae': analysis['stats']['mae'],
            'r2': analysis['stats']['r2'],
            'correlation': analysis['stats']['correlation'],
            'actual_fee_percent_mean': analysis['stats']['actual_fee_percent_mean'],
            'dynamic_fee_percent_mean': analysis['stats']['dynamic_fee_percent_mean'],
            'fee_percent_error_mean': analysis['stats']['fee_percent_error_mean'],
            'relative_fee_percent_error_mean': analysis['stats']['relative_fee_percent_error_mean']
        })
    
    # Create DataFrame from results
    sensitivity_df = pd.DataFrame(sensitivity_results)
    
    # Save results
    save_dataframe(sensitivity_df, 'parameter_sensitivity.csv', directory=ANALYSIS_DIR)
    
    # Visualize
    plt.figure(figsize=(14, 10))
    
    metrics = ['mse', 'mae', 'r2', 'correlation']
    titles = ['MSE (Mean Squared Error)', 'MAE (Mean Absolute Error)', 'R² (Coefficient of Determination)', 'Correlation Coefficient']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        plt.subplot(2, 2, i+1)
        plt.plot(sensitivity_df['slope'], sensitivity_df[metric], marker='o')
        plt.xlabel('Slope Value')
        plt.ylabel(title)
        plt.title(f'Change in {title} with Slope')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(plt, 'slope_sensitivity.png', directory=ANALYSIS_DIR)
    
    # Find optimal parameter
    best_r2 = sensitivity_df.loc[sensitivity_df['r2'].idxmax()]
    best_mse = sensitivity_df.loc[sensitivity_df['mse'].idxmin()]
    
    print(f"\nOptimal Slope (based on R²):")
    print(f"  Slope: {best_r2['slope']}")
    print(f"  R²: {best_r2['r2']:.4f}")
    
    print(f"\nOptimal Slope (based on MSE):")
    print(f"  Slope: {best_mse['slope']}")
    print(f"  MSE: {best_mse['mse']:.8f}")
    
    return sensitivity_df

def calculate_expected_revenue(df, model):
    """
    Calculate expected revenue when the dynamic fee model is applied.
    
    Args:
        df (pandas.DataFrame): Processed DataFrame
        model (DynamicFeeModel): Dynamic fee model
        
    Returns:
        dict: Revenue analysis results
    """
    # Calculate dynamic fees
    result_df = calculate_dynamic_fees(df, model)
    
    # Calculate original fee amount
    amount_col = "amount_in"
    if amount_col and 'actual_fee_percent' in result_df.columns:
        result_df['actual_fee_amount'] = result_df[amount_col] * result_df['actual_fee_percent']
        
        # Calculate dynamic fee amount
        result_df['dynamic_fee_amount'] = result_df[amount_col] * result_df['dynamic_fee_percent']
        
        # Calculate fee amount difference
        result_df['fee_amount_diff'] = result_df['dynamic_fee_amount'] - result_df['actual_fee_amount']
        
        # Calculate total
        total_actual_fee = result_df['actual_fee_amount'].sum()
        total_dynamic_fee = result_df['dynamic_fee_amount'].sum()
        total_fee_diff = result_df['fee_amount_diff'].sum()
        
        # Calculate revenue change percentage
        revenue_change_percent = (total_dynamic_fee / total_actual_fee - 1) * 100 if total_actual_fee > 0 else 0
        
        # Calculate pool-level revenue
        pool_revenue = result_df.groupby(['chain', 'pool_address']).agg({
            'actual_fee_amount': 'sum',
            'dynamic_fee_amount': 'sum',
            'fee_amount_diff': 'sum',
        }).reset_index()
        
        pool_revenue['revenue_change_percent'] = (
            pool_revenue['dynamic_fee_amount'] / pool_revenue['actual_fee_amount'] - 1
        ) * 100
        
        # Calculate daily revenue
        if 'timestamp' in result_df.columns:
            daily_revenue = result_df.groupby(result_df['timestamp'].dt.date).agg({
                'actual_fee_amount': 'sum',
                'dynamic_fee_amount': 'sum',
                'fee_amount_diff': 'sum'
            }).reset_index()
            
            daily_revenue['revenue_change_percent'] = (
                daily_revenue['dynamic_fee_amount'] / daily_revenue['actual_fee_amount'] - 1
            ) * 100
        else:
            daily_revenue = pd.DataFrame()
        
        # Visualize results
        plt.figure(figsize=(12, 10))
        
        # Daily revenue visualization
        if not daily_revenue.empty:
            plt.subplot(2, 2, 1)
            plt.plot(daily_revenue['timestamp'], daily_revenue['actual_fee_amount'], label='Actual Fee', marker='o')
            plt.plot(daily_revenue['timestamp'], daily_revenue['dynamic_fee_amount'], label='Dynamic Fee', marker='x')
            plt.xlabel('Date')
            plt.ylabel('Daily Fee Revenue')
            plt.title('Fee Revenue Over Time')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 2, 2)
            plt.bar(daily_revenue['timestamp'], daily_revenue['revenue_change_percent'])
            plt.xlabel('Date')
            plt.ylabel('Revenue Change (%)')
            plt.title('Daily Revenue Change')
            plt.grid(axis='y', alpha=0.3)
        
        # Pool-level revenue visualization
        top_pools = pool_revenue.nlargest(10, 'actual_fee_amount')
        
        plt.subplot(2, 2, 3)
        plt.bar(range(len(top_pools)), top_pools['actual_fee_amount'], label='Actual Fee')
        plt.bar(range(len(top_pools)), top_pools['dynamic_fee_amount'], label='Dynamic Fee', alpha=0.5)
        plt.xticks(range(len(top_pools)), [f"{chain}:{addr[:6]}..." for chain, addr in zip(top_pools['chain'], top_pools['pool_address'])], rotation=45, ha='right')
        plt.xlabel('Pool')
        plt.ylabel('Total Fee Revenue')
        plt.title('Fee Revenue Comparison for Top 10 Pools')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        plt.subplot(2, 2, 4)
        plt.bar(range(len(top_pools)), top_pools['revenue_change_percent'])
        plt.xticks(range(len(top_pools)), [f"{chain}:{addr[:6]}..." for chain, addr in zip(top_pools['chain'], top_pools['pool_address'])], rotation=45, ha='right')
        plt.xlabel('Pool')
        plt.ylabel('Revenue Change (%)')
        plt.title('Revenue Change for Top 10 Pools')
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        save_figure(plt, 'revenue_analysis.png', directory=ANALYSIS_DIR)
        
        return {
            'total_actual_fee': total_actual_fee,
            'total_dynamic_fee': total_dynamic_fee,
            'total_fee_diff': total_fee_diff,
            'revenue_change_percent': revenue_change_percent,
            'daily_revenue': daily_revenue,
            'pool_revenue': pool_revenue
        }
    else:
        print("Amount column or fee information is missing. Cannot calculate revenue.")
        return {}

def generate_summary_report(df, model, analysis_results, sensitivity_results=None, revenue_results=None):
    """
    Generate a summary report of the analysis results.
    
    Args:
        df (pandas.DataFrame): DataFrame with dynamic fees added
        model (DynamicFeeModel): Dynamic fee model used
        analysis_results (dict): Analysis results
        sensitivity_results (pandas.DataFrame): Sensitivity analysis results
        revenue_results (dict): Revenue analysis results
        
    Returns:
        str: Summary report
    """
    stats = analysis_results['stats']
    
    report = f"""
# Dynamic Fee Model Analysis Report

## 1. Dataset Overview
- Total Transactions: {len(df)}
- Number of Pools: {df['pool_address'].nunique()}
- Number of Chains: {df['chain'].nunique()}
- Pools per Chain:
{df.groupby('chain')['pool_address'].nunique().reset_index().rename(columns={'pool_address': 'Pool Count'}).to_string(index=False)}

## 2. Model Parameters
- Slope: {stats['model_params']['slope']}
- Max Fee: {stats['model_params']['max_fee']}
- Average Base Fee Rate: {stats['base_fee_percent_mean']:.6f} ({stats['base_fee_percent_mean']*100:.4f}%)
- Median Base Fee Rate: {stats['base_fee_percent_median']:.6f} ({stats['base_fee_percent_median']*100:.4f}%)

## 3. Performance Metrics
- MSE (Mean Squared Error): {stats['mse']:.8f}
- MAE (Mean Absolute Error): {stats['mae']:.8f}
- R² (Coefficient of Determination): {stats['r2']:.4f}
- Correlation Coefficient: {stats['correlation']:.4f}

## 4. Fee Statistics
- Average Actual Fee Rate: {stats['actual_fee_percent_mean']:.6f} ({stats['actual_fee_percent_mean']*100:.4f}%)
- Average Dynamic Fee Rate: {stats['dynamic_fee_percent_mean']:.6f} ({stats['dynamic_fee_percent_mean']*100:.4f}%)
- Median Actual Fee Rate: {stats['actual_fee_percent_median']:.6f} ({stats['actual_fee_percent_median']*100:.4f}%)
- Median Dynamic Fee Rate: {stats['dynamic_fee_percent_median']:.6f} ({stats['dynamic_fee_percent_median']*100:.4f}%)
- Actual Fee Rate Standard Deviation: {stats['actual_fee_percent_std']:.6f}
- Dynamic Fee Rate Standard Deviation: {stats['dynamic_fee_percent_std']:.6f}

## 5. Error Analysis
- Mean Error: {stats['fee_percent_error_mean']:.6f}
- Median Error: {stats['fee_percent_error_median']:.6f}
- Mean Absolute Error: {stats['fee_percent_error_abs_mean']:.6f}
- Mean Relative Error: {stats['relative_fee_percent_error_mean']:.2f}%
- Median Relative Error: {stats['relative_fee_percent_error_median']:.2f}%
"""
    
    # ΔD Analysis
    if 'positive_deltaD_count' in stats:
        report += f"""
## 6. ΔD Analysis
- ΔD > 0 (Increased Imbalance) Ratio: {stats['positive_deltaD_percent']:.2f}% ({stats['positive_deltaD_count']} transactions)
- ΔD < 0 (Decreased Imbalance) Ratio: {stats['negative_deltaD_percent']:.2f}% ({stats['negative_deltaD_count']} transactions)
- ΔD > 0 Average Actual Fee Rate: {stats['positive_deltaD_actual_fee_percent_mean']:.6f} ({stats['positive_deltaD_actual_fee_percent_mean']*100:.4f}%)
- ΔD > 0 Average Dynamic Fee Rate: {stats['positive_deltaD_dynamic_fee_percent_mean']:.6f} ({stats['positive_deltaD_dynamic_fee_percent_mean']*100:.4f}%)
- ΔD < 0 Average Actual Fee Rate: {stats['negative_deltaD_actual_fee_percent_mean']:.6f} ({stats['negative_deltaD_actual_fee_percent_mean']*100:.4f}%)
- ΔD < 0 Average Dynamic Fee Rate: {stats['negative_deltaD_dynamic_fee_percent_mean']:.6f} ({stats['negative_deltaD_dynamic_fee_percent_mean']*100:.4f}%)
"""
    
    # Revenue Analysis
    if revenue_results:
        report += f"""
## 7. Revenue Analysis
- Total Actual Fee: {revenue_results['total_actual_fee']:.2f}
- Total Dynamic Fee: {revenue_results['total_dynamic_fee']:.2f}
- Total Fee Difference: {revenue_results['total_fee_diff']:.2f}
- Revenue Change Percentage: {revenue_results['revenue_change_percent']:.2f}%
"""
    
    # Pool-level performance analysis
    pool_performance = analysis_results['pool_performance']
    if not pool_performance.empty:
        top_pools = pool_performance.nlargest(5, 'count')
        
        report += f"""
## 8. Top Pool Performance Analysis
"""
        
        for _, row in top_pools.iterrows():
            report += f"""
### {row['chain']}:{row['pool_address']}
- Transaction Count: {row['count']}
- Base Fee Rate: {row['base_fee_percent']:.6f} ({row['base_fee_percent']*100:.4f}%)
- Average Actual Fee Rate: {row['actual_fee_percent_mean']:.6f} ({row['actual_fee_percent_mean']*100:.4f}%)
- Average Dynamic Fee Rate: {row['dynamic_fee_percent_mean']:.6f} ({row['dynamic_fee_percent_mean']*100:.4f}%)
- MSE: {row['mse']:.8f}
- R²: {row['r2']:.4f}
- Correlation Coefficient: {row['correlation']:.4f}
"""
    
    # Sensitivity analysis results
    if sensitivity_results is not None:
        best_r2 = sensitivity_results.loc[sensitivity_results['r2'].idxmax()]
        best_mse = sensitivity_results.loc[sensitivity_results['mse'].idxmin()]
        
        report += f"""
## 9. Sensitivity Analysis Results
- Optimal Slope (based on R²):
  - Slope: {best_r2['slope']}
  - R²: {best_r2['r2']:.4f}

- Optimal Slope (based on MSE):
  - Slope: {best_mse['slope']}
  - MSE: {best_mse['mse']:.8f}
"""
    
    # Save report
    report_path = os.path.join(ANALYSIS_DIR, 'dynamic_fee_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"Report saved to {report_path}")
    
    return report

def run_analysis(slope=0.5, max_fee=0.95):
    """
    Run the dynamic fee model analysis.
    
    Args:
        output_dir (str): Output directory
        slope (float): Default slope value
        max_fee (float): Default max_fee value
        
    Returns:
        dict: Analysis results
    """
    # Create output directory
    ensure_directory_exists(ANALYSIS_DIR)
    
    # Create analysis dataset
    df_path = os.path.join(RESULTS_DIR, 'stable_surge_swaps.csv')
    
    if not os.path.exists(df_path):
        print("Creating analysis dataset...")
        df = create_stable_surge_swaps_dataset()
        if df.empty:
            print("Failed to create dataset. Analysis aborted.")
            return {}
    else:
        print(f"Loading existing dataset: {df_path}")
        df = pd.read_csv(df_path, parse_dates=['timestamp'] if 'timestamp' in pd.read_csv(df_path, nrows=1).columns else None)
    
    # Prepare data
    print("Preparing data...")
    processed_df = prepare_data(df)
    
    # Create basic model
    model = DynamicFeeModel(slope=slope, max_fee=max_fee)
    
    # Calculate dynamic fee rates
    print("Calculating dynamic fee rates...")
    result_df = calculate_dynamic_fees(processed_df, model)
    
    # Analyze model
    print("Analyzing model...")
    analysis_results = analyze_fee_model(result_df, model)
    
    # Visualize
    print("Visualizing results...")
    visualize_fee_model(result_df, analysis_results)
    
    # Parameter sensitivity analysis
    print("Performing parameter sensitivity analysis...")
    sensitivity_results = analyze_parameter_sensitivity(processed_df)
    
    # Revenue analysis
    print("Analyzing revenue...")
    revenue_results = calculate_expected_revenue(processed_df, model)
    
    # Generate report
    print("Generating report...")
    generate_summary_report(result_df, model, analysis_results, sensitivity_results, revenue_results)
    
    # Save processed data
    save_dataframe(result_df, 'processed_data.csv', directory=ANALYSIS_DIR)
    
    print(f"Analysis complete. Results saved to {ANALYSIS_DIR} directory.")
    
    return {
        'model': model,
        'analysis_results': analysis_results,
        'sensitivity_results': sensitivity_results,
        'revenue_results': revenue_results
    }

if __name__ == "__main__":
    # Run analysis
    run_analysis()