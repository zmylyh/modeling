import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import breaks_cusumolsresid
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_ind
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme()
def load_and_preprocess_data():
    print("\n=== 开始数据加载和预处理 ===")
    try:
        medals = pd.read_csv('data2025/summerOly_medal_counts.csv', encoding='utf-8')
        athletes = pd.read_csv('data2025/summerOly_athletes.csv', encoding='utf-8')
        print(f"成功加载数据 - 奖牌数据: {medals.shape}, 运动员数据: {athletes.shape}")
    except UnicodeDecodeError:
        medals = pd.read_csv('data2025/summerOly_medal_counts.csv', encoding='gbk')
        athletes = pd.read_csv('data2025/summerOly_athletes.csv', encoding='gbk')
        print("使用GBK编码重新加载数据")

    medals.columns = medals.columns.str.replace('ï»¿', '')
    athletes.columns = athletes.columns.str.replace('ï»¿', '')

    print("\n数据质量检查:")
    print(f"奖牌数据缺失值统计:\n{medals.isnull().sum()}")
    print(f"\n运动员数据缺失值统计:\n{athletes.isnull().sum()}")
    

    special_cases = {
        'ROC': 'Russia',
        'Soviet Union': 'Russia',
        'Unified Team': 'Russia',
        'EUN': 'Russia', 
        'URS': 'Russia',
    }

    print(f"\n合并前独特国家数量 - 奖牌数据: {medals['NOC'].nunique()}, 运动员数据: {athletes['NOC'].nunique()}")
    
    for old_noc, new_noc in special_cases.items():
        medals.loc[medals['NOC'] == old_noc, 'NOC'] = new_noc
        athletes.loc[athletes['NOC'] == old_noc, 'NOC'] = new_noc

    print(f"合并后独特国家数量 - 奖牌数据: {medals['NOC'].nunique()}, 运动员数据: {athletes['NOC'].nunique()}")
    valid_years = range(1896, 2021)
    medals = medals[medals['Year'].isin(valid_years)]
    athletes = athletes[athletes['Year'].isin(valid_years)]
    print("\n处理前数据范围:")
    print(f"年份范围: {medals['Year'].min()} - {medals['Year'].max()}")
    print(f"运动项目数量: {athletes['Sport'].nunique()}")
    medals_dups = medals.duplicated().sum()
    athletes_dups = athletes.duplicated().sum()
    medals = medals.drop_duplicates()
    athletes = athletes.drop_duplicates()
    print(f"\n移除的重复记录 - 奖牌数据: {medals_dups}, 运动员数据: {athletes_dups}")
    
    return medals, athletes
def create_sport_time_series(medals, athletes):
    print("\n=== 构建运动项目时间序列 ===")
    participation = athletes.groupby(['Year', 'NOC', 'Sport']).size().reset_index(name='Athletes')
    medals_data = athletes[athletes['Medal'].notna()].copy()
    medal_counts = medals_data.groupby(['Year', 'NOC', 'Sport', 'Medal']).size().reset_index(name='Count')
    medal_pivot = medal_counts.pivot_table(
        index=['Year', 'NOC', 'Sport'],
        columns='Medal',
        values='Count',
        fill_value=0
    ).reset_index()
    for medal_type in ['Gold', 'Silver', 'Bronze']:
        if medal_type not in medal_pivot.columns:
            medal_pivot[medal_type] = 0
    

    medal_pivot['Medals'] = medal_pivot[['Gold', 'Silver', 'Bronze']].sum(axis=1)
    sport_data = pd.merge(participation, medal_pivot[['Year', 'NOC', 'Sport', 'Medals']], 
                         on=['Year', 'NOC', 'Sport'], how='left')
    sport_data['Medals'] = sport_data['Medals'].fillna(0)
    sport_data['SuccessRate'] = sport_data['Medals'] / sport_data['Athletes']
    sport_data['SuccessRate'] = sport_data['SuccessRate'].fillna(0)
    success_rate_threshold = sport_data['SuccessRate'].quantile(0.95)  
    print(f"\n成功率95分位数: {success_rate_threshold:.4f}")
    sport_data.loc[sport_data['SuccessRate'] > success_rate_threshold, 'SuccessRate'] = success_rate_threshold
    print("\n数据统计:")
    print(f"总参赛人次: {sport_data['Athletes'].sum():,}")
    print(f"总获奖人次: {sport_data['Medals'].sum():,}")
    print(f"平均成功率: {sport_data['SuccessRate'].mean():.4f}")
    print(f"成功率中位数: {sport_data['SuccessRate'].median():.4f}")
    print(f"成功率标准差: {sport_data['SuccessRate'].std():.4f}")
    sport_stats = sport_data.groupby('Sport').agg({
        'Athletes': ['sum', 'mean'],
        'Medals': ['sum', 'mean'],
        'SuccessRate': ['mean', 'std', 'min', 'max']
    })
    sport_stats.columns = ['总参赛人数', '平均参赛人数/年', 
                          '总奖牌数', '平均奖牌数/年',
                          '平均成功率', '成功率标准差',
                          '最低成功率', '最高成功率']
    
    print("\n前10个最大项目的统计:")
    print(sport_stats.sort_values('总参赛人数', ascending=False).head(10))
    print("\n项目数量统计:")
    print(f"总项目数: {len(sport_data['Sport'].unique())}")
    print(f"平均每年参与国家数: {sport_data.groupby(['Year', 'Sport'])['NOC'].nunique().mean():.1f}")
    
    return sport_data
def detect_structural_breaks(time_series, dates):
    """使用改进的结构断点检测方法"""
    if len(time_series) < 8:  
        return None, None, None
    
    try:
        window_size = max(3, len(time_series) // 4)  
        rolling_mean = pd.Series(time_series).rolling(window=window_size, min_periods=1).mean()
        mean_changes = rolling_mean.diff()
        max_change_idx = np.argmax(np.abs(mean_changes[window_size:]))
        max_change_idx += window_size
        
        if max_change_idx >= len(dates):
            return None, None, None
        before_mean = np.mean(time_series[:max_change_idx])
        after_mean = np.mean(time_series[max_change_idx:])
        if len(time_series[:max_change_idx]) > 1 and len(time_series[max_change_idx:]) > 1:
            t_stat, p_value = ttest_ind(time_series[:max_change_idx], 
                                      time_series[max_change_idx:],
                                      equal_var=False)  
            
            if p_value < 0.1:  
                return dates[max_change_idx], t_stat, p_value
        return None, None, None
    except Exception as e:
        print(f"断点检测错误: {str(e)}")
        return None, None, None

def analyze_breakpoints(sport_data):
    """分析各国各项目的断点"""
    print("\n开始分析结构断点...")
    breakpoint_results = []
    for (noc, sport), group in sport_data.groupby(['NOC', 'Sport']):
        if len(group) >= 8:  
            group = group.sort_values('Year')
            breakpoint_year, t_stat, p_value = detect_structural_breaks(
                group['SuccessRate'].values, 
                group['Year'].values
            )
            
            if breakpoint_year is not None and p_value is not None and p_value < 0.1:  
                before_data = group[group['Year'] < breakpoint_year]
                after_data = group[group['Year'] >= breakpoint_year]
                
                if len(before_data) >= 3 and len(after_data) >= 3:  
                    before_mean = before_data['SuccessRate'].mean()
                    after_mean = after_data['SuccessRate'].mean()
                    improvement = after_mean - before_mean
                    pooled_std = np.sqrt(
                        (before_data['SuccessRate'].var() * (len(before_data) - 1) +
                         after_data['SuccessRate'].var() * (len(after_data) - 1)) /
                        (len(before_data) + len(after_data) - 2)
                    )
                    effect_size = improvement / pooled_std if pooled_std != 0 else 0
                    if abs(effect_size) >= 0.3:  
                        breakpoint_results.append({
                            'NOC': noc,
                            'Sport': sport,
                            'BreakpointYear': breakpoint_year,
                            'BeforeMean': before_mean,
                            'AfterMean': after_mean,
                            'Improvement': improvement,
                            'EffectSize': effect_size,
                            'TStatistic': t_stat,
                            'PValue': p_value
                        })
    
    results_df = pd.DataFrame(breakpoint_results)
    if not results_df.empty:
        print(f"\n找到 {len(results_df)} 个显著的结构断点")
        print("\n效应大小统计:")
        print(results_df['EffectSize'].describe())
        print("\n改进幅度统计:")
        print(results_df['Improvement'].describe())
    return results_df
def did_analysis(sport_data, breakpoint_results):
    """改进的双重差分分析"""
    print("\n开始双重差分分析...")
    did_results = []
    
    if breakpoint_results.empty:
        print("没有找到显著的结构断点，跳过DiD分析")
        return pd.DataFrame()
    
    for _, row in breakpoint_results.iterrows():
        noc = row['NOC']
        sport = row['Sport']
        breakpoint_year = row['BreakpointYear']
        
        try:
            treatment_data = sport_data[(sport_data['NOC'] == noc) & 
                                      (sport_data['Sport'] == sport)].copy()
            control_data = sport_data[(sport_data['Sport'] == sport) & 
                                    (sport_data['NOC'] != noc)].copy()
            
            if len(treatment_data) >= 5 and len(control_data) >= 5:
                treatment_data['Post'] = (treatment_data['Year'] >= breakpoint_year).astype(int)
                treatment_data['Treated'] = 1
                control_data['Post'] = (control_data['Year'] >= breakpoint_year).astype(int)
                control_data['Treated'] = 0
                combined_data = pd.concat([treatment_data, control_data])
                
                X = sm.add_constant(combined_data[['Treated', 'Post']])
                X['Interaction'] = X['Treated'] * X['Post']
                model = sm.OLS(combined_data['SuccessRate'], X).fit(cov_type='HC1') 
                
               
                did_effect = model.params['Interaction']
                std_error = model.bse['Interaction']
                t_stat = model.tvalues['Interaction']
                p_value = model.pvalues['Interaction']
                
                if p_value < 0.05:  
                    did_results.append({
                        'NOC': noc,
                        'Sport': sport,
                        'BreakpointYear': breakpoint_year,
                        'TreatmentEffect': did_effect,
                        'StandardError': std_error,
                        'TStatistic': t_stat,
                        'PValue': p_value,
                        'R2': model.rsquared
                    })
        except Exception as e:
            print(f"DiD分析错误 ({noc}, {sport}): {str(e)}")
            continue
    
    results_df = pd.DataFrame(did_results)
    if not results_df.empty:
        print(f"\n发现 {len(results_df)} 个显著的处理效应")
        print("\n处理效应统计:")
        print(results_df['TreatmentEffect'].describe())
        print("\n模型拟合优度 (R²) 统计:")
        print(results_df['R2'].describe())
    return results_df

def plot_coach_effect_analysis(sport_data, breakpoint_results, did_results):
    """绘制名帅效应分析图"""
    plt.figure(figsize=(20, 25))
    
    # 1. 断点分布
    ax1 = plt.subplot(111)
    sns.histplot(data=breakpoint_results, x='BreakpointYear', bins=20, ax=ax1)
    ax1.set_title('breakpoint year distribution', fontsize=12)
    ax1.set_xlabel('year')
    ax1.set_ylabel('count')
    plt.show()
    
    # 2. 改进幅度分布
    ax2 = plt.subplot(111)
    sns.boxplot(data=breakpoint_results, y='Sport', x='Improvement', ax=ax2)
    ax2.set_title('improvement after breakpoint', fontsize=12)
    ax2.set_xlabel('improvement')
    ax2.set_ylabel('program')
    plt.show()
    
    # 3. 前后表现对比
    ax3 = plt.subplot(111)
    improvement_data = breakpoint_results[['Sport', 'BeforeMean', 'AfterMean']].melt(
        id_vars=['Sport'], var_name='Period', value_name='Performance')
    sns.boxplot(data=improvement_data, x='Sport', y='Performance', hue='Period', ax=ax3)
    ax3.set_title('comparison', fontsize=12)
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)
    plt.show()
    
    # 4. DiD效应分布
    ax4 = plt.subplot(111)
    sns.scatterplot(data=did_results, x='BreakpointYear', y='TreatmentEffect', 
                    hue='Sport', size='TreatmentEffect', sizes=(50, 400), ax=ax4)
    ax4.set_title('did distribution', fontsize=12)
    ax4.set_xlabel('breakpoint year')
    ax4.set_ylabel('effect')
    plt.show()
    
    # 5. 主要项目时间趋势
    ax5 = plt.subplot(111)
    top_sports = breakpoint_results['Sport'].value_counts().head(5).index
    for sport in top_sports:
        sport_trend = sport_data[sport_data['Sport'] == sport].groupby('Year')['SuccessRate'].mean()
        plt.plot(sport_trend.index, sport_trend.values, marker='o', label=sport)
    ax5.set_title('success rate trend', fontsize=12)
    ax5.set_xlabel('year')
    ax5.set_ylabel('Average success rate')
    ax5.legend()
    plt.show()
    
    # 6. 国家-项目热力图
    ax6 = plt.subplot(111)
    pivot_data = breakpoint_results.pivot_table(
        index='NOC', columns='Sport', values='Improvement', aggfunc='mean')
    sns.heatmap(pivot_data, cmap='RdYlBu', center=0, ax=ax6)
    ax6.set_title('National Project Improvement Range Heat Map', fontsize=12)
    plt.show()
    
def main():
    medals, athletes = load_and_preprocess_data()   
    sport_data = create_sport_time_series(medals, athletes)
    
    
    print("\n=== 开始结构断点分析 ===")
    breakpoint_results = analyze_breakpoints(sport_data)
    
    if not breakpoint_results.empty:
        print("\n前10个最显著的结构断点:")
        significant_results = breakpoint_results.sort_values('EffectSize', ascending=False)
        print(significant_results[['NOC', 'Sport', 'BreakpointYear', 'Improvement', 'EffectSize', 'PValue']].head(10))
      
        print("\n=== 开始双重差分分析 ===")
        did_results = did_analysis(sport_data, breakpoint_results)
        
        if not did_results.empty:
            print("\n处理效应最大的10个案例:")
            top_effects = did_results.sort_values('TreatmentEffect', ascending=False)
            print(top_effects[['NOC', 'Sport', 'BreakpointYear', 'TreatmentEffect', 'PValue']].head(10))
          
            print("\n=== 生成可视化分析 ===")
            plot_coach_effect_analysis(sport_data, breakpoint_results, did_results)
     
        target_countries = ['United States', 'China', 'Great Britain']
        for country in target_countries:
            country_results = breakpoint_results[breakpoint_results['NOC'] == country]
            
            print(f"\n=== {country} 的项目投资建议 ===")
            if len(country_results) > 0:
        
                best_sports = country_results.sort_values('EffectSize', ascending=False)
                print(f"分析的项目数量: {len(country_results)}")
                print("最具潜力的项目:")
                for _, sport in best_sports.head(3).iterrows():
                    print(f"- {sport['Sport']}:")
                    print(f"  改进幅度: {sport['Improvement']:.2%}")
                    print(f"  效应大小: {sport['EffectSize']:.2f}")
                    print(f"  断点年份: {sport['BreakpointYear']}")
                    print(f"  显著性水平: {sport['PValue']:.4f}")
                    
                    if not did_results.empty:
                        did_effect = did_results[(did_results['NOC'] == country) & 
                                              (did_results['Sport'] == sport['Sport'])]
                        if not did_effect.empty:
                            print(f"  处理效应: {did_effect.iloc[0]['TreatmentEffect']:.2%}")
            else:
                print("暂无足够数据进行分析")
    else:
        print("\n未找到显著的结构断点，分析结束")

if __name__ == "__main__":
    main()
    print("\n=== 分析完成 ===")
    print("可视化结果已保存为 'coach_effect_analysis.png'") 