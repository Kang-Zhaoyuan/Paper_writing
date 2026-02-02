
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def calculate_rank(series, ascending=False):
    return series.rank(ascending=ascending, method='min')

def l1_distance(rank1, rank2):
    return np.sum(np.abs(rank1 - rank2))

def compute_bias_index(df):
    results = []
    
    # Group by Season and Week
    for (season, week), group in df.groupby(['赛季', '周数']):
        # Skip valid weeks with too few contestants if needed, but using all data is safer
        if len(group) < 2:
            continue
            
        # Get Scores and Votes
        judge_scores = group['裁判总分']
        fan_votes = group['估计粉丝票']
        
        # 1. Base Ranks
        # Score/Votes descending -> Rank 1 is best
        rank_judge = calculate_rank(judge_scores, ascending=False)
        rank_fan = calculate_rank(fan_votes, ascending=False)
        
        # 2. Ranking Method
        # Sum of ranks. Lower is better.
        # Note: In DWTS, it is strictly: Rank(Judge) + Rank(Fan).
        # We use the calculated ranks.
        combined_score_ranking = rank_judge + rank_fan
        final_rank_ranking = calculate_rank(combined_score_ranking, ascending=True)
        
        # 3. Percentage Method
        # Percentage of total points. Higher is better.
        # Judge Share + Fan Share
        if judge_scores.sum() == 0:
            judge_share = 0
        else:
            judge_share = judge_scores / judge_scores.sum()
            
        if fan_votes.sum() == 0:
            fan_share = 0
        else:
            fan_share = fan_votes / fan_votes.sum()
            
        combined_score_percent = judge_share + fan_share
        final_rank_percent = calculate_rank(combined_score_percent, ascending=False)
        
        # 4. Calculate Bias Index I
        # I = Dist(Final, Fan) / Dist(Final, Judge)
        # Using L1 distance (Manhattan) on ranks
        
        # Method 1: Ranking
        dist_final_fan_1 = l1_distance(final_rank_ranking, rank_fan)
        dist_final_judge_1 = l1_distance(final_rank_ranking, rank_judge)
        
        if dist_final_judge_1 == 0:
            I_ranking = np.nan # Avoid infinity, or handle as very large bias towards judge?
            # If Distance to Judge is 0, it means Final == Judge.
            # Then I -> Infinity (Extreme Judge Bias).
            # But here I > 1 means Judge Bias. I < 1 Fan Bias.
            # If Dist(Final, Fan) is also 0, then 0/0.
            if dist_final_fan_1 == 0:
                 I_ranking = 1.0 # Balanced/Identity
            else:
                 I_ranking = 10.0 # Cap at high value
        else:
            I_ranking = dist_final_fan_1 / dist_final_judge_1
            
        # Method 2: Percentage
        dist_final_fan_2 = l1_distance(final_rank_percent, rank_fan)
        dist_final_judge_2 = l1_distance(final_rank_percent, rank_judge)
        
        if dist_final_judge_2 == 0:
            if dist_final_fan_2 == 0:
                I_percent = 1.0
            else:
                I_percent = 10.0
        else:
            I_percent = dist_final_fan_2 / dist_final_judge_2
            
        results.append({
            'Season': season,
            'Week': week,
            'I_Ranking': I_ranking,
            'I_Percent': I_percent
        })
        
    return pd.DataFrame(results)

def main():
    # Load data
    file_path = '全赛季每周估计详情.csv' 
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    df = pd.read_csv(file_path)
    
    # Compute weekly I
    bias_df = compute_bias_index(df)
    
    # Aggregate by Season (Mean I)
    season_bias = bias_df.groupby('Season')[['I_Ranking', 'I_Percent']].mean().reset_index()
    
    # Plotting
    plt.figure(figsize=(14, 6))
    sns.set_style("whitegrid")
    
    # Width of bars
    bar_width = 0.35
    index = np.arange(len(season_bias))
    
    plt.bar(index, season_bias['I_Ranking'], bar_width, label='Ranking Rule', color='#4c72b0', alpha=0.9)
    plt.bar(index + bar_width, season_bias['I_Percent'], bar_width, label='Percentage Rule', color='#c44e52', alpha=0.9)
    
    plt.xlabel('Season', fontsize=12)
    plt.ylabel('Bias Index ($I$)\n($<1$: Fan Bias, $>1$: Judge Bias)', fontsize=12)
    plt.title('Comparison of Bias Index ($I$) Across Seasons: Ranking vs. Percentage Rule', fontsize=14)
    plt.xticks(index + bar_width / 2, season_bias['Season'].astype(int), rotation=90)
    plt.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, label='Balance ($I=1$)')
    plt.legend()
    
    plt.tight_layout()
    output_path = 'images/bias_index_compare.png'
    os.makedirs('images', exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
