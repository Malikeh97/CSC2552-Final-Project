import pandas as pd
import numpy as np
from statsmodels.stats.inter_rater import fleiss_kappa
import json
import warnings

def load_jsonl(file_path):
    """Load JSONL file into pandas DataFrame"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)


def calculate_fleiss_kappa_robust(all_ratings):
    """Calculate Fleiss' Kappa with robust error handling"""
    if not all_ratings or len(all_ratings) == 0:
        return np.nan
    
    if len(all_ratings) < 2:
        return np.nan
    
    all_vals = [rating for response in all_ratings for rating in response]
    
    if len(all_vals) < 2:
        return np.nan
    
    # Check for zero variance
    if len(set(all_vals)) == 1:
        return 1.0
    
    min_rating = min(all_vals)
    max_rating = max(all_vals)
    actual_categories = max_rating - min_rating + 1
    
    n_items = len(all_ratings)
    matrix = np.zeros((n_items, actual_categories))
    
    for i, ratings in enumerate(all_ratings):
        for rating in ratings:
            matrix[i, rating - min_rating] += 1
    
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            kappa = fleiss_kappa(matrix, method='fleiss')
            
            if np.isnan(kappa) or np.isinf(kappa):
                return 1.0
            
            return kappa
            
    except (RuntimeWarning, ZeroDivisionError, Warning):
        return 1.0
    except Exception as e:
        return np.nan


def calculate_response_quality_variance(all_ratings):
    """
    Calculate variance in response quality (mean ratings across responses)
    
    High variance = prompt produces inconsistent quality responses
    Low variance = prompt produces consistent quality responses
    
    Returns:
    --------
    dict with quality variance metrics
    """
    if not all_ratings or len(all_ratings) < 2:
        return {
            'response_mean_std': np.nan,
            'response_mean_variance': np.nan,
            'response_mean_range': np.nan,
            'response_mean_cv': np.nan
        }
    
    # Calculate mean rating for each response
    response_means = [np.mean(response) for response in all_ratings]
    
    # Variance metrics
    mean_std = np.std(response_means)
    mean_variance = np.var(response_means)
    mean_range = max(response_means) - min(response_means)
    
    # Coefficient of variation (normalized)
    overall_mean = np.mean(response_means)
    mean_cv = mean_std / overall_mean if overall_mean > 0 else 0
    
    return {
        'response_mean_std': mean_std,
        'response_mean_variance': mean_variance,
        'response_mean_range': mean_range,
        'response_mean_cv': mean_cv,
        'response_means': response_means
    }


def calculate_combined_controversy_score(kappa, response_quality_metrics):
    """
    Calculate a combined controversy score considering both:
    1. Intra-response disagreement (Fleiss' Kappa)
    2. Inter-response quality variance
    
    Returns:
    --------
    float: Combined controversy score (0 = not controversial, 1 = very controversial)
    """
    # Component 1: Annotator disagreement (inverse of kappa)
    # kappa = 1.0 → disagreement = 0
    # kappa = 0.0 → disagreement = 0.5
    # kappa < 0.0 → disagreement > 0.5
    if pd.isna(kappa):
        annotator_disagreement = 0.5
    else:
        annotator_disagreement = (1 - kappa) / 2  # Scale to 0-0.5
    
    # Component 2: Response quality variance
    # Normalize by max possible variance (e.g., 0-4 scale has max std ≈ 2)
    response_variance = response_quality_metrics['response_mean_std']
    if pd.isna(response_variance):
        quality_inconsistency = 0
    else:
        # Normalize: std > 1.5 is very high inconsistency
        quality_inconsistency = min(response_variance / 1.5, 1.0) * 0.5  # Scale to 0-0.5
    
    # Combined score (0 to 1)
    combined_score = annotator_disagreement + quality_inconsistency
    
    return combined_score


def classify_controversy_type(kappa, response_quality_metrics, combined_score):
    """
    Classify the type and level of controversy
    
    Returns:
    --------
    dict with classification details
    """
    response_std = response_quality_metrics['response_mean_std']
    
    # Thresholds
    KAPPA_HIGH = 0.6
    KAPPA_MEDIUM = 0.4
    RESPONSE_STD_HIGH = 1.0
    RESPONSE_STD_MEDIUM = 0.5
    
    # Determine controversy type
    has_annotator_disagreement = kappa < KAPPA_MEDIUM if not pd.isna(kappa) else False
    has_quality_variance = response_std > RESPONSE_STD_MEDIUM if not pd.isna(response_std) else False
    
    if pd.isna(kappa) or pd.isna(response_std):
        category = 'insufficient_data'
        controversy_type = 'unknown'
    elif kappa >= KAPPA_HIGH and response_std < RESPONSE_STD_MEDIUM:
        # High kappa, low response variance
        category = 'high_agreement'
        controversy_type = 'clear_objective'
    elif kappa >= KAPPA_HIGH and response_std >= RESPONSE_STD_HIGH:
        # High kappa, but high response variance
        category = 'controversial'
        controversy_type = 'quality_variance'  # YOUR EXAMPLE FALLS HERE
    elif kappa < KAPPA_MEDIUM and response_std < RESPONSE_STD_MEDIUM:
        # Low kappa, low response variance
        category = 'controversial'
        controversy_type = 'annotator_disagreement'
    elif kappa < KAPPA_MEDIUM and response_std >= RESPONSE_STD_HIGH:
        # Low kappa, high response variance
        category = 'controversial'
        controversy_type = 'both_disagreement_and_variance'
    else:
        # Medium cases
        category = 'medium_agreement'
        controversy_type = 'moderate'
    
    # Overall classification based on combined score
    if combined_score > 0.7:
        overall_category = 'highly_controversial'
    elif combined_score > 0.5:
        overall_category = 'controversial'
    elif combined_score > 0.3:
        overall_category = 'medium_agreement'
    else:
        overall_category = 'high_agreement'
    
    return {
        'category': overall_category,
        'controversy_type': controversy_type,
        'has_annotator_disagreement': has_annotator_disagreement,
        'has_quality_variance': has_quality_variance,
        'is_controversial': combined_score > 0.5
    }


def aggregate_prompt_controversy(df, rating_column='helpfulness'):
    """
    Aggregate all ratings for each unique prompt with FIXED controversy detection
    """
    results = []
    
    for prompt, group in df.groupby('prompt'):
        all_ratings = []
        
        # Collect all ratings
        for idx, row in group.iterrows():
            ratings = row[rating_column]
            if isinstance(ratings, list) and len(ratings) > 0:
                all_ratings.append(ratings)
        
        if not all_ratings or len(all_ratings) == 0:
            continue
        
        # Calculate Fleiss' Kappa (within-response agreement)
        kappa = calculate_fleiss_kappa_robust(all_ratings)
        
        # Calculate response quality variance (between-response consistency)
        quality_metrics = calculate_response_quality_variance(all_ratings)
        
        # Calculate combined controversy score
        combined_score = calculate_combined_controversy_score(kappa, quality_metrics)
        
        # Classify controversy
        classification = classify_controversy_type(kappa, quality_metrics, combined_score)
        
        # Calculate additional metrics
        all_ratings_flat = [rating for response in all_ratings for rating in response]
        
        overall_std = np.std(all_ratings_flat)
        overall_mean = np.mean(all_ratings_flat)
        overall_variance = np.var(all_ratings_flat)
        
        # Per-response metrics
        response_stds = [np.std(response) for response in all_ratings if len(response) > 1]
        avg_response_std = np.mean(response_stds) if response_stds else 0
        
        results.append({
            'prompt': prompt,
            'fleiss_kappa': kappa,
            'response_mean_std': quality_metrics['response_mean_std'],
            'response_mean_range': quality_metrics['response_mean_range'],
            'combined_controversy_score': combined_score,
            'category': classification['category'],
            'controversy_type': classification['controversy_type'],
            'is_controversial': classification['is_controversial'],
            'has_annotator_disagreement': classification['has_annotator_disagreement'],
            'has_quality_variance': classification['has_quality_variance'],
            'n_responses': len(all_ratings),
            'n_total_ratings': len(all_ratings_flat),
            'overall_mean': overall_mean,
            'overall_std': overall_std,
            'avg_response_std': avg_response_std,
            'response_means': quality_metrics['response_means']
        })
    
    results_df = pd.DataFrame(results)
    
    # Sort by combined controversy score (most controversial first)
    results_df = results_df.sort_values('combined_controversy_score', ascending=False)
    
    return results_df


def print_controversy_summary(results_df, metric_name='helpfulness'):
    """Print detailed summary with fixed controversy detection"""
    
    print("\n" + "="*100)
    print(f"FIXED PROMPT CONTROVERSY ANALYSIS ({metric_name.upper()})")
    print("="*100)
    
    print(f"\nTotal prompts analyzed: {len(results_df)}")
    print(f"Total responses: {results_df['n_responses'].sum()}")
    print(f"Total ratings: {results_df['n_total_ratings'].sum()}")
    
    print("\n" + "-"*100)
    print("CATEGORY DISTRIBUTION:")
    print("-"*100)
    category_dist = results_df['category'].value_counts()
    for cat, count in category_dist.items():
        pct = (count / len(results_df)) * 100
        print(f"  {cat:25s}: {count:4d} prompts ({pct:5.1f}%)")
    
    print("\n" + "-"*100)
    print("CONTROVERSY TYPE DISTRIBUTION:")
    print("-"*100)
    type_dist = results_df['controversy_type'].value_counts()
    for ctype, count in type_dist.items():
        pct = (count / len(results_df)) * 100
        print(f"  {ctype:35s}: {count:4d} prompts ({pct:5.1f}%)")
    
    print("\n" + "-"*100)
    print("KEY METRICS:")
    print("-"*100)
    print(f"  Mean Fleiss' Kappa: {results_df['fleiss_kappa'].mean():.4f}")
    print(f"  Mean Response Quality Std: {results_df['response_mean_std'].mean():.4f}")
    print(f"  Mean Combined Score: {results_df['combined_controversy_score'].mean():.4f}")
    
    print("\n" + "-"*100)
    print("INTERPRETATION GUIDE:")
    print("-"*100)
    print("  Controversy Types:")
    print("    - clear_objective: High agreement, consistent quality")
    print("    - annotator_disagreement: Annotators disagree on same response")
    print("    - quality_variance: Annotators agree, but responses vary wildly (YOUR CASE)")
    print("    - both_disagreement_and_variance: Both types of controversy")
    
    print("\n" + "-"*100)
    print("TOP 20 MOST CONTROVERSIAL PROMPTS:")
    print("-"*100)
    for i, (idx, row) in enumerate(results_df.head(20).iterrows(), 1):
        prompt_preview = row['prompt'][:65] + "..." if len(row['prompt']) > 65 else row['prompt']
        print(f"\n{i:2d}. {prompt_preview}")
        print(f"    Combined Score: {row['combined_controversy_score']:.3f} | Type: {row['controversy_type']}")
        print(f"    Fleiss κ: {row['fleiss_kappa']:.3f} | Response Std: {row['response_mean_std']:.3f}")
        print(f"    Response means: {[f'{m:.2f}' for m in row['response_means']]}")
    
    print("\n" + "-"*100)
    print("TOP 20 LEAST CONTROVERSIAL PROMPTS:")
    print("-"*100)
    for i, (idx, row) in enumerate(results_df.tail(20).iterrows(), 1):
        prompt_preview = row['prompt'][:65] + "..." if len(row['prompt']) > 65 else row['prompt']
        print(f"\n{i:2d}. {prompt_preview}")
        print(f"    Combined Score: {row['combined_controversy_score']:.3f} | Type: {row['controversy_type']}")
        print(f"    Fleiss κ: {row['fleiss_kappa']:.3f} | Response Std: {row['response_mean_std']:.3f}")
        print(f"    Response means: {[f'{m:.2f}' for m in row['response_means']]}")


def visualize_two_dimensional_controversy(results_df, 
                                         output_path='controversy_2d_analysis.png'):
    """
    Visualize the two-dimensional nature of controversy
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # 1. 2D scatter: Kappa vs Response Quality Variance
    scatter = axes[0, 0].scatter(results_df['fleiss_kappa'],
                                results_df['response_mean_std'],
                                c=results_df['combined_controversy_score'],
                                cmap='RdYlGn_r',
                                s=100,
                                alpha=0.6,
                                edgecolors='black',
                                linewidth=0.5)
    
    # Add quadrant lines
    axes[0, 0].axhline(0.5, color='red', linestyle='--', alpha=0.5, linewidth=2)
    axes[0, 0].axvline(0.4, color='red', linestyle='--', alpha=0.5, linewidth=2)
    
    # Label quadrants
    axes[0, 0].text(0.8, 1.5, 'Quality Variance\n(Your Example)', 
                   fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    axes[0, 0].text(0.2, 1.5, 'Both Issues', 
                   fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='red', alpha=0.5))
    axes[0, 0].text(0.8, 0.2, 'Clear & Consistent', 
                   fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='green', alpha=0.5))
    axes[0, 0].text(0.2, 0.2, 'Annotator\nDisagreement', 
                   fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='orange', alpha=0.5))
    
    axes[0, 0].set_xlabel("Fleiss' Kappa (Within-Response Agreement)", fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Response Quality Std (Between-Response Consistency)', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Two-Dimensional Controversy Space', fontsize=14, fontweight='bold')
    cbar = plt.colorbar(scatter, ax=axes[0, 0])
    cbar.set_label('Combined Controversy Score', fontsize=11)
    axes[0, 0].grid(alpha=0.3)
    
    # 2. Distribution of combined scores
    axes[0, 1].hist(results_df['combined_controversy_score'], bins=50, 
                   edgecolor='black', alpha=0.7, color='steelblue')
    axes[0, 1].axvline(0.5, color='red', linestyle='--', linewidth=2, label='Controversial threshold')
    axes[0, 1].set_xlabel('Combined Controversy Score', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Number of Prompts', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Distribution of Controversy Scores', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # 3. Controversy type distribution
    type_counts = results_df['controversy_type'].value_counts()
    colors_map = {
        'clear_objective': '#2ca02c',
        'annotator_disagreement': '#ff7f0e',
        'quality_variance': '#d62728',
        'both_disagreement_and_variance': '#8B0000',
        'moderate': '#1f77b4'
    }
    colors = [colors_map.get(t, '#gray') for t in type_counts.index]
    
    axes[1, 0].barh(range(len(type_counts)), type_counts.values, color=colors, edgecolor='black')
    axes[1, 0].set_yticks(range(len(type_counts)))
    axes[1, 0].set_yticklabels(type_counts.index, fontsize=10)
    axes[1, 0].set_xlabel('Number of Prompts', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Distribution by Controversy Type', fontsize=14, fontweight='bold')
    axes[1, 0].grid(axis='x', alpha=0.3)
    
    # 4. Category distribution
    category_counts = results_df['category'].value_counts()
    axes[1, 1].bar(range(len(category_counts)), category_counts.values, 
                  color='steelblue', edgecolor='black')
    axes[1, 1].set_xticks(range(len(category_counts)))
    axes[1, 1].set_xticklabels(category_counts.index, rotation=45, ha='right', fontsize=10)
    axes[1, 1].set_ylabel('Number of Prompts', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Overall Category Distribution', fontsize=14, fontweight='bold')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ 2D Visualization saved to: {output_path}")
    plt.close()


# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python analyze_controversy_fixed.py <input.jsonl>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    print("\n" + "="*100)
    print("FIXED TWO-DIMENSIONAL CONTROVERSY ANALYSIS")
    print("="*100)
    
    print(f"\nLoading data from: {file_path}")
    df = load_jsonl(file_path)
    
    print(f"✓ Data loaded!")
    print(f"  Rows: {len(df)}, Unique prompts: {df['prompt'].nunique()}")
    
    print("\nAnalyzing controversy with FIXED methodology...")
    results = aggregate_prompt_controversy(df, rating_column='helpfulness')
    
    # Print summary
    print_controversy_summary(results, metric_name='helpfulness')
    
    # Save results
    output_csv = '../../data/prompt_controversy.csv'
    results.to_csv(output_csv, index=False)
    print(f"\n✓ Results saved to: {output_csv}")
    
    # Visualize
    visualize_two_dimensional_controversy(results, '../../data/controversy_2d_analysis.png')
    
    # Print your specific example
    print("\n" + "="*100)
    print("CHECKING YOUR EXAMPLE:")
    print("="*100)
    example = results[results['prompt'].str.contains('chatgpt.*google', case=False, na=False)]
    if not example.empty:
        row = example.iloc[0]
        print(f"Prompt: {row['prompt']}")
        print(f"Response means: {row['response_means']}")
        print(f"Fleiss' Kappa: {row['fleiss_kappa']:.4f}")
        print(f"Response Quality Std: {row['response_mean_std']:.4f}")
        print(f"Combined Score: {row['combined_controversy_score']:.4f}")
        print(f"Category: {row['category']}")
        print(f"Controversy Type: {row['controversy_type']}")
        print(f"Is Controversial: {row['is_controversial']}")
    
    print("\n" + "="*100)
    print("✓✓✓ FIXED ANALYSIS COMPLETE! ✓✓✓")
    print("="*100)