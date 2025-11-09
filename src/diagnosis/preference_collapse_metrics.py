import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from collections import Counter
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class PreferenceCollapseMetrics:
    """
    Implements metrics to quantify preference collapse in language models.
    """
    
    def __init__(self, embedding_model='all-MiniLM-L6-v2'):
        """
        Initialize the metrics calculator.
        
        Args:
            embedding_model: Name of sentence-BERT model for semantic embeddings
        """
        self.embedding_model = SentenceTransformer(embedding_model)
        self.smoothing = SmoothingFunction()
    
    # ===== GENERATION DIVERSITY METRICS =====
    
    def calculate_self_bleu(self, responses: List[str], n_gram: int = 4) -> float:
        """
        Calculate Self-BLEU score (lower = more diverse).
        
        Each response is compared against all others using BLEU score.
        
        Args:
            responses: List of generated responses
            n_gram: Maximum n-gram size for BLEU calculation
            
        Returns:
            Average Self-BLEU score (lower indicates more diversity)
        """
        if len(responses) < 2:
            return 0.0
        
        tokenized_responses = [resp.lower().split() for resp in responses]
        bleu_scores = []
        
        for i, hypothesis in enumerate(tokenized_responses):
            # Compare against all other responses
            references = [tokenized_responses[j] for j in range(len(tokenized_responses)) if j != i]
            
            if references:
                # Calculate BLEU with smoothing for short sentences
                score = sentence_bleu(
                    references, 
                    hypothesis,
                    smoothing_function=self.smoothing.method1,
                    weights=[1/n_gram] * n_gram  # Uniform weights up to n_gram
                )
                bleu_scores.append(score)
        
        return np.mean(bleu_scores) if bleu_scores else 0.0
    
    def calculate_semantic_entropy(self, responses: List[str], n_clusters: int = None) -> float:
        """
        Calculate Semantic Entropy (higher = more diverse perspectives).
        
        Uses sentence-BERT embeddings and clustering to measure semantic diversity.
        
        Args:
            responses: List of generated responses
            n_clusters: Number of semantic clusters (default: min(5, len(responses)))
            
        Returns:
            Semantic entropy score
        """
        if len(responses) < 2:
            return 0.0
        
        # Filter and validate responses
        valid_responses = []
        invalid_count = 0
        for r in responses:
            if r and isinstance(r, str) and len(r.strip()) > 0:
                valid_responses.append(r.strip())
            else:
                invalid_count += 1
        
        if invalid_count > 0:
            print(f"  Semantic entropy: Filtered out {invalid_count} invalid response(s)")
        
        if len(valid_responses) < 2:
            return 0.0
        
        # Generate sentence embeddings
        try:
            embeddings = self.embedding_model.encode(valid_responses)
        except Exception as e:
            print(f"Warning: Failed to encode responses: {e}")
            return 0.0
        
        # Determine number of clusters
        if n_clusters is None:
            n_clusters = min(5, len(valid_responses))
        n_clusters = min(n_clusters, len(valid_responses))
        
        if n_clusters < 2:
            return 0.0
        
        # Cluster embeddings
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        
        # Calculate cluster distribution
        cluster_counts = Counter(labels)
        total = len(valid_responses)
        probabilities = [count / total for count in cluster_counts.values()]
        
        # Calculate entropy: H = -Σ p(i) * log(p(i))
        entropy = -sum(p * np.log(p) for p in probabilities if p > 0)
        
        return entropy
    
    def calculate_distinct_n(self, responses: List[str], n: int = 2) -> float:
        """
        Calculate Distinct-n: ratio of unique n-grams to total n-grams.
        
        Args:
            responses: List of generated responses
            n: Size of n-grams (default: 2 for bigrams)
            
        Returns:
            Distinct-n score (higher = more diverse)
        """
        all_ngrams = []
        
        for response in responses:
            tokens = response.lower().split()
            ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
            all_ngrams.extend(ngrams)
        
        if len(all_ngrams) == 0:
            return 0.0
        
        unique_ngrams = len(set(all_ngrams))
        total_ngrams = len(all_ngrams)
        
        return unique_ngrams / total_ngrams
    
    def compute_generation_diversity(self, responses: List[str]) -> Dict[str, float]:
        """
        Compute all generation diversity metrics.
        
        Args:
            responses: List of generated responses
            
        Returns:
            Dictionary with Self-BLEU, Semantic Entropy, and Distinct-2 scores
        """
        return {
            'self_bleu': self.calculate_self_bleu(responses),
            'semantic_entropy': self.calculate_semantic_entropy(responses),
            'distinct_2': self.calculate_distinct_n(responses, n=2)
        }
    
    # ===== PREFERENCE MARGINS =====
    
    def calculate_preference_margins(
        self, 
        log_probs: List[float],
        kappa: float = None
    ) -> Dict[str, float]:
        """
        Calculate preference margins: |log π(yᵢ|x) − log π(yⱼ|x)| across pairs.
        
        Args:
            log_probs: Log probabilities for each response given the prompt
            kappa: Agreement score (optional, for stratification)
            
        Returns:
            Dictionary with mean and std of preference margins
        """
        if len(log_probs) < 2:
            return {'mean_margin': 0.0, 'std_margin': 0.0, 'kappa': kappa}
        
        # Calculate all pairwise margins
        margins = []
        for i in range(len(log_probs)):
            for j in range(i+1, len(log_probs)):
                margin = abs(log_probs[i] - log_probs[j])
                margins.append(margin)
        
        return {
            'mean_margin': np.mean(margins),
            'std_margin': np.std(margins),
            'max_margin': np.max(margins),
            'min_margin': np.min(margins),
            'kappa': kappa
        }
    
    def estimate_log_probs_from_responses(
        self, 
        responses: List[str],
        use_length_normalization: bool = True
    ) -> List[float]:
        """
        Estimate log probabilities from responses using semantic similarity.
        
        This is a proxy when actual model log probabilities are unavailable.
        Higher semantic similarity to the centroid = higher estimated probability.
        
        Args:
            responses: List of generated responses
            use_length_normalization: Normalize by response length
            
        Returns:
            List of estimated log probabilities
        """
        if len(responses) == 0:
            return []
        
        # Filter and validate responses
        valid_responses = []
        valid_indices = []
        invalid_count = 0
        for i, r in enumerate(responses):
            if r and isinstance(r, str) and len(r.strip()) > 0:
                valid_responses.append(r.strip())
                valid_indices.append(i)
            else:
                invalid_count += 1
        
        if invalid_count > 0:
            print(f"  Log prob estimation: Filtered out {invalid_count} invalid response(s)")
        
        if len(valid_responses) == 0:
            return [0.0] * len(responses)
        
        # Get embeddings
        try:
            embeddings = self.embedding_model.encode(valid_responses)
        except Exception as e:
            print(f"Warning: Failed to encode responses for log prob estimation: {e}")
            return [0.0] * len(responses)
        
        # Calculate centroid
        centroid = np.mean(embeddings, axis=0)
        
        # Calculate similarities to centroid
        similarities = [
            np.dot(emb, centroid) / (np.linalg.norm(emb) * np.linalg.norm(centroid))
            for emb in embeddings
        ]
        
        # Convert to log probability scale
        # Add small constant to avoid log(0)
        log_probs_valid = [np.log(max(sim, 1e-10)) for sim in similarities]
        
        # Optional length normalization
        if use_length_normalization:
            lengths = [len(resp.split()) for resp in valid_responses]
            log_probs_valid = [lp / max(length, 1) for lp, length in zip(log_probs_valid, lengths)]
        
        # Map back to original indices
        log_probs = [0.0] * len(responses)
        for i, valid_idx in enumerate(valid_indices):
            log_probs[valid_idx] = log_probs_valid[i]
        
        return log_probs
    
    # ===== CONFIDENCE CALIBRATION =====
    
    def calculate_output_entropy(self, log_probs: List[float]) -> float:
        """
        Calculate model output entropy: H(π) = −(1/N) Σ log π(yᵢ|x)
        
        Args:
            log_probs: Log probabilities for each response
            
        Returns:
            Output entropy (higher = more uncertain)
        """
        if len(log_probs) == 0:
            return 0.0
        
        # Convert log probs to probabilities
        # Use log-sum-exp trick for numerical stability
        max_log_prob = max(log_probs)
        exp_log_probs = [np.exp(lp - max_log_prob) for lp in log_probs]
        sum_exp = sum(exp_log_probs)
        probs = [exp_lp / sum_exp for exp_lp in exp_log_probs]
        
        # Calculate entropy
        entropy = -sum(p * np.log(p) for p in probs if p > 0)
        
        return entropy
    
    def calculate_confidence_calibration(
        self, 
        entropies: List[float],
        kappas: List[float]
    ) -> Dict[str, float]:
        """
        Calculate Pearson correlation between model entropy and human agreement (κ).
        
        Args:
            entropies: List of model output entropies for different prompts
            kappas: List of corresponding human agreement scores
            
        Returns:
            Dictionary with correlation coefficient and p-value
        """
        if len(entropies) != len(kappas) or len(entropies) < 3:
            return {'pearson_r': 0.0, 'p_value': 1.0, 'n_samples': len(entropies)}
        
        # Remove any NaN or infinite values
        valid_pairs = [(e, k) for e, k in zip(entropies, kappas) 
                       if np.isfinite(e) and np.isfinite(k)]
        
        if len(valid_pairs) < 3:
            return {'pearson_r': 0.0, 'p_value': 1.0, 'n_samples': len(valid_pairs)}
        
        entropies_clean, kappas_clean = zip(*valid_pairs)
        
        r, p_value = pearsonr(entropies_clean, kappas_clean)
        
        return {
            'pearson_r': r,
            'p_value': p_value,
            'n_samples': len(valid_pairs)
        }
    
    # ===== FULL ANALYSIS =====
    
    def analyze_prompt(
        self,
        prompt: str,
        responses: List[str],
        kappa: float = None,
        log_probs: List[float] = None
    ) -> Dict[str, any]:
        """
        Perform complete preference collapse analysis for a single prompt.
        
        Args:
            prompt: The input prompt
            responses: List of generated responses
            kappa: Human agreement score (optional)
            log_probs: Model log probabilities (optional, will be estimated if None)
            
        Returns:
            Dictionary containing all metrics
        """
        # Generation diversity
        diversity_metrics = self.compute_generation_diversity(responses)
        
        # Estimate log probs if not provided
        if log_probs is None:
            log_probs = self.estimate_log_probs_from_responses(responses)
        
        # Preference margins
        margin_metrics = self.calculate_preference_margins(log_probs, kappa)
        
        # Output entropy
        output_entropy = self.calculate_output_entropy(log_probs)
        
        return {
            'prompt': prompt,
            'n_responses': len(responses),
            'kappa': kappa,
            **diversity_metrics,
            **margin_metrics,
            'output_entropy': output_entropy
        }
    
    def analyze_dataset(
        self,
        df: pd.DataFrame,
        prompt_col: str = 'prompt',
        response_col: str = 'response',
        kappa_col: str = None
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Analyze entire dataset for preference collapse.
        
        Args:
            df: DataFrame with prompts and responses
            prompt_col: Name of prompt column
            response_col: Name of response column (should contain a list of strings)
            kappa_col: Name of kappa (agreement) column (optional)
            
        Returns:
            Tuple of (detailed results DataFrame, summary statistics dict)
        """
        results = []
        all_entropies = []
        all_kappas = []
        
        print(f"Analyzing {len(df)} prompts...")
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing prompts"):
            try:
                prompt = row[prompt_col]
                
                # Get responses - expecting a list
                responses = row[response_col]
                
                # Parse if it's a string representation of a list
                if isinstance(responses, str):
                    try:
                        import ast
                        responses = ast.literal_eval(responses)
                    except:
                        # If parsing fails, treat as single response
                        responses = [responses]
                
                # Ensure it's a list and clean responses
                if not isinstance(responses, list):
                    responses = [responses]
                
                # Filter out None, empty strings, and non-string types
                invalid_responses = []
                for r in responses:
                    if r is None or not str(r).strip():
                        invalid_responses.append(r)
                
                # Print invalid responses if any found
                if invalid_responses:
                    print(f"\nRow {idx}: Found {len(invalid_responses)} invalid response(s):")
                    for i, invalid in enumerate(invalid_responses):
                        print(f"  Invalid response {i+1}: type={type(invalid)}, value={repr(invalid)[:100]}")
                
                responses = [str(r).strip() for r in responses if r is not None and str(r).strip()]
                
                # Skip if no valid responses
                if len(responses) == 0:
                    print(f"\nWarning: Row {idx} has no valid responses, skipping...")
                    continue
                
                # Get kappa if available
                kappa = row[kappa_col] if kappa_col and kappa_col in df.columns else None
                
                # Analyze this prompt
                prompt_results = self.analyze_prompt(prompt, responses, kappa)
                results.append(prompt_results)
                
                # Collect for calibration analysis
                all_entropies.append(prompt_results['output_entropy'])
                if kappa is not None:
                    all_kappas.append(kappa)
                    
            except Exception as e:
                print(f"\nError processing row {idx}: {e}")
                print(f"Response type: {type(row[response_col])}")
                print(f"Response preview: {str(row[response_col])[:200]}")
                continue
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Calculate confidence calibration if kappa available
        calibration = {}
        if all_kappas:
            calibration = self.calculate_confidence_calibration(all_entropies, all_kappas)
        
        # Summary statistics
        summary = {
            'mean_self_bleu': results_df['self_bleu'].mean(),
            'mean_semantic_entropy': results_df['semantic_entropy'].mean(),
            'mean_distinct_2': results_df['distinct_2'].mean(),
            'mean_preference_margin': results_df['mean_margin'].mean(),
            'mean_output_entropy': results_df['output_entropy'].mean(),
            **calibration
        }
        
        # Add stratified analysis by kappa if available
        if kappa_col and kappa_col in df.columns:
            # Low agreement (controversial) vs high agreement
            low_kappa = results_df[results_df['kappa'] < results_df['kappa'].median()]
            high_kappa = results_df[results_df['kappa'] >= results_df['kappa'].median()]
            
            summary['low_agreement'] = {
                'mean_self_bleu': low_kappa['self_bleu'].mean(),
                'mean_semantic_entropy': low_kappa['semantic_entropy'].mean(),
                'mean_distinct_2': low_kappa['distinct_2'].mean(),
                'mean_margin': low_kappa['mean_margin'].mean(),
            }
            
            summary['high_agreement'] = {
                'mean_self_bleu': high_kappa['self_bleu'].mean(),
                'mean_semantic_entropy': high_kappa['semantic_entropy'].mean(),
                'mean_distinct_2': high_kappa['distinct_2'].mean(),
                'mean_margin': high_kappa['mean_margin'].mean(),
            }
        
        return results_df, summary


def detect_preference_collapse(
    base_summary: Dict,
    dpo_summary: Dict,
    thresholds: Dict = None
) -> Dict[str, bool]:
    """
    Detect evidence of preference collapse by comparing Base and DPO models.
    
    Preference collapse is evidenced when:
    1. Diversity metrics decline on controversial prompts
    2. Preference margins remain large despite annotator disagreement
    3. Calibration deteriorates (r → 0)
    
    Args:
        base_summary: Summary statistics from base model
        dpo_summary: Summary statistics from DPO model
        thresholds: Custom thresholds for detection (optional)
        
    Returns:
        Dictionary indicating presence of collapse indicators
    """
    if thresholds is None:
        thresholds = {
            'diversity_decline': 0.15,  # 15% decline
            'margin_increase': 0.10,    # 10% increase
            'calibration_decline': 0.20  # 20% decline in correlation
        }
    
    indicators = {}
    
    # Check diversity decline
    if 'low_agreement' in base_summary and 'low_agreement' in dpo_summary:
        base_div = (
            base_summary['low_agreement']['mean_semantic_entropy'] +
            base_summary['low_agreement']['mean_distinct_2']
        ) / 2
        
        dpo_div = (
            dpo_summary['low_agreement']['mean_semantic_entropy'] +
            dpo_summary['low_agreement']['mean_distinct_2']
        ) / 2
        
        diversity_change = (base_div - dpo_div) / base_div if base_div > 0 else 0
        indicators['diversity_declined_on_controversial'] = (
            diversity_change > thresholds['diversity_decline']
        )
        indicators['diversity_change_pct'] = diversity_change * 100
    
    # Check if margins remain large on low agreement prompts
    if 'low_agreement' in dpo_summary:
        high_margin_low_agreement = (
            dpo_summary['low_agreement']['mean_margin'] > 
            dpo_summary.get('mean_preference_margin', 0) * 0.8
        )
        indicators['high_margins_despite_disagreement'] = high_margin_low_agreement
    
    # Check calibration deterioration
    if 'pearson_r' in base_summary and 'pearson_r' in dpo_summary:
        base_r = abs(base_summary['pearson_r'])
        dpo_r = abs(dpo_summary['pearson_r'])
        
        calibration_decline = (base_r - dpo_r) / base_r if base_r > 0 else 0
        indicators['calibration_deteriorated'] = (
            calibration_decline > thresholds['calibration_decline']
        )
        indicators['calibration_decline_pct'] = calibration_decline * 100
        indicators['approaching_zero_calibration'] = dpo_r < 0.2
    
    # Overall collapse detected
    indicators['preference_collapse_detected'] = any([
        indicators.get('diversity_declined_on_controversial', False),
        indicators.get('high_margins_despite_disagreement', False),
        indicators.get('calibration_deteriorated', False)
    ])
    
    return indicators


# Example usage
if __name__ == "__main__":
    # Example: Load your dataset
    df = pd.read_csv('/home/ehghaghi/scratch/ehghaghi/tulu_runs/outputs/both/tulu_Llama-3.1-Tulu-3-8B-DPO_responses_1412677.csv')
    
    # Initialize analyzer
    analyzer = PreferenceCollapseMetrics()
    
    # Analyze dataset
    results_df, summary = analyzer.analyze_dataset(
        df,
        prompt_col='prompt',
        response_col='responses',
        kappa_col='kappa'
    )
    
    print("=" * 80)
    print("PREFERENCE COLLAPSE ANALYSIS")
    print("=" * 80)
    
    print("\nDetailed Results:")
    print(results_df.to_string())
    
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    for key, value in summary.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for k, v in value.items():
                print(f"  {k}: {v:.4f}")
        else:
            print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")