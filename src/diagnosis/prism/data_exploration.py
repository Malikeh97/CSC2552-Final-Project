"""
Exploratory Data Analysis for PRISM Dataset
Analyzing the survey split to understand demographic distributions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


class PRISMSurveyEDA:
    """Exploratory Data Analysis for PRISM Survey dataset"""
    
    def __init__(self):
        """Initialize and load the PRISM survey dataset"""
        print("Loading PRISM survey dataset from HuggingFace...")

        # Load dataset dict
        self.dataset = load_dataset("HannahRoseKirk/prism-alignment", "survey")

        # Convert first split to pandas DataFrame
        split_name = list(self.dataset.keys())[0]  # typically 'train'
        self.df = self.dataset[split_name].to_pandas()

        # Expand if each row is a nested dict
        if len(self.df.columns) == 1 and isinstance(self.df.iloc[0, 0], dict):
            print("Expanding nested dictionaries inside the main column...")
            self.df = pd.json_normalize(self.df.iloc[:, 0])

        print(f"✓ Loaded {len(self.df)} survey responses with {len(self.df.columns)} columns\n")

        
    def basic_info(self):
        """Display basic dataset information"""
        print("="*80)
        print("BASIC DATASET INFORMATION")
        print("="*80)
        
        print(f"\nDataset shape: {self.df.shape}")
        print(f"Number of participants: {len(self.df)}")
        print(f"Number of columns: {len(self.df.columns)}")
        
        print("\n" + "-"*80)
        print("Column Names and Types:")
        print("-"*80)
        for col in self.df.columns:
            try:
                if self.df[col].apply(lambda x: isinstance(x, dict)).any():
                    nunique_val = self.df[col].apply(str).nunique()
                    print(f"  {col:30s} {str(self.df[col].dtype):15s} - {nunique_val} unique values (dicts converted to string)")
                else:
                    nunique_val = self.df[col].nunique()
                    print(f"  {col:30s} {str(self.df[col].dtype):15s} - {nunique_val} unique values")
            except Exception as e:
                print(f"  {col:30s} {str(self.df[col].dtype):15s} - ⚠️ Error computing unique values: {e}")

        
        print("\n" + "-"*80)
        print("First few rows:")
        print("-"*80)
        print(self.df.head())
        
        print("\n" + "-"*80)
        print("Missing Values:")
        print("-"*80)
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df) * 100).round(2)
        missing_df = pd.DataFrame({
            'Missing Count': missing,
            'Percentage': missing_pct
        })
        print(missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False))
        
        if missing_df['Missing Count'].sum() == 0:
            print("  No missing values found!")
    
    def analyze_demographics(self):
        """Analyze demographic distributions"""
        print("\n" + "="*80)
        print("DEMOGRAPHIC DISTRIBUTIONS")
        print("="*80)
        
        # Identify demographic columns (common demographic fields)
        demographic_cols = []
        possible_demo_names = ['age', 'gender', 'employment_status', 'education', 'marital_status', 
                              'english_proficiency', 'location', 'religion', 'ethnicity']
        
        for col in self.df.columns:
            col_lower = col.lower()
            if any(demo in col_lower for demo in possible_demo_names):
                demographic_cols.append(col)
        
        print(f"\nIdentified demographic columns: {demographic_cols}\n")
        
        # Analyze each demographic column
        for col in demographic_cols:
            print("-" * 80)
            print(f"Distribution of: {col}")
            print("-" * 80)

            # Detect if the column contains dictionaries
            if self.df[col].apply(lambda x: isinstance(x, dict)).any():
                print(f"⚠️ Column '{col}' contains dictionary values — flattening for readability.\n")

                # Extract 'categorised' if available, otherwise stringify
                self.df[col + '_parsed'] = self.df[col].apply(
                    lambda x: x.get('categorised') if isinstance(x, dict) and 'categorised' in x
                    else (x.get('self_described') if isinstance(x, dict) and 'self_described' in x
                        else str(x))
                )
                col_to_use = col + '_parsed'
            else:
                col_to_use = col

            value_counts = self.df[col_to_use].value_counts()
            value_pcts = (value_counts / len(self.df) * 100).round(2)

            dist_df = pd.DataFrame({
                'Count': value_counts,
                'Percentage': value_pcts
            })

            print(dist_df.head(20))
            print(f"\nTotal unique values: {self.df[col_to_use].nunique()}")
            print()
    
    def plot_demographic_distributions(self, save_dir="../../../data/prism_eda_plots"):
        """Create visualizations for demographic distributions"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        print("\n" + "="*80)
        print("GENERATING DEMOGRAPHIC VISUALIZATIONS")
        print("="*80 + "\n")
        
        # Identify demographic columns
        demographic_cols = []
        possible_demo_names = ['age', 'gender', 'employment_status', 'education', 'marital_status', 
                              'english_proficiency', 'location', 'religion', 'ethnicity']
        
        for col in self.df.columns:
            col_lower = col.lower()
            if any(demo in col_lower for demo in possible_demo_names):
                demographic_cols.append(col)
        
        # Plot each demographic
        for col in demographic_cols:
            print(f"Plotting: {col}")
            
            value_counts = self.df[col].value_counts()
            
            # Limit to top 20 for readability
            if len(value_counts) > 20:
                value_counts = value_counts.head(20)
                title_suffix = " (Top 20)"
            else:
                title_suffix = ""
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Create bar plot
            bars = ax.barh(range(len(value_counts)), value_counts.values, 
                          color='steelblue', alpha=0.7)
            ax.set_yticks(range(len(value_counts)))
            ax.set_yticklabels(value_counts.index)
            ax.set_xlabel('Count', fontsize=12)
            ax.set_title(f'Distribution of {col}{title_suffix}', 
                        fontsize=14, fontweight='bold')
            
            # Add value labels on bars
            for i, (bar, val) in enumerate(zip(bars, value_counts.values)):
                pct = val / len(self.df) * 100
                ax.text(val, i, f' {val} ({pct:.1f}%)', 
                       va='center', fontsize=9)
            
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            
            # Save figure
            safe_col_name = col.replace('/', '_').replace(' ', '_')
            plt.savefig(f"{save_dir}/distribution_{safe_col_name}.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"\n✓ Saved {len(demographic_cols)} visualizations to {save_dir}/")
    
    def analyze_user_distribution(self):
        """Analyze user_id distribution and uniqueness"""
        print("\n" + "="*80)
        print("USER ID ANALYSIS")
        print("="*80 + "\n")
        
        if 'user_id' in self.df.columns:
            print(f"Total number of user_ids: {self.df['user_id'].nunique()}")
            print(f"Total survey responses: {len(self.df)}")
            
            # Check for duplicates
            duplicates = self.df['user_id'].value_counts()
            duplicates = duplicates[duplicates > 1]
            
            if len(duplicates) > 0:
                print(f"\n⚠️  Found {len(duplicates)} user_ids with multiple survey responses:")
                print(duplicates.head(10))
            else:
                print("\n✓ Each user_id has exactly one survey response (1:1 mapping)")
        else:
            print("⚠️  'user_id' column not found in dataset")
            print("Available columns:", list(self.df.columns))
    
    def create_composite_groups(self):
        """Create composite demographic groups for analysis"""
        print("\n" + "="*80)
        print("COMPOSITE DEMOGRAPHIC GROUPS")
        print("="*80 + "\n")
        
        # Try to identify key demographic columns
        age_col = None
        education_col = None
        location_col = None
        
        for col in self.df.columns:
            col_lower = col.lower()
            if 'age' in col_lower and age_col is None:
                age_col = col
            if 'education' in col_lower and education_col is None:
                education_col = col
            if any(x in col_lower for x in ['country', 'region', 'location']) and location_col is None:
                location_col = col
        
        print("Identified columns for composite groups:")
        print(f"  Age: {age_col}")
        print(f"  Education: {education_col}")
        print(f"  Location: {location_col}")
        print()
        
        if age_col and education_col and location_col:
            # Create composite demographic group
            self.df['demographic_group'] = (
                self.df[location_col].astype(str) + "_" + 
                self.df[education_col].astype(str) + "_" + 
                self.df[age_col].astype(str)
            )
            
            print("Created composite 'demographic_group' column")
            print(f"Number of unique demographic groups: {self.df['demographic_group'].nunique()}")
            print()
            
            print("Top 20 demographic groups by size:")
            print("-"*80)
            group_counts = self.df['demographic_group'].value_counts().head(20)
            group_pcts = (group_counts / len(self.df) * 100).round(2)
            
            group_df = pd.DataFrame({
                'Count': group_counts,
                'Percentage': group_pcts
            })
            print(group_df)
            
            return age_col, education_col, location_col
        else:
            print("⚠️  Could not identify all required columns for composite groups")
            return None, None, None
    
    def analyze_cross_tabs(self):
        """Analyze cross-tabulations between demographic variables"""
        print("\n" + "="*80)
        print("CROSS-TABULATION ANALYSIS")
        print("="*80 + "\n")
        
        # Identify demographic columns
        demographic_cols = []
        for col in self.df.columns:
            col_lower = col.lower()
            if any(x in col_lower for x in ['age', 'gender', 'education', 'country', 'region']):
                if self.df[col].nunique() < 50:  # Only categorical with reasonable number of categories
                    demographic_cols.append(col)
        
        if len(demographic_cols) >= 2:
            # Create cross-tabs for first two demographic columns
            col1, col2 = demographic_cols[0], demographic_cols[1]
            
            print(f"Cross-tabulation: {col1} vs {col2}")
            print("-"*80)
            
            crosstab = pd.crosstab(self.df[col1], self.df[col2], margins=True)
            print(crosstab)
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(14, 10))
            
            # Remove margins for heatmap
            crosstab_no_margin = pd.crosstab(self.df[col1], self.df[col2])
            
            sns.heatmap(crosstab_no_margin, annot=True, fmt='d', 
                       cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Count'})
            ax.set_title(f'Cross-tabulation: {col1} vs {col2}', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel(col2, fontsize=12)
            ax.set_ylabel(col1, fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig("./prism_eda_plots/crosstab_heatmap.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            print("\n✓ Saved cross-tabulation heatmap")
    
    def summary_statistics(self):
        """Generate comprehensive summary statistics"""
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80 + "\n")
        
        summary = {
            'Total Participants': len(self.df),
            'Unique User IDs': self.df['user_id'].nunique() if 'user_id' in self.df.columns else 'N/A',
            'Number of Features': len(self.df.columns),
            'Memory Usage (MB)': self.df.memory_usage(deep=True).sum() / 1024**2
        }
        
        for key, value in summary.items():
            if isinstance(value, float):
                print(f"  {key:30s}: {value:.2f}")
            else:
                print(f"  {key:30s}: {value}")
        
        print()
    
    def generate_full_report(self, save_dir="./prism_eda_plots"):
        """Generate complete EDA report"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        print("\n" + "="*80)
        print("PRISM SURVEY DATASET - EXPLORATORY DATA ANALYSIS")
        print("="*80 + "\n")
        
        # Run all analyses
        self.basic_info()
        self.summary_statistics()
        self.analyze_user_distribution()
        self.analyze_demographics()
        self.create_composite_groups()
        self.plot_demographic_distributions(save_dir)
        self.analyze_cross_tabs()
        
        # Save dataframe info to text file
        report_path = os.path.join(save_dir, "eda_report.txt")
        with open(report_path, 'w') as f:
            f.write("PRISM SURVEY DATASET - EDA REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Total Participants: {len(self.df)}\n")
            f.write(f"Number of Features: {len(self.df.columns)}\n\n")
            
            f.write("Columns:\n")
            for col in self.df.columns:
                f.write(f"  - {col} ({self.df[col].dtype})\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("DEMOGRAPHIC DISTRIBUTIONS\n")
            f.write("="*80 + "\n\n")
            
            demographic_cols = [col for col in self.df.columns 
                              if any(x in col.lower() for x in ['age', 'gender', 'employment_status', 'education', 'marital_status', 
                              'english_proficiency', 'location', 'religion', 'ethnicity'])]
            
            for col in demographic_cols:
                f.write(f"\n{col}:\n")
                f.write("-"*40 + "\n")
                value_counts = self.df[col].value_counts().head(10)
                for val, count in value_counts.items():
                    pct = count / len(self.df) * 100
                    f.write(f"  {val}: {count} ({pct:.1f}%)\n")
        
        print(f"\n✓ Saved comprehensive report to {report_path}")
        print("="*80 + "\n")


def main():
    """Main execution function"""
    
    # Initialize EDA
    eda = PRISMSurveyEDA()
    
    # Generate full report
    eda.generate_full_report(save_dir="../../../data/prism_eda_plots")
    
    # Interactive exploration option
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nThe survey dataframe is available as: eda.df")
    print("You can explore it further in an interactive session.")
    print("\nExample explorations:")
    print("  - eda.df.head()")
    print("  - eda.df.describe()")
    print("  - eda.df['column_name'].value_counts()")
    print()
    
    return eda


if __name__ == "__main__":
    eda = main()