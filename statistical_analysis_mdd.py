#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
from pathlib import Path
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('default')
sns.set_palette("husl")

def load_actual_data():
    """Load the actual data from OverallCount of the Excel file"""
    xlsx_filename = 'risk_of_bias_statistical_tests.xlsx'
    sheet_name = 'OverallCount'
    
    # Check if file exists
    if not os.path.exists(xlsx_filename):
        print(f"Error: {xlsx_filename} not found in current directory!")
        print("Please ensure the Excel file is in the same folder as this script.")
        return None
    
    try:
        # Load Excel file OverallCount with proper column names
        df = pd.read_excel(xlsx_filename, sheet_name=sheet_name)
        
        # Fill empty Assessment field values with the previous value (forward fill)
        df['Assessment field'] = df['Assessment field'].fillna(method='ffill')
        
        # Clean up the data - handle mixed yes/no values in Factor-4
        df_clean = []
        
        for idx, row in df.iterrows():
            assessment_field = row['Assessment field']
            factor_type = row['TYPE']
            high_val = row['HIGH']
            moderate_val = row['MODERATE'] if pd.notna(row['MODERATE']) else 0
            low_val = row['LOW']
            
            # Skip Factor-4 (ML usage) - not needed for analysis
            if factor_type == 'Factor-4':
                continue
                
            # For all other factors, use the values as is
            df_clean.append({
                'Assessment_field': assessment_field,
                'Factor': factor_type,
                'Yes': 0,  # Not applicable for risk assessment factors
                'No': 0,   # Not applicable for risk assessment factors
                'High': high_val if pd.notna(high_val) else 0,
                'Moderate': moderate_val,
                'Low': low_val if pd.notna(low_val) else 0
            })
        
        # Convert to DataFrame
        df_processed = pd.DataFrame(df_clean)
        
        # Convert numeric columns
        df_processed['High'] = pd.to_numeric(df_processed['High'], errors='coerce').fillna(0)
        df_processed['Moderate'] = pd.to_numeric(df_processed['Moderate'], errors='coerce').fillna(0)
        df_processed['Low'] = pd.to_numeric(df_processed['Low'], errors='coerce').fillna(0)
        df_processed['Yes'] = pd.to_numeric(df_processed['Yes'], errors='coerce').fillna(0)
        df_processed['No'] = pd.to_numeric(df_processed['No'], errors='coerce').fillna(0)
        
        print(f"Successfully loaded and processed: {xlsx_filename} (Sheet: {sheet_name})")
        print("Data structure after processing:")
        print(df_processed)
        return df_processed
        
    except Exception as e:
        print(f"Error loading Excel file OverallCount: {e}")
        return None

# Sample data function removed - script will only work with actual data

def perform_statistical_analyses(df):
    """Perform comprehensive statistical analyses"""
    results = {}
    
    # 1. Chi-Square Test for Independence (Omics vs Neuroimaging vs Auxiliary)
    print("=" * 60)
    print("STATISTICAL ANALYSIS RESULTS")
    print("=" * 60)
    
    # Create contingency table for overall comparison - using average risk distribution
    omics_factors = df[df['Assessment_field'] == 'Omics']
    neuroimaging_factors = df[df['Assessment_field'] == 'Neuroimaging']
    auxiliary_factors = df[df['Assessment_field'] == 'Auxiliary']
    
    # Calculate average risk distribution per field (treating each factor as equal weight)
    omics_data = omics_factors[['Low', 'Moderate', 'High']].mean().round(0).astype(int)
    neuroimaging_data = neuroimaging_factors[['Low', 'Moderate', 'High']].mean().round(0).astype(int)
    auxiliary_data = auxiliary_factors[['Low', 'Moderate', 'High']].mean().round(0).astype(int)
    
    contingency_table = pd.DataFrame({
        'Omics': omics_data.values,
        'Neuroimaging': neuroimaging_data.values,
        'Auxiliary': auxiliary_data.values
    }, index=['Low', 'Moderate', 'High'])
    
    print("\n1. CONTINGENCY TABLE (Omics vs Neuroimaging vs Auxiliary):")
    print(contingency_table)
    
    # Chi-square test for all three fields (handle zero values)
    try:
        chi2_stat, chi2_p_value = stats.chi2_contingency(contingency_table)[:2]
        results['chi2_statistic'] = chi2_stat
        results['chi2_p_value'] = chi2_p_value
    except ValueError as e:
        print(f"Chi-square test failed due to zero values: {e}")
        print("Using alternative analysis approach...")
        # Use only non-zero rows for analysis
        non_zero_table = contingency_table.loc[contingency_table.sum(axis=1) > 0, 
                                             contingency_table.sum(axis=0) > 0]
        if non_zero_table.shape[0] > 1 and non_zero_table.shape[1] > 1:
            chi2_stat, chi2_p_value = stats.chi2_contingency(non_zero_table)[:2]
            results['chi2_statistic'] = chi2_stat
            results['chi2_p_value'] = chi2_p_value
        else:
            print("Insufficient non-zero data for chi-square test")
            results['chi2_statistic'] = 0.0
            results['chi2_p_value'] = 1.0
    
    print(f"\nChi-Square Test Results (All Three Fields):")
    print(f"Chi-Square Statistic: {chi2_stat:.4f}")
    print(f"P-value: {chi2_p_value:.6f}")
    print(f"Significance: {'Significant' if chi2_p_value < 0.05 else 'Not Significant'} (α=0.05)")
    
    # Pairwise comparisons
    print(f"\n2. PAIRWISE COMPARISONS:")
    
    # Helper function for safe chi-square test
    def safe_chi2_test(table, name):
        try:
            # Remove zero rows/columns
            non_zero_table = table.loc[table.sum(axis=1) > 0, table.sum(axis=0) > 0]
            if non_zero_table.shape[0] > 1 and non_zero_table.shape[1] > 1:
                chi2, p_val = stats.chi2_contingency(non_zero_table)[:2]
                return chi2, p_val
            else:
                print(f"  Warning: {name} - insufficient non-zero data for chi-square test")
                return 0.0, 1.0
        except ValueError as e:
            print(f"  Warning: {name} - chi-square test failed: {e}")
            return 0.0, 1.0
    
    # Omics vs Neuroimaging
    omics_vs_neuro_table = contingency_table[['Omics', 'Neuroimaging']]
    chi2_om_ne, p_om_ne = safe_chi2_test(omics_vs_neuro_table, "Omics vs Neuroimaging")
    print(f"\nOmics vs Neuroimaging:")
    print(f"  Chi-Square: {chi2_om_ne:.4f}, P-value: {p_om_ne:.6f}")
    print(f"  Significance: {'Yes' if p_om_ne < 0.05 else 'No'}")
    
    # Omics vs Auxiliary
    omics_vs_aux_table = contingency_table[['Omics', 'Auxiliary']]
    chi2_om_au, p_om_au = safe_chi2_test(omics_vs_aux_table, "Omics vs Auxiliary")
    print(f"\nOmics vs Auxiliary:")
    print(f"  Chi-Square: {chi2_om_au:.4f}, P-value: {p_om_au:.6f}")
    print(f"  Significance: {'Yes' if p_om_au < 0.05 else 'No'}")
    
    # Neuroimaging vs Auxiliary
    neuro_vs_aux_table = contingency_table[['Neuroimaging', 'Auxiliary']]
    chi2_ne_au, p_ne_au = safe_chi2_test(neuro_vs_aux_table, "Neuroimaging vs Auxiliary")
    print(f"\nNeuroimaging vs Auxiliary:")
    print(f"  Chi-Square: {chi2_ne_au:.4f}, P-value: {p_ne_au:.6f}")
    print(f"  Significance: {'Yes' if p_ne_au < 0.05 else 'No'}")
    
    # Store pairwise results
    results['pairwise_comparisons'] = {
        'omics_vs_neuroimaging': {'chi2': chi2_om_ne, 'p_value': p_om_ne, 'significant': p_om_ne < 0.05},
        'omics_vs_auxiliary': {'chi2': chi2_om_au, 'p_value': p_om_au, 'significant': p_om_au < 0.05},
        'neuroimaging_vs_auxiliary': {'chi2': chi2_ne_au, 'p_value': p_ne_au, 'significant': p_ne_au < 0.05}
    }
    return results

def create_visualizations(df, results):
    """Create individual high-quality visualizations and save detailed data"""
    print("\n" + "=" * 60)
    print("CREATING INDIVIDUAL HIGH-QUALITY VISUALIZATIONS")
    print("=" * 60)
    
    # Create results folder structure
    results_folder = "MDD_Risk_of_Bias_Results"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    # Create subfolders for different types of outputs
    plots_folder = os.path.join(results_folder, "Plots")
    data_folder = os.path.join(results_folder, "Data")
    
    for folder in [plots_folder, data_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)
    
    print(f"Results will be saved in: {results_folder}/")
    
    # Set publication-quality plotting parameters
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'dejavuserif'
    
    # Get data for all visualizations - calculate actual study counts
    # For each field, calculate the average risk distribution across all factors
    omics_factors = df[df['Assessment_field'] == 'Omics']
    neuroimaging_factors = df[df['Assessment_field'] == 'Neuroimaging']
    auxiliary_factors = df[df['Assessment_field'] == 'Auxiliary']
    
    # Calculate average risk distribution per field (treating each factor as equal weight)
    omics_data = omics_factors[['Low', 'Moderate', 'High']].mean().round(0).astype(int)
    neuroimaging_data = neuroimaging_factors[['Low', 'Moderate', 'High']].mean().round(0).astype(int)
    auxiliary_data = auxiliary_factors[['Low', 'Moderate', 'High']].mean().round(0).astype(int)
    
    
    # 1. Comprehensive Factor-wise Stacked Bar Chart (NEW)
    plt.figure(figsize=(22, 14))
    factors = ['Factor-1', 'Factor-2', 'Factor-3', 'Factor-5']
    factor_labels = ['Research Quality', 'Sample Size', 'Clinical Tests', 'Overall Quality']
    
    # Prepare data for each factor
    omics_low_counts = []
    omics_moderate_counts = []
    omics_high_counts = []
    neuro_low_counts = []
    neuro_moderate_counts = []
    neuro_high_counts = []
    aux_low_counts = []
    aux_moderate_counts = []
    aux_high_counts = []
    
    for factor in factors:
        try:
            omics_factor_data = df[(df['Assessment_field'] == 'Omics') & (df['Factor'] == factor)]
            neuro_factor_data = df[(df['Assessment_field'] == 'Neuroimaging') & (df['Factor'] == factor)]
            aux_factor_data = df[(df['Assessment_field'] == 'Auxiliary') & (df['Factor'] == factor)]
            
            # Standard handling for all factors
            if not omics_factor_data.empty and not neuro_factor_data.empty and not aux_factor_data.empty:
                omics_low_counts.append(omics_factor_data['Low'].iloc[0])
                omics_moderate_counts.append(omics_factor_data['Moderate'].iloc[0])
                omics_high_counts.append(omics_factor_data['High'].iloc[0])
                
                neuro_low_counts.append(neuro_factor_data['Low'].iloc[0])
                neuro_moderate_counts.append(neuro_factor_data['Moderate'].iloc[0])
                neuro_high_counts.append(neuro_factor_data['High'].iloc[0])
                
                aux_low_counts.append(aux_factor_data['Low'].iloc[0])
                aux_moderate_counts.append(aux_factor_data['Moderate'].iloc[0])
                aux_high_counts.append(aux_factor_data['High'].iloc[0])
            else:
                omics_low_counts.append(0)
                omics_moderate_counts.append(0)
                omics_high_counts.append(0)
                neuro_low_counts.append(0)
                neuro_moderate_counts.append(0)
                neuro_high_counts.append(0)
                aux_low_counts.append(0)
                aux_moderate_counts.append(0)
                aux_high_counts.append(0)
        except (IndexError, KeyError):
            omics_low_counts.append(0)
            omics_moderate_counts.append(0)
            omics_high_counts.append(0)
            neuro_low_counts.append(0)
            neuro_moderate_counts.append(0)
            neuro_high_counts.append(0)
            aux_low_counts.append(0)
            aux_moderate_counts.append(0)
            aux_high_counts.append(0)
    
    # Ensure we have data to plot
    if len(omics_low_counts) > 0 and len(neuro_low_counts) > 0 and len(aux_low_counts) > 0:
        # Create horizontal stacked bar chart
        y_pos = np.arange(len(factors))
        height = 0.25
        
        # Omics bars
        bars1_omics = plt.barh(y_pos - height, omics_low_counts, height, 
                               label='Omics - High Risk', color='#DC143C', alpha=0.9)
        bars2_omics = plt.barh(y_pos - height, omics_moderate_counts, height, 
                               left=omics_low_counts, label='Omics - Moderate Risk', 
                               color='#FFD700', alpha=0.9)
        bars3_omics = plt.barh(y_pos - height, omics_high_counts, height, 
                               left=[i+j for i,j in zip(omics_low_counts, omics_moderate_counts)], 
                               label='Omics - Low Risk', color='#32ba39', alpha=0.9)
        
        # Neuroimaging bars
        bars1_neuro = plt.barh(y_pos, neuro_low_counts, height, 
                               label='Neuroimaging - High Risk', color='#DC143C', alpha=0.9)
        bars2_neuro = plt.barh(y_pos, neuro_moderate_counts, height, 
                               left=neuro_low_counts, label='Neuroimaging - Moderate Risk', 
                               color='#FFD700', alpha=0.9)
        bars3_neuro = plt.barh(y_pos, neuro_high_counts, height, 
                               left=[i+j for i,j in zip(neuro_low_counts, neuro_moderate_counts)], 
                               label='Neuroimaging - Low Risk', color='#32ba39', alpha=0.9)
        
        # Auxiliary bars
        bars1_aux = plt.barh(y_pos + height, aux_low_counts, height, 
                             label='Auxiliary - High Risk', color='#DC143C', alpha=0.9)
        bars2_aux = plt.barh(y_pos + height, aux_moderate_counts, height, 
                             left=aux_low_counts, label='Auxiliary - Moderate Risk', 
                             color='#FFD700', alpha=0.9)
        bars3_aux = plt.barh(y_pos + height, aux_high_counts, height, 
                             left=[i+j for i,j in zip(aux_low_counts, aux_moderate_counts)], 
                             label='Auxiliary - Low Risk', color='#32ba39', alpha=0.9)
        
        # Add outline around each complete bar group
        for i in range(len(factors)):
            omics_total = omics_low_counts[i] + omics_moderate_counts[i] + omics_high_counts[i]
            neuro_total = neuro_low_counts[i] + neuro_moderate_counts[i] + neuro_high_counts[i]
            aux_total = aux_low_counts[i] + aux_moderate_counts[i] + aux_high_counts[i]
            
            # Outline for Omics bar
            if omics_total > 0:
                plt.barh(y_pos[i] - height, omics_total, height, 
                        color='none', edgecolor='black', linewidth=1, alpha=1)
                        
            # Outline for Neuroimaging bar
            if neuro_total > 0:
                plt.barh(y_pos[i], neuro_total, height, 
                        color='none', edgecolor='black', linewidth=1, alpha=1)
                        
            # Outline for Auxiliary bar
            if aux_total > 0:
                plt.barh(y_pos[i] + height, aux_total, height, 
                        color='none', edgecolor='black', linewidth=1, alpha=1)
        
        # Add field labels outside bars but inside plot boundary
        for i in range(len(factors)):
            # Calculate total width for each field to position labels properly
            omics_total = omics_low_counts[i] + omics_moderate_counts[i] + omics_high_counts[i]
            neuro_total = neuro_low_counts[i] + neuro_moderate_counts[i] + neuro_high_counts[i]
            aux_total = aux_low_counts[i] + aux_moderate_counts[i] + aux_high_counts[i]
            
            # Add field labels just outside each bar group within extended plot area
            if omics_total > 0:
                plt.text(omics_total + 0.5, y_pos[i] - height, 'Omics', ha='left', va='center', 
                        color='black', fontweight='normal', fontsize=20)
                        
            if neuro_total > 0:
                plt.text(neuro_total + 0.5, y_pos[i], 'Neuroimaging', ha='left', va='center', 
                        color='black', fontweight='normal', fontsize=20)
                        
            if aux_total > 0:
                plt.text(aux_total + 0.5, y_pos[i] + height, 'Auxiliary', ha='left', va='center', 
                        color='black', fontweight='normal', fontsize=20)
        
        plt.xlabel('NUMBER OF STUDIES', fontsize=20, fontweight='bold', labelpad=15)
        plt.ylabel('ASSESSMENT FACTORS', fontsize=20, fontweight='bold')
        plt.yticks(y_pos, factor_labels, fontsize=18, fontweight='bold')
        plt.xticks(fontsize=18, fontweight='bold')  # X-axis numerical labels: +1 size and bold
        
        # Create custom legend positioned at bottom with horizontal alignment
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#DC143C', label='High'),
            Patch(facecolor='#FFD700', label='Moderate'),
            Patch(facecolor='#32ba39', label='Low')
        ]
        plt.legend(handles=legend_elements, title='Risk Level',
                   title_fontproperties={'weight': 'bold', 'size': 22}, fontsize=22, loc='lower left', 
                   bbox_to_anchor=(0.0, -0.42, 1.0, 0.20), ncol=3,
                   frameon=False, fancybox=False, shadow=False, mode='expand')
        
        plt.grid(axis='x', alpha=0.3)
        
        # Set x-axis limit to 50
        plt.xlim(0, 50)  # Fixed x-axis limit
        
        # Save comprehensive factor comparison data
        comprehensive_factor_data = []
        for i, factor in enumerate(factors):
            comprehensive_factor_data.extend([
                {'Factor': factor, 'Factor_Label': factor_labels[i], 'Field': 'Omics', 
                 'Risk_Level': 'High Risk', 'Count': omics_low_counts[i]},
                {'Factor': factor, 'Factor_Label': factor_labels[i], 'Field': 'Omics', 
                 'Risk_Level': 'Moderate Risk', 'Count': omics_moderate_counts[i]},
                {'Factor': factor, 'Factor_Label': factor_labels[i], 'Field': 'Omics', 
                 'Risk_Level': 'Low Risk', 'Count': omics_high_counts[i]},
                {'Factor': factor, 'Factor_Label': factor_labels[i], 'Field': 'Neuroimaging', 
                 'Risk_Level': 'High Risk', 'Count': neuro_low_counts[i]},
                {'Factor': factor, 'Factor_Label': factor_labels[i], 'Field': 'Neuroimaging', 
                 'Risk_Level': 'Moderate Risk', 'Count': neuro_moderate_counts[i]},
                {'Factor': factor, 'Factor_Label': factor_labels[i], 'Field': 'Neuroimaging', 
                 'Risk_Level': 'Low Risk', 'Count': neuro_high_counts[i]},
                {'Factor': factor, 'Factor_Label': factor_labels[i], 'Field': 'Auxiliary', 
                 'Risk_Level': 'High Risk', 'Count': aux_low_counts[i]},
                {'Factor': factor, 'Factor_Label': factor_labels[i], 'Field': 'Auxiliary', 
                 'Risk_Level': 'Moderate Risk', 'Count': aux_moderate_counts[i]},
                {'Factor': factor, 'Factor_Label': factor_labels[i], 'Field': 'Auxiliary', 
                 'Risk_Level': 'Low Risk', 'Count': aux_high_counts[i]}
            ])
        
        comprehensive_factor_df = pd.DataFrame(comprehensive_factor_data)
        comprehensive_factor_df.to_csv(os.path.join(data_folder, 'comprehensive_factor_comparison_data.csv'), index=False)
        print("Comprehensive factor comparison data saved to: Data/comprehensive_factor_comparison_data.csv")
        
        plt.tight_layout()
        # Adjust layout to accommodate bottom legend with more spacing
        plt.subplots_adjust(bottom=0.32)  # Increased space for bottom legend with extra spacing
        plt.savefig(os.path.join(plots_folder, 'comprehensive_factor_comparison.png'), dpi=300, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        plt.close()
        print("Comprehensive factor comparison saved as: Plots/comprehensive_factor_comparison.png")
    else:
        print("Warning: No data available for comprehensive factor comparison visualization")
        # Create empty CSV file
        comprehensive_factor_data = []
        comprehensive_factor_df = pd.DataFrame(comprehensive_factor_data)
        comprehensive_factor_df.to_csv(os.path.join(data_folder, 'comprehensive_factor_comparison_data.csv'), index=False)
        print("Empty comprehensive factor comparison data saved to: Data/comprehensive_factor_comparison_data.csv")
    
    return results_folder

def save_results_to_csv(results, df, results_folder):
    """Save all results for future analysis"""
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)
    
    # Create subfolders if they don't exist
    data_folder = os.path.join(results_folder, "Data")
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    
    # 1. Save main statistical results
    main_results = {
        'Analysis_Type': ['Chi-Square Test (Overall)', 'Chi-Square Test (Overall)'],
        'Statistic': ['Chi-Square Statistic', 'P-Value'],
        'Value': [results['chi2_statistic'], results['chi2_p_value']],
        'Significance': ['Significant' if results['chi2_p_value'] < 0.05 else 'Not Significant'] * 2
    }
    
    main_results_df = pd.DataFrame(main_results)
    main_results_df.to_csv(os.path.join(data_folder, 'statistical_results_main.csv'), index=False)
    print("Main statistical results saved to: Data/statistical_results_main.csv")
    
   
    print(f"\nAll results saved in organized folder structure: {results_folder}/")

def main():
    """Main function to run the complete analysis"""
    print("MDD Risk-of-Bias Statistical Analysis")
    print("=" * 60)
    
    # Try to load the actual CSV data first
    print("Attempting to load actual data...")
    df = load_actual_data()
    
    # If actual data loading fails, show error and exit
    if df is None:
        print("\nERROR: Could not load actual data!")
        print("The Excel file exists but doesn't contain the expected study count data.")
        print("Expected format: Assessment_field | Factor | Low | Moderate | High | Yes | No")
        print("Please provide the actual study count data file.")
        return None
    else:
        print("Successfully loaded actual data from Excel file OverallCount!")
    
    if df is not None:
        print(f"\nDataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        print("\nFirst few rows of data:")
        print(df.head())
        
        # Perform statistical analyses
        results = perform_statistical_analyses(df)
        
        # Create visualizations
        results_folder = create_visualizations(df, results)
        
        # Save results to CSV files
        save_results_to_csv(results, df, results_folder)
    else:
        print("\nAnalysis terminated due to data loading issues.")
        print("Please provide a properly formatted data file with actual study counts.")
        return
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)
    print(f"Results organized in folder: {results_folder}/")
    print("\nFolder Structure:")
    print(f"├── {results_folder}/")
    print("│   ├── Plots/          # High-quality PNG visualizations")
    print("│   ├── Data/           # CSV files for external plotting")
    print("│   └── README.md       # Complete documentation")
    print("\nFiles generated:")
    print("📊 PLOTS (PNG files):")
    print("   1. comprehensive_factor_comparisonPercentage.png")
    print("\n📁 DATA (CSV files):")
    print("   1. comprehensive_factor_comparison_data.csv")
    print("   2. comprehensive_factor_comparison_percentage_data.csv")
    print("   3. statistical_results_main.csv")
    print(f"\nAll files are organized in: {results_folder}/")

if __name__ == "__main__":
    main()
