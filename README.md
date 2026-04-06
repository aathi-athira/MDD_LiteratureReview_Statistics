# MDD Risk-of-Bias Statistical Analysis

## Project Overview

This project conducts a comprehensive statistical analysis to evaluate **risk of bias** across different assessment methodologies used in Major Depressive Disorder (MDD) research. The analysis compares the quality and reliability of three major assessment fields through rigorous statistical testing and data visualization.

## Purpose

The main objective is to determine whether there are statistically significant differences in research quality (risk of bias) between:

- **Omics** approaches (Genomics, Proteomics, Metabolomics, Pharmacology)
- **Neuroimaging** techniques (EEG, NIRS, MRI/fMRI, DTI)
- **Auxiliary** methods (Scales/Questionnaires, Digital Tools, Therapy, ML/AI)

## Project Structure

```
RiskOfBiasOverall/
 statistical_analysis_mdd.py        # Main analysis script
 risk_of_bias_statistical_tests.xlsx # Input data (study counts by risk level)
 Overview.md                         # Literature review summary
 README.md                          # This file
 MDD_Risk_of_Bias_Results/          # Generated results
     Plots/
        comprehensive_factor_comparison.png
     Data/
         comprehensive_factor_comparison_data.csv
         statistical_results_main.csv
```

## Requirements

### System Requirements

- Python 3.7+
- Windows/macOS/Linux

### Python Dependencies

```bash
pip install pandas numpy matplotlib seaborn scipy openpyxl
```

Or create requirements.txt:

```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
scipy>=1.7.0
openpyxl>=3.0.0
```

## Usage

### 1. **Prepare Data**

Ensure `risk_of_bias_statistical_tests.xlsx` contains the study count data in the "OverallCount" sheet with columns:

- `Assessment field`: Omics, Neuroimaging, Auxiliary
- `TYPE`: Factor-1, Factor-2, Factor-3, Factor-5
- `HIGH`, `MODERATE`, `LOW`: Study counts for each risk level

### 2. **Run Analysis**

```bash
python statistical_analysis_mdd.py
```

### 3. **View Results**

Results are automatically saved in `MDD_Risk_of_Bias_Results/` folder.

## Analysis Components

### Risk Assessment Factors

The analysis evaluates **4 key quality factors**:

| Factor       | Description      | Purpose                           |
| ------------ | ---------------- | --------------------------------- |
| **Factor-1** | Research Quality | Overall methodological rigor      |
| **Factor-2** | Sample Size      | Adequacy of study sample sizes    |
| **Factor-3** | Clinical Tests   | Use of proper clinical validation |
| **Factor-4** | ML Usage(Yes/No) | Use of Machine Learning automation |
| **Factor-5** | Overall Quality  | Comprehensive quality assessment  |

※ Please note that **Factor-4** is being used in the qualitive assessment analysis and not in the risk of bias calculation/visualization.

### Risk Levels

Each factor is rated on **3 risk levels**:

- **High Risk** - Poor quality/high bias risk
- **Moderate Risk** - Acceptable quality/moderate bias
- **Low Risk** - Excellent quality/minimal bias

## Generated Outputs

### 1. **Statistical Results** (`statistical_results_main.csv`)

```
Analysis_Type,Statistic,Value,Significance
Chi-Square Test (Overall),Chi-Square Statistic,0.9284477325158028,Not Significant
Chi-Square Test (Overall),P-Value,0.920444523146606,Not Significant
```

### 2. **Visualization** (`comprehensive_factor_comparison.png`)

- Horizontal stacked bar chart (22x14 inches, 300 DPI)
- Factor-wise comparison across all assessment fields
- Color-coded risk levels with study counts

## Technical Details

### Visualization Features

- Publication-quality figures (300 DPI)
- Color-blind friendly palette (#DC143C, #FFD700, #32ba39)
- Professional serif typography
- Horizontal stacked bars with clear legends
- Field labels and risk level indicators

## Important Notes

### Data Requirements

- Excel file must be in same directory as script
- "OverallCount" sheet must exist with proper structure

## License

This project is part of academic research on Major Depressive Disorder assessment methodologies.

---

**Project**: MDD Risk-of-Bias Assessment Analysis  
**Created**: 2025  
**Dependencies**: Python 3.7+, pandas, numpy, matplotlib, seaborn, scipy
**Output**: Statistical analysis + publication-quality visualizations
