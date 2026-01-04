# SVD Analysis Report

## Overview
**SVD Analysis for Collaborative Filtering** - Full Dataset Implementation

---

## Dataset Statistics

| Metric | Value |
|--------|-------|
| **Users** | 147,914 |
| **Items** | 11,123 |
| **Global Mean Rating** | 3.7432 |
| **Latent Factors (k)** | 100 |

---

## Section 2: SVD Decomposition Results

### Variance Explained
| Top-k | Cumulative Variance |
|-------|-------------------|
| 10 | ~18.8% |
| 20 | ~31.2% |
| 50 | ~61.6% |
| 75 | ~82.0% |
| 100 | 100% (of computed) |

### Orthogonality Verification
- **U^T @ U ≈ I**: ✓ Verified (max error ~10⁻¹⁴)
- **V^T @ V ≈ I**: ✓ Verified (max error ~10⁻¹⁴)

### Visualizations
- ![SVD Full Analysis](../plots/SVD_full_analysis.png)

---

## Section 3: Truncated SVD Evaluation

### Reconstruction Error (k=100)
| Metric | Value |
|--------|-------|
| MAE (mean-centered) | 0.5779 |
| RMSE | 0.7807 |

---

## Section 4: Rating Prediction

### Target Users & Items
Predictions made for target users on target items using:
- **k = 100** latent factors
- **Mean-centered approach** (global mean = 3.7432)
- Clipped to [1.0, 5.0] range

---

## Section 5: SVD vs PCA Comparison

See: [Part3_Discussion_Conclusion.md](Part3_Discussion_Conclusion.md)

---

## Section 6: Latent Factor Interpretation

Top latent factors extracted and visualized:
- First 3 components represent primary rating patterns
- ![Latent Space](../plots/latent_space_visualization.png)

---

## Section 7: Sensitivity Analysis

### Error vs k Value
| k | MAE | RMSE | Variance |
|---|-----|------|----------|
| 10 | 0.6475 | 0.8305 | 18.8% |
| 20 | 0.6375 | 0.8239 | 31.2% |
| 50 | 0.6111 | 0.8057 | 61.6% |
| 75 | 0.5937 | 0.7929 | 82.0% |
| 100 | 0.5779 | 0.7807 | 100% |

### Visualization
- ![Sensitivity Analysis](../plots/SVD_sensitivity_analysis.png)

---

## Section 8: Cold-Start Analysis

### User Distribution
| Type | Count | MAE |
|------|-------|-----|
| Cold-start (≤5 ratings) | 78,983 | 0.7069 |
| Warm (≥20 ratings) | 29,104 | 0.4185 |

### Cold-Start Penalty
- **+68.9% higher error** for cold-start users

### Mitigation Strategy
- Use item popularity weighting for cold-start users

### Visualization
- ![Cold-Start Analysis](../plots/SVD_cold_start_analysis.png)

---

## Output Files

### Results Directory
| File | Description |
|------|-------------|
| `full_svd_singular_values.csv` | Singular values, eigenvalues, variance |
| `sensitivity_robustness_full.csv` | Error at different k values |
| `cold_start_analysis_full.csv` | Cold vs warm user comparison |
| `SVD_terminal_output.txt` | Full terminal output |

### Plots Directory
| File | Description |
|------|-------------|
| `SVD_full_analysis.png` | Singular values & variance plots |
| `SVD_sensitivity_analysis.png` | Error vs k comparison |
| `SVD_cold_start_analysis.png` | Cold-start user analysis |
| `latent_space_visualization.png` | Latent factor visualization |
