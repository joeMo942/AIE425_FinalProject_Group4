# Final Comparison: PCA Mean-Filling vs PCA MLE

## Overview

This document provides a comprehensive comparison of two PCA-based dimensionality reduction methods for rating prediction in a recommendation system.

**Dataset:** Amazon Movies & TV Reviews
- **Users:** 147,914
- **Items:** 11,123
- **Ratings:** 2,149,655

**Target Users:** U1=134471, U2=27768, U3=16157
**Target Items:** I1=1333 (mean=3.69), I2=1162 (mean=3.74)

---

## 1. Outcomes

### Part 1: PCA with Mean-Filling

| Configuration | Avg Error | Min Error | Max Error |
|--------------|-----------|-----------|-----------|
| Top-5 PCs | 0.507 | 0.29 | 0.84 |
| Top-10 PCs | 0.385 | 0.14 | 0.50 |

**Key Findings:**
- Top-10 PCs achieves 24% lower error than Top-5 PCs
- Variance explained: Top-5 = 1.37%, Top-10 = 2.30%
- All predictions are below item mean (downward bias)
- High neighbor similarity (0.92-1.0) in reduced space

---

### Part 2: PCA with MLE

| Configuration | Avg Error | Min Error | Max Error |
|--------------|-----------|-----------|-----------|
| Top-5 PCs | 0.033 | 0.00 | 0.07 |
| Top-10 PCs | 0.013 | 0.00 | 0.03 |

**Key Findings:**
- Top-10 PCs achieves 60% lower error than Top-5 PCs
- Variance explained: Top-5 = 13.84%, Top-10 = 21.86%
- Predictions cluster around item mean (stable)
- Near-perfect predictions (3 out of 6 with 0.00 error using Top-10)

---

### Cross-Method Comparison

| Comparison | Mean-Fill Error | MLE Error | MLE Improvement |
|------------|-----------------|-----------|-----------------|
| Top-5 PCs | 0.507 | 0.033 | **93% better** |
| Top-10 PCs | 0.385 | 0.013 | **97% better** |

**MLE wins in ALL 12 prediction cases.**

---

## 2. Summary and Comparison

### Prediction Accuracy

| Metric | Part 1 (Mean-Fill) | Part 2 (MLE) | Winner |
|--------|-------------------|--------------|--------|
| Best Avg Error | 0.385 (Top-10) | **0.013** (Top-10) | MLE |
| Best Single Prediction | 0.14 | **0.00** | MLE |
| Worst Prediction | 0.84 | 0.07 | MLE |
| Zero-Error Predictions | 0 | 3 | MLE |

### Variance Captured

| PCs | Mean-Fill | MLE |
|-----|-----------|-----|
| Top-5 | 1.37% | **13.84%** |
| Top-10 | 2.30% | **21.86%** |

MLE captures **~10x more variance** than Mean-Fill with the same number of principal components.

---

### Pros and Cons

#### Part 1: PCA with Mean-Filling

**Pros:**
- Simple implementation
- Consistent approach to handling missing data
- Works with complete matrices

**Cons:**
- Artificial values distort covariance structure
- Low variance explained (1-2%)
- Downward prediction bias (all predictions below mean)
- Poor accuracy (avg error 0.39-0.51)
- k-NN vulnerable to neighbor selection bias

---

#### Part 2: PCA with MLE

**Pros:**
- Uses only observed data (statistically rigorous)
- High variance explained (14-22%)
- Excellent accuracy (avg error 0.01-0.03)
- Predictions naturally regress to item mean
- Reconstruction formula is robust

**Cons:**
- Computationally more expensive
- Zero covariance for item pairs with <2 common users
- May underestimate covariance for rare item pairs

---

## 3. Conclusion

### Impact of Maximum Likelihood Estimation

The PCA MLE method provides a **dramatic improvement** over the traditional mean-filling approach:

1. **93-97% Error Reduction:** MLE achieves near-perfect predictions compared to Mean-Fill's significant errors.

2. **10x Better Variance Capture:** By using only observed data, MLE preserves the true statistical structure of the rating matrix, capturing 13.84% (Top-5) vs 1.37% variance.

3. **Elimination of Prediction Bias:** Mean-Fill shows consistent downward bias (all predictions below mean), while MLE predictions center around the true item mean.

4. **Statistical Rigor:** MLE's approach of dividing by the number of common raters (not total users) produces a covariance matrix that better reflects actual user behavior.

### Recommendation

**PCA MLE with Top-10 Principal Components** is the recommended approach for rating prediction, achieving:
- Average error of only 0.013
- 3 out of 6 predictions with zero error
- 97% improvement over Mean-Fill

The computational overhead of MLE is justified by its significantly superior prediction accuracy and statistical validity.

### Final Verdict

| Aspect | Winner |
|--------|--------|
| Prediction Accuracy | **MLE** |
| Variance Explained | **MLE** |
| Statistical Validity | **MLE** |
| Computational Simplicity | Mean-Fill |
| Overall | **MLE** |

**Maximum Likelihood Estimation fundamentally transforms PCA-based recommendation** by respecting the observed data structure, leading to dramatically better rating predictions.
