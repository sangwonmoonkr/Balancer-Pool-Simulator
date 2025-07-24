# Dynamic Fee Formula Comparison

This document explains a dynamic fee formula that combines linear and quadratic terms, as presented in an image, and compares it to a purely linear fee model. It covers the formula's structure, its purpose, and the differences between the two approaches with examples. Mathematical expressions are written in plain text for compatibility.

## Understanding the Fee Formula

The fee formula from the image is applied when an action (e.g., deposit or swap) increases the deviation from an optimal state (postdiff - curdiff > 0):

```text
fee = base_fee + (postdiff - curdiff) * slope + (1/2) * (postdiff - curdiff)^2 * slope^2
```
### Components
- **postdiff - curdiff**: The change in the difference (e.g., a distance metric) between the current state and an optimal state before and after an action. Let's call this Delta D.
- **slope**: A scaling factor that controls the fee's rate of increase (e.g., slope = 0.01).
- **slope^2**: Part of the quadratic term, amplifying the fee for larger deviations (slope^2 means slope squared).
- **1/2**: A constant, often arising in quadratic optimization or Taylor expansion.

### Structure
- **Linear Term**: (postdiff - curdiff) * slope
  - Provides a baseline fee proportional to the deviation. Small deviations incur a modest fee.
- **Quadratic Term**: (1/2) * (postdiff - curdiff)^2 * slope^2
  - Grows quadratically, making the fee increase rapidly for larger deviations, discouraging significant deviations.

### Purpose
- **Quadratic Form**: Suggests derivation from a cost function (e.g., least squares optimization), common in balancing systems.
- **Balance**: The linear term ensures fairness for small deviations, while the quadratic term penalizes large deviations.
- **Flexibility**: The slope parameter allows adjustment of the fee's steepness.

## Model Overview Compared with Balancer V3 StableSurge

Balancer V3 StableSurge is a new specialized AMM designed for stable assets with improved capital efficiency and dynamic fee mechanisms. The StableSurge pools utilize specialized pricing functions optimized for low-slippage trades while maintaining tight price ranges for stable assets. The dynamic fee mechanism automatically adjusts swap fees based on pool imbalance levels to incentivize rebalancing actions.

Our simulation evaluates how our proposed dynamic fee formula compares to Balancer V3's existing fee structure using historical transaction data from live pools.

<details>
<summary>Dataset Details</summary>

## Dataset Overview
- Total Transactions: 415,076
- Number of Pools: 91
- Number of Chains: 5
- Pools per Chain:
  | chain     | Pool Count |
  |-----------|------------|
  | arbitrum  | 14         |
  | avalanche | 8          |
  | base      | 60         |
  | ethereum  | 7          |
  | gnosis    | 2          |

</details>

### Model Parameters
- Slope: 0.5
- Max Fee: 0.95

## Statistically Verified Benefits

### Fee Level Alignment
- Average Actual Fee Rate: 0.2070% vs. Dynamic Fee Rate: 0.1510% (–27.1%)
- Median Actual Fee Rate: 0.0100% vs. Median Dynamic Fee Rate: 0.0173% (+73%)

### Smoothed Dispersion
- Actual σ: 0.017602 → Dynamic σ: 0.015524
- Reduction: 11.8% tighter fee swings

### Balanced Imbalance Penalty
- ΔD > 0 (50.66% of trades): Actual = 0.2528% vs. Dynamic = 0.1768%
- ΔD < 0 (49.34% of trades): Actual = 0.1599% vs. Dynamic = 0.1246%

### Revenue Impact
- Total Actual Fees: 137,682.27 units
- Total Dynamic Fees: 726,044.03 units
- Change: +427.33%

## Visualization Results

### Fee vs Imbalance Relationships

This chart shows how fees relate to pool imbalances (ΔD), comparing actual fees from historical data (left) with our dynamic fee model (right).

![ΔD vs Fee Relationship](results/analysis/all/deltaD_vs_fees.png)

### Average Fee Comparison by Imbalance Direction

This visualization compares average fee rates between imbalance-increasing trades (ΔD>0), rebalancing trades (ΔD<0), and overall averages.

![ΔD Fee Comparison](results/analysis/all/deltaD_fee_comparison.png)

### Fee Rate Difference vs Imbalance Level

This scatter plot visualizes how fee rate differences between dynamic and actual relate to imbalance levels (ΔD).

![Fee Difference vs ΔD](results/analysis/all/deltaD_fee_error.png)

### Fee Change Distribution

This visualization shows the distribution of transactions where dynamic fees are higher, lower, or equal to actual fees.

![Fee Change Distribution](results/analysis/all/fee_change_distribution.png)

### Fee Distribution Comparison

This chart shows how fees are distributed across different fee bins, comparing the frequency of actual vs. dynamic fees.

![Fee Bin Comparison](results/analysis/all/fee_bin_comparison.png)

### Fee Correlation Analysis

The scatter plot shows the correlation between actual historical fees and dynamic model fees, with the red dashed line representing perfect correlation.

![Actual vs Dynamic Fees](results/analysis/all/actual_vs_dynamic_fees.png)
