# FlowState probes (released weights, ETTm1/OT, 16 test-period windows, L=1024, H=96)

FS1. Relative change of the median forecast at common physical times, and of the last encoder state, vs native. Median [min, max] over windows. zohR = every sample repeated r times, lossless for FlowState's own right-hold ZOH kernel; refine = FOH virtual knots; down = every r-th real sample (lossy).

| revision | params | grid change | lossless for FlowState's rule? | forecast change | state change |
|---|---|---|---|---|---|
| v1.0 (main) | 9.1M | ZOH repeat x2 | yes | 4.2% [1.2, 14.3] | 31.8% [17.3, 44.1] |
| v1.0 (main) | 9.1M | ZOH repeat x4 | yes | 4.2% [1.3, 15.0] | 33.3% [18.1, 46.6] |
| v1.0 (main) | 9.1M | FOH virtual knots x2 | no (FOH path) | 4.7% [1.4, 15.2] | 34.3% [19.1, 49.6] |
| v1.0 (main) | 9.1M | FOH virtual knots x4 | no (FOH path) | 4.8% [1.3, 15.1] | 36.8% [20.7, 55.3] |
| v1.0 (main) | 9.1M | real downsampling /2 | no (lossy) | 2.2% [1.0, 12.2] | 22.2% [16.2, 42.6] |
| v1.0 (main) | 9.1M | real downsampling /4 | no (lossy) | 4.8% [2.2, 16.9] | 54.3% [37.3, 70.5] |
| v1.0 (main) | | repeat of native input | yes | 0.0e+00 | |

v1.0 (main): forecast MAE native 0.678 vs FOH-refined x2 0.741 (median over windows).

| r1.1 | 18.5M | ZOH repeat x2 | yes | 7.0% [1.3, 23.9] | 45.3% [23.7, 63.7] |
| r1.1 | 18.5M | ZOH repeat x4 | yes | 6.9% [1.6, 20.6] | 43.0% [24.4, 64.8] |
| r1.1 | 18.5M | FOH virtual knots x2 | no (FOH path) | 6.7% [1.5, 19.9] | 36.9% [25.8, 58.2] |
| r1.1 | 18.5M | FOH virtual knots x4 | no (FOH path) | 6.7% [1.6, 18.8] | 39.8% [32.3, 59.8] |
| r1.1 | 18.5M | real downsampling /2 | no (lossy) | 3.9% [0.7, 13.7] | 38.5% [29.5, 52.1] |
| r1.1 | 18.5M | real downsampling /4 | no (lossy) | 9.4% [2.3, 19.5] | 74.8% [63.3, 98.9] |
| r1.1 | | repeat of native input | yes | 0.0e+00 | |

r1.1: forecast MAE native 0.988 vs FOH-refined x2 0.859 (median over windows).

FS2. Attribution on 8 windows: causal RevIN (running sample-count mean/std) vs fixed per-context statistics (affine, grid-independent). Per-layer relative change of the encoder output at the last context step.

| revision | RevIN | grid change | forecast change | per-layer change at last step (layer 0 -> 5) |
|---|---|---|---|---|
| v1.0 (main) | causal_revin | zohR2 | 6.8% | 0.2% 2.7% 8.6% 17% 27% 34% |
| v1.0 (main) | causal_revin | foh2 | 6.6% | 5.1% 7.4% 9.7% 19% 29% 36% |
| v1.0 (main) | fixed_revin | zohR2 | 5.1% | 0.002% 2.2% 8.8% 16% 24% 27% |
| v1.0 (main) | fixed_revin | foh2 | 5.2% | 4.9% 7.1% 9.4% 19% 24% 28% |
| r1.1 | causal_revin | zohR2 | 11.7% | 0.096% 1.3% 4.7% 37% 51% 46% |
| r1.1 | causal_revin | foh2 | 9.0% | 1.4% 5% 8.3% 31% 44% 40% |
| r1.1 | fixed_revin | zohR2 | 12.4% | 0.0038% 1.2% 4.1% 34% 47% 43% |
| r1.1 | fixed_revin | foh2 | 10.9% | 1.5% 4.7% 7.9% 26% 38% 36% |

FS3. Encoder in float32 vs float64 (weights cast), ZOH repeat x2, fixed RevIN statistics, 4 windows: median per-layer relative change at the last context step.

| revision | dtype | layer 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|---|
| v1.0 (main) | float32 | 1.9e-05 | 2.4e-02 | 9.1e-02 | 1.8e-01 | 2.6e-01 | 3.0e-01 |
| v1.0 (main) | float64 | 5.7e-14 | 2.4e-02 | 9.1e-02 | 1.8e-01 | 2.6e-01 | 3.0e-01 |
| r1.1 | float32 | 9.9e-06 | 2.0e-02 | 6.9e-02 | 3.2e-01 | 4.9e-01 | 4.9e-01 |
| r1.1 | float64 | 4.5e-14 | 2.0e-02 | 6.9e-02 | 3.2e-01 | 4.9e-01 | 4.9e-01 |
