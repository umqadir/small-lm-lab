# Cross-backend validation gate: noise-floor adjudication

## Summary

- New backend required to reproduce a known training run within a tolerance fixed in advance. Failed.
- Five-seed control: two runs of the identical configuration, differing only in random seed, also fail it, all ten pairs, tightest by a factor of 4.76.
- Tolerance was calibrated in a setting where identical initial weights were guaranteed, then applied in one where they were not.
- The failing comparison was tighter than every same-configuration pair measured.

## Gate and tolerance provenance

Completed work, including the learning-rate selection, trained with MLX on Apple silicon. Move to NVIDIA hardware changed framework and device. Gate registered in advance: rerun a known 500-step training run on the new backend, compare loss curve against the banked original over steps 400 to 500.

| statistic | tolerance |
|---|---|
| absolute loss difference at step 500 | 0.02 |
| mean absolute difference over steps 400 to 500 | 0.015 |

Set for MLX on one device to MLX on another: same framework and seed, identical initial weights, differing only in device arithmetic. Measured that way, end difference 0.001119, a factor of 17.9 inside tolerance.

Plan changed to PyTorch on CUDA; gate carried across unchanged, still describing a same-framework comparison, now applied cross-framework.

## Measured failure

| statistic | measured | tolerance | verdict |
|---|---|---|---|
| end difference at step 500 | 0.061059 | 0.02 | fail |
| mean difference over window | 0.043621 | 0.015 | fail |

3.05 times the end bound, 2.91 times the mean bound. Run: 500 steps, clean exit, about 30,300 tokens per second.

## Precision audit

- Suspect: reduced-precision arithmetic. NVIDIA GPUs from the Ampere generation onward run nominally float32 matmuls in TF32: exponent range kept, 10 explicit mantissa bits instead of 23, relative error near 1e-3 against 1e-7 for float32. Tensors still report `float32`.
- Codebase never controlled TF32. Precision flag suppressed bfloat16 autocast, said nothing about TF32. Fixed: TF32 now refused explicitly, policy recorded in each run's metadata.
- Not the cause. Canary: float32 matmul against a float64 reference, exact framework defaults the failing gate used, measured relative Frobenius error 5.75e-07 against a 1e-05 threshold, genuine float32. Matmuls already true float32 by default; only the convolution path defaults to TF32, and a decoder-only transformer performs no convolutions. Fix explains none of the 0.061059.

## Difference between the two runs

PyTorch and MLX seed weights from different pseudorandom generators: one through PyTorch's generator, the other through MLX's. Same distribution, same nominal seed, different numbers.

The two runs never shared a starting point: different trajectories of the same recipe, not one trajectory on two backends. Gate was built on the second assumption.

## Noise-floor measurement

Backend, precision, data, code, and step budget held fixed; seed varied. Five runs, ten pairs, same steps 400 to 500 window.

Within-framework, PyTorch against PyTorch, seed the only difference:

| pair | end | mean |
|---|---|---|
| s1 vs s2 | 0.119850 | 0.318902 |
| s1 vs s3 | 0.711391 | 0.300632 |
| s1 vs s4 | 0.616208 | 0.307768 |
| s1 vs s5 | 0.502398 | 0.377310 |
| s2 vs s3 | 0.831241 | 0.424031 |
| s2 vs s4 | 0.736058 | 0.339431 |
| s2 vs s5 | 0.622248 | 0.518523 |
| s3 vs s4 | 0.095183 | 0.271766 |
| s3 vs s5 | 0.208993 | 0.281497 |
| s4 vs s5 | 0.113810 | 0.391563 |

end: min 0.095183, median 0.559303, max 0.831241.
mean: min 0.271766, median 0.329166, max 0.518523.

Cross-framework, PyTorch against the banked MLX reference:

| pair | end | mean |
|---|---|---|
| s1 vs MLX | 0.078285 | 0.045959 |
| s2 vs MLX | 0.041564 | 0.318941 |
| s3 vs MLX | 0.789676 | 0.302346 |
| s4 vs MLX | 0.694493 | 0.331207 |
| s5 vs MLX | 0.580684 | 0.358306 |

## Result

- All ten within-framework pairs fail both tolerances. Smallest end difference among same-configuration runs: 0.095183, 4.76 times the 0.02 bound.
- Tolerance unreachable by construction: no pair of separately initialized runs could have passed it. The gate measured initialization noise and reported it as a backend defect.
- Cross-framework comparison that failed the gate, at 0.061059, was tighter by a factor of 1.6 than the tightest same-framework pair at 0.095183.
- Consecutive logged steps within a single run move by 0.3 to 0.5. The gate required two noisy point estimates to agree to 0.02.

## Registered replacement rule

Registered before the five-seed data existed.

Rule: tolerance = 1.5 times the maximum observed within-framework difference, per statistic. Maximum used instead of a quantile: ten pairs cannot estimate a high quantile. 1.5 factor fixed in advance.

Two abandonment branches, registered with equal weight:

- If the derived tolerance would exceed 0.25 end-difference: adopt no replacement.
- If the within-framework spread turned out small relative to the observed 0.061059: no replacement adopted, backend reported unsuitable until a cause is found.

Applying it:

```
tol_end  = 1.5 x 0.831241 = 1.246862
tol_mean = 1.5 x 0.518523 = 0.777784
```

1.246862 exceeds 0.25: no replacement gate adopted. Divergence branch does not fire: the within-framework floor of 0.095183 is larger than the cross-framework difference of 0.061059, not smaller.

## Basis for backend fidelity

Curve comparison retired as uninformative at this scale. Remaining evidence, on identical weights:

- Converting weights between the two implementations and comparing outputs: agrees to 3.5e-06 maximum absolute logit difference.
- TF32 canary: 5.75e-07 relative Frobenius error against a float64 reference, on a 1e-05 threshold.
- Full test suite: passes identically on both machines.

Limitation: no passing cross-backend curve gate; agreement established only on identical weights and the test suite. An effect appearing solely in the optimizer trajectory would not be caught.

## Conclusions

1. Tolerance changed meaning when the comparison changed from same-framework to cross-framework, while the number stayed at 0.02.
2. The original gate never established what agreement to expect from two runs that ought to agree.
3. Two abandonment branches were fixed in the registration before the five-seed data existed.
4. The uncontrolled-TF32 explanation fit the facts and was wrong; measuring it cost minutes and changed the conclusion.
