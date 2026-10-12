# Part 6 — Comparing LoRA rank 8 and rank 64

We now have the most important experimental finding so far: the rank-64 LoRA model achieves 65.9% best validation accuracy, compared with 65.0% for dense fine-tuning, while using 75% fewer trainable parameters.

There is an important distinction between the two LoRA runs you pasted. The first result, 52.3%, belongs to the rank-8 experiment. The later result, 65.9%, belongs to a rank-64 experiment initialised from the already fine-tuned dense baseline. These are not equivalent experimental conditions, so we should interpret them separately.

## 1. Results at a glance

## Best validation accuracy

Dense fine-tuning

# 65.0%

LoRA, rank 64

# 65.9%

Accuracy difference

## +0.9 percentage points

Trainable parameter reduction

## 75.0%

The rank-64 result is promising: it retains approximately the same validation performance as the dense baseline, with a modest numerical improvement. However, the 0.9-percentage-point difference alone does not establish that LoRA is statistically better.

## 2. Why rank matters

The rank $r$ determines the capacity of the low-rank adapter. Increasing the rank gives the model more freedom to modify the pretrained weights, but also increases the number of trainable parameters.

For each adapted linear layer, the adapter contains

$$
N_{\text{adapter}}=r(d_{\text{in}}+d_{\text{out}})
$$

parameters.

The rank-8 implementation uses relatively small adapters. The rank-64 implementation uses adapters eight times larger in each adapted layer, because the parameter count scales linearly with rank.

This gives the model more capacity to learn the target task, but it reduces the efficiency advantage over dense fine-tuning.

The two reported accuracy results are consistent with the possibility that rank 8 provides insufficient adaptation capacity while rank 64 is more expressive. Nevertheless, we cannot attribute the difference to rank alone because the two runs also use different starting model weights.

## 3. Comparing parameter efficiency

The rank-64 output reports the following:

| Metric | Dense baseline | LoRA, rank 64 |
| --- | --- | --- |
| Total parameters | 1,332,741 | 1,627,653 |
| Trainable parameters | 1,183,237 | 296,197 |
| Trainable fraction of total | 88.8% | 18.2% |
| Best validation accuracy | 65.0% | 65.9% |

The LoRA model has more total parameters because it retains the original weights and adds adapter matrices. However, only 296,197 parameters are trainable.

The reduction in trainable parameters is:

$$
\begin{aligned}
\text{Reduction}
&=\left(1-\frac{296197}{1183237}\right)\times100\\
&\approx75.0\%
\end{aligned}
$$

This is a substantial reduction in the number of trainable parameters. It supports the central motivation for LoRA, although the notebook has not yet provided measurements of actual training time or GPU memory usage.

## 4. An important experimental limitation

The rank-64 experiment uses:

python
```
lora_model = copy.deepcopy(baseline_ft_model)
```

This is a critical detail. It means that the rank-64 model starts from the already fine-tuned dense baseline, not directly from the original CIFAR-10 pretrained checkpoint.

The experiment then freezes the MLP weights and adds zero-initialised LoRA adapters. Consequently, the initial model already has the dense baseline's learned target-task weights and classifier.

This makes the rank-64 run different from the rank-8 run, which was set up from the pretrained checkpoint with a newly initialised classifier.

Therefore:

- The rank-8 result of 52.3% and the rank-64 result of 65.9% cannot be treated as a controlled comparison of ranks.
- The rank-64 result demonstrates that LoRA adapters can be trained on top of a dense fine-tuned solution while retaining competitive validation performance.
- It does not establish that rank 64 is inherently better than rank 8 when both start from identical pretrained weights.

For a scientifically stronger comparison, both ranks should start from the same pretrained checkpoint, use the same classifier initialisation and data split, and be trained under otherwise identical conditions.

## 5. Interim conclusion

The results currently support three observations:

1. Dense fine-tuning reaches 65.0% best validation accuracy with 1,183,237 trainable parameters.
2. The rank-8 LoRA run reaches 52.3%, which is 12.7 percentage points below the baseline.
3. The rank-64 LoRA run reaches 65.9%, 0.9 percentage points above the baseline, while reducing the trainable parameter count by 75%.

The third result is encouraging, but the different initialisation makes it important not to overstate the comparison. The evidence supports LoRA's parameter-efficiency potential, while the effect of rank itself remains inconclusive.

Please paste the next cell and output. We should check whether the notebook includes additional rank experiments, timing measurements or a final comparison. Those will help us produce a rigorous final report with appropriately qualified conclusions.
