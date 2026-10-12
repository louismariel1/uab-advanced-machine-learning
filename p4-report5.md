# Part 5 — Understanding and implementing LoRA

We have now reached the central experiment of the practical. The objective is to replace dense updates to the fully connected layers with low-rank trainable adapters, while keeping the pretrained weights frozen.

The key question is whether LoRA can approach the baseline's 65.0% best validation accuracy while training substantially fewer parameters.

## 1. The mathematical idea behind LoRA

In ordinary fine-tuning, a linear layer uses a weight matrix $W$ and bias $b$:

$$
z=Wx+b
$$

During training, the weight matrix can be updated directly. If $W$ has millions of entries, updating all those parameters can be expensive.

The notebook first introduces an intermediate abstraction, `AbstractLayer`, which freezes $W$ and adds a separate, trainable update matrix $\Delta W$:

$$
z=Wx+\Delta Wx+b
$$

Because $\Delta W$ starts at zero, the initial output is identical to that of the original layer. The model can then learn an update without modifying the pretrained matrix $W$.

LoRA takes this idea further by expressing the update as the product of two smaller matrices:

$$
\Delta W=BA
$$

For a linear layer with $d_{\text{in}}$ input features and $d_{\text{out}}$ output features, the shapes are:

$$
A\in\mathbb{R}^{r\times d_{\text{in}}}
$$

$$
B\in\mathbb{R}^{d_{\text{out}}\times r}
$$

Here, $r$ is the rank, which is deliberately much smaller than the dimensions of the original weight matrix.

### The two approaches

Dense fine-tuning

$$
W'=W+\Delta W
$$

The original weight matrix is updated directly.

LoRA fine-tuning

$$
W'=W+\frac{BA}{r}
$$

The original weight matrix is frozen; only the low-rank matrices A and B are trained.

The notebook's implementation scales the adapter update by $1/r$. This is a common LoRA scaling choice, although other implementations use a separate scaling hyperparameter.

The important point is that the number of trainable adapter parameters is

$$
N_{\mathrm{LoRA}}=r(d_{\text{in}}+d_{\text{out}})
$$

rather than

$$
N_{\mathrm{dense}}=d_{\text{in}}d_{\text{out}}
$$

for the original weight matrix. This difference can be substantial when $r$ is small.

## 2. Why test `AbstractLayer` first?

The notebook replaces the four fully connected layers with `AbstractLayer` and checks whether the new model produces exactly the same outputs as the original pretrained model.

The reported output confirms:

- The original fully connected layer and its abstracted version produce identical outputs.
- The complete pretrained network also produces identical outputs after the replacements.

This is an important implementation check. It demonstrates that adding a zero-initialised update matrix does not change the network's initial predictions.

The distinction is that `AbstractLayer` still introduces a full-sized update matrix. It explains the mathematical idea, but does not deliver the parameter savings of LoRA. Those savings come from replacing the full-sized update with the product of two low-rank matrices.

## 3. How the notebook implements `LoraLayer`

The implementation contains three important operations.

### A. Freeze the original weights and bias

python
```
self.original_layer.weight.requires_grad_(False)
self.original_layer.bias.requires_grad_(False)
```

This prevents gradients from updating the original linear layer's weight and bias. The original layer still participates in the forward pass, so its pretrained knowledge is retained.

### B. Initialise the low-rank matrices

python
```
self.A = nn.Parameter(
    torch.randn(r, self.in_features) * 0.01
)

self.B = nn.Parameter(
    torch.zeros(self.out_features, r)
)
```

The matrix $A$ starts with small random values, while $B$ starts at zero.

Therefore,

$$
BA=0
$$

at initialisation, and the adapted layer initially behaves exactly like the original layer.

This matters because it avoids an arbitrary change to the pretrained model's outputs at the start of fine-tuning.

### C. Add the scaled adapter update

python
```
original_out = self.original_layer(x)
update = (x @ self.A.T) @ self.B.T

return original_out + update / self.r
```

This implements

$$
y=Wx+b+\frac{BAx}{r}
$$

The original linear layer computes the first part, and the low-rank matrices compute the trainable adjustment. The result is their sum.

## 4. Experimental design: matching the baseline

The notebook sets:

python
```
r = 8
```

It wraps `fc1`, `fc2`, `fc3` and `fc4` with LoRA adapters, leaving the final classifier trainable as a conventional linear layer.

Both approaches freeze `conv1` and `conv2`. Both use the same target dataset, number of epochs, learning rate and weight decay.

| Experimental setting | Dense baseline | LoRA |
| --- | --- | --- |
| Target task | Five vehicle classes | Same |
| Pretrained feature weights | Reused | Reused |
| Convolutional layers | Frozen | Frozen |
| MLP weight matrices | Trainable | Frozen |
| MLP adaptation | Full weight updates | Rank-8 adapters |
| Classifier head | Trainable | Trainable |
| Training epochs | 30 | 30 |
| Learning rate | 0.0001 | 0.0001 |
| Weight decay | 0.01 | 0.01 |

This is a useful controlled comparison: the major intended difference is how the MLP weights are adapted.

The baseline starts with pretrained MLP weights and a newly initialised classifier. The LoRA model does the same, with the additional low-rank adapters initially contributing zero to the forward pass.

## 5. Expected parameter savings at rank 8

We can calculate the expected parameter counts directly from the architecture and the implementation.

| Adapted layer | Original weight parameters | Rank-8 adapter parameters |
| --- | --- | --- |
| `fc1` (256 → 1024) | 262,144 | 10,240 |
| `fc2` (1024 → 512) | 524,288 | 12,288 |
| `fc3` (512 → 512) | 262,144 | 8,192 |
| `fc4` (512 → 256) | 131,072 | 6,144 |
| Total | 1,179,648 | 36,864 |

These counts concern the weight matrices; the biases and classifier are handled separately.

The four LoRA adapters together contain only 36,864 trainable parameters. The classifier contributes another 1,285 trainable parameters, giving:

- Expected total trainable parameters: 38,149
- Expected trainable fraction of the complete LoRA model: approximately 2.8%
- Expected reduction in trainable parameters relative to the dense baseline: approximately 96.8%

The total parameter count of the LoRA model is actually slightly higher than the baseline's, because the original weights are retained and the adapter parameters are added. LoRA reduces the number of parameters that must be trained, not necessarily the total number stored.

These are calculated expectations from the code, not reported notebook measurements. We should confirm them against the output of the parameter-counting cell.

## 6. Interpreting the LoRA training experiment

The final cell trains the LoRA model for the same 30 epochs as the dense baseline and prints:

python
```
print(f'Best LoRA-FT accuracy: {lora_metrics.best_val_acc:.1%}')

diff = (
    lora_metrics.best_val_acc
    - baseline_ft_metrics.best_val_acc
)
```

This difference is measured in percentage points, not relative percent improvement.

For example, if LoRA achieved 63% and the baseline achieved 65%, the difference would be −2 percentage points. That would indicate a small loss in validation accuracy alongside the expected reduction in trainable parameters.

The actual LoRA training output and final accuracy have not been included in your message yet, so we cannot determine whether LoRA matched the baseline or performed worse or better.

One further distinction will matter for our final conclusion: fewer trainable parameters do not automatically mean proportionally less wall-clock training time. The low-rank matrix operations introduce additional computation, and actual speed depends on hardware and implementation. We can establish the parameter savings from the code, but measured speed improvements require timing results.

## 7\. What to paste next

Please share the output produced by the LoRA implementation and training cells, particularly:

- The reported total and trainable parameter counts.
- The printed `Best LoRA-FT accuracy`.
- The percentage-point difference from the dense baseline.
- Any training curves or prediction visualisations that appeared.

With those results, we can make the direct experimental comparison and determine how strong the evidence is for LoRA's efficiency–performance trade-off.
