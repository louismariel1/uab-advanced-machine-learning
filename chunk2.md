## Practical 4 — Chunk 2: Exam Compression Notes

### 1\. LoRA core idea

LoRA (**Low-Rank Adaptation**) freezes the original weight matrix and learns a low-rank update.

Instead of updating:

\[ W \]

we use:

\[ W' = W + BA \]

where:

- (W) = original pretrained weight, **frozen**
- (A) = trainable low-rank matrix
- (B) = trainable low-rank matrix
- (r) = LoRA rank, usually (r \\ll d)

For a layer:

\[ xW^T \]

LoRA computes:

\[ xW^T + xA^TB^T \]

### 2\. Forward pass — know this

```
z_frozen = self.original_layer(x)

z_lora = torch.matmul(
    torch.matmul(x, self.A.T),
    self.B.T
)

return z_frozen + z_lora
```

**Exam explanation:**

1. Pass (x) through the frozen original layer.
2. Project (x) down from `in_features → r` using (A).
3. Project back from `r → out_features` using (B).
4. Add the LoRA update to the original output.

So:

\[ \\boxed{y = W x + BAx} \]

(up to the row/column convention used in the code).

---

## 3\. Why LoRA is parameter-efficient

For a normal layer:

\[ W \\in \\mathbb{R}^{d_{out}\\times d_{in}} \]

Number of weight parameters:

\[ d_{out}d_{in} \]

LoRA replaces learning all of these with:

\[ A \\in \\mathbb{R}^{r\\times d\_{in}} \]

and

\[ B \\in \\mathbb{R}^{d\_{out}\\times r} \]

Trainable parameters:

\[ \\boxed{r(d_{in}+d_{out})} \]

Since:

\[ r \\ll d_{in},d_{out} \]

the number of trainable parameters is much smaller.

**Key exam phrase:**

> LoRA keeps the pretrained full-rank weights frozen and learns only a low-rank update, greatly reducing the number of trainable parameters.

---

## 4\. What gets trained?

For this practical:

**Frozen:**

- Original FC weight matrices
- Convolutional layers
- Other pretrained parameters not explicitly made trainable

**Trainable:**

- LoRA (A) matrices
- LoRA (B) matrices
- Classifier weight and bias

The assertion checks this:

```
assert lora_model.fc1.original_layer.weight.requires_grad == False
```

Meaning:

\[ \\boxed{\\text{original }W\\text{ is frozen}} \]

---

## 5\. Rank (r)

`r` controls the capacity of the LoRA update.

Examples tested:

```
lora_ranks = [2, 4, 8, 16, 32]
```

### Low (r)

- Very few trainable parameters
- More parameter-efficient
- Less expressive
- May underfit

### High (r)

- More trainable parameters
- More expressive
- Potentially better accuracy
- Less parameter-efficient

**Main trade-off:**

\[ \\boxed{\\text{higher }r \\rightarrow \\text{more parameters + potentially better performance}} \]

---

## 6\. Dense FT vs LoRA

### Dense fine-tuning

Updates the full model's trainable weights.

\[ \\boxed{\\text{Many trainable parameters}} \]

### LoRA fine-tuning

Updates only low-rank adapters.

\[ \\boxed{\\text{Far fewer trainable parameters}} \]

The practical compares:

- Validation accuracy
- Number of trainable parameters
- State-dict size
- Optimizer state size
- Forward/backward time
- Different LoRA ranks

---

## 7\. Important parameter-count comparison

The code calculates:

```
total_params, trainable_params = num_parameters(model)
```

Then:

```
1 - lora_trainable / dense_trainable
```

gives the **percentage reduction in trainable parameters**.

For example, if:

- Dense = 100,000 trainable
- LoRA = 20,000 trainable

then:

\[ 1-\\frac{20,000}{100,000}=0.8 \]

so:

\[ \\boxed{80%\\text{ reduction}} \]

---

## 8\. Why use a fresh model for every rank?

For each:

\[ r \\in {2,4,8,16,32} \]

the code creates a new model.

This is important because you **must not continue training one already-trained LoRA model** when comparing ranks.

Otherwise the experiment is not controlled.

Each rank should have:

- Same pretrained weights
- Same dataset
- Same train/validation split
- Same epochs
- Same learning rate
- Same regularisation
- Only (r) changes

Therefore differences are mainly attributable to the rank.

---

## 9\. Why set the same seed?

```
torch.manual_seed(0)
```

This makes initialization reproducible.

**Exam answer:**

> The same random seed is used to make the rank comparison more controlled and reproducible.

---

# 10\. State dict size

The code:

```
def get_state_dict_size(state_dict):
    param_count = 0

    for key, values in state_dict.items():
        param_count += values.numel()

    return param_count
```

counts the total number of tensor elements in the state dict.

### Important distinction

A model's **state dict** contains model parameters/buffers.

LoRA's state dict can still contain the original frozen weights.

Therefore:

> LoRA does **not automatically mean a much smaller complete model state dict** if the frozen pretrained weights are still saved.

To save only the adaptation, you can save only:

\[ \\boxed{A\\text{ and }B} \]

which is much smaller.

---

# 11\. Optimizer state dict

The code counts optimizer state:

```
def get_opt_state_dict_size(state_dict):
    grad_count = 0

    for i, param in state_dict['state'].items():
        for key, values in param.items():
            grad_count += values.numel()

    return grad_count
```

### Why is LoRA optimizer state smaller?

The optimizer only needs states for parameters being optimized.

Dense FT:

\[ \\boxed{\\text{optimizer tracks many trainable parameters}} \]

LoRA:

\[ \\boxed{\\text{optimizer tracks only }A,B\\text{ (+ classifier)}} \]

Therefore LoRA generally has a **much smaller optimizer state**.

This is an important exam distinction:

> The LoRA model's full state dict may still contain the frozen pretrained model, but its optimizer state is much smaller because only trainable parameters need optimizer states.

---

# 12\. Forward and backward speed

LoRA is **not necessarily faster**.

Although it has fewer trainable parameters, the forward pass now contains extra operations:

\[ x \\rightarrow xA^T \\rightarrow xA^TB^T \]

in addition to:

\[ x \\rightarrow Wx \]

So LoRA introduces extra computation.

### Forward

LoRA can be:

\[ \\boxed{\\text{slightly slower}} \]

because it computes both:

- original layer
- LoRA update

### Backward

LoRA can require less gradient computation because (W) is frozen.

But the adapter operations themselves still require gradients.

So actual timing depends on:

- layer dimensions
- rank (r)
- hardware
- batch size
- implementation

**Exam-safe answer:**

> LoRA reduces trainable parameters and optimizer/gradient computation, but its forward pass adds the low-rank adapter computation. Therefore it is not guaranteed to be faster; timing should be measured experimentally.

---

# 13\. Why does increasing (r) increase computation?

LoRA performs:

\[ xA^T \]

followed by:

\[ (xA^T)B^T \]

The intermediate dimension is (r).

Therefore larger (r) means more operations.

\[ \\boxed{r\\uparrow \\Rightarrow \\text{parameters}\\uparrow \\Rightarrow \\text{computation}\\uparrow} \]

---

# 14\. Rank sweep experiment

The code tests:

```
r = 2, 4, 8, 16, 32
```

and records:

```
best_val_acc
trainable_params
trainable_fraction
```

The best rank is found using:

```
best_lora_idx = rank_results['best_val_acc'].idxmax()
```

So:

\[ \\boxed{\\text{best rank}=\\arg\\max\_r(\\text{validation accuracy})} \]

Then it compares against dense FT:

```
accuracy_difference = best_lora_acc - dense_acc
```

If positive:

> LoRA beats the dense baseline.

If negative:

> Dense fine-tuning performs better.

---

# 15\. Most important plots

### Plot 1: Accuracy vs rank

Shows:

\[ r \\rightarrow \\text{validation accuracy} \]

Purpose:

> Determine how LoRA capacity affects performance.

Dense FT is shown as a horizontal reference line.

---

### Plot 2: Accuracy vs trainable parameters

Shows:

\[ \\text{trainable parameters} \\rightarrow \\text{validation accuracy} \]

This is arguably the most important LoRA plot because LoRA's goal is **parameter efficiency**, not simply maximum accuracy.

You want to identify a point giving:

\[ \\boxed{\\text{good accuracy with very few trainable parameters}} \]

---

# 16\. `trainable_fraction`

Calculated as:

```
trainable_params / total_params
```

Example:

\[ \\frac{10,000}{100,000}=0.1 \]

Therefore:

\[ \\boxed{10%} \]

of the model parameters are trainable.

---

# 17\. LoRA merging — optional extension

After training:

\[ W' = W + BA \]

The LoRA matrices can be merged into the original weight.

Then inference becomes an ordinary layer:

\[ \\boxed{y=W'x} \]

### Why merge?

After merging:

- No separate LoRA computation
- Same architecture as the original layer
- No extra inference operations
- Same output as the LoRA version, assuming the merge is implemented correctly

**Key concept:**

\[ \\boxed{W\_{\\text{merged}}=W+BA} \]

---

# 18\. Saving only adapters

Instead of saving the complete model, save only:

```
A
B
```

for each adapted layer.

This is useful because the pretrained model can be shared/reused, while each fine-tuning task only stores its small LoRA adapters.

Conceptually:

\[ \\boxed{\\text{checkpoint}= \\text{pretrained model}+\\text{small adapters}} \]

---

# 19\. GPU memory

The provided:

```
torch.cuda.memory_allocated(0)
torch.cuda.memory_reserved(0)
torch.cuda.max_memory_reserved(0)
```

measure GPU memory usage.

### Important distinction

- `memory_allocated` → currently allocated GPU memory
- `memory_reserved` → memory PyTorch has reserved
- `max_memory_reserved` → peak reserved memory

`torch.cuda.empty_cache()` releases unused cached memory, but **doesn't delete live tensors/models**.

For a fair model comparison, restart the kernel/session as suggested by the practical.

---

# 20\. High-probability exam questions

### Q: What does LoRA do?

> LoRA freezes pretrained weights and learns a low-rank weight update using two small matrices (A) and (B).

### Q: What is the LoRA equation?

\[ \\boxed{W'=W+BA} \]

### Q: Why is LoRA parameter efficient?

\[ d_{out}d_{in} \]

parameters become approximately:

\[ \\boxed{r(d_{in}+d_{out})} \]

trainable parameters.

### Q: What does (r) represent?

> The rank of the low-rank adaptation and therefore the capacity of the LoRA update.

### Q: What happens when (r) increases?

> More trainable parameters, more expressive adaptation, and potentially higher accuracy, but also more computation.

### Q: Are the original weights trained?

> No. They are frozen.

### Q: Why can LoRA have a similar full state-dict size to dense FT?

> Because the frozen pretrained weights may still be included in the state dict.

### Q: Why is LoRA's optimizer state smaller?

> Only trainable parameters require optimizer state, and LoRA has far fewer trainable parameters.

### Q: Is LoRA always faster?

> No. It reduces backward/optimization work but adds low-rank computations to the forward pass.

### Q: How do you merge LoRA?

\[ \\boxed{W\_{\\text{merged}}=W+BA} \]

### Q: Why test multiple ranks?

> To find the best trade-off between adaptation capacity, accuracy, and parameter efficiency.

---

## 21\. One-minute memorisation sheet

If you only have time to memorise **this**, learn:

\[ \\boxed{W'=W+BA} \]

- **W** = frozen pretrained weight
- **A, B** = trainable LoRA matrices
- **r** = low rank
- Original weights → **frozen**
- A/B → **trainable**
- Trainable parameters:

\[ \\boxed{r(d_{in}+d_{out})} \]

- Higher (r) → more capacity + more parameters/computation
- Lower (r) → fewer parameters + potentially less capacity
- LoRA greatly reduces **trainable parameters**
- Optimizer state is therefore much smaller
- Full state dict may **not** be much smaller because frozen (W) can still be stored
- LoRA forward pass has **extra adapter computation**
- Merge after training:

\[ \\boxed{W\\leftarrow W+BA} \]

- Main experimental goal:

\[ \\boxed{\\text{high accuracy with few trainable parameters}} \]

- Compare LoRA against dense FT using **validation accuracy, parameters, state/optimizer size, and timing**.
