# P4 - Practical Lora solution
The practical is essentially asking you to **implement LoRA correctly, fine-tune the CIFAR-10-pretrained model on a 5-class CIFAR-100 subset, and demonstrate that you can achieve similar accuracy while training far fewer parameters**.

 Below is a clean solution you can adapt directly into the notebook.

 ## 1\. What you need to achieve

 Your pipeline is:

```
CIFAR-10
   │
   ▼
Pre-train ConvMLP
   │
   ▼
Save pretrained weights
   │
   ├───────────────┐
   ▼               ▼
Dense FT          LoRA FT
   │               │
   ▼               ▼
Target CIFAR-100  Target CIFAR-100
5 classes         5 classes
   │               │
   ▼               ▼
Baseline accuracy  LoRA accuracy
```

 The important comparison is:

 |  | Dense fine-tuning | LoRA |
| --- | --- | --- |
| Pretrained weights | Yes | Yes |
| Conv layers | Frozen | Frozen |
| FC weights | Trainable | Frozen |
| Adaptation | Full weight matrices | Low-rank A/B matrices |
| Trainable parameters | Much larger | Much smaller |
| Target accuracy | Baseline | Ideally close to baseline |

---

 # 2\. The key part: implement `LoraLayer`

 The notebook has:

```
class LoraLayer(nn.Module):
    def __init__(self,
                 original_layer,
                 r,
                 ):
        super().__init__()
        self.verbose = verbose

        self.original_layer = original_layer

        # define the A and B matrices...
```

 There is actually a small problem here:

```
self.verbose = verbose
```

 `verbose` isn't an argument to the provided constructor.

 You should remove that line.

 Use this implementation:

```
class LoraLayer(nn.Module):
    def __init__(self, original_layer, r):
        super().__init__()

        self.original_layer = original_layer
        self.r = r

        # Dimensions of original Linear layer
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features

        # Freeze original weight and bias
        self.original_layer.weight.requires_grad_(False)
        self.original_layer.bias.requires_grad_(False)

        # LoRA matrices:
        # A: r x input_dim
        # B: output_dim x r
        self.A = nn.Parameter(
            torch.randn(r, self.in_features) * 0.01
        )

        # B is initialized to zero so that BA = 0 initially
        self.B = nn.Parameter(
            torch.zeros(self.out_features, r)
        )

        # LoRA scaling
        self.scaling = 1.0 / r

    def forward(self, x):

        # Original frozen transformation
        original_output = self.original_layer(x)

        # Low-rank update:
        # BAx
        lora_update = F.linear(
            F.linear(x, self.A),
            self.B
        )

        # Combine original W x + BA x
        return original_output + self.scaling * lora_update
```

 ### Why this works

 The original layer computes:

 $$
Wx+b
$$

 LoRA changes this to:

 $$
Wx+b + BAx
$$

 where:

 $$
W \in \mathbb{R}^{d_{out}\times d_{in}}
$$

 but instead of learning all $d_{out}d_{in}$ parameters, we learn:

 $$
A \in \mathbb{R}^{r\times d_{in}}
$$

 and

 $$
B \in \mathbb{R}^{d_{out}\times r}
$$

 with:

 $$
r \ll d_{in},d_{out}
$$

 Therefore the number of trainable LoRA parameters is:

 $$
r(d_{in}+d_{out})
$$

 instead of:

 $$
d_{in}d_{out}
$$

---

 # 3\. Choose the rank

 For the first run, I recommend:

```
r = 8
```

 So put this **before** replacing the layers:

```
r = 8
```

 Then:

```
lora_model.fc1 = LoraLayer(lora_model.fc1, r)
lora_model.fc2 = LoraLayer(lora_model.fc2, r)
lora_model.fc3 = LoraLayer(lora_model.fc3, r)
lora_model.fc4 = LoraLayer(lora_model.fc4, r)
```

---

 # 4\. Important: initialize the LoRA model correctly

 The notebook already gives you most of this:

```
torch.manual_seed(0)

lora_model = ConvMLP(
    num_classes=target_data.num_classes
).to(device)

lora_model.load_state_dict(
    pt_state_dict,
    strict=False
)
```

 Then freeze convolutional layers:

```
lora_model.conv1.requires_grad_(False)
lora_model.conv2.requires_grad_(False)
```

 Then replace the FC layers:

```
r = 8

lora_model.fc1 = LoraLayer(lora_model.fc1, r)
lora_model.fc2 = LoraLayer(lora_model.fc2, r)
lora_model.fc3 = LoraLayer(lora_model.fc3, r)
lora_model.fc4 = LoraLayer(lora_model.fc4, r)
```

---

 # 5\. Check the trainable parameters

 Run:

```
lora_total, lora_trainable = num_parameters(lora_model)

print(f'Total parameters: {lora_total:,}')
print(f'Trainable parameters: {lora_trainable:,}')
print(f'Trainable fraction: {lora_trainable/lora_total:.2%}')

print("\nTrainable parameters:")
for name, param in lora_model.named_parameters():
    if param.requires_grad:
        print(name, tuple(param.shape))
```

 You should see something conceptually like:

```
fc1.A
fc1.B
fc2.A
fc2.B
fc3.A
fc3.B
fc4.A
fc4.B
classifier.weight
classifier.bias
```

 The original FC weights should **not** appear as trainable.

---

 # 6\. Add an explicit test that LoRA initially behaves like the original model

 This is a very useful test and I strongly recommend adding it.

 Because `B` starts at zero:

 $$
BA=0
$$

 Therefore, immediately after creating the LoRA layer:

 $$
Wx+b+BAx = Wx+b
$$

 So test it:

```
# Create a random input
test_x = torch.randn(
    16,
    lora_model.fc1.in_features,
    device=device
)

# Original pretrained fc1
original_fc1 = pt_model.fc1

# LoRA fc1
lora_fc1 = lora_model.fc1

with torch.no_grad():
    original_output = original_fc1(test_x)
    lora_output = lora_fc1(test_x)

torch.testing.assert_close(
    original_output,
    lora_output,
    rtol=1e-5,
    atol=1e-5
)

print("✓ Initial LoRA output matches original layer")
```

 This confirms your initialization is correct.

---

 # 7\. Train the LoRA model

 Use exactly the same fine-tuning settings as the dense baseline:

```
lora_metrics = train_model(
    lora_model,
    train_loader_target,
    val_loader_target,
    num_ft_epochs,
    lr,
    l2_reg,
    title=f'LoRA Fine-tuning ({target_data.name})',
    baseline_metrics=baseline_ft_metrics
)

print(
    f'Best LoRA-FT accuracy: '
    f'{lora_metrics.best_val_acc:.1%}'
)
```

 Then compare:

```
baseline_acc = baseline_ft_metrics.best_val_acc
lora_acc = lora_metrics.best_val_acc

diff = lora_acc - baseline_acc

print(f"Dense baseline: {baseline_acc:.1%}")
print(f"LoRA:           {lora_acc:.1%}")
print(f"Difference:     {diff:+.1%}")
```

---

 # 8\. What result should you expect?

 The assignment explicitly says the LoRA validation accuracy should normally be **within a few percentage points of the dense baseline**.

 Don't worry if LoRA is slightly worse.

 For example:

```
Dense FT: 72.5%
LoRA:     70.8%
Difference: -1.7 percentage points
```

 would be a perfectly reasonable result.

 The objective isn't necessarily:

 > LoRA must beat dense fine-tuning.

 The objective is:

 > LoRA should retain comparable performance while dramatically reducing the number of trainable parameters.

---

 # 9\. Try different values of `r`

 This is explicitly required by the assignment.

 Try:

```
r = 2
r = 4
r = 8
r = 16
r = 32
```

 Record the results.

 For example:

 | Rank $r$ | Trainable parameters | Validation accuracy |
| --- | --- | --- |
| 2 | ... | ... |
| 4 | ... | ... |
| 8 | ... | ... |
| 16 | ... | ... |
| 32 | ... | ... |
| Dense | ... | ... |

Then make a plot:

```
r_values = [2, 4, 8, 16, 32]
accuracies = [
    acc_r2,
    acc_r4,
    acc_r8,
    acc_r16,
    acc_r32
]

plt.figure(figsize=(7, 4))

plt.plot(
    r_values,
    np.array(accuracies) * 100,
    marker='o',
    label='LoRA'
)

plt.axhline(
    baseline_ft_metrics.best_val_acc * 100,
    color='red',
    linestyle='--',
    label='Dense FT baseline'
)

plt.xlabel('LoRA rank r')
plt.ylabel('Validation accuracy (%)')
plt.title('LoRA rank vs validation accuracy')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

---

 # 10\. Why does increasing `r` matter?

 This is an important conceptual point for the report.

 For each layer:

 $$
\Delta W = BA
$$

 The maximum rank of $BA$ is $r$.

 Therefore:

 - $r=1$: extremely constrained update
- $r=2$: very low-rank update
- $r=8$: more expressive
- $r=32$: much more expressive
- large $r$: approaches full-rank adaptation

 But increasing $r$ also increases trainable parameters.

 So LoRA involves a trade-off:

```
small r
│
├── fewer parameters
├── cheaper training
├── less expressive adaptation
│
large r
│
├── more parameters
├── more computation
├── more expressive adaptation
└── potentially better accuracy
```

---

 # 11\. Compare model parameter counts

 The assignment asks you to compare the models.

 Use:

```
print("DENSE FINE-TUNING")
dense_total, dense_trainable = num_parameters(baseline_ft_model)

print(f"Total:     {dense_total:,}")
print(f"Trainable: {dense_trainable:,}")

print("\nLoRA")
lora_total, lora_trainable = num_parameters(lora_model)

print(f"Total:     {lora_total:,}")
print(f"Trainable: {lora_trainable:,}")

print(
    f"\nReduction in trainable parameters: "
    f"{1 - lora_trainable/dense_trainable:.1%}"
)
```

 This is one of your strongest results for the report.

---

 # 12\. Compare state-dict sizes

 The notebook gives you:

```
get_state_dict_size()
```

 Use:

```
dense_state_size = get_state_dict_size(
    baseline_ft_model.state_dict()
)

lora_state_size = get_state_dict_size(
    lora_model.state_dict()
)

print(f"Dense state dict parameters: {dense_state_size:,}")
print(f"LoRA state dict parameters:  {lora_state_size:,}")
```

 However, there is an important subtlety.

 ### The LoRA model's full state dict may not be dramatically smaller

 Why?

 Because your LoRA model still contains:

```
original_layer.weight
original_layer.bias
A
B
```

 The original frozen weights are still stored.

 So:

 > **Trainable parameter reduction ≠ necessarily model checkpoint size reduction.**

 This is an excellent point to mention in your report.

 If you save only A/B adapters, then the fine-tuning checkpoint can become dramatically smaller.

---

 # 13\. Compare optimiser state

 AdamW maintains extra state for trainable parameters, typically including quantities such as:

```
first moment
second moment
```

 Therefore, if LoRA has dramatically fewer trainable parameters, its optimizer state should also be dramatically smaller.

 You can obtain the optimizers explicitly:

```
dense_optimizer = torch.optim.AdamW(
    baseline_ft_model.parameters(),
    lr=lr,
    weight_decay=l2_reg
)

lora_optimizer = torch.optim.AdamW(
    lora_model.parameters(),
    lr=lr,
    weight_decay=l2_reg
)
```

 But note that these are **new optimizers**, so their state dictionaries won't contain the accumulated Adam states until training has occurred.

 A better approach is to modify `train_model()` if you need the optimizer after training, or create the optimizer outside the function.

 For the assignment, the conceptual explanation is:

 > AdamW stores optimizer states for trainable parameters. Since LoRA trains far fewer parameters, its optimizer state requires substantially less memory.

---

 # 14\. Compare forward/backward time

 The assignment asks an interesting question:

 > Is LoRA faster or slower?

 You might expect LoRA to be faster because it trains fewer parameters.

 But **the forward pass can actually be slower**.

 Why?

 Dense layer:

 $$
Wx
$$

 LoRA:

 $$
Wx + BAx
$$

 So LoRA performs the original frozen matrix multiplication **plus additional low-rank matrix multiplications**.

 Therefore:

```
Forward:
Dense → Wx

LoRA → Wx + BAx
             ↑
       additional operations
```

 LoRA can therefore have **more forward computation** even though it has fewer trainable parameters.

 The major savings are in:

 - gradient computation
- optimizer updates
- optimizer state
- trainable parameter memory

 This distinction is very important.

---

 # 15\. A simple timing experiment

 You can use:

```
import time

def benchmark_forward_backward(
    model,
    loader,
    num_batches=20
):
    model.train()

    loss_func = nn.CrossEntropyLoss()

    times = []

    iterator = iter(loader)

    for _ in range(num_batches):

        x, y = next(iterator)

        if device == 'cuda':
            torch.cuda.synchronize()

        start = time.perf_counter()

        model.zero_grad()

        pred = model(x)

        loss = loss_func(pred, y)

        loss.backward()

        if device == 'cuda':
            torch.cuda.synchronize()

        end = time.perf_counter()

        times.append(end - start)

    return np.mean(times), np.std(times)
```

 Then:

```
dense_time = benchmark_forward_backward(
    baseline_ft_model,
    train_loader_target
)

lora_time = benchmark_forward_backward(
    lora_model,
    train_loader_target
)

print("Dense:", dense_time)
print("LoRA:", lora_time)
```

 Don't be surprised if the difference is small or if LoRA is even slightly slower per batch.

---

 # 16\. One issue in the supplied notebook

 There is another potential problem in this section:

```
lora_model.load_state_dict(pt_state_dict, strict=False)
```

 Remember that earlier they did:

```
del pt_state_dict['classifier.weight']
del pt_state_dict['classifier.bias']
```

 So `pt_state_dict` no longer contains the classifier.

 That's intentional.

 The model gets:

```
pretrained convolution layers
pretrained MLP layers
new random classifier
```

 Then LoRA replaces the MLP layers.

 That's exactly what you want.

---

 # 17\. Recommended complete LoRA section

 If you want a compact replacement for the incomplete section in the notebook, use this:

```
# ============================================================
# LoRA FINE-TUNING
# ============================================================

torch.manual_seed(0)

# Create target model
lora_model = ConvMLP(
    num_classes=target_data.num_classes
).to(device)

# Load pretrained source-domain weights
lora_model.load_state_dict(
    pt_state_dict,
    strict=False
)

# Freeze convolutional layers
lora_model.conv1.requires_grad_(False)
lora_model.conv2.requires_grad_(False)

class LoraLayer(nn.Module):

    def __init__(self, original_layer, r):
        super().__init__()

        self.original_layer = original_layer
        self.r = r

        # Dimensions
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features

        # Freeze original parameters
        self.original_layer.weight.requires_grad_(False)
        self.original_layer.bias.requires_grad_(False)

        # A: r x input
        self.A = nn.Parameter(
            torch.randn(
                r,
                self.in_features,
                device=original_layer.weight.device
            ) * 0.01
        )

        # B: output x r
        # Zero initialization ensures BA = 0 initially
        self.B = nn.Parameter(
            torch.zeros(
                self.out_features,
                r,
                device=original_layer.weight.device
            )
        )

        # Scaling
        self.scaling = 1.0 / r

    def forward(self, x):

        # Frozen original layer
        original_output = self.original_layer(x)

        # LoRA update BAx
        lora_output = F.linear(
            F.linear(x, self.A),
            self.B
        )

        return original_output + self.scaling * lora_output

# ------------------------------------------------------------
# Choose LoRA rank
# ------------------------------------------------------------

r = 8

# ------------------------------------------------------------
# Replace fully-connected layers
# ------------------------------------------------------------

lora_model.fc1 = LoraLayer(
    lora_model.fc1, r
)

lora_model.fc2 = LoraLayer(
    lora_model.fc2, r
)

lora_model.fc3 = LoraLayer(
    lora_model.fc3, r
)

lora_model.fc4 = LoraLayer(
    lora_model.fc4, r
)

# ------------------------------------------------------------
# Check parameters
# ------------------------------------------------------------

lora_total, lora_trainable = num_parameters(lora_model)

print(
    f"LoRA model total parameters: "
    f"{lora_total:,}"
)

print(
    f"LoRA trainable parameters: "
    f"{lora_trainable:,}"
)

print(
    f"Trainable fraction: "
    f"{lora_trainable/lora_total:.2%}"
)

print("\nTrainable parameters:")

for name, param in lora_model.named_parameters():
    if param.requires_grad:
        print(
            f"  {name}: {tuple(param.shape)}"
        )

# ------------------------------------------------------------
# Verify original weights are frozen
# ------------------------------------------------------------

assert not lora_model.fc1.original_layer.weight.requires_grad
assert not lora_model.fc2.original_layer.weight.requires_grad
assert not lora_model.fc3.original_layer.weight.requires_grad
assert not lora_model.fc4.original_layer.weight.requires_grad

print("\n✓ Original FC weights are frozen")

# ------------------------------------------------------------
# Train
# ------------------------------------------------------------

lora_metrics = train_model(
    lora_model,
    train_loader_target,
    val_loader_target,
    num_ft_epochs,
    lr,
    l2_reg,
    title=f'LoRA Fine-tuning ({target_data.name})',
    baseline_metrics=baseline_ft_metrics
)

print(
    f"\nBest LoRA-FT accuracy: "
    f"{lora_metrics.best_val_acc:.1%}"
)

# ------------------------------------------------------------
# Compare against dense baseline
# ------------------------------------------------------------

baseline_acc = baseline_ft_metrics.best_val_acc
lora_acc = lora_metrics.best_val_acc

difference = lora_acc - baseline_acc

print(
    f"\nDense baseline: {baseline_acc:.1%}"
)

print(
    f"LoRA:           {lora_acc:.1%}"
)

print(
    f"Difference:     {difference:+.1%}"
)

print(
    f"Trainable parameter reduction: "
    f"{1 - lora_trainable/dense_trainable:.1%}"
)
```

---

 # 18\. What your report should say

 The report can be structured very simply.

 ### Method

 Explain:

 > The model was first pretrained on CIFAR-10 and then transferred to a five-class subset of CIFAR-100. For the dense baseline, the convolutional layers were frozen while the fully connected layers and classifier were fine-tuned. For LoRA, the pretrained fully connected weight matrices were frozen and adapted using low-rank matrices $A$ and $B$, such that the effective weight update was $\Delta W = BA$.

 ### Results

 Include a table:

 | Method | Trainable parameters | Val. accuracy |
| --- | --- | --- |
| Dense FT | XXX | XX.X% |
| LoRA r=2 | XXX | XX.X% |
| LoRA r=4 | XXX | XX.X% |
| LoRA r=8 | XXX | XX.X% |
| LoRA r=16 | XXX | XX.X% |

Then include your training curves.

 ### Discussion

 A good conclusion would be along these lines:

 > LoRA achieved validation accuracy close to the dense fine-tuning baseline while updating substantially fewer parameters. The reduction is achieved by representing the weight update as a low-rank product $BA$, rather than directly updating the full weight matrix. Increasing the rank generally provides greater adaptation capacity but also increases the number of trainable parameters. An important observation is that reducing trainable parameters does not necessarily make each forward pass faster, because LoRA introduces additional matrix multiplications. The main computational benefits are reduced gradient and optimiser-state requirements.

---

 ## The three things I would focus on for the practical

 If you're preparing for an assessment, make sure you can explain these **without looking at the notebook**:

 1. **Why LoRA works**\

   $$
   W' = W + BA
   $$
2. **Why it saves parameters**\

   $$
   d_{out}d_{in}
      \quad\rightarrow\quad
      r(d_{in}+d_{out})
   $$
3. **Why `B` is initialized to zero**
    So that initially:\

   $$
   BA=0
   $$
   \
    meaning the LoRA model initially behaves exactly like the pretrained model rather than abruptly changing its predictions.

 Those three ideas are essentially the heart of the assignment.
