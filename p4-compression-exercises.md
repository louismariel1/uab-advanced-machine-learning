Absolutely. Based on the three chunks, I would structure the practical work as a **progressive lab series**: start with measuring model size/performance, then implement quantisation and pruning, and finish with PEFT/LoRA. The key is that students should not merely call a library function—they should **measure the trade-offs** between accuracy, memory, training cost, and inference speed.

 ## Suggested practical exercise sequence

 | Exercise | Main concept | Difficulty | What students practise |
| --- | --- | --- | --- |
| 1 | Baseline model | ⭐ | Parameters, memory, accuracy, inference |
| 2 | Weight quantisation | ⭐⭐ | FP32 → INT8, error and memory |
| 3 | PTQ vs QAT | ⭐⭐⭐ | Calibration, fake quantisation |
| 4 | Magnitude pruning | ⭐⭐ | Sparsity and accuracy |
| 5 | Structured pruning | ⭐⭐⭐ | Channels/neurons and real speedups |
| 6 | Iterative pruning | ⭐⭐⭐ | Pruning + fine-tuning |
| 7 | Combine pruning + quantisation | ⭐⭐⭐ | Compression pipeline |
| 8 | Layer freezing | ⭐⭐ | Parameter-efficient fine-tuning |
| 9 | Build an adapter | ⭐⭐⭐ | Bottlenecks and residual adaptation |
| 10 | Implement LoRA from scratch | ⭐⭐⭐⭐ | Low-rank updates |
| 11 | QLoRA-style experiment | ⭐⭐⭐⭐ | Quantisation + LoRA |
| 12 | Compression challenge | ⭐⭐⭐⭐⭐ | Choose and justify a complete strategy |

---

 # Exercise 1 — Establish a baseline

 ### Goal

 Before compressing anything, students need to know what they are trying to improve.

 Use a small CNN on MNIST or CIFAR-10.

```
import torch
import torch.nn as nn

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def parameter_memory(model, bytes_per_parameter=4):
    return count_parameters(model) * bytes_per_parameter

model = MyCNN()

params = count_parameters(model)
memory_mb = parameter_memory(model) / 1024**2

print(f"Parameters: {params:,}")
print(f"Approx. FP32 weight memory: {memory_mb:.2f} MB")
```

 Students then measure:

 - number of parameters
- FP32 model size
- validation/test accuracy
- inference time
- training time

 ### Questions

 1. How many parameters does the model contain?
2. How much memory do the weights require in FP32?
3. What happens if we halve the number of parameters?
4. Is fewer parameters automatically equivalent to faster inference?

 ### Solution

 For $N$ parameters stored using FP32:

 $$
\text{memory}=4N\text{ bytes}.
$$

 This establishes the baseline against which every compression method is evaluated.

---

 # Exercise 2 — Implement simple quantisation

 This is an excellent exercise because students can understand quantisation **mathematically before using PyTorch's more sophisticated implementations**.

 Suppose:

 $$
x_{\min}=-2,\qquad x_{\max}=2
$$

 and we want to represent $x$ using that conventional hardware can process efficiently. Random individual zeros may require specialised sparse kernels to produce an signed INT8:

 $$
[-128,127].
$$

 A simple affine quantisation scheme is:

 $$
s=\frac{x_{\max}-x_{\min}}{q_{\max}-q_{\min}}
$$

 and

 $$
z=q_{\min}-\frac{x_{\min}}{s}.
$$

 Then:

 $$
q=\operatorname{round}\left(\frac{x}{s}+z\right).
$$

 Dequantisation is:

 $$
\hat{x}=s(q-z).
$$

 ### Student task

 Implement:

```
def quantise(x, num_bits=8):
    qmin = -(2 ** (num_bits - 1))
    qmax = (2 ** (num_bits - 1)) - 1

    xmin = x.min()
    xmax = x.max()

    scale = (xmax - xmin) / (qmax - qmin)
    zero_point = qmin - xmin / scale

    q = torch.round(x / scale + zero_point)
    q = torch.clamp(q, qmin, qmax)

    return q, scale, zero_point
```

 Then implement dequantisation.

 ### Questions

 - What information is lost?
- What happens if the tensor contains an extreme outlier?
- What happens when you reduce from 8 bits to 4 bits?
- What happens to the reconstruction error?

 ### Important experiment

 Run:

```
for bits in [8, 6, 4, 2]:
    ...
```

 and plot:

 $$
\text{quantisation error}
$$

 against

 $$
\text{number of bits}.
$$

 Students should discover the fundamental trade-off:

 > **Fewer bits → less memory → larger quantisation error.**

---

 # Exercise 3 — Post-training quantisation

 Now students use an actual trained network.

 ### Task

 Train an FP32 model and then apply post-training quantisation.

 Compare:

```
FP32 model
      ↓
PTQ
      ↓
INT8 model
```

 Measure:

 | Metric | FP32 | INT8 |
| --- | --- | --- |
| Accuracy |  |  |
| Model size |  |  |
| Inference time |  |  |
| Memory |  |  |

### Questions

 1. Did accuracy decrease?
2. Did model size decrease?
3. Did inference become faster?
4. If model size decreased but inference did not become faster, why?

 ### Expected conclusion

 Students should discover the distinction between:

 > **compression benefit**

 and

 > **hardware acceleration benefit**.

 Quantisation can reduce storage even when the hardware does not provide a meaningful speedup.

---

 # Exercise 4 — PTQ vs QAT

 This is a very good exam/practical exercise.

 Create three models:

```
Model A: FP32
Model B: PTQ
Model C: QAT
```

 Compare their accuracy.

 ### Student task

 Train:

```
FP32 model
    ↓
train
    ↓
PTQ
```

 and separately:

```
FP32 model
    ↓
insert fake quantisation
    ↓
train
    ↓
quantised model
```

 ### Questions

 - Why can QAT outperform PTQ?
- What does "fake quantisation" mean?
- Why is QAT more expensive?
- If PTQ already gives almost identical accuracy, would you choose QAT?

 ### Expected answer

 QAT exposes the model to quantisation effects during training, allowing it to learn parameters that are robust to those errors.

 But QAT costs additional training time, so PTQ is preferable when it already meets the accuracy requirement.

---

 # Exercise 5 — Magnitude pruning

 This should be one of the first pruning exercises.

 Train a model and examine the weight distribution:

```
weights = model.fc1.weight.detach()

plt.hist(weights.flatten().numpy(), bins=100)
plt.show()
```

 Students should notice that many weights may be close to zero.

 ### Task

 Prune the smallest 20%:

```
threshold = torch.quantile(weights.abs(), 0.20)

mask = weights.abs() > threshold

weights *= mask
```

 Then calculate:

```
sparsity = (weights == 0).float().mean()

print(sparsity)
```

 ### Repeat

 Try:

```
10%
20%
40%
60%
80%
90%
```

 and plot:

 $$
\text{sparsity} \quad\text{vs}\quad \text{accuracy}.
$$

 ### The important discovery

 Students should see that accuracy does not necessarily decrease dramatically at first.

 This leads directly to the lecture's question:

 > **How can we remove so many parameters without destroying the model?**

---

 # Exercise 6 — Pruning + fine-tuning

 The previous exercise leaves the model damaged.

 Now students perform:

```
Train
 ↓
Prune
 ↓
Fine-tune
 ↓
Evaluate
```

 Compare this with:

```
Train
 ↓
Prune
 ↓
Evaluate
```

 ### Questions

 - How much accuracy is recovered?
- Why does fine-tuning help?
- Why might iterative pruning be better than pruning everything at once?

 ### Extension

 Implement:

```
Train
 ↓
Prune 20%
 ↓
Fine-tune
 ↓
Prune another 20%
 ↓
Fine-tune
 ↓
...
```

 This recreates the basic idea of **iterative pruning** from the lecture.

---

 # Exercise 7 — Structured vs unstructured pruning

 This is particularly valuable because students often misunderstand pruning.

 Compare:

 ### Unstructured

```
[0.2, 0, -0.4, 0, 0.1, 0, ...]
```

 with:

 ### Structured

```
remove entire neuron/channel/filter
```

 ### Practical experiment

 Train a CNN.

 Then remove:

 - individual weights
- entire filters/channels

 Measure:

 - number of parameters
- FLOPs
- model size
- inference time
- accuracy

 ### Key question

 > If both models remove 50% of the parameters, why might structured pruning give a larger real-world speedup?

 ### Answer

 Because structured pruning produces smaller dense tensors that conventional hardware can process efficiently. Random individual zeros may require specialised sparse kernels to produce an actual speedup.

---

 # Exercise 8 — Layer freezing

 Now move into parameter-efficient fine-tuning.

 Take a pretrained network and freeze everything except the final layer:

```
for param in model.parameters():
    param.requires_grad = False

for param in model.classifier.parameters():
    param.requires_grad = True
```

 Calculate:

```
total = sum(p.numel() for p in model.parameters())

trainable = sum(
    p.numel() for p in model.parameters()
    if p.requires_grad
)

print(total)
print(trainable)
print(trainable / total)
```

 ### Students answer

 - What percentage of parameters are trainable?
- How much optimiser state is required?
- Does freezing reduce inference cost?

 The last question is particularly important:

 > **No. Freezing is primarily a training-time optimisation.**

---

 # Exercise 9 — Build an adapter

 Now students implement the architecture from the lecture.

 A simple adapter:

 $$
x
\rightarrow
W_{\text{down}}
\rightarrow
\text{nonlinearity}
\rightarrow
W_{\text{up}}
\rightarrow
x+\Delta x.
$$

 For example:

```
class Adapter(nn.Module):
    def __init__(self, d_model, bottleneck):
        super().__init__()

        self.down = nn.Linear(d_model, bottleneck)
        self.up = nn.Linear(bottleneck, d_model)

    def forward(self, x):
        return x + self.up(torch.relu(self.down(x)))
```

 Freeze the original model and train only the adapter.

 ### Experiment

 Try:

```
bottleneck = 4
bottleneck = 8
bottleneck = 16
bottleneck = 32
```

 Plot:

 $$
\text{trainable parameters}
$$

 against

 $$
\text{accuracy}.
$$

 This gives students an intuitive understanding of the **parameter-efficiency trade-off**.

---

 # Exercise 10 — Implement LoRA from scratch

 This should probably be the **capstone coding exercise** because it connects directly to the lecture.

 Start with:

```
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_features, in_features)
        )

        self.weight.requires_grad = False

        self.A = nn.Parameter(
            torch.randn(rank, in_features)
        )

        self.B = nn.Parameter(
            torch.zeros(out_features, rank)
        )

    def forward(self, x):
        base = x @ self.weight.T
        update = x @ self.A.T @ self.B.T

        return base + update
```

 The central equation is:

 $$
W' = W + BA.
$$

 Students should verify the dimensions:

 $$
W\in\mathbb{R}^{d\times d}
$$

 $$
A\in\mathbb{R}^{r\times d}
$$

 $$
B\in\mathbb{R}^{d\times r}.
$$

 Therefore:

 $$
BA\in\mathbb{R}^{d\times d}.
$$

 ### Critical experiment

 Set:

```
rank = 1
rank = 2
rank = 4
rank = 8
rank = 16
rank = 32
```

 Measure:

 - trainable parameters
- accuracy
- training time
- memory

 Students should see why:

 $$
2dr \ll d^2
$$

 when

 $$
r\ll d.
$$

---

 # Exercise 11 — LoRA weight merging

 This exercise tests whether students really understand why LoRA has essentially no inference overhead after merging.

 During training:

 $$
W' = W + BA.
$$

 After training, construct:

```
merged_weight = weight + B @ A
```

 Then replace the original weight.

 Compare:

```
Base model + LoRA
```

 against:

```
Merged model
```

 They should produce essentially the same outputs.

 ### Question

 Why is LoRA different from an adapter here?

 ### Answer

 A conventional adapter introduces additional computation into the forward pass.

 LoRA's low-rank update can be mathematically merged into the original weight matrix:

 $$
W_{\text{merged}}=W+BA.
$$

 Therefore, after merging, the architecture can be identical to the original model.

---

 # Exercise 12 — QLoRA-style experiment

 This is an excellent final technical exercise.

 Students combine:

 $$
\boxed{\text{quantisation}+\text{LoRA}}
$$

 Conceptually:

```
Large pretrained model
        ↓
     quantise
        ↓
 frozen low-precision model
        +
     LoRA A,B
        ↓
    fine-tuning
```

 Students compare:

 | Method | Base precision | Trainable parameters | Accuracy | Memory |
| --- | --- | --- | --- | --- |
| Full FT | FP32 | 100% |  |  |
| Frozen | FP32 | \~0% |  |  |
| LoRA | FP32 | small |  |  |
| Quantised | INT8/4 | small/0 |  |  |
| QLoRA | INT8/4 | small |  |  |

The key question is:

 > **Why can combining two individually useful techniques produce a much larger practical benefit?**

 Because the methods attack different bottlenecks:

 - quantisation reduces the representation cost of the **base model**
- LoRA reduces the number of parameters that need to be **updated/stored for adaptation**

---

 # Final exercise — The compression challenge

 I would make this the final assessed practical.

 ## Scenario

 You have a pretrained model with:

 - 1 billion parameters
- FP32 weights
- limited GPU memory
- a new downstream task
- target accuracy within 1% of the original model
- multiple downstream tasks need to be supported.

 Students are told:

 > You may use any techniques from the lecture. Design and implement a compression/fine-tuning strategy.

 They must justify their choices.

 ### Possible solution

 A strong solution is:

```
                 PRETRAINED MODEL
                       │
                       ▼
                  Quantisation
                       │
                       ▼
              Frozen low-precision
                  base model
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
       LoRA A/B      LoRA A/B     LoRA A/B
       Task 1        Task 2       Task 3
```

 The base model is stored once.

 Each task only requires a small LoRA module.

 At deployment:

```
Base model + Task 1 LoRA
Base model + Task 2 LoRA
Base model + Task 3 LoRA
```

 can be selected as required.

---

 # One particularly important exercise: "Which method should I choose?"

 I would give students scenarios rather than simply asking them to implement algorithms.

 ### Scenario A

 > The model is already trained. You want to deploy it to a phone and cannot afford much memory.

 **Best candidates:** quantisation and possibly pruning.

---

 ### Scenario B

 > You have a huge pretrained LLM and need to fine-tune it for one task, but your GPU cannot store all optimiser states.

 **Best candidate:** LoRA/PEFT.

---

 ### Scenario C

 > PTQ causes a significant accuracy drop and you have access to training data.

 **Best candidate:** QAT.

---

 ### Scenario D

 > You need actual GPU speedup, not merely a smaller file.

 **Best candidate:** structured pruning and/or hardware-supported quantisation.

---

 ### Scenario E

 > You need 50 different task-specific versions of the same LLM.

 **Best candidate:** frozen base model \+ separate LoRA/adapters.

---

 ### Scenario F

 > You want the smallest possible training footprint for a large model.

 **Best candidate:** quantised base model \+ LoRA/QLoRA-style approach.

---

 ## What I would assess

 For a university practical, I would not assess merely whether the code runs. I would give marks for **measurement and reasoning**:

 | Component | Weight |
| --- | --- |
| Correct implementation | 25% |
| Accuracy evaluation | 15% |
| Memory/parameter analysis | 15% |
| Runtime analysis | 10% |
| Understanding trade-offs | 15% |
| Experimental design | 10% |
| Explanation/interpretation | 10% |

The most important principle for the whole practical series is:

 $$
\boxed{
\text{Compression is not simply “make the model smaller”.}
}
$$

 Students should learn to ask:

 > **What resource am I trying to save?**

 > **What information am I throwing away or approximating?**

 > **Does the hardware actually benefit from my compression?**

 > **What happens to accuracy?**

 > **Is the saving during training, inference, storage, or all three?**

 > **Can two techniques be combined?**

 Those questions tie together essentially the entire lecture: **quantisation, pruning, layer freezing, adapters, low-rank factorisation, and LoRA**.
