# Step-by-step solutions: Model Compression

 We'll cover:

 1. Baseline model and performance measurement
2. Quantisation — PTQ
3. Quantisation-aware training
4. Unstructured pruning
5. Structured pruning
6. Iterative pruning
7. Combining pruning + quantisation
8. LoRA from scratch
9. LoRA parameter-count comparison
10. Comparing all compression methods

---

 # Exercise 1 — Establish a baseline

 ### Goal

 Before compressing anything, students should answer:

 > **How large is my model, how accurate is it, and how fast is it?**

 We'll use MNIST and a simple MLP.

 ### Step 1 — Imports

```
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
```

 ### Step 2 — Load MNIST

```
transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128)
```

---

 ### Step 3 — Create the model

```
class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.network(x)
```

 Create it:

```
model = MLP()
```

---

 ### Step 4 — Count parameters

```
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

print(count_parameters(model))
```

 The important concept is:

 $$
\text{number of parameters}
=
\sum_l \text{parameters in layer } l
$$

 This gives us our **baseline**.

---

 ### Step 5 — Train

```
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(5):

    model.train()

    for x, y in train_loader:

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        output = model(x)

        loss = criterion(output, y)

        loss.backward()

        optimizer.step()

    print(f"Epoch {epoch+1} complete")
```

---

 ### Step 6 — Measure accuracy

```
def evaluate(model, loader):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for x, y in loader:

            x = x.to(device)
            y = y.to(device)

            output = model(x)

            predictions = output.argmax(dim=1)

            correct += (predictions == y).sum().item()
            total += y.size(0)

    return correct / total
```

 Then:

```
accuracy = evaluate(model, test_loader)

print("Accuracy:", accuracy)
```

---

 ### Step 7 — Measure inference time

```
def measure_inference_time(model, loader):

    model.eval()

    start = time.time()

    with torch.no_grad():

        for x, _ in loader:

            x = x.to(device)

            _ = model(x)

    end = time.time()

    return end - start
```

```
baseline_time = measure_inference_time(model, test_loader)

print("Inference time:", baseline_time)
```

 Now students have three important baseline measurements:

 | Metric | Baseline |
| --- | --- |
| Parameters | ... |
| Accuracy | ... |
| Inference time | ... |

### Teaching point

 This is essential.

 Compression is **not automatically successful** just because the model has fewer parameters.

 We want:

 $$
\boxed{
\text{smaller model}
+
\text{similar accuracy}
+
\text{possibly faster inference}
}
$$

---

 # Exercise 2 — Post-training quantisation

 ## Question

 > Can we represent the same model using fewer bits?

 Normally a neural-network weight might be:

 $$
\text{float32}=32\text{ bits}
$$

 Quantisation might use:

 $$
\text{int8}=8\text{ bits}
$$

 So, ignoring overhead:

 $$
\frac{32}{8}=4
$$

 times less memory for the quantised values.

---

 ## Step 1 — Save the baseline

```
torch.save(model.state_dict(), "model_fp32.pth")
```

 Check its size:

```
import os

size_fp32 = os.path.getsize("model_fp32.pth")

print("FP32 size:", size_fp32 / 1024, "KB")
```

---

 ## Step 2 — Dynamic quantisation

 For this MLP, PyTorch provides a simple form of post-training quantisation:

```
quantized_model = torch.ao.quantization.quantize_dynamic(
    model.cpu(),
    {nn.Linear},
    dtype=torch.qint8
)
```

 Notice something important:

 > We did **not retrain the model**.

 That is why this is a form of:

 **Post-training quantisation (PTQ).**

---

 ## Step 3 — Save the quantised model

```
torch.save(
    quantized_model.state_dict(),
    "model_int8.pth"
)
```

```
size_int8 = os.path.getsize("model_int8.pth")

print("INT8 size:", size_int8 / 1024, "KB")
```

---

 ## Step 4 — Compare sizes

```
compression_ratio = size_fp32 / size_int8

print("Compression ratio:", compression_ratio)
```

---

 ## Step 5 — Check accuracy

```
quantized_accuracy = evaluate(
    quantized_model,
    test_loader
)

print("Quantized accuracy:", quantized_accuracy)
```

 You may need a CPU version of the evaluation function because this model is now on CPU.

---

 ## What should students conclude?

 Ask:

 > **Why didn't the accuracy necessarily fall by 4× just because the representation uses 4× fewer bits?**

 Because quantisation is not simply "throwing away 75% of the information."

 Instead, many floating-point values are mapped to a discrete set of values.

 Conceptually:

 $$
w_{\text{float}}
\rightarrow
q_{\text{int8}}
$$

 and during computation:

 $$
q_{\text{int8}}
\rightarrow
\text{approximately }w_{\text{float}}
$$

 The important word is **approximately**.

---

 # Exercise 3 — Quantisation error

 Now make the students see the quantisation error directly.

 Suppose:

 $$
w = 0.731
$$

 and the quantised representation gives:

 $$
\hat w = 0.73
$$

 Then:

 $$
e=w-\hat w
$$

 so:

 $$
e=0.001
$$

---

 ## Exercise

 Create a simple quantiser.

```
def simple_quantize(x, scale):

    q = torch.round(x / scale)

    return q
```

 Test it:

```
x = torch.tensor([
    -1.0,
    -0.75,
    -0.25,
    0.0,
    0.32,
    0.71,
    1.0
])

scale = 0.1

q = simple_quantize(x, scale)

x_hat = q * scale

print("Original:")
print(x)

print("Quantized:")
print(q)

print("Reconstructed:")
print(x_hat)
```

 Then:

```
error = x - x_hat

print("Quantization error:")
print(error)
```

---

 ## Exam question

 > Why can quantisation reduce memory without necessarily producing a large accuracy loss?

 ### Answer

 Because many model weights are redundant or do not require full 32-bit precision. Quantisation maps floating-point values to a smaller discrete representation while preserving the important structure of the learned parameters. If the quantisation error is sufficiently small, the model's predictions remain approximately unchanged.

---

 # Exercise 4 — Post-training quantisation vs QAT

 Now introduce the key distinction.

 ### PTQ

```
Train FP32 model
       ↓
Quantise
       ↓
Deploy
```

 ### QAT

```
Train
 ↓
Simulate quantisation
 ↓
Model learns to compensate
 ↓
Deploy quantised model
```

---

 ## Question

 > Why might QAT achieve higher accuracy than PTQ?

 ### Answer

 Because during QAT the model is exposed to quantisation noise during training.

 The model can modify its parameters so that the final quantisation error has less effect on predictions.

 In simplified form:

 $$
W
\rightarrow
Q(W)
$$

 during training.

 The model therefore learns parameters that are robust to the eventual quantisation.

---

 # Exercise 5 — Unstructured pruning

 Now move from:

 > "Can we make each parameter smaller?"

 to:

 > "Do we need all the parameters at all?"

---

 ## Step 1 — Inspect the weights

```
for name, parameter in model.named_parameters():

    if "weight" in name:
        print(name)
        print(parameter.data.abs().mean())
```

---

 ## Step 2 — Magnitude pruning

 The lecture describes:

 > Remove weights with small magnitude.

 For example:

 $$
|w| \approx 0
$$

 is considered less important.

 Using PyTorch:

```
import torch.nn.utils.prune as prune
```

 Apply pruning:

```
for module in model.modules():

    if isinstance(module, nn.Linear):

        prune.l1_unstructured(
            module,
            name="weight",
            amount=0.5
        )
```

 This removes 50% of the weights according to magnitude.

---

 ## Step 3 — Measure sparsity

```
def calculate_sparsity(model):

    total = 0
    zeros = 0

    for parameter in model.parameters():

        total += parameter.numel()
        zeros += (parameter == 0).sum().item()

    return zeros / total
```

```
print(
    "Sparsity:",
    calculate_sparsity(model)
)
```

 You should see approximately:

```
Sparsity ≈ 0.5
```

---

 ## Step 4 — Evaluate

```
accuracy_pruned = evaluate(model, test_loader)

print("Pruned accuracy:", accuracy_pruned)
```

 The interesting question is:

 > We deleted half the weights. Why might accuracy remain surprisingly high?

 Because the network is often **overparameterised**.

 Many parameters are not individually essential.

---

 # Exercise 6 — Fine-tuning after pruning

 Pruning normally does not have to be the final step.

 The model can recover from pruning.

 Conceptually:

```
Train
 ↓
Prune
 ↓
Fine-tune
 ↓
Prune again
 ↓
Fine-tune
```

 For the simple experiment:

```
for epoch in range(2):

    model.train()

    for x, y in train_loader:

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        output = model(x)

        loss = criterion(output, y)

        loss.backward()

        optimizer.step()
```

 Then evaluate again.

 ### Key question

 > Why can fine-tuning recover accuracy after pruning?

 Because pruning changes the model's effective function. Fine-tuning allows the remaining weights to adapt and compensate for the removed connections.

---

 # Exercise 7 — Iterative pruning

 Instead of immediately removing 90%:

```
90% pruning
```

 try:

```
50%
 ↓
fine-tune

70%
 ↓
fine-tune

80%
 ↓
fine-tune

90%
 ↓
fine-tune
```

 For example:

```
pruning_levels = [0.3, 0.5, 0.7, 0.9]
```

 At every stage:

 1. prune
2. fine-tune
3. evaluate
4. record accuracy

 Create a table:

 | Sparsity | Accuracy |
| --- | --- |
| 0% | baseline |
| 30% | ... |
| 50% | ... |
| 70% | ... |
| 90% | ... |

### Learning objective

 Students should discover experimentally that:

 > **Compression is often a trade-off rather than a binary success/failure.**

---

 # Exercise 8 — Structured pruning

 Unstructured pruning produces:

```
[1, 0, 1, 0, 1, 0, ...]
```

 The matrix still has the same dimensions.

 Therefore, ordinary dense hardware may still process the zeros.

 Structured pruning instead removes whole units.

 For example:

```
512 neurons
       ↓
384 neurons
```

 Now the actual matrix becomes smaller.

---

 ## Example

 Original:

```
nn.Linear(784, 512)
```

 After pruning 128 neurons:

```
nn.Linear(784, 384)
```

 The next layer must also change:

```
nn.Linear(384, 256)
```

 This illustrates why structured pruning is more complicated.

---

 ## Key question

 > Why can structured pruning provide an actual speedup while unstructured pruning may not?

 Because structured pruning changes the dimensions of the computation.

 The hardware can perform a genuinely smaller dense matrix multiplication.

---

 # Exercise 9 — Compare unstructured and structured pruning

 Ask students to compare:

 ### Unstructured

```
same matrix dimensions
+
many zeros
```

 versus:

 ### Structured

```
smaller matrix dimensions
+
no unnecessary zero entries
```

 Complete this table experimentally:

 | Method | Parameters | Sparsity | Accuracy | Inference time |
| --- | --- | --- | --- | --- |
| Baseline |  | 0% |  |  |
| Unstructured |  | 50% |  |  |
| Structured |  | — |  |  |

### Expected conclusion

 Unstructured pruning can achieve impressive **parameter sparsity**, but the practical speed/memory benefit depends heavily on sparse storage and hardware support.

 Structured pruning usually gives a more direct computational benefit.

---

 # Exercise 10 — Low-rank approximation with SVD

 Now investigate the lecture's discussion of low-rank factorisation.

 Take a matrix:

```
W = torch.randn(512, 512)
```

 Perform SVD:

```
U, S, Vh = torch.linalg.svd(W)
```

 We have:

 $$
W = U\Sigma V^T
$$

---

 ## Step 1 — Keep only rank $r$

```
r = 32

W_approx = (
    U[:, :r]
    @ torch.diag(S[:r])
    @ Vh[:r, :]
)
```

---

 ## Step 2 — Measure reconstruction error

```
error = torch.norm(W - W_approx) / torch.norm(W)

print("Relative error:", error.item())
```

---

 ## Step 3 — Count parameters

 Original:

 $$
512\times512=262144
$$

 Low-rank representation:

 $$
512r+r+r512
$$

 approximately:

 $$
2(512)(32)=32768
$$

 So:

 $$
262144 \rightarrow 32768
$$

 approximately an **8× reduction**.

---

 ## Important lecture connection

 But the lecture makes an important warning:

 > **The weights of modern neural networks are not necessarily sufficiently low-rank for this to work well.**

 So ask:

 > If low-rank compression is mathematically possible, why isn't it always a good compression method?

 ### Answer

 Because a low-rank approximation may introduce substantial reconstruction error. Neural-network weights can be sensitive to SVD truncation, so a large parameter reduction may cause a significant loss in model performance.

 This motivates LoRA.

---

 # Exercise 11 — Implement LoRA from scratch

 This is the most important practical exercise.

 Suppose:

 $$
W\in\mathbb{R}^{d\times d}
$$

 Normal fine-tuning learns:

 $$
W' = W+\Delta W
$$

 LoRA instead assumes:

 $$
\Delta W \approx BA
$$

 where:

 $$
A\in\mathbb{R}^{r\times d}
$$

 and:

 $$
B\in\mathbb{R}^{d\times r}
$$

 with:

 $$
r\ll d
$$

---

 ## Step 1 — Create a LoRA layer

```
class LoRALinear(nn.Module):

    def __init__(self, in_features, out_features, rank=8):

        super().__init__()

        self.linear = nn.Linear(
            in_features,
            out_features
        )

        # Freeze original weights
        for p in self.linear.parameters():
            p.requires_grad = False

        self.A = nn.Parameter(
            torch.randn(rank, in_features) * 0.01
        )

        self.B = nn.Parameter(
            torch.zeros(out_features, rank)
        )

    def forward(self, x):

        base = self.linear(x)

        update = x @ self.A.T @ self.B.T

        return base + update
```

---

 # Why initialise B to zero?

 At the beginning:

 $$
B=0
$$

 therefore:

 $$
BA=0
$$

 and:

 $$
W'=W
$$

 So the LoRA model initially behaves exactly like the pretrained model.

 This is the residual-learning idea from the lecture.

---

 # Exercise 12 — Count LoRA parameters

 Suppose:

 $$
d=4096
$$

 and:

 $$
r=8
$$

 Full matrix:

 $$
4096^2
=
16,777,216
$$

 LoRA parameters:

 $$
4096(8)+8(4096)
$$

 $$
=65,536
$$

 Therefore:

 $$
\frac{65,536}{16,777,216}
\approx0.0039
$$

 or about:

 $$
\boxed{0.39\%}
$$

 of the original matrix.

 This demonstrates the fundamental LoRA idea:

 > **The model is large, but the required adaptation can be tiny.**

---

 # Exercise 13 — Train only LoRA parameters

 Suppose:

```
model = LoRALinear(
    512,
    512,
    rank=8
)
```

 Check trainable parameters:

```
for name, parameter in model.named_parameters():

    print(
        name,
        parameter.requires_grad,
        parameter.numel()
    )
```

 You should see that the original linear layer is frozen while:

```
A
B
```

 are trainable.

 Then:

```
trainable = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

total = sum(
    p.numel()
    for p in model.parameters()
)

print("Trainable:", trainable)
print("Total:", total)
print("Percentage:", 100 * trainable / total)
```

---

 # Exercise 14 — LoRA rank experiment

 This is an excellent experiment because it directly tests one of the lecture's claims.

 Train models with:

```
r = 1
r = 2
r = 4
r = 8
r = 16
r = 32
```

 Record:

 | Rank | Trainable parameters | Accuracy |
| --- | --- | --- |
| 1 |  |  |
| 2 |  |  |
| 4 |  |  |
| 8 |  |  |
| 16 |  |  |
| 32 |  |  |

Then ask:

 > What happens as rank increases?

 Usually:

 - trainable parameters increase
- adaptation capacity increases
- accuracy may improve
- memory/training cost increases

 The important point is that **very small ranks can sometimes perform surprisingly well**.

---

 # Exercise 15 — LoRA vs full fine-tuning

 Now compare three approaches.

 ### Model A

 Full fine-tuning:

```
100% of parameters trainable
```

 ### Model B

 Layer freezing:

```
only selected layers trainable
```

 ### Model C

 LoRA:

```
tiny A and B matrices trainable
```

 Measure:

 | Method | Trainable parameters | Accuracy | Training memory | Inference architecture |
| --- | --- | --- | --- | --- |
| Full FT |  |  |  |  |
| Frozen layers |  |  |  |  |
| LoRA |  |  |  |  |

---

 # Exercise 16 — LoRA merging

 The lecture makes a particularly important point:

 $$
W' = W+BA
$$

 Once training is finished, calculate:

```
merged_weight = (
    model.linear.weight.data
    + model.B @ model.A
)
```

 Then put that into a normal linear layer.

 Conceptually:

```
Before merging:

x → W → output
 \
  → A → B → /

After merging:

x → (W + BA) → output
```

 Therefore LoRA can have:

 $$
\boxed{\text{zero additional inference layers}}
$$

 after merging.

 This distinguishes LoRA from ordinary adapter modules.

---

 # Exercise 17 — Compare adapters and LoRA

 Ask students to explain:

 > Why do adapter modules introduce inference overhead whereas merged LoRA does not?

 ### Answer

 Adapters introduce additional neural-network operations into the forward pass:

 $$
x
\rightarrow
\text{base layer}
\rightarrow
\text{adapter}
\rightarrow
...
$$

 LoRA can instead modify the existing weight matrix:

 $$
W' = W+BA
$$

 After merging, the model is simply:

 $$
xW'
$$

 so the architecture is unchanged.

---

 # Exercise 18 — Combine quantisation and LoRA

 Now combine two ideas:

 ### Quantisation

 Compress the **base model**.

 ### LoRA

 Train a small **low-rank update**.

 Conceptually:

```
Large pretrained model
        ↓
     INT4/INT8
        ↓
 Frozen base model
        +
   small LoRA matrices
        ↓
     fine-tuning
```

 This is the idea behind **QLoRA**.

 The key insight is:

 > Quantisation reduces the memory required to store the base model, while LoRA reduces the number of parameters that need to be trained.

---

 # Exercise 19 — The ultimate comparison

 Have students perform all four techniques on the same task:

 1. FP32 baseline
2. Quantisation
3. Pruning
4. LoRA

 Record:

 | Technique | Model size | Trainable parameters | Accuracy | Training cost | Inference cost |
| --- | --- | --- | --- | --- | --- |
| FP32 |  |  |  |  |  |
| INT8 |  |  |  |  |  |
| 50% pruning |  |  |  |  |  |
| LoRA |  |  |  |  |  |
| QLoRA |  |  |  |  |  |

Then answer:

 > Which technique should we use?

 There is **no universal answer**.

---

 # Exercise 20 — The final exam-style practical question

 Give students this scenario:

 > You have a 7-billion-parameter language model. You need to adapt it to a new domain, but your GPU does not have enough memory to perform conventional full fine-tuning. You also need to deploy several domain-specific versions of the model.

 Ask:

 ### Q1. What is the problem?

 Full fine-tuning requires updating and storing optimiser states for billions of parameters, making it very memory-intensive.

 ### Q2. Would pruning solve the main problem?

 Not necessarily.

 Pruning primarily reduces the model's parameter count/storage and potentially inference cost. It is not primarily a solution for efficiently **training multiple task-specific versions**.

 ### Q3. Would layer freezing help?

 Yes.

 It reduces the number of trainable parameters and therefore optimiser-state memory.

 But the adaptation is constrained by which layers are allowed to change.

 ### Q4. Why might LoRA be better?

 LoRA freezes the original model and learns:

 $$
\Delta W=BA
$$

 with:

 $$
r\ll d
$$

 Thus only a tiny number of parameters need to be trained and stored.

 ### Q5. Why is LoRA especially suitable for multiple domains?

 Because we can maintain:

```
             ┌── LoRA domain A
             │
Base model ──┼── LoRA domain B
             │
             └── LoRA domain C
```

 The expensive base model is shared.

 Only the small adaptation modules differ.

 ### Q6. What if the base model itself does not fit in GPU memory?

 Use quantisation.

 For example:

```
quantised base model
+
LoRA adapters
```

 This leads naturally to QLoRA.

---

 # The conceptual progression students should learn

 The entire practical sequence can be taught as one story:

```
                MODEL TOO LARGE / EXPENSIVE
                           │
             ┌─────────────┴─────────────┐
             │                           │
        Reduce precision             Remove parameters
             │                           │
       Quantisation                   Pruning
             │                           │
       FP32 → INT8/INT4           Delete unnecessary weights
             │                           │
             └─────────────┬─────────────┘
                           │
                    MODEL COMPRESSION
                           │
                           ↓
               But what about fine-tuning?
                           │
                           ↓
                  Don't update everything
                           │
          ┌────────────────┼────────────────┐
          │                │                │
     Layer freezing     Adapters          LoRA
          │                │                │
     Freeze layers      Add small         Learn low-rank
                         modules           updates
                                             │
                                             ↓
                                           QLoRA
                                             │
                              Quantisation + LoRA
```

 And the central principle behind the whole lecture is:

 $$
\boxed{
\text{Large neural networks contain substantial redundancy}
}
$$

 Different techniques exploit different forms of that redundancy:

 | Technique | What redundancy does it exploit? |
| --- | --- |
| Quantisation | Numerical precision |
| Pruning | Unnecessary parameters/connections |
| Structured pruning | Unnecessary neurons/channels |
| Layer freezing | Unnecessary parameter updates |
| Adapters | Low-dimensional task adaptation |
| Low-rank factorisation | Matrix structure |
| LoRA | Low-dimensional **parameter updates** |
| QLoRA | Low precision + low-dimensional adaptation |

And this gives students the most important distinction to remember:

 > **Model compression asks: "How can I represent/use the model more cheaply?"**
>
>  **Parameter-efficient fine-tuning asks: "How can I adapt the model without changing all of its parameters?"**

 They are related, but they solve **different problems**.
