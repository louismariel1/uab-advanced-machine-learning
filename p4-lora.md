Absolutely. This assignment is essentially asking you to **implement LoRA from scratch in PyTorch, compare it against ordinary dense fine-tuning, and explain why LoRA is more parameter-efficient**.

 The key idea is:

 > **Keep the pretrained model weights frozen and learn only a small low-rank update to those weights.**

 Below I’ll walk through the notebook in the order you encounter it, then explain exactly what you need to implement and what you need to report.

---

 # 1\. What is the overall goal?

 The assignment has three stages:

 1. **Pre-train** a neural network on CIFAR-10.
2. **Dense fine-tune** that pretrained network on a small subset of CIFAR-100.
3. **LoRA fine-tune** the same pretrained network on the same target data, but update far fewer parameters.

 Then you compare:

 |  | Dense fine-tuning | LoRA |
| --- | --- | --- |
| Pretrained weights | Partly frozen | Frozen |
| MLP weights | Updated directly | Frozen |
| Extra parameters | None | Small A/B matrices |
| Trainable parameters | Many | Few |
| Target accuracy | Baseline | Ideally close to baseline |
| Optimizer memory | Larger | Smaller |
| Forward/backward speed | Usually faster | May be slower |
| Main advantage | Simplicity | Parameter/memory efficiency |

The assignment is therefore **not just "make LoRA work."**

 You're investigating the trade-off:

 > **Can we get almost the same fine-tuning performance while training dramatically fewer parameters?**

---

 # 2\. The datasets: CIFAR-10 → CIFAR-100

 The first important concept is **transfer learning**.

 You start with:

 ### Source domain

 CIFAR-10:

 - 50,000 training images
- 10 classes
- Examples: airplane, automobile, bird, cat, etc.

 The model learns general visual features from this relatively large dataset.

 Then you move to:

 ### Target domain

 A small subset of CIFAR-100.

 The notebook lets you choose one CIFAR-100 superclass:

```
TARGET_CATEGORY = 'fish'
```

 This gives you:

```
aquarium_fish
flatfish
ray
shark
trout
```

 So your target task becomes a **5-class classification problem**.

 The important transfer-learning idea is:

```
CIFAR-10
   ↓
pretrained model
   ↓
transfer knowledge
   ↓
CIFAR-100 subset
```

 The model has already learned useful visual representations, so you don't need to train the entire network from scratch.

---

 # 3\. Why choose a small target dataset?

 This is important for understanding why parameter-efficient fine-tuning is useful.

 Imagine:

```
Pretrained model
100 million parameters
        ↓
Target dataset
only a few thousand examples
```

 If you update all 100 million parameters, you're doing a huge amount of computation and potentially overfitting the small target dataset.

 LoRA asks:

 > What if the pretrained model is already mostly correct, and we only need a relatively small adjustment?

 That adjustment is what LoRA learns.

---

 # 4\. Understanding the model

 The model is:

```
class ConvMLP(nn.Module):
```

 Its architecture is roughly:

```
Image
  ↓
Conv1
  ↓
MaxPool
  ↓
Conv2
  ↓
Global Average Pool
  ↓
fc1: 256 → 1024
  ↓
fc2: 1024 → 512
  ↓
fc3: 512 → 512
  ↓
fc4: 512 → 256
  ↓
classifier: 256 → number of classes
```

 The important part for this assignment is:

```
fc1
fc2
fc3
fc4
```

 These are large fully-connected layers.

 The assignment deliberately uses large MLP layers because LoRA is particularly easy to understand on a matrix multiplication.

---

 # 5\. Understanding a Linear layer mathematically

 A PyTorch linear layer essentially performs:

 $$
z = Wx+b
$$

 where:

 - $x$ = input
- $W$ = weight matrix
- $b$ = bias
- $z$ = output

 For example:

```
self.fc2 = nn.Linear(1024, 512)
```

 means:

 $$
W \in \mathbb{R}^{512\times1024}
$$

 So there are:

 $$
512\times1024 = 524,288
$$

 weight parameters.

 Dense fine-tuning changes all of them.

 LoRA does not.

---

 # 6\. Stage 1 — Pre-training

 This section:

```
pt_model = ConvMLP(num_classes = source_data.num_classes)
```

 creates a 10-class model.

 Then:

```
num_pt_epochs = 20
lr = 2e-3
l2_reg = 1e-3
```

 and:

```
pt_metrics = train_model(...)
```

 trains it on CIFAR-10.

 The important outcome is:

```
pt_model
```

 which contains useful pretrained weights.

 Then:

```
torch.save(pt_model.state_dict(), pt_save_name)
```

 saves those weights.

 So conceptually:

```
Random ConvMLP
     ↓
CIFAR-10 training
     ↓
Pretrained ConvMLP
```

 This is your starting point for both experiments.

---

 # 7\. Stage 2 — Dense fine-tuning baseline

 Now we need a baseline to compare LoRA against.

 The model is created with five target classes:

```
baseline_ft_model = ConvMLP(num_classes = target_data.num_classes)
```

 Since the target dataset has five classes:

```
fish
 ├── aquarium_fish
 ├── flatfish
 ├── ray
 ├── shark
 └── trout
```

 the classifier becomes:

```
nn.Linear(256, 5)
```

 The old CIFAR-10 classifier can't be reused because it predicts 10 classes.

---

 # 8\. Why are the convolutional layers frozen?

 The notebook does:

```
baseline_ft_model.conv1.requires_grad_(False)
baseline_ft_model.conv2.requires_grad_(False)
```

 So:

```
Conv1 → frozen
Conv2 → frozen

fc1 → trainable
fc2 → trainable
fc3 → trainable
fc4 → trainable
classifier → trainable
```

 This is deliberate.

 The assignment wants to compare:

```
Dense updates to MLP
            vs
LoRA updates to MLP
```

 rather than introducing another difference involving the convolutional layers.

---

 # 9\. Dense fine-tuning

 The pretrained weights are loaded:

```
pt_state_dict = torch.load(pt_save_name)
```

 but:

```
del pt_state_dict['classifier.weight']
del pt_state_dict['classifier.bias']
```

 The classifier is removed because the target task has five classes rather than ten.

 Then:

```
baseline_ft_model.load_state_dict(pt_state_dict, strict=False)
```

 loads all the pretrained weights except the classifier.

 Now the model looks like:

```
Pretrained:
Conv1 ───── frozen
Conv2 ───── frozen
fc1  ────── trainable
fc2  ────── trainable
fc3  ────── trainable
fc4  ────── trainable
classifier ─ randomly initialized + trainable
```

 This is your **dense fine-tuning baseline**.

---

 # 10\. What does "dense" mean?

 Suppose:

 $$
W\in\mathbb{R}^{512\times1024}
$$

 Dense fine-tuning directly modifies:

 $$
W
$$

 using gradient descent:

 $$
W_{new}=W-\eta\frac{\partial L}{\partial W}
$$

 So every element of $W$ can change.

 If $W$ has 524,288 parameters, all 524,288 potentially require gradients.

---

 # 11\. The key idea behind LoRA

 Now we get to the heart of the assignment.

 Instead of changing:

 $$
W
$$

 directly, LoRA freezes it.

 We introduce:

 $$
\Delta W
$$

 and calculate:

 $$
W' = W+\Delta W
$$

 Therefore:

 $$
z=(W+\Delta W)x+b
$$

 or:

 $$
z=Wx+\Delta Wx+b
$$

 The important point:

 $$
W
$$

 doesn't change.

---

 # 12\. But isn't ΔW just as big as W?

 Yes!

 And that's exactly why the `AbstractLayer` is only an intermediate step.

 The notebook first creates:

```
self.dW = torch.nn.Parameter(torch.zeros_like(self.W))
```

 If:

 $$
W\in\mathbb{R}^{512\times1024}
$$

 then:

 $$
\Delta W\in\mathbb{R}^{512\times1024}
$$

 So this doesn't save parameters.

 The `AbstractLayer` is teaching you the mathematical idea:

```
original:
        W

abstract:
        W + ΔW
```

 The pretrained `W` is frozen and only `ΔW` is trained.

---

 # 13\. LoRA's clever trick

 Instead of storing the entire:

 $$
\Delta W
$$

 we approximate it using two small matrices:

 $$
\Delta W \approx BA
$$

 where:

 $$
A\in\mathbb{R}^{r\times d_{in}}
$$

 and

 $$
B\in\mathbb{R}^{d_{out}\times r}
$$

 with:

 $$
r \ll d_{in},d_{out}
$$

 Therefore:

 $$
BA\in\mathbb{R}^{d_{out}\times d_{in}}
$$

 which has exactly the same shape as $W$.

---

 # 14\. Example of the parameter saving

 Take:

```
fc2 = nn.Linear(1024, 512)
```

 The original weight contains:

 $$
512\times1024=524,288
$$

 parameters.

 Suppose:

 $$
r=8
$$

 Then:

 $$
A\in\mathbb{R}^{8\times1024}
$$

 and:

 $$
B\in\mathbb{R}^{512\times8}
$$

 Parameter count:

 $$
8(1024)+512(8)
$$

 $$
=8192+4096
$$

 $$
=12,288
$$

 instead of:

 $$
524,288
$$

 That's about **42.7× fewer parameters** for this layer's weight update.

 And the original 524,288 parameters remain frozen.

---

 # 15\. The LoRA forward pass

 This is the most important equation in the whole assignment.

 Normal layer:

 $$
z=Wx+b
$$

 LoRA:

 $$
z=Wx+BAx+b
$$

 You can rewrite that as:

 $$
z=(W+BA)x+b
$$

 but you don't actually have to construct $W+BA$.

 Instead calculate:

```
original path:
x → W → Wx

LoRA path:
x → A → Ax → B → BAx

then:

Wx + BAx + b
```

 This is the entire LoRA implementation conceptually.

---

 # 16\. What exactly are A and B?

 Suppose:

```
original_layer = nn.Linear(in_features, out_features)
```

 Then:

```
W = [out_features × in_features]
```

 Your matrices should be:

```
A = [r × in_features]

B = [out_features × r]
```

 Then:

```
B @ A
```

 has shape:

```
[out_features × r] @ [r × in_features]
```

 giving:

```
[out_features × in_features]
```

 which matches $W$.

---

 # 17\. Initialization is important

 The assignment explicitly tells you:

 > Initialize B with zeros and A with random normal values.

 Why?

 Initially:

 $$
B=0
$$

 Therefore:

 $$
BA=0
$$

 So initially:

 $$
Wx+BAx+b = Wx+b
$$

 This means your LoRA model initially behaves **exactly like the pretrained model**.

 That's desirable.

 You're starting with:

```
pretrained model
      +
zero adaptation
```

 and allowing the adaptation to grow during training.

---

 # 18\. Why not initialize both randomly?

 If both $A$ and $B$ were random, then:

 $$
BA\neq0
$$

 and your model would immediately change its behavior before training.

 That's generally undesirable because you've just destroyed the exact pretrained starting point.

 With:

 $$
B=0
$$

 you start from:

 $$
\Delta W=0
$$

---

 # 19\. The actual `LoraLayer`

 The missing code is the central part of your assignment.

 Conceptually it needs to do four things:

 ### 1\. Keep the original layer

```
self.original_layer = original_layer
```

 ### 2\. Freeze its weights

```
self.original_layer.weight.requires_grad = False
```

 ### 3\. Create A and B

 For a layer:

```
original_layer.in_features
original_layer.out_features
```

 you want:

```
A: r × in_features
B: out_features × r
```

 ### 4\. Implement:

 $$
Wx+BAx+b
$$

---

 # 20\. One subtle point: the bias

 The assignment says you can leave the bias alone.

 So:

```
W → frozen
b → can remain as-is
A → trainable
B → trainable
```

 In this practical, the expected trainable parameters after replacing the FC layers should essentially be:

```
A matrices
B matrices
classifier.weight
classifier.bias
```

 The convolutional layers are also frozen.

---

 # 21\. What does `r` mean?

 `r` is the **rank**.

 For example:

```
r = 8
```

 means each LoRA update is constrained to have rank at most 8.

 Compare:

 ### Dense

 $$
\Delta W\in\mathbb{R}^{d_{out}\times d_{in}}
$$

 ### LoRA

 $$
\Delta W=BA
$$

 where:

 $$
B\in\mathbb{R}^{d_{out}\times8}
$$

 and:

 $$
A\in\mathbb{R}^{8\times d_{in}}
$$

 Small `r` means:

```
fewer parameters
less memory
less expressive update
```

 Large `r` means:

```
more parameters
more memory
more expressive update
```

 This is one of the things you're explicitly asked to investigate.

---

 # 22\. Why does LoRA work?

 This is an important conceptual question for your report.

 The assumption behind LoRA is that although a model may contain millions or billions of parameters, the **change required to adapt it to a new task may have a relatively low-dimensional structure**.

 In other words:

```
Huge pretrained weight matrix
          ↓
Only a relatively small adjustment is needed
          ↓
That adjustment can be approximated by a low-rank matrix
```

 So rather than learning every element of:

 $$
\Delta W
$$

 we learn a compact representation:

 $$
BA
$$

---

 # 23\. What does the optimizer actually train?

 This is a very important part of the assignment.

 The training code says:

```
opt = torch.optim.AdamW(model.parameters(), ...)
```

 At first glance this looks like it trains everything.

 But PyTorch parameters with:

```
requires_grad=False
```

 don't receive gradients.

 So for the LoRA model, the optimizer effectively updates only trainable parameters.

 For example:

```
Conv1 W     frozen
Conv2 W     frozen

fc1 W       frozen
fc1 A       TRAINABLE
fc1 B       TRAINABLE

fc2 W       frozen
fc2 A       TRAINABLE
fc2 B       TRAINABLE

fc3 W       frozen
fc3 A       TRAINABLE
fc3 B       TRAINABLE

fc4 W       frozen
fc4 A       TRAINABLE
fc4 B       TRAINABLE

classifier W TRAINABLE
classifier b TRAINABLE
```

---

 # 24\. Why the optimizer state becomes much smaller

 This is one of the things you have to measure.

 AdamW doesn't just store the model weights.

 It maintains optimizer state such as:

 - first moment
- second moment

 for trainable parameters.

 So if dense fine-tuning has:

```
1,000,000 trainable parameters
```

 AdamW needs state associated with those parameters.

 If LoRA has:

```
100,000 trainable parameters
```

 the optimizer state is dramatically smaller.

 This is one of LoRA's major practical advantages.

---

 # 25. Model state dict vs optimizer state dict

 The assignment asks you to compare both.

 This distinction is important.

 ## Model state dict

 Contains model parameters/buffers.

 Because your LoRA model still contains the frozen pretrained weights, its `state_dict` can still contain the large original matrices.

 So don't be surprised if:

```
LoRA model state_dict
```

 is **not dramatically smaller** than the dense model's state dict.

 The full pretrained weights are still there.

---

 ## Optimizer state dict

 This is different.

 The optimizer only maintains state for parameters it is optimizing.

 Therefore:

```
Dense:
large W gradients
→ large Adam states

LoRA:
small A/B gradients
→ much smaller Adam states
```

 This is one of the most important observations you should discuss in your report.

---

 # 26\. Why might LoRA actually be slower?

 This is a great trick in the assignment.

 You might initially think:

 > Fewer parameters = faster.

 Not necessarily.

 Dense layer:

 $$
Wx
$$

 LoRA layer:

 $$
Wx+BAx
$$

 You've added another computation path.

 Instead of one matrix multiplication, you're effectively doing:

```
Wx
Ax
BAx
add
```

 So although the **backward pass computes fewer parameter gradients**, the forward pass has additional matrix operations.

 Therefore LoRA can actually be:

 > **slower per iteration than dense fine-tuning**, especially for relatively small networks like this one.

 The assignment is trying to teach you that:

 **parameter efficiency ≠ automatically faster wall-clock execution.**

---

 # 27\. Why does LoRA still make sense then?

 Because in real-world models the major concern isn't necessarily just raw time for one forward pass.

 Consider a huge transformer:

```
billions of pretrained parameters
```

 You don't want to update all of them for every downstream task.

 LoRA gives you:

```
shared pretrained model
        +
small task-specific adapters
```

 This can drastically reduce:

 - trainable parameters
- gradient memory
- optimizer memory
- storage for task-specific checkpoints

---

 # 28\. What you need to experiment with

 The assignment specifically asks you to vary:

```
r
```

 For example, try:

```
r = 1
r = 2
r = 4
r = 8
r = 16
r = 32
```

 You don't necessarily need exactly those values, but this is a sensible experiment.

 You want to investigate:

```
rank
 ↓
number of trainable parameters
 ↓
training behaviour
 ↓
validation accuracy
```

---

 # 29\. Expected relationship with rank

 Generally:

```
r ↑
    ↓
more LoRA parameters
    ↓
more expressive adaptation
    ↓
potentially better accuracy
```

 But there is no guarantee that larger `r` always produces better validation accuracy.

 You might observe something like:

 | r | Trainable params | Best validation accuracy |
| --- | --- | --- |
| 1 | very low | 70% |
| 2 | low | 74% |
| 4 | low | 77% |
| 8 | moderate | 78% |
| 16 | higher | 78% |
| 32 | much higher | 77% |

The exact numbers will depend on your run.

 The interesting question is:

 > What is the smallest rank that achieves performance close to, or better than, dense fine-tuning?

---

 # 30\. What does "within a few percentage points" mean?

 Suppose your dense baseline achieves:

```
78.5%
```

 and LoRA achieves:

```
76.8%
```

 That's:

 $$
78.5-76.8=1.7
$$

 percentage points.

 That's likely a perfectly reasonable LoRA result.

 But:

```
Dense = 78.5%
LoRA = 55%
```

 would suggest something is wrong.

 Possible causes:

 - incorrect matrix dimensions
- A/B reversed
- incorrect forward pass
- weights not frozen
- bad initialization
- wrong labels
- wrong target model
- wrong rank
- learning-rate issues

---

 # 31\. Important distinction: percentage points vs percent

 If:

```
baseline = 80%
LoRA = 76%
```

 you should say:

 > LoRA is **4 percentage points** below the baseline.

 Not:

 > LoRA is 4% below.

 The relative decrease would actually be:

 $$
\frac{80-76}{80}=5\%
$$

 This distinction is useful in your report.

---

 # 32\. What the training curves should show

 The assignment asks you to show plots comparing:

```
Dense fine-tuning
vs
LoRA fine-tuning
```

 The notebook already provides:

```
baseline_ft_metrics
lora_metrics
```

 and the plotting code can overlay them.

 You should discuss things like:

 ### Training loss

 Does it decrease?

 ### Validation loss

 Does it decrease or start increasing?

 ### Training accuracy

 Does the model fit the target dataset?

 ### Validation accuracy

 Does LoRA approach the dense baseline?

 For example, you might observe:

```
Dense:
training accuracy ↑ quickly
validation accuracy ↑ quickly

LoRA:
training accuracy ↑ more slowly
validation accuracy ↑
eventually approaches dense
```

 That would be an interesting result.

---

 # 33\. What is `AbstractLayer` teaching you?

 This class:

```
class AbstractLayer(nn.Module):
```

 is basically a stepping stone.

 It demonstrates:

 $$
W \rightarrow W+\Delta W
$$

 The forward pass:

```
z_frozen = torch.matmul(x, self.W.T)
z_update = torch.matmul(x, self.dW.T)

z = z_frozen + z_update
return z + self.b
```

 corresponds exactly to:

 $$
z=Wx+\Delta Wx+b
$$

 LoRA then replaces:

 $$
\Delta W
$$

 with:

 $$
BA
$$

 So:

```
AbstractLayer

W + dW

        ↓

LoRA

W + BA
```

 That's the conceptual progression.

---

 # 34\. The biggest implementation mistake to avoid

 Be careful about matrix orientation.

 PyTorch stores:

```
nn.Linear(in_features, out_features)
```

 with:

```
weight.shape == (out_features, in_features)
```

 Therefore if:

```
in_features = 1024
out_features = 512
r = 8
```

 you need:

```
A = (8, 1024)
B = (512, 8)
```

 not the other way around.

 Then:

 $$
BA:
(512,8)(8,1024)
=
(512,1024)
$$

 which matches `W`.

---

 # 35\. Another important implementation detail

 Your initial LoRA output should match the original layer.

 Because:

 $$
B=0
$$

 you should get:

 $$
BA=0
$$

 so:

 $$
Wx+BAx+b=Wx+b
$$

 You should ideally test this explicitly.

 For example, conceptually:

```
original_output = original_layer(x)
lora_output = lora_layer(x)

assert torch.allclose(original_output, lora_output)
```

 immediately after initialization.

 That's an excellent sanity check.

---

 # 36\. What does the final assignment actually expect from you?

 Your submission has several components.

 ## Part A — Implement LoRA

 Implement:

```
class LoraLayer(nn.Module):
```

 with:

 - frozen original weights
- A matrix
- B matrix
- appropriate initialization
- correct forward pass

 Then replace:

```
fc1
fc2
fc3
fc4
```

 with your LoRA layers.

---

 # 37\. Part B — Show parameter reduction

 You should report something like:

```
Dense model:
Total parameters: X
Trainable parameters: Y

LoRA model:
Total parameters: X'
Trainable parameters: Y'

Trainable parameter reduction: Z%
```

 The exact values depend on your implementation/rank.

 The important comparison is:

 $$
\text{trainable fraction}
=
\frac{\text{trainable parameters}}
{\text{total parameters}}
$$

---

 # 38\. Part C — Compare accuracy

 Report:

```
Dense FT best validation accuracy: XX.X%

LoRA best validation accuracy: YY.Y%

Difference: ZZ.Z percentage points
```

 Then explain whether LoRA successfully maintained performance.

---

 # 39\. Part D — Compare training curves

 Include plots showing at least:

```
Dense loss vs LoRA loss
Dense validation loss vs LoRA validation loss

Dense accuracy vs LoRA accuracy
Dense validation accuracy vs LoRA validation accuracy
```

 Then describe what you see.

---

 # 40\. Part E — State dict size

 Use:

```
get_state_dict_size(model.state_dict())
```

 for both models.

 Then compare them.

 Important insight:

 > The LoRA model's complete state dict can remain large because it still contains the frozen pretrained weights.

 If you save the entire LoRA model, you're still saving those weights.

 The real storage advantage comes when you save **only the adapter parameters**.

 That's exactly why the optional extension asks you to save only A/B.

---

 # 41\. Part F — Optimizer state size

 Use:

```
get_opt_state_dict_size(optimizer.state_dict())
```

 and compare.

 You should expect a much bigger difference here because AdamW maintains state for trainable parameters.

 Conceptually:

```
Dense:
Adam state
 ├── fc1 weights
 ├── fc2 weights
 ├── fc3 weights
 ├── fc4 weights
 └── classifier

LoRA:
Adam state
 ├── fc1 A/B
 ├── fc2 A/B
 ├── fc3 A/B
 ├── fc4 A/B
 └── classifier
```

---

 # 42\. Part G — Forward/backward timing

 You need to compare the computational cost.

 You can measure:

```
forward time
backward time
```

 for dense and LoRA.

 You should use GPU synchronization when timing CUDA operations, because GPU execution is asynchronous.

 Otherwise your timing may be misleading.

 The conceptual expectation is:

```
Forward:
LoRA may be slower

Backward:
LoRA can have fewer trainable gradients,
but extra operations are involved

Overall:
not necessarily faster per iteration
```

 This is a very important conclusion.

---

 # 43\. Part H — Try different ranks

 This is probably the most experimental part.

 Create a table such as:

 | Rank | Trainable parameters | Best val accuracy | Training time |
| --- | --- | --- | --- |
| 1 | ... | ... | ... |
| 2 | ... | ... | ... |
| 4 | ... | ... | ... |
| 8 | ... | ... | ... |
| 16 | ... | ... | ... |

Then identify:

 > Which value of `r` gave the best trade-off?

 And specifically:

 > Did any rank outperform dense fine-tuning?

---

 # 44\. Optional extension: merging LoRA

 After training:

 $$
W'=W+BA
$$

 You can actually permanently add the LoRA update to the original weights.

 For example:

```
original W
    +
trained BA
    ↓
merged W
```

 Then you can discard A and B.

 The resulting model is just an ordinary Linear layer again:

 $$
z=W'x+b
$$

 This is extremely useful because inference no longer needs:

```
Wx
+
BAx
```

 It only needs:

```
W'x
```

 So LoRA can be used during training and then merged for deployment.

---

 # 45\. Optional extension: adapter-only checkpoint

 Instead of saving:

```
entire pretrained model
+
A/B
```

 you could save only:

```
fc1.A
fc1.B
fc2.A
fc2.B
fc3.A
fc3.B
fc4.A
fc4.B
```

 Then a user who already has the pretrained model can load these adapters.

 Conceptually:

```
Base model
     +
Task A adapters
     ↓
Task A model

Base model
     +
Task B adapters
     ↓
Task B model
```

 This is one of the major practical benefits of LoRA.

---

 # 46\. A good mental model for the whole assignment

 Think of the entire practical as this:

 ### Step 1

 Train:

```
Random model
      ↓
CIFAR-10
      ↓
Pretrained model
```

 ### Step 2

 Adapt normally:

```
Pretrained model
      ↓
freeze CNN
      ↓
update all MLP weights
      ↓
Dense fine-tuned model
```

 ### Step 3

 Adapt with LoRA:

```
Pretrained model
      ↓
freeze CNN
      ↓
freeze MLP W
      ↓
add A/B adapters
      ↓
train A/B
      ↓
LoRA model
```

 ### Step 4

 Compare:

```
                 Dense       LoRA
                 -----       ----
Accuracy           ?           ?
Parameters         ?           ?
Optimizer size     ?           ?
Forward time       ?           ?
Backward time      ?           ?
```

 ### Step 5

 Experiment with:

```
r = 1, 2, 4, 8, 16, ...
```

 and find the best trade-off.

---

 # 47\. What your report should explain

 A strong report could have this structure:

 ## 1\. Introduction

 Explain:

 - transfer learning
- parameter-efficient fine-tuning
- purpose of LoRA

 ## 2\. Dense fine-tuning baseline

 Explain:

 - pretrained CIFAR-10 model
- target CIFAR-100 subset
- frozen convolutional layers
- trainable MLP/classifier

 ## 3\. LoRA implementation

 Explain:

 $$
z=Wx+b
$$

 becomes:

 $$
z=Wx+BAx+b
$$

 Explain:

 - A dimensions
- B dimensions
- rank $r$
- initialization
- frozen W

 ## 4\. Results

 Include:

 - parameter counts
- accuracy
- loss curves
- accuracy curves
- state-dict sizes
- optimizer-state sizes
- timing

 ## 5\. Rank experiment

 Table:

```
r | trainable parameters | validation accuracy
```

 ## 6\. Discussion

 Explain:

 - why LoRA uses fewer trainable parameters
- why optimizer state is smaller
- why forward pass might not be faster
- how rank affects performance
- whether LoRA matches dense fine-tuning

 ## 7\. Conclusion

 Summarize the trade-off.

---

 # 48\. The central equations to remember

 If you only remember five things for this assignment, remember these:

 ### Normal linear layer

 $$
z=Wx+b
$$

 ### Dense fine-tuning

 $$
W\leftarrow W-\eta\nabla_WL
$$

 ### Full residual adaptation

 $$
z=Wx+\Delta Wx+b
$$

 ### LoRA approximation

 $$
\Delta W\approx BA
$$

 ### LoRA forward pass

 $$
\boxed{z=Wx+BAx+b}
$$

 where:

 $$
A\in\mathbb{R}^{r\times d_{in}}
$$

 and:

 $$
B\in\mathbb{R}^{d_{out}\times r}
$$

 with:

 $$
r\ll d_{in},d_{out}
$$

 and typically:

 $$
A\sim\mathcal N(0,\sigma^2),\qquad B=0
$$

 at initialization.

---

 ## The single most important thing you're being graded on

 Your implementation needs to demonstrate that you understand this transformation:

```
                 Dense fine-tuning

             W  ←── TRAINED
             │
             ↓
             Wx + b

                    ↓

                    LoRA

             W  ←── FROZEN
             │
             ├──────→ Wx
             │
             A  ←── TRAINED
             ↓
             B  ←── TRAINED
             │
             ↓
             BAx

             Wx + BAx + b
```

 So **LoRA is not a completely new model**. It is a way of representing the _change_ you want to make to an existing pretrained model using a much smaller number of trainable parameters.

 If you understand that diagram and the equation

 $$
\boxed{W_{\text{effective}}=W+BA}
$$

 you understand the core of the practical.
