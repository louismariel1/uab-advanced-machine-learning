 # Question-Driven Teaching Script: Practical 4 — LoRA

 ## Opening: What are we trying to solve?

 **Teacher:**

 “Today we are going to study a technique called **LoRA**, or **Low-Rank Adaptation**.

 Before we look at any code, I want to start with a question.

 ### Question 1

 Suppose I have a neural network that has already been trained on a large dataset, and I now want to adapt it to a new, smaller dataset.

 **Do I really need to change every parameter in the neural network?**

 Take a few seconds to think about that.”

 **Answer:**

 “No. And that is the central motivation behind this practical.

 If the original model has learned useful general features, we would like to **keep most of those learned parameters fixed** and modify only a small number of parameters needed for the new task.

 This is called **parameter-efficient fine-tuning**, or PEFT.”

---

 ### Question 2

 “What is the difference between **pre-training** and **fine-tuning**?”

 **Answer:**

 “Pre-training means learning useful representations from a large source dataset.

 Fine-tuning means taking that already-trained model and adapting it to a new target task or domain.

 In this practical:

 - our **source dataset** is CIFAR-10;
- our **target dataset** is a small subset of CIFAR-100;
- we first train on CIFAR-10;
- then we adapt the model to five related CIFAR-100 classes.”

---

 # Part 1 — Understanding the experimental setup

 ### Question 3

 “Why do you think the notebook uses CIFAR-10 first and CIFAR-100 second?”

 **Answer:**

 “We want to demonstrate **transfer learning**.

 CIFAR-10 and CIFAR-100 both contain small colour images. Although their classes are different, the visual information is related.

 For example, a model trained on CIFAR-10 can learn useful low-level and mid-level features such as:

 - edges,
- textures,
- shapes,
- colour patterns,
- object parts.

 Those features can potentially be reused for CIFAR-100.”

---

 ### Question 4

 “CIFAR-100 has 100 classes. Are we going to fine-tune on all 100?”

 **Answer:**

 “No.

 We select one **coarse superclass**, such as `fish`.

 The notebook defines:

```
"fish": ["aquarium_fish", "flatfish", "ray", "shark", "trout"]
```

 So our target problem becomes a **5-class classification problem**.”

---

 ### Question 5

 “If we choose `TARGET_CATEGORY = 'fish'`, what are the five target classes?”

 **Answer:**

 “ Aquarium fish, flatfish, ray, shark and trout.”

---

 ### Question 6

 “Why does the notebook need to remap the class labels?”

 **Answer:**

 “Originally, CIFAR-100 has labels ranging from 0 to 99.

 Our selected five classes may have completely different original numerical labels.

 But our new classifier has only five outputs.

 Therefore we remap the selected classes to:

```
0, 1, 2, 3, 4
```

 This is necessary because the final classification layer will produce five logits.”

---

 # Part 2 — Train/validation split

 ### Question 7

 “Why do we split the data into training and validation sets?”

 **Answer:**

 “The training set is used to update the model's parameters.

 The validation set is **not** used to update parameters. Instead, it tells us how well the model generalises to examples it did not train on.”

---

 ### Question 8

 “The notebook uses `val_frac = 0.2`. What does that mean?”

 **Answer:**

 “Twenty percent of each dataset is used for validation and eighty percent for training.”

---

 ### Question 9

 “What is the purpose of a `DataLoader`?”

 **Answer:**

 “A `DataLoader` gives us batches of examples rather than processing the entire dataset at once.

 For example:

```
batch_size = 256
```

 means that approximately 256 images are processed in each batch.”

---

 ### Question 10

 “What does this transformation do?”

```
transforms.ToTensor()
```

 **Answer:**

 “It converts the image into a PyTorch tensor and changes the representation into the format expected by PyTorch neural-network layers.”

---

 ### Question 11

 “What does this line do?”

```
transforms.Normalize(
    (0.5, 0.5, 0.5),
    (0.2, 0.2, 0.2)
)
```

 **Answer:**

 “It normalises the three colour channels.

 For each channel, it approximately performs:

 $$
x' = \frac{x-\mu}{\sigma}
$$

 Here the mean is 0.5 and standard deviation is 0.2 for each RGB channel.”

---

 # Part 3 — Understanding the model

 **Teacher:**

 “Now let's talk about the architecture.”

 ### Question 12

 “What are the major components of `ConvMLP`?”

 **Answer:**

 “There are four conceptual parts:

 1. convolutional preprocessing;
2. global pooling;
3. a large fully connected MLP;
4. a final classification layer.”

---

 ### Question 13

 “What does `conv1` do?”

```
self.conv1 = nn.Conv2d(input_channels, 64, 3, padding=1)
```

 **Answer:**

 “It takes the three RGB channels and produces 64 feature maps.

 The kernel is 3 by 3.

 The convolution begins extracting visual features.”

---

 ### Question 14

 “What does `MaxPool2d(2)` do?”

 **Answer:**

 “It reduces the spatial dimensions, normally by a factor of two.

 This reduces computation and gives the network some degree of spatial invariance.”

---

 ### Question 15

 “What does the second convolution do?”

```
self.conv2 = nn.Conv2d(64, 256, 3, padding=1)
```

 **Answer:**

 “It takes the 64 feature maps and produces 256 feature maps.

 The network can therefore construct increasingly rich representations.”

---

 ### Question 16

 “What does `AdaptiveAvgPool2d((1,1))` accomplish?”

 **Answer:**

 “It reduces every feature map to a single number.

 So the 256 feature maps become a 256-dimensional feature vector.”

---

 ### Question 17

 “Why is this architecture intentionally unusual?”

 **Answer:**

 “The notebook tells us that it is not necessarily an optimal architecture for CIFAR.

 The important point is that it contains relatively large **fully connected weight matrices**.

 That makes the mathematics and implementation of LoRA easier to understand.

 In real modern applications, LoRA is particularly important for very large Transformer models.”

---

 # Part 4 — Understanding a Linear Layer

 ### Question 18

 “Let's forget neural networks for a moment.

 What does a fully connected layer mathematically compute?”

 **Answer:**

 “A linear layer computes approximately:

 $$
z = Wx+b
$$

 where:

 - $x$ is the input;
- $W$ is the weight matrix;
- $b$ is the bias;
- $z$ is the output.”

---

 ### Question 19

 “Suppose $W$ has dimensions $512\times1024$.

 How many weight parameters does it contain?”

 **Answer:**

 $$
512\times1024 = 524,288
$$

 So just that matrix contains **524,288 parameters**, before even considering the bias.”

---

 ### Question 20

 “During ordinary fine-tuning, what happens to $W$?”

 **Answer:**

 “We calculate gradients with respect to $W$, and the optimiser updates the matrix.

 Conceptually:

 $$
W_{\text{new}}=W_{\text{old}}+\Delta W
$$

 The entire matrix can potentially change.”

---

 # Part 5 — Pre-training

 ### Question 21

 “What is the first actual learning stage in this notebook?”

 **Answer:**

 “Pre-training on CIFAR-10.”

---

 ### Question 22

 “What does this create?”

```
pt_model = ConvMLP(num_classes=source_data.num_classes)
```

 **Answer:**

 “It creates a model whose final classifier has ten outputs because CIFAR-10 has ten classes.”

---

 ### Question 23

 “What does this do?”

```
pt_metrics = train_model(
    pt_model,
    train_loader_source,
    val_loader_source,
    num_pt_epochs,
    lr,
    l2_reg
)
```

 **Answer:**

 “It trains the model on CIFAR-10.

 The training loop performs:

 1. forward pass;
2. loss calculation;
3. backward pass;
4. optimiser update;
5. metric recording;
6. validation after each epoch.”

---

 ### Question 24

 “What loss function is used?”

 **Answer:**

```
nn.CrossEntropyLoss()
```

 “It is appropriate for multi-class classification when the model outputs logits.”

---

 ### Question 25

 “Does the model need to explicitly apply softmax before `CrossEntropyLoss`?”

 **Answer:**

 “No.

 `CrossEntropyLoss` expects logits and internally handles the relevant log-softmax operation.

 So this is correct:

```
loss = loss_func(pred, y)
```

 where `pred` contains raw logits.”

---

 # Part 6 — Understanding the training loop

 ### Question 26

 “What is the purpose of this?”

```
opt.zero_grad()
```

 **Answer:**

 “PyTorch accumulates gradients by default.

 Therefore, before calculating gradients for the current batch, we clear the gradients from the previous batch.”

---

 ### Question 27

 “What happens here?”

```
pred = model(x)
```

 **Answer:**

 “This is the **forward pass**.

 The input images travel through the neural network and produce predictions.”

---

 ### Question 28

 “What does this do?”

```
batch_loss.backward()
```

 **Answer:**

 “This performs **backpropagation**.

 PyTorch calculates the gradients of the loss with respect to the trainable parameters.”

---

 ### Question 29

 “What does this do?”

```
opt.step()
```

 **Answer:**

 “This tells the optimiser to use the gradients to update the trainable parameters.”

---

 ### Question 30

 “So what is the complete learning cycle?”

 **Answer:**

 “The essential cycle is:

 $$
\boxed{\text{Forward}
\rightarrow
\text{Loss}
\rightarrow
\text{Backward}
\rightarrow
\text{Update}}
$$

 And this happens repeatedly over batches and epochs.”

---

 # Part 7 — Dense fine-tuning baseline

 **Teacher:**

 “Now we have a trained CIFAR-10 model.

 Here's the crucial question.”

 ### Question 31

 “Can we simply use the CIFAR-10 classifier for our five fish classes?”

 **Answer:**

 “No.

 The original classifier produces ten outputs corresponding to CIFAR-10.

 Our target task has five classes.

 Therefore, we need a new five-output classifier.”

---

 ### Question 32

 “What happens here?”

```
baseline_ft_model = ConvMLP(
    num_classes=target_data.num_classes
)
```

 **Answer:**

 “We create a new model with a five-class classifier.”

---

 ### Question 33

 “But if we create a new model, don't we lose the pre-trained weights?”

 **Answer:**

 “We would, unless we load them.

 The notebook loads the pre-trained weights from the checkpoint.”

---

 ### Question 34

 “Why are the classifier weights deleted from the checkpoint?”

```
del pt_state_dict['classifier.weight']
del pt_state_dict['classifier.bias']
```

 **Answer:**

 “Because the old classifier belongs to the ten-class CIFAR-10 task.

 We don't want those ten-class weights.

 We want the new five-class classifier to start with newly initialised parameters.”

---

 ### Question 35

 “What does `strict=False` allow?”

```
baseline_ft_model.load_state_dict(
    pt_state_dict,
    strict=False
)
```

 **Answer:**

 “It allows us to load all the matching pre-trained parameters while ignoring the missing classifier parameters.”

---

 ### Question 36

 “Why freeze the convolutional layers?”

```
baseline_ft_model.conv1.requires_grad_(False)
baseline_ft_model.conv2.requires_grad_(False)
```

 **Answer:**

 “To create a controlled comparison.

 The practical wants us to compare **dense fine-tuning of the MLP** against **LoRA adaptation of the MLP**.

 The convolutional feature extractor is therefore kept fixed.”

---

 # Part 8 — What exactly is the LoRA problem?

 **Teacher:**

 “Now we reach the most important part.

 Imagine our pre-trained weight matrix is $W$.”

 ### Question 37

 “During dense fine-tuning, what happens to $W$?”

 **Answer:**

 “It changes directly.”

---

 ### Question 38

 “Can we instead freeze $W$ and introduce another matrix that represents the change we want?”

 **Answer:**

 “Yes.

 We can write:

 $$
W_{\text{new}}=W+\Delta W
$$

 where $W$ stays frozen and $\Delta W$ is trainable.”

---

 ### Question 39

 “What is the problem with doing that directly?”

 **Answer:**

 “$\Delta W$ has exactly the same number of parameters as $W$.

 So we have not actually saved many trainable parameters.”

---

 # Part 9 — The key idea of LoRA

 ### Question 40

 “So what if we approximate $\Delta W$ using two much smaller matrices?”

 **Answer:**

 “That is exactly the LoRA idea.”

 We write:

 $$
\boxed{\Delta W \approx BA}
$$

 where:

 $$
A\in\mathbb{R}^{r\times d_{\text{in}}}
$$

 and

 $$
B\in\mathbb{R}^{d_{\text{out}}\times r}
$$

 with:

 $$
r\ll d_{\text{in}},d_{\text{out}}.
$$

 Therefore:

 $$
\boxed{Wx+BAx+b}
$$

 is our LoRA computation.”

---

 ### Question 41

 “What does $r$ mean?”

 **Answer:**

 “$r$ is the **rank** or, more precisely in this implementation, the bottleneck dimension of the low-rank adapter.

 It controls the capacity of the LoRA update.”

---

 ### Question 42

 “Suppose a layer has a weight matrix of size $1024\times512$, and we choose $r=8$.

 How many parameters does the original weight matrix have?”

 **Answer:**

 $$
1024\times512=524,288.
$$

---

 ### Question 43

 “How many parameters do the LoRA matrices contain?”

 **Answer:**

 For $A$:

 $$
8\times512=4,096.
$$

 For $B$:

 $$
1024\times8=8,192.
$$

 Total:

 $$
4,096+8,192=12,288.
$$

 So instead of training **524,288** weights, we train only **12,288** adapter parameters.

 That is the fundamental parameter-saving mechanism.”

---

 ### Question 44

 “What happens if I increase $r$?”

 **Answer:**

 “The adapter becomes larger and more expressive.

 But we lose some of the parameter and computation savings.

 So $r$ controls a trade-off:

 $$
\text{smaller }r
\rightarrow
\text{fewer parameters, less capacity}
$$

 and

 $\text{larger }r
\rightarrow
\text{more parameters, greater capacity}.$”

---

 # Part 10 — Understanding `AbstractLayer`

 **Teacher:**

 “Before implementing LoRA, the notebook gives us an intermediate abstraction.”

 ### Question 45

 “What does `AbstractLayer` do?”

 **Answer:**

 “It takes an ordinary linear layer and replaces its trainable weight update with an explicit trainable matrix called `dW`.”

---

 ### Question 46

 “What happens to the original weight?”

```
self.W.requires_grad = False
```

 **Answer:**

 “It is frozen.

 No gradient update is applied to it.”

---

 ### Question 47

 “What is this?”

```
self.dW = torch.nn.Parameter(
    torch.zeros_like(self.W),
    requires_grad=True
)
```

 **Answer:**

 “This creates a trainable matrix having exactly the same shape as $W$.

 It represents $\Delta W$.”

---

 ### Question 48

 “What does the forward pass calculate?”

```
z_frozen = torch.matmul(x, self.W.T)
z_update = torch.matmul(x, self.dW.T)
z = z_frozen + z_update
return z + self.b
```

 **Answer:**

 Mathematically:

 $$
z = xW^T+x\Delta W^T+b.
$$

 Or conceptually:

 $$
\boxed{z=(W+\Delta W)x+b}
$$

 depending on the orientation convention.”

---

 ### Question 49

 “Why does the notebook test that `AbstractLayer` gives the same output as the original layer?”

 **Answer:**

 “Because replacing a layer must not accidentally change its behaviour.

 Since `dW` starts at zero:

 $$
\Delta W=0
$$

 so initially:

 $$
W+\Delta W=W.
$$

 Therefore the outputs should be identical.”

---

 # Part 11 — Now build LoRA

 **Teacher:**

 “Now you are asked to implement `LoraLayer`.

 Let's reason through the code rather than memorising it.”

 ### Question 50

 “What information does a LoRA layer need?”

 **Answer:**

 “At minimum:

 1. the original linear layer;
2. the rank $r$;
3. the frozen original weight;
4. trainable matrices $A$ and $B$.”

---

 ### Question 51

 “Suppose the original linear layer has:

```
in_features = 512
out_features = 1024
```

 What shape should $A$ have?”

 **Answer:**

 Using the convention in the notebook:

 $$
A\in\mathbb{R}^{r\times512}.
$$

---

 ### Question 52

 “What shape should $B$ have?”

 **Answer:**

 $$
B\in\mathbb{R}^{1024\times r}.
$$

---

 ### Question 53

 “What shape does $BA$ have?”

 **Answer:**

 $$
(1024\times r)(r\times512)
=
1024\times512.
$$

 Exactly the same shape as the original weight matrix.”

---

 # Part 12 — The most important initialisation question

 ### Question 54

 “When we first replace a trained layer with LoRA, should its output suddenly change?”

 **Answer:**

 “No.

 We want the LoRA model initially to behave exactly like the pre-trained model.”

---

 ### Question 55

 “How can we guarantee that?”

 **Answer:**

 “We initialise the LoRA update to zero:

 $$
BA=0.
$$

 A common strategy is:

 $$
B=0
$$

 and initialise $A$ randomly.”

---

 ### Question 56

 “Why not initialise both $A$ and $B$ randomly?”

 **Answer:**

 “Because then:

 $$
BA\neq0
$$

 in general.

 The adapter would immediately alter the pre-trained model's behaviour before any training.”

---

 ### Question 57

 “Why can $A$ be random if $B$ is zero?”

 **Answer:**

 “Because:

 $$
BA=B A=0A=0.
$$

 So the initial update is still zero.

 But during training, $A$ already contains useful random variation while $B$ can begin learning the update.”

---

 # Part 13 — LoRA forward pass

 ### Question 58

 “What should the LoRA forward pass calculate?”

 **Answer:**

 “The original frozen linear transformation plus the low-rank update.”

 Conceptually:

 $$
\boxed{
z=Wx+BAx+b
}
$$

 or, using the PyTorch linear-layer orientation:

 $$
\boxed{
z=xW^T+x(BA)^T+b
}
$$

---

 ### Question 59

 “Which parameters should receive gradients?”

 **Answer:**

 “Only:

 - $A$;
- $B$;
- and, in this practical, the new classifier weights and bias.

 The original MLP weights are frozen.

 The convolutional layers are also frozen.”

---

 # Part 14 — Reading the parameter-count test

 ### Question 60

 “The notebook prints:

```
lora_trainable
```

 What are we hoping to see?”

 **Answer:**

 “A much smaller number of trainable parameters than the dense baseline.”

---

 ### Question 61

 “If the output says that the original model has many parameters but only a small fraction are trainable under LoRA, what does that demonstrate?”

 **Answer:**

 “It demonstrates **parameter-efficient fine-tuning**.

 We still have the original model parameters, but we are not calculating/storing optimiser updates for all of them.”

---

 ### Question 62

 “What should this assertion verify?”

```
assert lora_model.fc1.original_layer.weight.requires_grad == False
```

 **Answer:**

 “It verifies that the original full-rank weight matrix really is frozen.

 If it were accidentally trainable, the experiment would no longer be a proper LoRA comparison.”

---

 # Part 15 — Dense fine-tuning versus LoRA

 **Teacher:**

 “Now let's compare the two approaches.”

 ### Question 63

 “In dense fine-tuning, what is updated?”

 **Answer:**

 “The selected full-rank MLP weights, plus the classifier.”

---

 ### Question 64

 “In LoRA, what is updated?”

 **Answer:**

 “The low-rank adapter matrices $A$ and $B$, plus the classifier.”

---

 ### Question 65

 “Which method should have fewer trainable parameters?”

 **Answer:**

 “LoRA.”

---

 ### Question 66

 “Does fewer trainable parameters automatically mean fewer total model parameters?”

 **Answer:**

 “No.

 The original model still exists.

 LoRA primarily reduces the number of **trainable parameters** and therefore the optimiser state and gradient-related memory.

 The frozen base weights still occupy memory.”

---

 # Part 16 — Why LoRA can still be slower

 ### Question 67

 “Here is a subtle question.

 If LoRA has fewer trainable parameters, must its forward pass always be faster?”

 **Answer:**

 “No.”

---

 ### Question 68

 “Why?”

 **Answer:**

 “Because the forward pass still needs to calculate the original transformation:

 $$
Wx
$$

 and also the adapter transformation:

 $$
BAx.
$$

 So there can be extra matrix operations.

 The parameter reduction does not automatically imply a faster forward pass.”

---

 ### Question 69

 “What about the backward pass?”

 **Answer:**

 “LoRA has fewer trainable parameters, so the amount of gradient computation and optimiser work can be much smaller.

 However, the exact timing depends on the implementation, hardware, matrix sizes, and framework overhead.

 Therefore we should **measure**, not assume.”

---

 # Part 17 — Model state dictionary

 ### Question 70

 “What is a model `state_dict`?”

 **Answer:**

 “It is essentially a mapping containing the model's parameter tensors and relevant persistent buffers.

 It can be saved to disk and later loaded to reconstruct the learned parameter values.”

---

 ### Question 71

 “Will a LoRA model's state dict necessarily contain only $A$ and $B$?”

 **Answer:**

 “No.

 The model still contains the frozen base model parameters.

 Unless we deliberately design a special saving mechanism, the state dict can contain the base weights as well as the LoRA parameters.”

---

 ### Question 72

 “Then why does LoRA still make deployment/storage interesting?”

 **Answer:**

 “Because after training, we can choose to save **only the adapter parameters**.

 The same pre-trained base model can then be reused with different adapters for different tasks.”

---

 # Part 18 — Optimiser state

 ### Question 73

 “Why should the optimiser state be much smaller for LoRA?”

 **Answer:**

 “Because the optimiser only tracks trainable parameters.”

 The notebook uses:

```
torch.optim.AdamW(
    model.parameters(),
    ...
)
```

 but frozen parameters do not receive gradients and do not receive AdamW update states in the same way as trainable parameters.”

---

 ### Question 74

 “What additional information does Adam typically maintain?”

 **Answer:**

 “For trainable parameters, Adam maintains running statistics such as estimates related to the first and second moments of the gradients.

 Therefore, reducing the number of trainable parameters can substantially reduce optimiser memory.”

---

 # Part 19 — A crucial distinction: parameters versus computation

 ### Question 75

 “If LoRA trains only 5% of the parameters, does that mean it performs only 5% of the forward computation?”

 **Answer:**

 “No.

 This is a very important distinction.

 **Parameter efficiency** and **computational efficiency** are related but not identical.

 The frozen base model still has to perform its forward computation.”

---

 # Part 20 — The rank experiment

 ### Question 76

 “What happens if we choose $r=1$?”

 **Answer:**

 “The adapter has extremely limited capacity.

 It can represent only a very restricted low-rank update.”

---

 ### Question 77

 “What happens as $r$ increases?”

 **Answer:**

 “The adapter can represent increasingly complex updates.”

---

 ### Question 78

 “Could a larger $r$ eventually approach dense fine-tuning?”

 **Answer:**

 “Yes.

 As the rank becomes sufficiently large, the low-rank representation can represent increasingly general matrices.

 But then the parameter-saving advantage decreases.”

---

 ### Question 79

 “Why does the assignment ask you to try several values of $r$?”

 **Answer:**

 “Because rank is a hyperparameter.

 We want to discover empirically whether a smaller or larger adapter provides the best trade-off between:

 - trainable parameters;
- computation;
- training behaviour;
- validation accuracy.”

---

 # Part 21 — Understanding the final comparison

 ### Question 80

 “What is the main success criterion for the LoRA model?”

 **Answer:**

 “It should achieve validation accuracy reasonably close to the dense fine-tuning baseline while using substantially fewer trainable parameters.”

---

 ### Question 81

 “Suppose dense fine-tuning achieves 80% validation accuracy and LoRA achieves 79%.

 Is LoRA a failure?”

 **Answer:**

 “Not necessarily.

 If LoRA uses dramatically fewer trainable parameters and achieves nearly the same accuracy, that may be an excellent result.

 The whole point is the **trade-off between performance and efficiency**.”

---

 ### Question 82

 “What if LoRA achieves 80.5%?”

 **Answer:**

 “That is even more interesting.

 The assignment specifically asks us to try different ranks to see whether LoRA can outperform the dense baseline.

 But we should not conclude that LoRA is universally better from one experiment. We would need controlled experiments and ideally multiple random seeds.”

---

 # Part 22 — Reading the training curves

 ### Question 83

 “What should we look for in the loss curves?”

 **Answer:**

 “Generally, we expect training loss to decrease.

 Validation loss should ideally decrease initially and then stabilise.

 If validation loss starts increasing while training loss continues decreasing, that can indicate overfitting.”

---

 ### Question 84

 “What should we look for in accuracy curves?”

 **Answer:**

 “We want training and validation accuracy to improve.

 The important comparison is whether LoRA eventually reaches a validation accuracy close to the dense baseline.”

---

 ### Question 85

 “If training accuracy is very high but validation accuracy is much lower, what might be happening?”

 **Answer:**

 “Potentially overfitting.

 The model is fitting the training examples well but generalising poorly.”

---

 # Part 23 — Understanding the timing experiment

 ### Question 86

 “If the assignment asks us to compare forward and backward time, what exactly should we measure?”

 **Answer:**

 “We should measure the time required for:

 1. a forward pass;
2. loss calculation;
3. backward pass.

 Ideally, we should perform repeated measurements and average them rather than relying on one noisy measurement.”

---

 ### Question 87

 “Why is GPU timing tricky?”

 **Answer:**

 “GPU operations are often asynchronous.

 The CPU may continue before the GPU has actually completed the computation.

 Therefore, when timing CUDA operations, we generally need synchronisation, for example with:

```
torch.cuda.synchronize()
```

 before reading the elapsed time.”

---

 # Part 24 — Extension: merging LoRA

 **Teacher:**

 “Now imagine we have finished training.”

 ### Question 88

 “We have:

 $$
W+BA.
$$

 Can we combine those into one ordinary weight matrix?”

 **Answer:**

 “Yes.”

 We calculate:

 $$
\boxed{W_{\text{merged}}=W+BA}
$$

 and replace the original weight with the merged weight.”

---

 ### Question 89

 “What is the advantage?”

 **Answer:**

 “After merging, we no longer need to calculate the separate LoRA branch during inference.

 The resulting layer can behave like an ordinary linear layer.”

---

 ### Question 90

 “Does merging change the mathematical output?”

 **Answer:**

 “No, assuming the merge is performed correctly.

 Before merging:

 $$
Wx+BAx.
$$

 After merging:

 $$
(W+BA)x.
$$

 By distributivity:

 $$
(W+BA)x=Wx+BAx.
$$

 So the outputs are mathematically identical, apart from ordinary floating-point numerical differences.”

---

 # Part 25 — Saving only adapters

 ### Question 91

 “Why might we want to save only $A$ and $B$?”

 **Answer:**

 “Because the base model may already exist.

 Suppose one base model is adapted to:

 - medical images;
- satellite images;
- animals;
- vehicles.

 We could potentially store separate small adapters rather than separate complete copies of the entire base model.”

---

 ### Question 92

 “What would be needed to load an adapter later?”

 **Answer:**

 “We need:

 1. the same compatible base model;
2. the adapter matrices $A$ and $B$;
3. the rank and architecture information;
4. the target classifier if the task requires a different output head.”

---

 # Part 26 — GPU memory

 ### Question 93

 “Does freezing a parameter remove its memory from the GPU?”

 **Answer:**

 “Not necessarily.

 The frozen parameter still has to be stored because the model needs it for the forward pass.”

---

 ### Question 94

 “So where can LoRA save memory?”

 **Answer:**

 “Primarily by reducing:

 - gradient storage for trainable parameters;
- optimiser state;
- potentially activation-related backward requirements depending on implementation;
- and, if we save adapters only, disk storage.”

---

 # Part 27 — Debugging questions

 **Teacher:**

 “Now let's imagine your LoRA implementation produces bad results.”

 ### Question 95

 “What is the first thing you should check?”

 **Answer:**

 “Check whether the initial LoRA model produces the same output as the corresponding pre-trained model.

 Because initially:

 $$
BA=0.
$$

 So the adapter should initially make no change.”

---

 ### Question 96

 “If the outputs are different immediately, what could be wrong?”

 **Answer:**

 “Possible causes include:

 - incorrect $A$ shape;
- incorrect $B$ shape;
- incorrect matrix multiplication order;
- incorrect transpose;
- incorrect bias handling;
- $B$ not initialised to zero;
- base weights not copied correctly;
- base weights accidentally modified.”

---

 ### Question 97

 “If training accuracy never improves, what should you inspect?”

 **Answer:**

 “Check:

 - Are $A$ and $B$ `nn.Parameter`s?
- Do they have `requires_grad=True`?
- Are they actually included in the optimiser?
- Is the forward pass using them?
- Is the learning rate appropriate?
- Is the classifier trainable?
- Are the labels correct?
- Is the model receiving the correct target data?”

---

 ### Question 98

 “If the LoRA parameter count is almost the same as dense fine-tuning, what is probably wrong?”

 **Answer:**

 “Most likely the original full-rank weights have not been frozen, or the adapter matrices were accidentally created at full rank.”

---

 # Part 28 — One complete conceptual walkthrough

 **Teacher:**

 “Let's see if we can now explain the entire practical in one sequence.”

 ### Question 99

 “Step one: what do we do?”

 **Answer:**

 “Train a ConvMLP on CIFAR-10.”

---

 ### Question 100

 “Step two?”

 **Answer:**

 “Save the pre-trained model.”

---

 ### Question 101

 “Step three?”

 **Answer:**

 “Select a five-class subset of CIFAR-100 as our target domain.”

---

 ### Question 102

 “Step four?”

 **Answer:**

 “Create a baseline model with the pre-trained feature/MLP weights and a new five-class classifier.”

---

 ### Question 103

 “Step five?”

 **Answer:**

 “Freeze the convolutional layers and densely fine-tune the MLP and classifier.”

---

 ### Question 104

 “Why do we need this baseline?”

 **Answer:**

 “To answer the question:

 **How much performance do we lose, if any, when we replace dense fine-tuning with LoRA?**”

---

 ### Question 105

 “Step six?”

 **Answer:**

 “Create LoRA versions of the fully connected layers.”

---

 ### Question 106

 “What happens to the original weights?”

 **Answer:**

 “They are frozen.”

---

 ### Question 107

 “What replaces their trainable updates?”

 **Answer:**

 “Two small matrices:

 $$
A
$$

 and

 $$
B.
$$

 Their product approximates the update:

 $\Delta W\approx BA.$”

---

 ### Question 108

 “What is the resulting forward equation?”

 **Answer:**

 $$
\boxed{
z=Wx+BAx+b
}
$$

 with $W$ frozen and $A,B$ trainable.”

---

 ### Question 109

 “Step seven?”

 **Answer:**

 “Fine-tune the LoRA model on the same target data using the same general training setup.”

---

 ### Question 110

 “What do we compare?”

 **Answer:**

 “We compare:

 - validation accuracy;
- training curves;
- number of trainable parameters;
- model state-dict size;
- optimiser state-dict size;
- forward-pass time;
- backward-pass time;
- GPU memory, if doing the extension;
- and different values of $r$.”

---

 # Part 29 — The final conceptual test

 **Teacher:**

 “I'm going to give you a series of statements. Decide whether each is true or false.”

 ### Question 111

 “LoRA modifies the original pre-trained weight matrix directly.”

 **Answer: False.**

 “The original weight matrix is frozen. LoRA learns an additional low-rank update.”

---

 ### Question 112

 “LoRA replaces the update $\Delta W$ with $BA$.”

 **Answer: True.**

 $$
\Delta W\approx BA.
$$

---

 ### Question 113

 “$r$ should normally be much smaller than the dimensions of the original weight matrix.”

 **Answer: True.**

 “That is what gives LoRA its parameter efficiency.”

---

 ### Question 114

 “If $B=0$, then $BA=0$, regardless of $A$.”

 **Answer: True.**

 “That is why zero-initialising $B$ is useful.”

---

 ### Question 115

 “The LoRA model must have fewer total parameters than the original model.”

 **Answer: False.**

 “The frozen base model is still present. What LoRA dramatically reduces is the number of **trainable parameters**.”

---

 ### Question 116

 “LoRA always makes the forward pass faster.”

 **Answer: False.**

 “The frozen base computation still occurs, and the LoRA branch introduces additional computation.”

---

 ### Question 117

 “LoRA can reduce optimiser memory.”

 **Answer: True.**

 “Because far fewer parameters are trainable.”

---

 ### Question 118

 “After training, $W+BA$ can be merged into one weight matrix.”

 **Answer: True.**

 “That gives an ordinary layer with equivalent mathematical behaviour.”

---

 ### Question 119

 “Choosing a larger $r$ always gives better validation accuracy.”

 **Answer: False.**

 “A larger rank gives greater capacity, but it may not improve generalisation and reduces the efficiency advantage.”

---

 # Part 30 — The students' final challenge

 **Teacher:**

 “Now I want you to answer this without looking at the notebook.”

 ### Question 120

 **Why does LoRA make fine-tuning parameter-efficient?**

 Pause here and formulate your answer.

 The ideal answer is:

 > “Instead of updating a large pre-trained weight matrix $W$, LoRA freezes $W$ and learns a low-rank update $BA$. Because the rank $r$ is much smaller than the input and output dimensions, $A$ and $B$ contain far fewer trainable parameters than $W$. This reduces the number of gradients and optimiser states required during fine-tuning, while still allowing the model to adapt to the target task.”

---

 # Final five-question recap

 **Teacher:**

 “Let's finish with five questions. If you can answer these, you understand the core of the practical.”

 ### Question 1

 “What is transfer learning?”

 **Answer:**

 “Reusing knowledge learned on a source task to help solve a target task.”

 ### Question 2

 “What is dense fine-tuning?”

 **Answer:**

 “Updating selected full-rank model parameters using their gradients.”

 ### Question 3

 “What is LoRA?”

 **Answer:**

 “A parameter-efficient fine-tuning method that freezes the original weights and learns a low-rank update.”

 ### Question 4

 “What is the key equation?”

 **Answer:**

 $$
\boxed{
W_{\text{adapted}}=W+BA
}
$$

 and consequently:

 $$
\boxed{
z=Wx+BAx+b
}
$$

 ### Question 5

 “What is the fundamental trade-off controlled by $r$?”

 **Answer:**

 “Rank $r$ controls adapter capacity versus efficiency:

 $$
\boxed{
\text{small }r
\rightarrow
\text{efficient but less expressive}
}
$$

 $$
\boxed{
\text{large }r
\rightarrow
\text{more expressive but less parameter-efficient}
}
$$

 And that is the central idea behind Practical 4: **can we adapt a pre-trained model to a new domain without having to update all of its large weight matrices?**”

---

 ## One sentence I would want students to remember

 > **“LoRA freezes the knowledge we already have and learns a small, low-rank correction to that knowledge.”**

 That sentence captures the entire practical: **pre-training → transfer learning → dense baseline → frozen weights → low-rank update → parameter-efficient fine-tuning → comparison of accuracy, computation and memory.**
