# Tutorial: How to Read an AI/ML Program

 We'll use **PyTorch** as the primary framework, because it exposes the underlying ideas quite clearly. Once you understand the concepts, TensorFlow becomes much easier to read.

 We'll progressively analyze a small neural network rather than throwing a large program at you.

---

 ## Part 1 — The fundamental question

 ### Q1. What is an AI/ML program actually manipulating?

 At the lowest level, an ML program is manipulating **data structures containing numbers**.

 For example:

```
x = [2, 4, 6, 8]
```

 But neural networks generally don't work with ordinary Python lists.

 They work with structures such as:

```
x = torch.tensor([2, 4, 6, 8])
```

 This gives us our first important concept:

 > **Tensor = a multidimensional container of numbers.**

 So, at the beginning, we can think:

```
Python program
      ↓
creates numerical structures
      ↓
performs mathematical operations on them
      ↓
produces new numerical structures
      ↓
eventually produces a prediction
```

---

 # Part 2 — What structures should I look for?

 When reading a PyTorch neural-network program, I want you to train yourself to identify these major structures:

```
1. Input data
2. Labels / targets
3. Tensors
4. Dataset
5. DataLoader
6. Model
7. Layers
8. Parameters / weights
9. Intermediate activations
10. Prediction
11. Loss
12. Gradients
13. Optimizer
```

 Don't worry about understanding all of them yet.

 We're going to discover them one by one.

---

 # Part 3 — Start with a tiny problem

 Let's create the smallest meaningful neural-network problem.

 Suppose we have:

```
Input → Output

1 → 3
2 → 5
3 → 7
4 → 9
```

 Can you see the relationship?

 The output is:

```
output = 2 × input + 1
```

 Our goal is to create a neural network that **learns this relationship from examples**.

---

 ## Q2. What are our inputs?

 Our inputs are:

```
x = [1, 2, 3, 4]
```

 And the desired outputs are:

```
y = [3, 5, 7, 9]
```

 We can represent them as tensors:

```
import torch

x = torch.tensor([1., 2., 3., 4.])
y = torch.tensor([3., 5., 7., 9.])
```

 Now pause.

 ### What structures exist?

 We have:

```
x
│
└── Tensor containing input numbers

y
│
└── Tensor containing desired answers
```

 Already, you can start reading ML code differently.

 Instead of thinking:

 > "Why are they creating these weird tensors?"

 think:

 > **"They are creating numerical data structures that will be manipulated by the neural network."**

---

 # Part 4 — What is the neural network manipulating?

 Let's create a very simple model:

```
import torch.nn as nn

model = nn.Linear(1, 1)
```

 This one line hides something extremely important.

 ## Q3. What is `nn.Linear(1, 1)`?

 It represents a mathematical operation approximately like:

```
y = wx + b
```

 where:

```
x = input
w = weight
b = bias
y = output
```

 So the model contains two important things:

```
weight
bias
```

 Initially, PyTorch doesn't know the correct values of `w` and `b`.

 It might start with something like:

```
w = 0.37
b = -0.82
```

 Those numbers are initially not what we want.

 Our training process will manipulate them.

---

 # Part 5 — The central idea

 This gives us one of the most important mental models for neural networks:

```
INPUT
  │
  ▼
┌──────────────┐
│ Neural       │
│ Network      │
│              │
│ weights      │
│ biases       │
└──────────────┘
  │
  ▼
PREDICTION
```

 But we don't know whether the prediction is good.

 So we need another structure.

---

 # Part 6 — What is the prediction?

 Suppose our model currently has:

```
weight = 0.37
bias   = -0.82
```

 For:

```
x = 4
```

 the model produces:

```
y_pred = 0.37 × 4 - 0.82
       = 0.66
```

 But the correct answer is:

```
9
```

 So:

```
prediction = 0.66
actual      = 9
```

 Clearly, our model is bad.

 Now we have another important question.

---

 # Part 7 — How does the program know that the prediction is bad?

 ## Q4. What is "loss"?

 We calculate a number representing **how wrong the model is**.

 For example:

```
loss = (prediction - actual) ** 2
```

 If:

```
prediction = 0.66
actual = 9
```

 then:

```
loss = (0.66 - 9)²
```

 which is large.

 If instead:

```
prediction = 8.9
actual = 9
```

 then:

```
loss = (8.9 - 9)²
```

 which is very small.

 So:

 > **Loss is a numerical measurement of how badly the model's prediction differs from the desired answer.**

 Now our program has a very important flow:

```
INPUT
  ↓
MODEL
  ↓
PREDICTION
  ↓
LOSS
```

 But we're not done.

---

 # Part 8 — How does the model improve?

 This is where neural networks become particularly interesting.

 We have:

```
input
  ↓
model
  ↓
prediction
  ↓
loss
```

 The loss tells us:

 > "Your current weights produced a bad result."

 But we need to answer:

 > **Which weights should change, and by how much?**

 That's the job of **gradients**.

---

 # Part 9 — What is a gradient?

 Imagine our model has:

```
weight = w
```

 and the loss depends on that weight:

```
loss = f(w)
```

 The gradient tells us roughly:

 > **"If I change this weight slightly, which direction will make the loss decrease?"**

 For example:

```
gradient = +5
```

 means roughly:

```
increase weight → loss increases
decrease weight → loss decreases
```

 So we want to move the weight in the opposite direction of the gradient.

 This is the basic idea behind **gradient descent**.

---

 # Part 10 — What is backpropagation?

 This is one of the most important concepts to understand.

 During the forward pass:

```
input
  ↓
layer
  ↓
layer
  ↓
prediction
  ↓
loss
```

 The information travels **forward**.

 Then we calculate gradients by moving backward through the computation:

```
loss
  ↓
gradient
  ↓
previous operation
  ↓
gradient
  ↓
previous operation
  ↓
...
```

 This is called:

 > **Backpropagation**

 PyTorch can automatically calculate these gradients for us.

 For example:

```
loss.backward()
```

 That tiny line can perform a substantial amount of mathematical work.

---

 # Part 11 — Who actually changes the weights?

 Now we have:

```
weights
      ↓
forward pass
      ↓
prediction
      ↓
loss
      ↓
backpropagation
      ↓
gradients
```

 But the gradients themselves don't change the weights.

 Something needs to use them.

 That is the **optimizer**.

 For example:

```
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)
```

 The optimizer looks at the gradients and updates the model's parameters.

 Conceptually:

```
weight_new =
    weight_old - learning_rate × gradient
```

 So now our entire training process becomes:

```
             ┌──────────────┐
             │    INPUT     │
             └──────┬───────┘
                    ↓
             ┌──────────────┐
             │    MODEL     │
             │ weights etc. │
             └──────┬───────┘
                    ↓
             ┌──────────────┐
             │  PREDICTION  │
             └──────┬───────┘
                    ↓
             ┌──────────────┐
             │     LOSS     │
             └──────┬───────┘
                    ↓
             ┌──────────────┐
             │BACKPROPAGATION│
             └──────┬───────┘
                    ↓
             ┌──────────────┐
             │  GRADIENTS   │
             └──────┬───────┘
                    ↓
             ┌──────────────┐
             │  OPTIMIZER   │
             └──────┬───────┘
                    ↓
             UPDATE WEIGHTS
                    │
                    └───────────→ repeat
```

 **This diagram is probably the single most important thing to understand before diving deeply into PyTorch.**

---

 # Part 12 — Now let's look at actual PyTorch code

 Here's a complete tiny training program:

```
import torch
import torch.nn as nn

# 1. Data
x = torch.tensor([[1.], [2.], [3.], [4.]])
y = torch.tensor([[3.], [5.], [7.], [9.]])

# 2. Model
model = nn.Linear(1, 1)

# 3. Loss function
loss_fn = nn.MSELoss()

# 4. Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 5. Training
for epoch in range(1000):

    # Forward pass
    prediction = model(x)

    # Calculate error
    loss = loss_fn(prediction, y)

    # Calculate gradients
    optimizer.zero_grad()
    loss.backward()

    # Update weights
    optimizer.step()
```

 Instead of trying to memorize this code, let's **read it as a sequence of manipulations**.

---

 # Part 13 — Read the program as a sequence of operations

 ### Q5. What happens first?

```
x = ...
y = ...
```

 We create:

```
INPUT DATA
TARGET DATA
```

---

 ### Q6. What happens next?

```
model = nn.Linear(1, 1)
```

 We create a structure containing learnable parameters:

```
MODEL
 ├── weight
 └── bias
```

---

 ### Q7. What happens next?

```
prediction = model(x)
```

 The input tensor is passed through the model.

 So:

```
x
↓
model
↓
prediction
```

 This is the **forward pass**.

---

 ### Q8. What happens next?

```
loss = loss_fn(prediction, y)
```

 We compare:

```
prediction
     vs
desired answer
```

 and produce:

```
loss
```

 So:

```
prediction + target
        ↓
       loss
```

---

 ### Q9. What does this do?

```
loss.backward()
```

 PyTorch calculates gradients for the model's learnable parameters.

 Conceptually:

```
loss
 ↓
gradient of loss
 ↓
weight gradient
bias gradient
```

---

 ### Q10. What does this do?

```
optimizer.step()
```

 The optimizer uses those gradients to modify:

```
weight
bias
```

 So:

```
old weights
    ↓
gradients
    ↓
optimizer
    ↓
new weights
```

---

 # Part 14 — Why is there a loop?

 This:

```
for epoch in range(1000):
```

 means:

 > **Repeat the learning process many times.**

 One iteration:

```
data
 ↓
prediction
 ↓
loss
 ↓
gradient
 ↓
weight update
```

 Then do it again:

```
data
 ↓
prediction
 ↓
loss
 ↓
gradient
 ↓
weight update
```

 Again.

 Again.

 Again.

 Eventually we hope:

```
loss ↓↓↓↓↓↓↓
```

 and:

```
prediction → correct answer
```

---

 # Part 15 — The really important structures

 At this point, we can create a useful map.

 | Structure | What it contains/represents | What happens to it? |
| --- | --- | --- |
| Input tensor | Input numbers | Passed through model |
| Target tensor | Correct answers | Compared with prediction |
| Model | Mathematical transformations + parameters | Processes input |
| Parameters | Weights and biases | Updated during training |
| Prediction | Model's current answer | Compared with target |
| Loss | How wrong prediction is | Used to calculate gradients |
| Gradients | Direction of parameter improvement | Used by optimizer |
| Optimizer | Parameter-update mechanism | Changes weights |

This is the **object-and-operation view** of an ML program.

---

 # Part 16 — Now introduce layers

 Real neural networks are more complicated than:

```
x → Linear → prediction
```

 They might look like:

```
Input
  ↓
Linear
  ↓
ReLU
  ↓
Linear
  ↓
ReLU
  ↓
Linear
  ↓
Output
```

 For example:

```
model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)
```

 Don't look at this as mysterious Python syntax.

 Read it as:

```
Tensor
  ↓
Linear transformation
  ↓
Tensor
  ↓
ReLU transformation
  ↓
Tensor
  ↓
Linear transformation
  ↓
Tensor
  ↓
ReLU transformation
  ↓
Tensor
  ↓
Linear transformation
  ↓
Output tensor
```

 This is a **pipeline of transformations of tensors**.

 That perspective is extremely useful.

---

 # Part 17 — The deeper pattern

 Eventually, when you encounter a complicated AI program, you should be able to ask these questions:

 ### 1\. What are the objects?

```
What tensors exist?
What is the model?
What are the parameters?
What is the dataset?
What is the loss?
What is the optimizer?
```

 ### 2\. What does each object contain?

 For example:

```
Tensor
 ├── values
 ├── shape
 ├── dtype
 └── device
```

 And:

```
Model
 ├── layers
 ├── parameters
 └── operations
```

 ### 3\. What operation is being performed?

 Examples:

```
matrix multiplication
addition
activation function
normalization
reshaping
indexing
concatenation
convolution
attention
```

 ### 4\. What is the input to the operation?

 For example:

```
Tensor A
    +
Tensor B
    ↓
Tensor C
```

 ### 5\. What is the output?

```
Tensor C
```

 ### 6\. Why is the operation happening?

 This is perhaps the most important question.

 Not merely:

 > "What does `torch.matmul()` do?"

 but:

 > **"Why does this neural network need matrix multiplication at this point?"**

---

 # Part 18 — And eventually we reach the full picture

 A typical training program can be understood as:

```
                DATA
                 │
                 ▼
            ┌─────────┐
            │ Tensors │
            └────┬────┘
                 │
                 ▼
            ┌─────────┐
            │  Model  │
            └────┬────┘
                 │
                 ▼
           PREDICTION
                 │
                 ▼
            ┌─────────┐
            │  Loss   │
            └────┬────┘
                 │
                 ▼
          BACKPROPAGATION
                 │
                 ▼
            GRADIENTS
                 │
                 ▼
            OPTIMIZER
                 │
                 ▼
        UPDATED PARAMETERS
                 │
                 └──────────┐
                            │
                            ▼
                         REPEAT
```

 And the ultimate goal is:

```
             INPUT
               ↓
          trained model
               ↓
          useful prediction
```

---

 # Our tutorial roadmap

 I would recommend learning this in **layers**, rather than trying to learn PyTorch API functions individually.

 ### Stage 1 — Numerical structures

 We'll understand:

 - Python variables
- lists
- NumPy arrays
- PyTorch tensors
- tensor dimensions
- shape
- dtype
- device
- CPU vs GPU

 ### Stage 2 — Tensor manipulation

 We'll take tensors and ask:

 - How are they created?
- How are they indexed?
- How are they reshaped?
- How are they combined?
- What does broadcasting mean?
- What does matrix multiplication actually do?

 ### Stage 3 — Mathematical operations

 We'll understand:

```
addition
multiplication
matrix multiplication
dot product
mean
sum
activation functions
```

 and, importantly, **why neural networks need them**.

 ### Stage 4 — Neural-network structures

 We'll investigate:

```
Layer
Neuron
Weight
Bias
Activation
Parameter
Model
```

 ### Stage 5 — Forward propagation

 We'll trace an actual tensor:

```
input tensor
    ↓
layer 1
    ↓
activation
    ↓
layer 2
    ↓
activation
    ↓
output
```

 We'll actually track **shapes and values** through the network.

 ### Stage 6 — Loss

 We'll answer:

 > How does a bunch of numbers called "loss" tell the network that it made a bad prediction?

 ### Stage 7 — Gradients and backpropagation

 This is where we'll slow down considerably.

 We'll follow:

```
output
 ↓
loss
 ↓
gradient
 ↓
weight gradient
 ↓
parameter update
```

 rather than treating `loss.backward()` as magic.

 ### Stage 8 — Optimizers

 We'll understand:

```
SGD
learning rate
momentum
Adam
```

 and what these actually manipulate.

 ### Stage 9 — Dataset and DataLoader

 Then we'll introduce:

```
Dataset
    ↓
DataLoader
    ↓
batches
    ↓
model
```

 and understand why ML programs don't normally throw the entire dataset into the model at once.

 ### Stage 10 — A complete PyTorch program

 Finally we'll take a real but small program and dissect **every line** using the same questions:

 > **What structure is this?**\
>  **What does it contain?**\
>  **What operation is happening?**\
>  **What is the input?**\
>  **What is the output?**\
>  **Why are we doing this?**\
>  **What happens next?**

 Then we can do exactly the same thing with a **CNN**, followed by a **Transformer/LLM**, where this mental model becomes particularly powerful.

 If you want to follow this tutorial interactively, **we should start with Stage 1: tensors** and build everything from there, using tiny pieces of PyTorch code and questions before moving to the next concept.
