# Advanced Machine Learning — LoRA Practice Exam

This practice exam applies the structure and reasoning style of the official Autumn 2026 past paper to Low-Rank Adaptation (LoRA), parameter-efficient fine-tuning (PEFT), and transfer learning.

It has two parts:

- TD1 — Practice exam: Questions only, with no answers revealed. Attempt this under exam conditions.
- TD2 — Comprehensive solutions: Model answers, mathematical derivations, explanations, and mark allocations for every question.

The questions emphasize conceptual understanding, interpreting algorithms and code, explaining mathematical relationships, and diagnosing experimental outcomes. No code writing is required.

## TD1 — Practice exam

# Exam conditions

Total marks

# 50

Points

Suggested duration

# 90 min

Pen-and-paper style

Question types

## MCQ + written

Materials

## One double-sided sheet

- For multiple-choice questions, select all correct answers. The number of correct options is not disclosed.
- Explain your reasoning in written answers; definitions alone may not earn full marks.
- You may describe algorithms in natural language, pseudocode, or equations.
- No code writing is required, but you may need to interpret code or diagnose an error.
- Experimental results are hypothetical unless explicitly stated.

## Section A — Multiple choice (10 points)

### Question 1 — Transfer learning and fine-tuning (2 points)

Which statements about adapting a pretrained model to a target task are correct? Select all that apply.

- A. Fine-tuning can benefit from pretrained features when the source and target tasks have useful similarities.
- B. The source and target tasks must have exactly the same number of classes.
- C. A target classifier with a different output dimension may need to be replaced or reinitialized.
- D. Freezing a layer removes its weights from the model.
- E. A smaller learning rate is often useful when adapting pretrained parameters.

### Question 2 — LoRA fundamentals (2 points)

Which statements about Low-Rank Adaptation are correct? Select all that apply.

- A. LoRA freezes the original weights of the layers being adapted and learns a low-rank update.
- B. LoRA always reduces the total number of parameters stored in the complete model.
- C. A low-rank update can reduce the number of trainable parameters substantially.
- D. LoRA guarantees higher validation accuracy than dense fine-tuning.
- E. LoRA can be used with selected fully connected layers while leaving convolutional layers frozen.

### Question 3 — LoRA factorization and initialization (2 points)

Let a linear layer have weight matrix $W\in\mathbb{R}^{d_{\text{out}}\times d_{\text{in}}}$. Consider the update

$$
W_{\text{effective}}=W+BA.
$$

Which statements are correct? Select all that apply.

- A. $A$ can have shape $r\times d_{\text{in}}$.
- B. $B$ can have shape $d_{\text{out}}\times r$.
- C. The product $BA$ has the same shape as $W$.
- D. Setting $B=0$ initially makes the adapter update zero.
- E. Both $A$ and $B$ should always be initialized to zero to maximize gradient flow.

### Question 4 — Training and gradient computation (2 points)

Which statements are correct for the LoRA practical? Select all that apply.

- A. Frozen parameters can still participate in the forward pass.
- B. `requires_grad=False` prevents the corresponding parameter from receiving ordinary autograd gradients.
- C. The original frozen weights must be deleted before the adapter can operate.
- D. An optimizer should be configured to update the intended trainable parameters.
- E. `model.eval()` by itself disables all gradient computation.

### Question 5 — Interpreting experimental evidence (2 points)

A dense fine-tuning baseline and a LoRA model achieve validation accuracies of 82% and 80%, respectively. The LoRA model has substantially fewer trainable parameters.

Which conclusions are justified? Select all that apply.

- A. LoRA is necessarily the better method for every deployment scenario.
- B. LoRA may provide a useful trade-off between predictive performance and parameter efficiency.
- C. The results alone establish that LoRA has lower inference latency.
- D. The two percentage values differ by 2 percentage points.
- E. The result proves that the LoRA implementation is mathematically identical to dense fine-tuning.

## Section B — Short written answers (12 points)

### Question 6 — Transfer learning and the target classifier (3 points)

The Practical 4 notebook pretrains a convolutional MLP on CIFAR-10 and adapts it to the five fine-grained classes in the CIFAR-100 `fish` superclass.

(a) Explain why reusing the pretrained feature extractor may be beneficial. (1 point)

(b) Explain why the source classifier cannot simply be reused unchanged. (1 point)

(c) Explain why the target labels must be remapped to consecutive indices from 0 to 4. (1 point)

### Question 7 — Dense fine-tuning versus LoRA (4 points)

In the notebook, the convolutional layers are frozen in both the dense fine-tuning baseline and the LoRA model.

(a) Identify which parameters are trainable in the dense baseline. (1 point)

(b) Identify which parameters are trainable in the intended LoRA model. (1 point)

(c) Explain the fundamental difference between the updates learned by the two methods. (2 points)

## Section C — Algorithm interpretation and application (12 points)

### Question 8 — Interpret the LoRA algorithm (4 points)

Consider the following training procedure:

1. Load a pretrained model.
2. Freeze the convolutional layers.
3. Replace each selected fully connected layer with a wrapper that preserves its original weights and introduces two trainable matrices $A$ and $B$.
4. Initialize $A$ randomly and $B$ to zero.
5. For each training batch, calculate the base layer output and the adapter output, then add them.
6. Compute cross-entropy against the target labels and update the trainable adapter parameters and target classifier.
7. Evaluate validation accuracy after each epoch.

(a) Identify the learning technique being implemented. (1 point)

(b) Explain why $B$ is initialized to zero. (1 point)

(c) Explain why the original weights should remain frozen. (1 point)

(d) Explain why the classifier remains trainable even though the original fully connected weights are frozen in the adapted layers. (1 point)

### Question 9 — Diagnose an implementation bug (4 points)

A student implements the following conceptual wrapper for a linear layer:

python

```
class LoraLayer(nn.Module):
    def __init__(self, original_layer, rank):
        super().__init__()
        self.original_layer = original_layer
        self.A = nn.Parameter(torch.randn(rank, original_layer.in_features))
        self.B = nn.Parameter(torch.zeros(original_layer.out_features, rank))
        self.verbose = verbose

    def forward(self, x):
        return self.original_layer(x) + F.linear(F.linear(x, self.A), self.B)
```

(a) Identify the immediate error that prevents successful initialization. (1 point)

(b) Explain two additional implementation requirements for this to be a correct LoRA layer. (2 points)

(c) Does the forward expression need to explicitly construct the full matrix $BA$? Explain. (1 point)

### Question 10 — Rank and parameter efficiency (4 points)

A linear layer has 1,024 input features and 512 output features. Ignore biases and assume a standard LoRA update with rank $r=8$.

(a) Calculate the number of weights in the original matrix. (1 point)

(b) Calculate the number of trainable adapter weights. (1 point)

(c) Calculate the percentage reduction in the number of weights being trained, relative to training the full matrix. (1 point)

(d) Explain one trade-off associated with increasing the rank. (1 point)

## Section D — Theoretical reasoning and quantitative analysis (16 points)

### Question 11 — Why does low-rank adaptation work? (5 points)

Suppose a pretrained linear layer uses

$$
y=Wx+b.
$$

LoRA replaces its effective transformation with

$$
y=(W+BA)x+b,
$$

where $A\in\mathbb{R}^{r\times d_{\text{in}}}$ and $B\in\mathbb{R}^{d_{\text{out}}\times r}$.

(a) Explain the intuition behind restricting the update to a low-rank matrix. (2 points)

(b) Explain why this restriction reduces the number of trainable parameters. (1 point)

(c) Explain what happens if $r$ is increased until the factorization can represent any matrix-shaped update. Does this automatically guarantee that LoRA training will produce the same result as dense fine-tuning? (2 points)

### Question 12 — Evaluate a practical experiment (5 points)

A researcher obtains the following hypothetical results on the same target validation set:

| Metric | Dense baseline | LoRA |
| --- | --- | --- |
| Best validation accuracy | 84% | 82% |
| Trainable parameters | 1,500,000 | 90,000 |
| Training time | 60 seconds | 75 seconds |
| Full checkpoint size | 6.0 MB | 6.2 MB |

(a) Calculate the reduction in trainable parameters as a percentage. (1 point)

(b) Calculate the absolute validation accuracy difference in percentage points. (1 point)

(c) Explain why LoRA might take longer to train despite using fewer trainable parameters. (1 point)

(d) Explain why the full LoRA checkpoint might not be smaller than the dense baseline checkpoint. (1 point)

(e) Give a balanced conclusion based on these measurements. (1 point)

### Question 13 — Freezing, gradients, and optimization (6 points)

A student intends to train LoRA adapters but discovers that the original fully connected weights are changing during training. The convolutional layers remain frozen, and the model achieves reasonable validation accuracy.

(a) Explain why this observation indicates that the intended LoRA experiment may not have been implemented correctly. (2 points)

(b) Describe two checks that could help identify the cause. (2 points)

(c) Explain why achieving reasonable validation accuracy does not establish that the implementation is correct or parameter-efficient. (2 points)

## TD2 — Comprehensive solutions and marking guide

Important: The solutions below are deliberately separated from the exam questions conceptually, but they are included in this response because you requested both tasks. For realistic self-testing, complete TD1 before reading further.

# Section A — Multiple-choice answers

## Question 1 — Transfer learning (2 points)

Correct answers: A, C, E.

- A — Correct. Pretrained features can transfer useful visual representations from the source task to the target task.
- B — Incorrect. The source and target tasks can have different numbers of classes. A classifier head can be replaced to match the target output dimension.
- C — Correct. A 10-class source classifier cannot directly serve as a 5-class target classifier without modification.
- D — Incorrect. Freezing prevents parameter updates; it does not remove the weights or stop the layer from performing its forward computation.
- E — Correct. A lower learning rate can help preserve useful pretrained representations while adapting them to the new task.

Marking: 2 points for the correct selection set. No partial credit unless the examiner's marking scheme allows it.

## Question 2 — LoRA fundamentals (2 points)

Correct answers: A, C, E.

- A — Correct. LoRA freezes the original weight matrix in the adapted layer and learns an additive low-rank update.
- B — Incorrect. The complete model still stores its frozen base weights. LoRA reduces trainable parameters, but this does not necessarily reduce the complete model's stored parameter count.
- C — Correct. Two smaller matrices can represent the update using substantially fewer parameters than a full matrix.
- D — Incorrect. LoRA does not guarantee superior accuracy. Its performance depends on the rank, architecture, task, optimization, and implementation.
- E — Correct. LoRA can be applied selectively to fully connected layers while leaving the convolutional layers frozen, as in the notebook.

## Question 3 — Factorization and initialization (2 points)

Correct answers: A, B, C, D.

The factor shapes are

$$
A\in\mathbb{R}^{r\times d_{\text{in}}},
\qquad
B\in\mathbb{R}^{d_{\text{out}}\times r}.
$$

Consequently,

$$
BA\in\mathbb{R}^{d_{\text{out}}\times d_{\text{in}}},
$$

which matches the shape of $W$.

- A — Correct. This is the standard shape of $A$.
- B — Correct. This is the standard shape of $B$.
- C — Correct. The dimensions of the product match the original weight matrix.
- D — Correct. If $B=0$, then $BA=0$ regardless of the initial value of $A$.
- E — Incorrect. Initializing both matrices to zero makes the initial gradients of the bilinear product with respect to both factors zero under the ordinary formulation. Random $A$ and zero $B$ allow the adapter to begin learning.

## Question 4 — Training and gradients (2 points)

Correct answers: A, B, D.

- A — Correct. Frozen layers must still compute outputs used by subsequent layers.
- B — Correct. `requires_grad=False` prevents ordinary autograd from calculating parameter gradients for that parameter.
- C — Incorrect. The original weights must remain available because the pretrained transformation is part of the model.
- D — Correct. The optimizer should update the intended trainable parameters, including the adapter factors and target classifier.
- E — Incorrect. `model.eval()` changes the behavior of modules such as dropout and batch normalization. It does not itself disable gradient tracking. `torch.no_grad()` or another gradient-control mechanism is used for that purpose.

## Question 5 — Experimental interpretation (2 points)

Correct answers: B, D.

- A — Incorrect. A parameter-efficiency advantage does not automatically make LoRA preferable for every use case.
- B — Correct. The measured result suggests a trade-off: slightly lower validation accuracy with substantially fewer trainable parameters.
- C — Incorrect. Training time and trainable parameter count do not establish inference latency. Inference must be measured.
- D — Correct. $82\%-80\%$ is not the comparison here; the difference between 82% and 80% is 2 percentage points.
- E — Incorrect. A low-rank update generally restricts the space of possible updates. Similar results do not establish mathematical equivalence.

# Section B — Short written answers

## Question 6 — Transfer learning and target classifier (3 points)

### (a) Why reuse the pretrained feature extractor? (1 point)

Pretraining on CIFAR-10 enables the network to learn visual features such as edges, textures, shapes, and object parts. Some of these representations are useful for distinguishing CIFAR-100 fish classes, so reusing them can reduce the amount of target-task training needed.

Full credit requires explaining the reuse of learned representations, rather than merely stating that transfer learning is faster.

### (b) Why replace the source classifier? (1 point)

The source classifier produces 10 logits, corresponding to the 10 CIFAR-10 classes. The target task contains five fish classes, so its classifier must produce five logits.

The source classifier's weight matrix has shape $10\times256$, while the target classifier's weight matrix has shape $5\times256$. These shapes are incompatible for direct loading into the target classifier.

### (c) Why remap labels to 0–4? (1 point)

Cross-entropy classification expects each target label to be a valid class index for the output logits. With five outputs, the valid indices are

$$
\{0,1,2,3,4\}.
$$

The original CIFAR-100 class IDs are not necessarily consecutive within the selected superclass. Remapping creates a consistent five-class label space.

## Question 7 — Dense fine-tuning versus LoRA (4 points)

### (a) Trainable parameters in the dense baseline (1 point)

The convolutional layers `conv1` and `conv2` are frozen. The fully connected layers `fc1`–`fc4` and the new five-class classifier remain trainable, including their ordinary weights and biases.

### (b) Trainable parameters in the LoRA model (1 point)

The convolutional layers are frozen. The original weights in the adapted fully connected layers are also frozen, and their LoRA matrices $A$ and $B$ are trainable. The target classifier remains trainable.

In the intended notebook setup, the original fully connected biases can remain frozen while the target classifier's weight and bias are updated.

### (c) Fundamental difference between the updates (2 points)

Dense fine-tuning learns a full weight update for each trainable fully connected layer. If the weight matrix has $d_{\text{out}}d_{\text{in}}$ entries, all of those entries can change.

LoRA freezes the original matrix $W$ and learns an update

$$
\Delta W=BA.
$$

The update is constrained to have rank at most $r$. Its trainable parameter count is

$$
r(d_{\text{in}}+d_{\text{out}}),
$$

rather than $d_{\text{out}}d_{\text{in}}$.

This constraint reduces the number of trainable parameters but also restricts the updates the model can represent.

Marking guidance: Award one point for identifying the full-matrix versus low-rank update distinction and one point for explaining the parameter-efficiency or representational trade-off.

# Section C — Algorithm interpretation and application

## Question 8 — Interpret the LoRA algorithm (4 points)

### (a) Identify the technique (1 point)

This is Low-Rank Adaptation (LoRA), a form of parameter-efficient fine-tuning. The model retains its pretrained base weights and learns low-rank adapter updates.

### (b) Why initialize $B=0$? (1 point)

Since $B=0$,

$$
BA=0.
$$

Therefore, at initialization,

$$
W_{\text{effective}}=W.
$$

The adapted layer initially behaves like the original pretrained layer. Random initialization of $A$ provides a nonzero factor through which learning can begin.

### (c) Why freeze the original weights? (1 point)

Freezing the original weights ensures that the adaptation is performed through the low-rank factors rather than by simultaneously changing the complete base matrix.

This preserves the pretrained transformation and reduces the number of trainable parameters, which is the central objective of LoRA.

### (d) Why train the classifier? (1 point)

The source classifier was designed to distinguish CIFAR-10 classes, while the target classifier must distinguish five CIFAR-100 fish classes. Its weights need to adapt to the new label space. Freezing the original feature-extraction weights does not remove the need to train a new target head.

## Question 9 — Diagnose an implementation bug (4 points)

### (a) Immediate error (1 point)

The line

python

```
self.verbose = verbose
```

references `verbose`, but `verbose` is neither a constructor argument nor a locally defined variable.

Unless a suitable name is explicitly supplied in the surrounding scope, initialization raises a `NameError`. The fix is to remove this line or add an appropriate `verbose` argument with a default value.

### (b) Two additional requirements (2 points)

Any two well-explained requirements earn full credit.

1. Freeze the base layer's parameters.
   The wrapper should ensure that the original weight and, in the intended setup, its bias have `requires_grad=False`. Otherwise, the experiment may update the base weights as well as the adapter.
2. Use the correct factor shapes and initialization.
   For rank $r$, $A$ must have shape $(r,d_{\text{in}})$, and $B$ must have shape $(d_{\text{out}},r)$. A standard initialization uses random $A$ and zero $B$.
3. Ensure the optimizer sees the intended trainable parameters.
   The adapter factors and target classifier must be included in the optimizer. Frozen parameters should not be inadvertently updated.
4. Handle optional scaling consistently.
   If the implementation uses a factor such as $\alpha/r$, it must apply that factor in the forward pass and use the same convention when merging or evaluating the adapter.

### (c) Must the full matrix $BA$ be constructed? (1 point)

No. The update can be computed using two linear transformations:

python

```
update = F.linear(F.linear(x, A), B)
```

This computes the low-rank contribution directly, avoiding the need to materialize the full $d_{\text{out}}\times d_{\text{in}}$ update matrix in every forward pass.

The complete output is the original layer's output plus this adapter contribution.

## Question 10 — Rank and parameter efficiency (4 points)

Given

$$
d_{\text{in}}=1024,\quad
d_{\text{out}}=512,\quad r=8.
$$

### (a) Original weight count (1 point)

$$
N_{\text{base}}=d_{\text{out}}d_{\text{in}}
$$

$$
=512\times1024
=\boxed{524,288}.
$$

### (b) Adapter weight count (1 point)

$$
N_{\text{LoRA}}=r(d_{\text{in}}+d_{\text{out}})
$$

$$
=8(1024+512)
=8\times1536
=\boxed{12,288}.
$$

### (c) Percentage reduction (1 point)

The reduction relative to training the full matrix is

$$
\text{Reduction}=\frac{N_{\text{base}}-N_{\text{LoRA}}}{N_{\text{base}}}\times100\%.
$$

Substituting the values:

$$
=\frac{524,288-12,288}{524,288}\times100\%
$$

$$
=\frac{512,000}{524,288}\times100\% \approx\boxed{97.66\%}.
$$

Thus, the adapter uses approximately 97.66% fewer trainable weight parameters than training the complete matrix.

### (d) Trade-off of increasing rank (1 point)

Increasing rank increases the capacity of the update, potentially improving adaptation when a very low rank is too restrictive. However, it also increases the number of trainable parameters, optimizer state, and adapter computation.

A larger rank is not guaranteed to improve validation accuracy, so rank selection should be evaluated experimentally.

# Section D — Theoretical reasoning and quantitative analysis

## Question 11 — Why does low-rank adaptation work? (5 points)

### (a) Intuition behind a low-rank update (2 points)

A pretrained model already contains useful representations. The target task may not require changing every entry in every pretrained weight matrix to adapt those representations.

LoRA assumes that a useful task-specific update can be represented or approximated by a low-rank matrix:

$$
\Delta W=BA.
$$

The factors represent the update through a smaller intermediate dimension $r$. Instead of learning an unrestricted change to the original transformation, the model learns a constrained update that can capture useful directions of adaptation.

A low-rank update does not guarantee that every task can be adapted equally well; it is a parameter-efficiency assumption that works well when the useful change can be approximated within the chosen rank.

### (b) Why fewer parameters? (1 point)

The full matrix contains

$$
d_{\text{out}}d_{\text{in}}
$$

weights.

The two adapter matrices contain

$$
rd_{\text{in}}+d_{\text{out}}r
=r(d_{\text{in}}+d_{\text{out}})
$$

weights.

When $r$ is small relative to the input and output dimensions, the adapter contains far fewer trainable parameters.

### (c) What if the rank is sufficiently large? (2 points)

A matrix of dimensions $d_{\text{out}}\times d_{\text{in}}$ has rank at most

$$
\min(d_{\text{out}},d_{\text{in}}).
$$

A factorization with a sufficiently large intermediate dimension can represent any matrix-shaped update. This removes the basic rank restriction on the update.

However, it does not automatically guarantee identical results to dense fine-tuning. The two methods have different parameterizations and optimization dynamics. Initialization, learning rates, regularization, optimizer states, and the optimization trajectory can still differ.

The distinction is between being able to represent an update and actually learning the same update during training.

## Question 12 — Evaluate a practical experiment (5 points)

The results are hypothetical and should be interpreted only as given.

### (a) Trainable parameter reduction (1 point)

Dense baseline:

$$
N_D=1,500,000.
$$

LoRA:

$$
N_L=90,000.
$$

The reduction is

$$
\frac{1,500,000-90,000}{1,500,000}\times100\%
=\boxed{94\%}.
$$

LoRA uses 94% fewer trainable parameters.

### (b) Validation accuracy difference (1 point)

$$
84\%-82\%=\boxed{2\text{ percentage points}}.
$$

LoRA achieves two percentage points lower best validation accuracy than the dense baseline.

This is an absolute difference, not a 2% relative reduction. The relative reduction in accuracy would be calculated differently.

### (c) Why might LoRA be slower? (1 point)

LoRA introduces additional adapter operations, including extra linear transformations. On a relatively small model, the overhead of these operations and framework execution can outweigh the savings from computing gradients for fewer parameters.

The measured times also depend on hardware, batch size, implementation, and timing methodology. Fewer trainable parameters do not guarantee shorter runtime.

### (d) Why might the complete checkpoint be larger? (1 point)

The complete LoRA checkpoint may contain the original frozen model weights as well as the learned adapter matrices. The dense baseline checkpoint may contain only the weights of its trained model.

Consequently, a full LoRA checkpoint can be similar in size or slightly larger. Saving only the adapter weights separately can produce a much smaller adapter-only artifact, but that artifact still requires access to the base model.

### (e) Balanced conclusion (1 point)

LoRA substantially reduces the number of trainable parameters, from 1.5 million to 90,000, while sacrificing two percentage points of best validation accuracy in this hypothetical experiment.

However, it takes longer to train and has a slightly larger full checkpoint. Therefore, it demonstrates parameter efficiency, but the measurements do not establish runtime or complete-checkpoint advantages. Whether it is preferable depends on the deployment requirements and whether the accuracy trade-off is acceptable.

## Question 13 — Freezing, gradients, and optimization (6 points)

### (a) Why is this a potential implementation error? (2 points)

The intended LoRA setup freezes the original fully connected weight matrices and learns through the adapter factors. If the original weights change during training, the model is not following that intended parameterization.

It may be performing a mixture of dense fine-tuning and adapter training. That can undermine the experiment's parameter-efficiency comparison and make the reported number of trainable parameters incorrect.

Reasonable validation accuracy does not rule out this error: the model can still learn a useful classifier while using a different optimization procedure than intended.

### (b) Two diagnostic checks (2 points)

Any two correctly explained checks earn full credit.

1. Inspect `requires_grad` flags. Examine the named parameters and verify that the original adapted-layer weights are frozen, while $A$, $B$, and the target classifier parameters are trainable.
2. Compare weights before and after an optimizer step. Save a copy of an original frozen weight matrix and check whether its values change after training. A change indicates that the freezing or optimization setup is not behaving as intended.
3. Inspect the optimizer's parameter groups. Check that the optimizer contains the intended trainable parameters and does not accidentally include parameters that should remain frozen. Including frozen parameters alone does not make them update if their gradients remain absent, but inspecting the groups helps detect configuration mistakes.
4. Check the wrapper's base layer. Verify that the original linear layer has been frozen after replacement and that no later initialization step or wrapper operation re-enables gradients.

### (c) Why accuracy alone is insufficient (2 points)

Accuracy measures predictive performance, not whether the intended training method was implemented correctly. A model that accidentally trains its full weight matrices may achieve good accuracy while using many more trainable parameters than reported.

A correct experimental evaluation should verify the parameter-freezing conditions, count trainable parameters, check optimizer configuration, and measure accuracy alongside runtime and memory use. The implementation must satisfy the experimental design before its results can support a claim about LoRA's parameter efficiency.

# Final revision checklist

Use this checklist to test whether you are prepared for an official-style question on LoRA.

## Revision progress

### 0 of 12

[ ]Explain why transfer learning can help on the five-class CIFAR-100 fish task.

[ ]Distinguish frozen base parameters from trainable parameters.

[ ]Derive the dimensions of A, B, and BA.

[ ]Explain why random A and zero B give a zero initial update.

[ ]Calculate LoRA parameter counts and percentage reductions.

[ ]Explain the effect of changing the rank.

[ ]Interpret an incomplete LoRA implementation and identify likely bugs.

[ ]Explain why fewer trainable parameters do not guarantee faster training.

[ ]Distinguish adapter-only checkpoint size from full model checkpoint size.

[ ]Interpret validation accuracy, training time, and parameter counts together.

[ ]Explain when a sufficiently high-rank LoRA update can represent a full matrix update.

[ ]Describe checks that verify the implementation actually freezes the intended weights.

Reset checklist

## The central exam takeaway

LoRA freezes pretrained weights and learns a low-rank update through smaller matrices, reducing the number of trainable parameters. The official exam is likely to reward more than remembering $W_{\text{effective}}=W+BA$: you should be able to explain why the method is useful, what its assumptions restrict, how the implementation should behave, and what the experimental results do—and do not—demonstrate.

For a fair comparison, always distinguish trainable parameter count, total model parameters, checkpoint size, optimizer memory, training time, inference latency, and predictive performance. They are related, but they are not interchangeable measurements.
