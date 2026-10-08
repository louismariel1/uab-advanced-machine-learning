Absolutely. Below are the **full model solutions for Mock Exams 1–3**, with calculations shown and explanations written at the level I would expect for a strong exam answer.

I’ve also included the **reasoning behind the answers**, because for this syllabus the distinction between merely knowing a definition and being able to apply it is likely to matter.

# Mock Exam 1 — Full Solutions

:::writing{variant="document" id="64182" title="Advanced Machine Learning — Mock Exam 1 Full Solutions"}

# Advanced Machine Learning

## Mock Examination 1 — Full Model Solutions

**Total: 100 marks**

---

# Section A — Multiple Choice

### Q1. Transfer learning

**Answer: B — Reusing knowledge learned from one task/domain to help another task.**

Transfer learning starts with a model that has learned useful representations from a source task/domain and adapts those representations to a target task/domain.

---

### Q2. Regularisation

**Answer: C — Increasing an appropriate regularisation mechanism.**

Regularisation discourages the model from fitting training data too specifically and can improve generalisation.

Examples include:

- weight decay;
- dropout;
- data augmentation;
- early stopping.

Simply increasing model capacity would generally not be described as a regularisation mechanism.

---

### Q3. Validation data

**Answer: B — To estimate performance during model selection without using the test set for tuning.**

The usual division is:

\[ \\text{Training set} \\rightarrow \\text{learn parameters} \]

\[ \\text{Validation set} \\rightarrow \\text{choose/tune model} \]

\[ \\text{Test set} \\rightarrow \\text{final unbiased-ish evaluation} \]

If the test set is repeatedly used for decisions, it effectively becomes part of the model-selection process.

---

### Q4. Quantisation

**Answer: B — It represents parameters using fewer numerical values/bits.**

For example:

\[ FP32\\rightarrow INT8 \]

reduces the number of bits used per parameter from 32 to 8.

Quantisation does not inherently remove parameters.

---

### Q5. 1 billion FP32 parameters

**Answer: C — 4 GB.**

Each FP32 parameter requires 32 bits:

\[ 1\\times10^9\\times32 = 32\\times10^9\\text{ bits} \]

Convert bits to bytes:

\[ \\frac{32\\times10^9}{8} = 4\\times10^9\\text{ bytes} \]

So approximately:

\[ \\boxed{4\\text{ GB}} \]

ignoring overhead and the distinction between decimal GB and GiB.

---

### Q6. LoRA

**Answer: C — It freezes pretrained weights and learns a low-rank task-specific update.**

The key equation is:

\[ \\boxed{W'=W+BA} \]

where (W) is frozen and (BA) is trainable.

---

### Q7. LoRA parameter count

**Answer: B —****(****2dr****)****.**

For a square matrix:

\[ W\\in\\mathbb{R}^{d\\times d} \]

LoRA uses:

\[ A\\in\\mathbb{R}^{r\\times d} \]

and:

\[ B\\in\\mathbb{R}^{d\\times r} \]

Therefore:

\[ N\_A=rd \]

\[ N\_B=dr \]

so:

\[ \\boxed{N\_{LoRA}=2dr} \]

---

### Q8. Freezing and inference

**Answer: B — Frozen layers still perform their forward computations.**

Freezing means their parameters are not updated during training.

It does **not** mean that the layer disappears.

Thus:

\[ \\boxed{\\text{freezing}\\neq\\text{removing computation}} \]

---

### Q9. Structured pruning

**Answer: B — Structured pruning removes regular structures such as channels, filters or neurons.**

Unstructured pruning can remove arbitrary individual weights.

Structured pruning removes larger regular units.

Regular structures are generally easier for hardware and dense linear-algebra libraries to exploit.

---

### Q10. SVD versus LoRA

**Answer: B — SVD compresses the existing matrix; LoRA represents the adaptation as a low-rank update.**

SVD-style compression:

\[ W\\approx W\_{low-rank} \]

LoRA:

\[ \\boxed{W'=W+\\Delta W} \]

with:

\[ \\boxed{\\Delta W\\approx BA} \]

The distinction between (W) and (\\Delta W) is critical.

---

# Section B — Short Answer

## Q11. Training, validation and test performance

### Model answer

The **training set** is used to learn the model parameters. The model directly minimises its training objective using this data.

The **validation set** is used during development to compare models and tune hyperparameters such as learning rate, regularisation strength, architecture or LoRA rank.

The **test set** should ideally be used only for the final evaluation of the chosen model.

Repeatedly tuning the model based on test-set performance causes the test set to influence model selection. Consequently, the test set is no longer an independent measure of generalisation.

A good summary is:

\[ \\boxed{ \\text{Training}\\rightarrow\\text{learn} } \]

\[ \\boxed{ \\text{Validation}\\rightarrow\\text{select/tune} } \]

\[ \\boxed{ \\text{Test}\\rightarrow\\text{final evaluation} } \]

---

## Q12. Two regularisation mechanisms

### Example 1: Weight decay

Weight decay penalises large weights.

Conceptually, the objective becomes:

\[ L_{total}=L_{data}+\\lambda|W|^2 \]

The additional penalty discourages unnecessarily large parameter values and can reduce overfitting.

### Example 2: Dropout

During training, dropout randomly removes/zeroes some activations.

This prevents the network from relying too heavily on particular pathways or features and encourages more robust representations.

### Other acceptable answers

Other valid mechanisms include:

- data augmentation;
- early stopping;
- suitable architectural constraints.

The important point is to explain **how** the mechanism reduces overfitting rather than merely naming it.

---

## Q13. FP32, FP16, INT8 and INT4

FP32 uses 32 bits per parameter.

FP16 uses 16 bits.

INT8 uses 8 bits.

INT4 uses 4 bits.

Therefore, ignoring overhead:

\[ FP32:32\\text{ bits} \]

\[ FP16:16\\text{ bits} \]

\[ INT8:8\\text{ bits} \]

\[ INT4:4\\text{ bits} \]

Relative to FP32:

\[ FP16\\rightarrow2\\times\\text{less storage} \]

\[ INT8\\rightarrow4\\times\\text{less storage} \]

\[ INT4\\rightarrow8\\times\\text{less storage} \]

Quantisation can reduce accuracy because continuous/high-precision values are mapped to a smaller set of representable values:

\[ w\\rightarrow Q(w) \]

so generally:

\[ Q(w)\\neq w \]

The resulting quantisation error can affect model outputs.

---

## Q14. Pruning types

### Magnitude pruning

Removes weights with small magnitude:

\[ |w|\\text{ small}\\Rightarrow w\\rightarrow0 \]

It assumes small weights are less important.

### Similarity pruning

Looks for redundant neurons/features.

If:

\[ h_1(x)\\approx h_2(x) \]

one may be redundant and potentially removable.

### Unstructured pruning

Removes individual weights at arbitrary locations.

It can create a highly sparse matrix.

The problem is that arbitrary sparsity may not be efficiently exploited by hardware.

### Structured pruning

Removes complete structures such as:

- channels;
- filters;
- neurons;
- blocks.

This produces more regular smaller computations.

Therefore structured pruning often has better practical hardware implications.

The key distinction is:

\[ \\boxed{\\text{parameter reduction}\\neq\\text{automatic runtime reduction}} \]

---

## Q15. Why PEFT reduces training memory

PEFT freezes most pretrained parameters.

Frozen parameters generally do not need:

- trainable gradients;
- optimizer states.

For example, Adam-type optimisers maintain additional state for trainable parameters.

With full fine-tuning:

\[ \\text{many parameters} \\rightarrow \\text{many gradients} \\rightarrow \\text{large optimizer state} \]

With PEFT:

\[ \\text{few trainable parameters} \\rightarrow \\text{fewer gradients} \\rightarrow \\text{smaller optimizer state} \]

However, the frozen model still needs to be stored and still participates in the forward pass.

Thus:

\[ \\boxed{\\text{PEFT primarily reduces training cost, not necessarily complete model storage or inference computation}} \]

---

# Section C — Calculations

## Q16. Quantisation calculation

Given:

\[ N=2.5\\times10^9 \]

parameters.

### (a) FP32

Each parameter requires 32 bits:

\[ 2.5\\times10^9\\times32 = 80\\times10^9\\text{ bits} \]

Convert to bytes:

\[ \\frac{80\\times10^9}{8} = 10\\times10^9 \]

Therefore:

\[ \\boxed{10\\text{ GB}} \]

approximately.

---

### (b) INT8

Each parameter requires 8 bits:

\[ 2.5\\times10^9\\times8 = 20\\times10^9\\text{ bits} \]

\[ \\frac{20\\times10^9}{8} = 2.5\\times10^9\\text{ bytes} \]

Therefore:

\[ \\boxed{2.5\\text{ GB}} \]

---

### (c) INT4

Each parameter requires 4 bits:

\[ 2.5\\times10^9\\times4 = 10\\times10^9\\text{ bits} \]

\[ \\frac{10\\times10^9}{8} = 1.25\\times10^9\\text{ bytes} \]

Therefore:

\[ \\boxed{1.25\\text{ GB}} \]

---

### (d) Reduction factors

FP32 → INT8:

\[ \\frac{32}{8}=4 \]

Therefore:

\[ \\boxed{4\\times} \]

FP32 → INT4:

\[ \\frac{32}{4}=8 \]

Therefore:

\[ \\boxed{8\\times} \]

---

### (e) Why real memory can be larger

The calculation considers only the raw weights.

Real systems can additionally require:

- quantisation scales;
- zero points/metadata;
- activations;
- buffers;
- framework overhead;
- temporary tensors;
- other model components.

Therefore:

\[ \\boxed{\\text{theoretical weight storage}\<\\text{real deployment memory}} \]

in general.

---

## Q17. LoRA parameter calculation

Given:

\[ d=4096 \]

and square matrices.

Dense parameters:

\[ d^2=4096^2 \]

\[ =16,777,216 \]

So:

\[ \\boxed{16,777,216} \]

---

### (b) (r=8)

LoRA requires:

\[ 2dr \]

\[ =2(4096)(8) \]

\[ =65,536 \]

Therefore:

\[ \\boxed{65,536} \]

---

### (c) (r=32)

\[ 2(4096)(32) \]

\[ =262,144 \]

Therefore:

\[ \\boxed{262,144} \]

---

### (d) Percentage for rank 8

\[ \\frac{65,536}{16,777,216}\\times100 \]

Since:

\[ \\frac{2r}{d} = \\frac{16}{4096} = 0.00390625 \]

therefore:

\[ 0.00390625\\times100 = 0.390625% \]

Thus:

\[ \\boxed{\\approx0.391%} \]

of the dense matrix.

---

### (e) (r=8) versus (r=32)

Rank 32 has:

\[ \\frac{32}{8}=4 \]

times as many LoRA parameters.

It therefore provides greater adaptation capacity, but also increases:

- trainable parameters;
- gradient memory;
- optimizer state;
- adapter computation.

Rank 8 is more parameter-efficient but may have insufficient capacity for the target task.

Therefore:

\[ \\boxed{r\\uparrow\\Rightarrow\\text{capacity}\\uparrow,\\quad\\text{cost}\\uparrow} \]

The best rank must be determined empirically.

---

## Q18. Dense fine-tuning versus LoRA

### (a) Dense parameters

For:

\[ W\\in\\mathbb{R}^{d\\times d} \]

the full update contains:

\[ \\boxed{d^2} \]

trainable parameters.

---

### (b) LoRA parameters

We have:

\[ A\\in\\mathbb{R}^{r\\times d} \]

so:

\[ N\_A=rd \]

and:

\[ B\\in\\mathbb{R}^{d\\times r} \]

so:

\[ N\_B=dr \]

Therefore:

\[ \\boxed{N\_{LoRA}=2dr} \]

---

### (c) Ratio

\[ \\frac{N_{LoRA}}{N_{dense}} = \\frac{2dr}{d^2} \]

Cancel (d):

\[ \\boxed{\\frac{2r}{d}} \]

---

### (d) Why the ratio becomes small

If:

\[ r\\ll d \]

then:

\[ 2r\\ll d \]

and therefore:

\[ 2dr\\ll d^2 \]

For example, if (r=8) and (d=4096):

\[ \\frac{2r}{d} = \\frac{16}{4096} \\approx0.0039 \]

Only about 0.39% of the full matrix's parameter count is trainable through the LoRA update.

---

### (e) Why this does not mean (W) is low rank

LoRA leaves:

\[ W \]

unchanged.

The low-rank assumption applies to:

\[ \\boxed{\\Delta W} \]

where:

\[ \\Delta W\\approx BA \]

Therefore:

\[ \\boxed{\\text{LoRA does not require }W\\text{ itself to be low rank}} \]

This is one of the most important distinctions in the course.

---

# Section D — Applied Scenarios

## Q19. Deployment scenario

### (a) Techniques to prioritise

The most appropriate techniques are:

- quantisation;
- pruning.

The problem concerns deployment:

- memory;
- latency;
- redundant parameters.

Quantisation reduces numerical precision/storage.

Pruning reduces the number of parameters.

Layer freezing and LoRA are primarily adaptation/training techniques rather than direct solutions to the deployment-compression problem.

---

### (b) Technique primarily intended for adaptation

\[ \\boxed{\\text{LoRA}} \]

LoRA is a PEFT method.

It freezes the pretrained model and learns a task-specific low-rank update.

---

### (c) Why 90% pruning does not imply 10× speedup

Suppose 90% of weights become zero.

This means:

\[ \\boxed{90%\\text{ parameter sparsity}} \]

but the hardware/software must actually exploit those zeros.

If the implementation still performs dense matrix operations, the zero values may not provide proportional computational savings.

Therefore:

\[ \\boxed{\\text{sparsity}\\neq\\text{automatic speedup}} \]

Structured sparsity or suitable sparse hardware can make the reduction more useful.

---

### (d) Why INT8 is attractive

The hardware has strong INT8 support.

Therefore the theoretical memory reduction can potentially translate into practical computational benefits.

FP32 → INT8 gives:

\[ \\boxed{4\\times} \]

less raw weight storage.

Because the hardware efficiently supports INT8 arithmetic, this scenario is particularly favourable.

---

## Q20. Twenty-task fine-tuning scenario

A strong solution would be:

\[ \\boxed{\\text{one shared pretrained base}+\\text{20 task-specific LoRA adapters}} \]

### Step 1 — Freeze the base

Keep:

\[ W \]

frozen.

This avoids updating the huge pretrained parameter set.

### Step 2 — Train LoRA adapters

For each task (i):

\[ W_i=W+B_iA\_i \]

Only (A_i,B_i) need to be trained.

### Step 3 — Optimizer savings

The optimizer tracks only the trainable parameters.

Therefore each task requires much less optimizer state than full fine-tuning.

### Step 4 — Storage

Instead of storing:

\[ 20\\times\\text{full model} \]

store:

\[ \\boxed{\\text{one base model}+20\\text{ small adapters}} \]

This is particularly attractive when the base model is shared.

### Step 5 — Inference

After training, the adapter can be merged:

\[ \\boxed{W_i'=W+B_iA\_i} \]

The resulting model can then operate as an ordinary layer without a separate LoRA branch.

### Strong overall answer

A highly effective solution would combine:

\[ \\boxed{\\text{quantised shared base}+\\text{LoRA per task}} \]

if the accuracy requirements tolerate the quantisation.

---

# Key lessons from Mock 1

The highest-priority concepts demonstrated by this paper are:

\[ \\boxed{\\text{Quantisation}\\rightarrow\\text{fewer bits}} \]

\[ \\boxed{\\text{Pruning}\\rightarrow\\text{fewer parameters}} \]

\[ \\boxed{\\text{PEFT}\\rightarrow\\text{fewer trainable parameters}} \]

\[ \\boxed{\\text{LoRA}\\rightarrow\\Delta W\\approx BA} \]

and especially:

\[ \\boxed{\\text{SVD compresses }W,\\quad\\text{LoRA compresses }\\Delta W} \] :::

# Mock Exam 2 — Full Solutions

:::writing{variant="document" id="30856" title="Advanced Machine Learning — Mock Exam 2 Full Solutions"}

# Advanced Machine Learning

## Mock Examination 2 — Full Model Solutions

**Total: 100 marks**

---

# Section A — Conceptual MCQs

### Q1

**Answer: B — The model may be overfitting.**

A large gap between training and validation performance can indicate that the model has learned the training data too specifically.

The conclusion should be phrased as "may be overfitting" rather than "definitely overfitting", because other factors can also contribute.

---

### Q2

**Answer: C — Increasing suitable layer width/depth.**

Increasing capacity can allow the network to represent more complex functions.

The other choices generally reduce capacity.

---

### Q3

**Answer: B — Quantisation introduces numerical approximation.**

Quantisation maps high-precision values to a smaller set of representable values:

\[ w\\rightarrow Q(w) \]

This can introduce an error:

\[ Q(w)-w \]

which can affect predictions.

---

### Q4

**Answer: B — It can be applied after training without full quantisation-aware retraining.**

PTQ means:

\[ \\boxed{\\text{Train}\\rightarrow\\text{quantise}} \]

It is attractive because it can be considerably simpler than retraining with quantisation effects incorporated.

---

### Q5

**Answer: A — Structured pruning.**

Structured pruning removes regular computational structures such as channels, filters or neurons.

---

### Q6

**Answer: C — It generally does not receive gradient-based updates.**

A frozen parameter can still be used during the forward pass.

It simply does not participate in the normal parameter-update process.

---

### Q7

**Answer: C — 0.**

Because:

\[ B=0 \]

therefore:

\[ BA=0 \]

regardless of (A).

This makes the initial LoRA update zero.

---

### Q8

**Answer: C — Increasing LoRA rank increases adapter capacity and cost, but the accuracy effect is empirical.**

Higher rank gives more parameters and therefore potentially greater adaptation capacity.

However:

\[ r\\uparrow\\not\\Rightarrow\\text{accuracy necessarily increases} \]

The relationship must be measured experimentally.

---

# Section B — Explain the Claim

## Q9

> "Our model has 80% fewer trainable parameters after freezing layers, therefore inference is 80% cheaper."

### Answer: Incorrect.

Freezing parameters reduces the number of parameters that are updated during training.

It can reduce:

- gradient storage;
- optimizer state;
- backpropagation/update work.

However, frozen layers still participate in the forward pass.

Therefore their computations still occur during inference.

The correct conclusion is:

\[ \\boxed{\\text{freezing can reduce training cost without proportionally reducing inference cost}} \]

If the frozen layers are actually removed or compressed, that is a different operation.

---

## Q10

> "We pruned 95% of the model's weights, so our application will definitely run approximately 20 times faster."

### Answer: Incorrect.

95% sparsity means only 5% of the original weights remain nonzero.

The theoretical number of nonzero parameters is:

\[ 0.05N \]

However, speedup depends on whether the hardware and software can exploit the sparsity.

If arbitrary sparse matrices are represented inefficiently or processed using dense operations, the theoretical parameter reduction may produce little practical speedup.

Structured pruning can be more hardware-friendly.

Therefore:

\[ \\boxed{\\text{95% sparsity does not guarantee 20× speedup}} \]

---

## Q11

> "LoRA works because the pretrained weight matrix is low rank."

### Answer: Incorrect.

LoRA does **not** require:

\[ W\\approx W\_{low-rank} \]

Instead:

\[ W'=W+\\Delta W \]

and LoRA assumes:

\[ \\boxed{\\Delta W\\approx BA} \]

The pretrained matrix (W) can remain full rank.

The low-dimensional assumption concerns the **task-specific update**, not necessarily the original representation.

---

## Q12

> "Our LoRA checkpoint is almost the same size as the dense model, so LoRA has failed to reduce parameter efficiency."

### Answer: Incorrect/incomplete.

There is an important distinction between:

1. total model storage;
2. trainable parameter count;
3. optimizer state.

A complete LoRA state dict can contain the frozen base weights:

\[ W \]

as well as:

\[ A,B \]

Therefore the complete checkpoint can still be large.

However, LoRA can dramatically reduce:

- trainable parameters;
- gradients;
- optimizer state;
- task-specific adapter storage.

Thus a large complete state dict does not imply that LoRA failed at parameter-efficient fine-tuning.

---

# Section C — Technical Calculations

## Q13. Parameter-efficiency analysis

We have:

\[ W\_1:2048\\times2048 \]

\[ W\_2:4096\\times4096 \]

\[ W\_3:1024\\times1024 \]

---

### (a) Dense parameter count

For (W\_1):

\[ 2048^2=4,194,304 \]

For (W\_2):

\[ 4096^2=16,777,216 \]

For (W\_3):

\[ 1024^2=1,048,576 \]

Total:

\[ 4,194,304+16,777,216+1,048,576 \]

\[ =\\boxed{22,020,096} \]

---

### (b) LoRA with (r=16)

For a square matrix:

\[ N\_{LoRA}=2dr \]

#### (W\_1)

\[ 2(2048)(16)=65,536 \]

#### (W\_2)

\[ 2(4096)(16)=131,072 \]

#### (W\_3)

\[ 2(1024)(16)=32,768 \]

Total:

\[ 65,536+131,072+32,768 \]

\[ =\\boxed{229,376} \]

---

### (c) Reduction factor

\[ \\frac{22,020,096}{229,376} \\approx95.99 \]

Therefore approximately:

\[ \\boxed{96\\times} \]

fewer trainable parameters.

Equivalently, LoRA uses approximately:

\[ \\frac{229,376}{22,020,096}\\times100 \\approx1.04% \]

of the dense parameter count.

---

### (d) Why the largest matrix benefits most

For a square matrix:

Dense:

\[ d^2 \]

LoRA:

\[ 2dr \]

The absolute difference is:

\[ d^2-2dr \]

As (d) grows, the dense parameter count grows quadratically:

\[ O(d^2) \]

whereas the LoRA parameter count grows linearly with (d) for fixed (r):

\[ O(dr) \]

Therefore the largest matrix obtains the largest absolute saving.

---

## Q14. Quantisation and combined compression

Given:

\[ N=8\\times10^9 \]

parameters.

### (a) FP32

\[ 8\\times10^9\\times32 = 256\\times10^9\\text{ bits} \]

Divide by 8:

\[ 32\\times10^9\\text{ bytes} \]

Therefore:

\[ \\boxed{32\\text{ GB}} \]

---

### (b) INT8

\[ 8\\times10^9\\times8 = 64\\times10^9\\text{ bits} \]

\[ \\frac{64\\times10^9}{8} = 8\\times10^9 \]

Therefore:

\[ \\boxed{8\\text{ GB}} \]

---

### (c) INT4

\[ 8\\times10^9\\times4 = 32\\times10^9\\text{ bits} \]

\[ \\frac{32\\times10^9}{8} = 4\\times10^9 \]

Therefore:

\[ \\boxed{4\\text{ GB}} \]

---

### (d) 75% pruning + INT4

75% pruning leaves:

\[ 25% \]

of the parameters.

Remaining parameter count:

\[ 8\\times10^9\\times0.25 = 2\\times10^9 \]

At INT4:

\[ 2\\times10^9\\times4 = 8\\times10^9\\text{ bits} \]

Convert to bytes:

\[ \\frac{8\\times10^9}{8} = 1\\times10^9 \]

Therefore:

\[ \\boxed{1\\text{ GB}} \]

of theoretical remaining weight storage.

Relative to the original 32 GB FP32 representation:

\[ \\frac{32}{1}=32 \]

So theoretically this combination gives:

\[ \\boxed{32\\times} \]

less raw weight storage.

---

### (e) Why this does not predict speed

Storage and computation are different.

Actual latency depends on:

- sparse representation;
- hardware support;
- INT4 kernel efficiency;
- memory bandwidth;
- matrix dimensions;
- overhead;
- whether sparsity is structured;
- software implementation.

Therefore:

\[ \\boxed{\\text{32× storage reduction}\\neq\\text{32× inference speedup}} \]

---

## Q15. LoRA rank sweep

| Rank | Trainable parameters | Validation accuracy |
| --- | --- | --- |
| 2 | 40,000 | 81.1% |
| 4 | 80,000 | 82.3% |
| 8 | 160,000 | 83.0% |
| 16 | 320,000 | 83.2% |
| 32 | 640,000 | 83.3% |

### (a)

Highest validation accuracy:

\[ \\boxed{r=32} \]

with:

\[ 83.3% \]

---

### (b)

If parameter efficiency is strongly prioritised, (r=8) or (r=16) may be preferable.

For example:

\[ r=16 \]

gets 83.2%, only 0.1 percentage points below rank 32, while using half the parameters.

This is an engineering trade-off.

---

### (c)

Choosing the largest rank solely because it gives the highest accuracy ignores:

- memory;
- training cost;
- optimizer state;
- adapter computation;
- deployment/storage requirements.

The improvement from (r=16) to (r=32) is only:

\[ 83.3-83.2=0.1 \]

percentage points.

But the trainable parameters double:

\[ 320,000\\rightarrow640,000 \]

Therefore rank 16 may provide a better accuracy/efficiency trade-off.

---

### (d)

Useful additional measurements include:

- training time;
- peak GPU memory;
- optimizer-state size;
- complete checkpoint size;
- adapter-only checkpoint size;
- inference latency;
- inference throughput;
- repeated-seed variance.

The best model is not necessarily the one with the highest validation accuracy alone.

---

# Section D — Experimental Design

## Q16. Fair rank experiment

A good experimental design would be:

### 1\. Use identical pretrained starting weights

Every rank should begin from the same pretrained model.

Otherwise differences may come from different starting points.

---

### 2\. Use the same dataset split

The train/validation/test split should be identical for all ranks.

---

### 3\. Keep hyperparameters controlled

Keep constant:

- learning rate;
- optimiser;
- batch size;
- epochs;
- regularisation;
- data preprocessing;
- evaluation procedure.

The main experimental variable should be:

\[ \\boxed{r} \]

---

### 4\. Use controlled random seeds

Using the same seed improves comparability and reproducibility.

Ideally, a robust experiment would repeat configurations over multiple seeds and report variation.

---

### 5\. Start each rank from a fresh model

Do not train:

\[ r=2\\rightarrow r=4\\rightarrow r=8 \]

using the same already-adapted model.

Each experiment should independently start from the same pretrained checkpoint.

---

### 6\. Evaluate on validation data during selection

Use validation accuracy to choose:

\[ r^\* \]

Do not repeatedly optimise the choice using the test set.

---

### 7\. Measure parameter counts

Record:

- total parameters;
- trainable parameters;
- trainable fraction.

---

### 8\. Measure memory

Record appropriate quantities such as:

- allocated memory;
- reserved memory;
- peak memory.

Memory measurements should be performed from a controlled/clean state.

---

### 9\. Measure efficiency

Record:

- training time;
- time per batch;
- optimizer-state size;
- checkpoint size.

---

### 10\. Final test evaluation

After selecting the rank based on validation performance and efficiency, evaluate the final chosen model on the held-out test set.

---

## Q17. Diagnosing unexpected LoRA behaviour

The model performs much worse immediately before training.

At least three possible causes are:

### 1\. Incorrect LoRA initialization

Normally:

\[ B=0 \]

so:

\[ BA=0 \]

initially.

Therefore:

\[ W'=W \]

and the initial model should reproduce the pretrained model's behaviour, assuming the implementation is otherwise correct.

If (B) is randomly initialized rather than zero, the adapter may immediately perturb the pretrained model.

---

### 2\. Incorrect forward equation

The implementation should effectively calculate:

\[ Wx+BAx+b \]

or the equivalent expression under the tensor orientation used.

A transpose or multiplication-order error can produce incorrect outputs.

---

### 3\. Incorrect freezing

The base weight matrix should generally have:

\[ requires\_grad=False \]

If the wrong parameters are frozen or trainable, the experiment may not represent LoRA correctly.

---

### 4\. Incorrect loading of pretrained weights

If the model did not correctly load the pretrained checkpoint, the comparison is invalid.

---

### 5\. Incorrect classifier adaptation

If the target task has a different number of classes, the classifier needs to be appropriately adapted.

---

### 6\. Scaling implementation

Incorrect LoRA scaling, such as incorrectly applying:

\[ \\frac{\\alpha}{r} \]

can alter the magnitude of the update.

---

### Strongest immediate diagnostic

Compare the pretrained model's output and LoRA model's output **before training**.

They should be essentially identical if:

\[ BA=0 \]

and the architecture is implemented correctly.

---

# Key lessons from Mock 2

Mock 2 is particularly testing whether you understand that:

\[ \\boxed{\\text{parameter efficiency}\\neq\\text{automatic runtime efficiency}} \]

and:

\[ \\boxed{\\text{complete checkpoint size}\\neq\\text{trainable parameter count}} \]

and:

\[ \\boxed{r\\uparrow\\Rightarrow\\text{capacity/cost}\\uparrow} \]

but:

\[ \\boxed{r\\uparrow\\not\\Rightarrow\\text{accuracy necessarily}\\uparrow} \]

The experimental-design questions are especially important because they test whether you understand **how to establish that a result is actually meaningful**. :::

# Mock Exam 3 — Full Solutions

:::writing{variant="document" id="57294" title="Advanced Machine Learning — Mock Exam 3 Full Solutions"}

# Advanced Machine Learning

## Mock Examination 3 — Full Model Solutions

**Total: 100 marks**

---

# Section A — Rapid-Fire Questions

## Q1. Why can an overparameterised neural network still generalise well?

Overparameterisation means the model has many more parameters than are strictly necessary to represent the training data.

However, the model may contain substantial redundancy and the optimisation process may favour solutions that generalise well.

This motivates ideas such as:

- pruning;
- intrinsic dimensionality;
- low-dimensional adaptation.

A useful conceptual statement is:

\[ \\boxed{\\text{raw parameter dimensionality can greatly exceed effective dimensionality}} \]

---

## Q2. Purpose of validation set

The validation set is used for model selection and hyperparameter tuning without directly updating the model parameters on those examples.

Examples include selecting:

- learning rate;
- architecture;
- regularisation;
- LoRA rank.

---

## Q3. Quantisation versus pruning

Quantisation reduces the **precision/bits used to represent parameters**.

Pruning removes **parameters or computational structures**.

Therefore:

\[ \\boxed{\\text{quantisation}\\rightarrow\\text{fewer bits}} \]

\[ \\boxed{\\text{pruning}\\rightarrow\\text{fewer parameters}} \]

---

## Q4. Why INT4 reduces memory

INT4 uses:

\[ 4\\text{ bits/parameter} \]

instead of:

\[ 32\\text{ bits/parameter} \]

for FP32.

Therefore the theoretical storage ratio is:

\[ \\frac{32}{4}=8 \]

so:

\[ \\boxed{8\\times\\text{ less raw weight storage}} \]

---

## Q5. Why PTQ may reduce accuracy

PTQ introduces quantisation after training.

The trained model was not necessarily optimised to compensate for the resulting numerical approximation.

Thus:

\[ w\\rightarrow Q(w) \]

can introduce quantisation error and cause accuracy degradation.

---

## Q6. Magnitude versus similarity pruning

Magnitude pruning examines individual weights:

\[ |w|\\text{ small}\\Rightarrow\\text{candidate for removal} \]

Similarity pruning examines redundancy between features/neurons.

For example:

\[ h_1(x)\\approx h_2(x) \]

may indicate that one feature is redundant.

---

## Q7. Structured pruning and hardware

Structured pruning removes regular structures such as channels or filters.

This creates smaller, regular tensor operations that standard hardware can often process more efficiently.

Unstructured sparsity can create irregular memory-access and sparse-computation patterns.

Therefore:

\[ \\boxed{\\text{structured sparsity is often easier to exploit efficiently}} \]

---

## Q8. Central LoRA equation

\[ \\boxed{W'=W+BA} \]

where:

\[ W=\\text{frozen pretrained weight} \]

and:

\[ BA\\approx\\Delta W \]

is the trainable low-rank update.

---

## Q9. Why LoRA reduces optimizer-state memory

Only trainable parameters require optimiser state.

LoRA freezes the large pretrained matrix and trains only (A), (B), and any other explicitly trainable parameters.

Thus:

\[ \\boxed{\\text{fewer trainable parameters}\\rightarrow\\text{less optimizer state}} \]

---

## Q10. SVD versus LoRA

SVD compression approximates the **existing weight matrix**:

\[ \\boxed{W\\approx W\_{low-rank}} \]

LoRA leaves (W) intact and approximates the **fine-tuning update**:

\[ \\boxed{\\Delta W\\approx BA} \]

This distinction is fundamental.

---

# Section B — Integrated Short Answers

## Q11. Compression hierarchy

We have:

\[ \\text{FP32 dense} \\rightarrow \\text{INT8 dense} \\rightarrow \\text{INT8 pruned} \]

### FP32 → INT8

This exploits **excess numerical precision**.

Instead of 32 bits per parameter:

\[ 32\\rightarrow8 \]

Therefore raw weight storage falls by approximately:

\[ \\boxed{4\\times} \]

---

### INT8 dense → INT8 pruned

This exploits **parameter redundancy**.

Some weights or structures are removed.

The model now has:

\[ \\boxed{\\text{fewer parameters}} \]

while the remaining parameters retain INT8 representation.

---

### Why latency does not scale proportionally

Memory reduction does not automatically equal computation reduction.

Latency depends on:

- hardware;
- kernels;
- memory bandwidth;
- sparse support;
- data movement;
- implementation overhead;
- matrix dimensions.

For pruning specifically:

\[ \\boxed{\\text{sparsity must be efficiently exploited}} \]

For quantisation:

\[ \\boxed{\\text{hardware must efficiently support low-precision computation}} \]

Therefore a 4× storage reduction need not produce a 4× latency reduction.

---

## Q12. Fine-tuning hierarchy

### 1\. Full fine-tuning

The pretrained parameters are updated.

\[ W'=W+\\Delta W \]

where (\\Delta W) is full-sized.

**Advantage:** maximum unrestricted adaptation.

**Disadvantage:** expensive in gradients, optimizer state and training memory.

---

### 2\. Layer freezing

Some layers remain frozen while selected layers are trained.

**Advantage:** fewer trainable parameters.

**Disadvantage:** frozen layers still execute their forward computation.

---

### 3\. Adapters

Small trainable modules are inserted into or alongside frozen layers.

Conceptually:

\[ y=x+\\text{adapter}(x) \]

**Advantage:** small task-specific parameter set.

**Limitation:** adapters add additional computations unless otherwise optimised.

---

### 4\. LoRA

The original weights remain frozen and the update is:

\[ \\Delta W\\approx BA \]

**Advantage:** extremely small trainable update and reduced optimizer state.

**Limitation:** adaptation capacity depends on rank and adapter computation is present unless merged.

---

# Q13. Low-dimensionality

### Overparameterisation

A network can have many more parameters than are apparently necessary to solve a task.

Therefore:

\[ \\text{raw parameter dimension} \\gg \\text{effective task dimension} \]

---

### Redundancy

If many parameters/features contain overlapping information, removing or compressing some of them may not cause catastrophic performance loss.

This motivates pruning.

---

### Intrinsic/effective dimensionality

The useful solutions or adaptations may occupy a much lower-dimensional region than the full parameter space.

Conceptually:

\[ \\boxed{\\text{effective dimension}\\ll\\text{parameter dimension}} \]

---

### Low-rank representations

A matrix can sometimes be approximated using a small number of directions:

\[ W\\approx UV \]

where:

\[ r\\ll d \]

This reduces parameter count.

---

### LoRA

LoRA applies this idea specifically to the **adaptation**:

\[ \\Delta W\\approx BA \]

Therefore the model does not need a completely unrestricted full-dimensional update.

---

### Why this does not mean every (W) should be low rank

The fact that useful task adaptations can be low dimensional does not imply that the pretrained weight matrix itself can always be aggressively compressed without loss.

Directly replacing:

\[ W \]

with:

\[ W\_{low-rank} \]

can remove important information.

LoRA instead preserves (W) and learns a low-rank correction.

Therefore:

\[ \\boxed{ \\text{low-dimensional adaptation} \\neq \\text{low-rank pretrained weights} } \]

---

# Q14. "10× fewer parameters means 10× faster and 10× less memory"

This statement is **too strong**.

## Trainable parameters

This part can be approximately true.

LoRA can reduce trainable parameters dramatically:

\[ d^2\\rightarrow2dr \]

when:

\[ r\\ll d \]

---

## Gradients

Because most pretrained weights are frozen, fewer gradient tensors need to be stored.

Therefore gradient memory can be significantly reduced.

---

## Optimizer state

This is one of the clearest savings.

The optimizer only tracks trainable parameters.

Therefore:

\[ \\boxed{\\text{LoRA}\\rightarrow\\text{much smaller optimizer state}} \]

---

## Forward computation

The frozen model still performs its normal forward computation.

LoRA also adds:

\[ BAx \]

to the forward pass.

Therefore LoRA does not reduce the base model's entire forward computation.

It can even add computation relative to the frozen base.

---

## Complete model storage

The complete state may contain:

\[ W+A+B \]

Therefore it does not automatically become 10× smaller.

If only adapters are stored, then task-specific storage can be dramatically smaller.

---

## Wall-clock training time

Wall-clock speed depends on:

- hardware;
- batch size;
- implementation;
- memory bandwidth;
- base-model forward cost;
- adapter computation.

Therefore:

\[ \\boxed{\\text{10× fewer trainable parameters}\\not\\Rightarrow10×\\text{ faster training}} \]

---

# Section C — Multi-Step Problems

## Q15. Complete LoRA derivation

Given:

\[ d\_{in}=3072 \]

\[ d\_{out}=4096 \]

\[ r=16 \]

---

### (a) Original weight parameters

\[ N_W=d_{out}d\_{in} \]

\[ =4096\\times3072 \]

Calculate:

\[ 4096\\times3000=12,288,000 \]

\[ 4096\\times72=294,912 \]

Therefore:

\[ \\boxed{12,582,912} \]

parameters.

---

### (b) LoRA parameters

The general formula is:

\[ N_{LoRA}=r(d_{in}+d\_{out}) \]

Therefore:

\[ 16(3072+4096) \]

\[ =16(7168) \]

\[ =\\boxed{114,688} \]

---

### (c) Fraction of original parameters

\[ \\frac{114,688}{12,582,912} \]

Approximately:

\[ 0.0091146 \]

or:

\[ \\boxed{0.911%} \]

Thus LoRA trains less than 1% of the original matrix's parameter count.

---

### (d) FP32 memory for LoRA parameters

Each parameter:

\[ 32\\text{ bits}=4\\text{ bytes} \]

Therefore:

\[ 114,688\\times4 = 458,752\\text{ bytes} \]

Thus approximately:

\[ \\boxed{0.459\\text{ MB}} \]

using decimal MB.

---

### (e) Why this is not total training memory

Training memory also includes:

- the frozen pretrained model;
- activations;
- gradients for trainable parameters;
- optimizer state;
- temporary buffers;
- framework overhead.

Therefore:

\[ \\boxed{\\text{LoRA parameter memory}\\neq\\text{total GPU training memory}} \]

---

### (f) Merging

After training:

\[ \\boxed{W\_{merged}=W+BA} \]

The two components can be combined into one weight matrix.

The resulting layer can then perform the ordinary computation:

\[ y=W\_{merged}x+b \]

without a separate LoRA branch.

---

# Q16. Memory-efficient adaptation system

We have a 100-billion-parameter pretrained model and four target tasks.

### (a) Full FT or PEFT?

Use:

\[ \\boxed{\\text{PEFT}} \]

because full fine-tuning requires updating an enormous number of parameters.

This creates substantial:

- gradient memory;
- optimizer-state memory;
- training computation;
- task-specific storage.

---

### (b) Quantised base + LoRA?

Yes.

A QLoRA-style strategy is appropriate:

\[ \\boxed{\\text{quantised frozen base}+\\text{LoRA}} \]

The quantised base reduces the memory required to store the model during adaptation.

LoRA reduces the number of trainable parameters and associated optimizer state.

---

### (c) Storage organisation

Use:

\[ \\boxed{\\text{one shared base model}+4\\text{ task-specific adapters}} \]

rather than four complete model copies.

---

### (d) What is stored for each task?

For task (i), store its learned LoRA matrices:

\[ A_i,B_i \]

and any other task-specific trainable components, such as an adapted classifier if applicable.

---

### (e) Deployment

After training:

\[ W_i=W+ B_iA\_i \]

The adapter can be merged into the base weights for that task.

This removes the need for a separate adapter branch during inference.

---

### (f) Rank trade-off

Higher (r):

\[ \\text{capacity}\\uparrow \]

but also:

\[ \\text{parameters}\\uparrow \]

\[ \\text{memory}\\uparrow \]

\[ \\text{computation}\\uparrow \]

Lower (r) is more efficient but may underfit the target task.

The best rank should therefore be selected experimentally using validation performance and efficiency measurements.

---

# Q17. Compression decision problem

Strategies:

| Strategy | Precision | Sparsity | Adaptation | Hardware |
| --- | --- | --- | --- | --- |
| A | FP32 | Dense | Full FT | General GPU |
| B | INT8 | Dense | Frozen | INT8 accelerator |
| C | INT4 | 75% sparse | LoRA | Sparse-support GPU |
| D | FP16 | Structured pruning | LoRA | General GPU |

Requirements:

- limited memory;
- minimal training cost;
- good accuracy;
- practical speed.

---

### (a) First strategy to investigate

A strong first candidate is:

\[ \\boxed{\\text{Strategy C}} \]

because it combines:

- INT4;
- substantial sparsity;
- LoRA;
- sparse hardware support.

It directly addresses both memory and adaptation cost.

However, this should be an **experimental hypothesis**, not an automatic conclusion.

---

### (b) Why attractive?

INT4:

\[ \\boxed{\\text{very low memory per parameter}} \]

75% sparsity:

\[ \\boxed{\\text{fewer active parameters}} \]

LoRA:

\[ \\boxed{\\text{few trainable parameters}} \]

Sparse-support GPU:

\[ \\boxed{\\text{potential to exploit sparsity}} \]

Thus all four requirements are potentially addressed.

---

### (c) Required experimental evidence

We would need to measure:

- target-task validation/test accuracy;
- peak GPU memory;
- training time;
- inference latency;
- throughput;
- actual storage;
- adapter size;
- perhaps energy consumption if relevant.

Compare C against suitable baselines, particularly B and D.

---

### (d) Why a smaller model might not have the best latency

### Reason 1: Hardware utilisation

A theoretically smaller representation may not map efficiently to hardware.

### Reason 2: Sparse overhead

Sparse operations can introduce indexing and irregular-memory overhead.

Therefore:

\[ \\boxed{\\text{fewer operations}\\neq\\text{necessarily lower wall-clock time}} \]

Other factors include:

- memory bandwidth;
- kernel implementation;
- data movement;
- launch overhead;
- batch size.

---

### (e) If INT4 causes unacceptable accuracy degradation

Possible alternatives include:

- INT8;
- FP16;
- QAT;
- less aggressive quantisation;
- less aggressive pruning;
- structured rather than arbitrary pruning;
- LoRA with a different rank;
- combinations of these.

If accuracy is the main problem with PTQ-style quantisation, QAT may be investigated.

---

# Section D — Grand Synthesis

## Q18. End-to-end system design

A strong answer would begin with:

\[ \\boxed{\\text{shared pretrained base}+\\text{parameter-efficient task adaptations}} \]

and then select compression according to empirical constraints.

---

## A. Starting model

Start from the pretrained model rather than training each target model from scratch.

The pretrained model already contains useful representations:

\[ \\boxed{\\text{pretrained knowledge}\\rightarrow\\text{target-task adaptation}} \]

This is transfer learning.

---

## B. Model compression

If memory is constrained, investigate quantisation.

For example:

\[ FP32\\rightarrow INT8 \]

or potentially:

\[ FP32\\rightarrow INT4 \]

if accuracy remains acceptable.

Pruning can also be investigated if the model contains substantial redundancy.

However, the decision should be empirical.

A strong solution would test:

\[ \\boxed{\\text{quantisation}} \]

and potentially:

\[ \\boxed{\\text{quantisation + structured pruning}} \]

rather than assuming the most aggressive compression is automatically best.

---

## C. Adaptation

Use PEFT, preferably LoRA for the scenario described.

Freeze:

\[ W \]

and learn:

\[ \\Delta W\\approx BA \]

Thus:

\[ \\boxed{W'=W+BA} \]

This dramatically reduces trainable parameters.

---

## D. Rank selection

Run a rank sweep, for example:

\[ r\\in{2,4,8,16,32} \]

For every rank:

- start from the same pretrained checkpoint;
- use the same data split;
- keep training conditions controlled;
- measure validation accuracy;
- measure trainable parameters;
- measure memory;
- measure training time.

Choose:

\[ \\boxed{r^\*} \]

based on the desired accuracy/efficiency trade-off rather than simply choosing the largest rank.

---

## E. Memory

With LoRA:

### Gradients

Only trainable parameters require gradients, so gradient memory decreases.

### Optimizer state

Only trainable parameters need optimizer state.

Therefore:

\[ \\boxed{\\text{optimizer memory}\\downarrow} \]

### Model weights

The frozen base model still has to be stored.

Therefore LoRA does not by itself eliminate the memory cost of the pretrained model.

Quantisation can address this separately.

---

## F. Storage

There are two approaches.

### Complete adapted model

Store:

\[ W+BA \]

or the corresponding complete model parameters.

This can be large.

### Shared base + adapters

Store:

\[ \\boxed{\\text{one base model}+{A_i,B_i}\_{i=1}^N} \]

This is much more attractive when many tasks share the same base.

The task-specific components are tiny relative to the complete model.

---

## G. Inference

After training, merge:

\[ \\boxed{W\_{merged}=W+BA} \]

The inference layer can then operate normally:

\[ y=W\_{merged}x+b \]

This removes the need for a separate LoRA branch.

Thus LoRA can provide training/storage benefits while avoiding additional adapter architecture during the final deployment.

---

## H. Experimental validation

The final system should be compared with a baseline.

At minimum measure:

### Accuracy

\[ \\text{validation/test performance} \]

### Training efficiency

\[ \\text{training time} \]

### Memory

\[ \\text{peak GPU memory} \]

### Parameter efficiency

\[ \\text{trainable parameters} \]

### Storage

\[ \\text{complete model size} \]

and:

\[ \\text{adapter-only size} \]

### Inference

\[ \\text{latency} \]

and potentially:

\[ \\text{throughput} \]

The final decision should consider the whole trade-off.

---

## Model answer — concise exam version

A very strong exam answer could be compressed to:

> I would retain the pretrained model as a shared base, freeze it and use LoRA for task-specific adaptation. LoRA represents the required update as (\\Delta W\\approx BA), so the number of trainable parameters changes from (d^2) to (2dr) for a square matrix, substantially reducing gradient and optimizer-state memory. If base-model memory is also a constraint, I would investigate quantisation, potentially using a QLoRA-style approach. I would perform a controlled rank sweep and select the rank based on validation accuracy together with memory, parameter count and training-time measurements rather than simply selecting the largest rank. For multiple tasks, I would store one shared base model and separate small LoRA adapters. After training, each adapter can be merged using (W'=W+BA), allowing normal inference without a separate LoRA branch. Finally, I would compare the proposed system against dense fine-tuning using accuracy, training time, peak memory, storage and inference latency. This is necessary because reductions in parameter count or theoretical storage do not automatically translate into proportional real-world speedups.

---

# Final high-priority concepts across all three exams

If you were revising immediately before the exam, I would prioritise these distinctions:

### 1\. Quantisation

\[ \\boxed{\\text{fewer bits per parameter}} \]

FP32 → INT8:

\[ \\boxed{4\\times\\text{ less raw weight storage}} \]

FP32 → INT4:

\[ \\boxed{8\\times\\text{ less raw weight storage}} \]

---

### 2\. Pruning

\[ \\boxed{\\text{remove parameters/structures}} \]

But:

\[ \\boxed{\\text{parameter reduction}\\neq\\text{automatic speedup}} \]

---

### 3\. Structured versus unstructured pruning

\[ \\boxed{\\text{structured}\\rightarrow\\text{regular hardware-friendly structures}} \]

\[ \\boxed{\\text{unstructured}\\rightarrow\\text{arbitrary individual weights}} \]

---

### 4\. PTQ versus QAT

\[ \\boxed{\\text{PTQ: train}\\rightarrow\\text{quantise}} \]

\[ \\boxed{\\text{QAT: train while accounting for quantisation effects}} \]

If PTQ causes unacceptable accuracy loss:

\[ \\boxed{\\text{consider QAT}} \]

---

### 5\. Freezing

\[ \\boxed{\\text{training efficiency}} \]

not automatically:

\[ \\boxed{\\text{inference efficiency}} \]

---

### 6\. LoRA

The equation to know:

\[ \\boxed{W'=W+BA} \]

with:

\[ \\boxed{W\\text{ frozen}} \]

and:

\[ \\boxed{A,B\\text{ trainable}} \]

---

### 7\. LoRA parameter count

General case:

\[ \\boxed{r(d_{in}+d_{out})} \]

Square matrix:

\[ \\boxed{2dr} \]

Dense:

\[ \\boxed{d^2} \]

Ratio:

\[ \\boxed{\\frac{2r}{d}} \]

---

### 8\. LoRA initialization

\[ \\boxed{B=0} \]

therefore:

\[ \\boxed{BA=0} \]

so initially:

\[ \\boxed{W'=W} \]

---

### 9\. SVD versus LoRA

This is probably the **single most important conceptual distinction**:

\[ \\boxed{\\text{SVD: compress }W} \]

versus:

\[ \\boxed{\\text{LoRA: compress }\\Delta W} \]

LoRA does **not** require (W) itself to be low rank.

---

### 10\. State dict versus optimizer state

LoRA may still store:

\[ W+A+B \]

so:

\[ \\boxed{\\text{complete checkpoint need not be dramatically smaller}} \]

But only (A,B) and other trainable parameters require optimizer state:

\[ \\boxed{\\text{optimizer state can be dramatically smaller}} \]

---

### 11\. Rank

\[ r\\uparrow \\Rightarrow \\begin{cases} \\text{adaptation capacity}\\uparrow\\ \\text{trainable parameters}\\uparrow\\ \\text{optimizer state}\\uparrow\\ \\text{adapter computation}\\uparrow \\end{cases} \]

But:

\[ \\boxed{r\\uparrow\\not\\Rightarrow\\text{accuracy necessarily}\\uparrow} \]

---

### 12\. Final synthesis

The entire four-chunk compression/PEFT story can be remembered as:

\[ \\boxed{ \\begin{array}{c} \\text{Quantisation}\\ \\downarrow\\ \\text{reduce precision} \\end{array} } \\qquad \\boxed{ \\begin{array}{c} \\text{Pruning}\\ \\downarrow\\ \\text{remove redundancy} \\end{array} } \]

and for adaptation:

\[ \\boxed{ \\text{PEFT} \\rightarrow \\text{freeze pretrained knowledge} \\rightarrow \\text{learn a small task-specific change} } \]

with LoRA giving:

\[ \\boxed{ W'=W+BA } \]

and the fundamental conceptual distinction:

\[ \\boxed{ \\underbrace{W}_{\\text{pretrained model}} \\neq \\underbrace{\\Delta W}_{\\text{task-specific adaptation}} } \]

That distinction ties together the most demanding questions in all three mock exams. :::
