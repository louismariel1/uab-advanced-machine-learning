# **90-minute / 100-mark format** with a mixture of:

- conceptual MCQs,
- short-answer explanations,
- calculations/derivations,
- code/implementation reasoning,
- scenario-based questions,
- and a higher-level synthesis question.

The difficulty will progressively vary: **Mock 1 = balanced**, **Mock 2 = harder/application-heavy**, **Mock 3 = exam-pressure/high-discrimination**.

# Mock Exam 1 — Balanced

:::writing{variant="document" id="48317" title="Advanced Machine Learning — 90-Minute Mock Exam 1"}

# Advanced Machine Learning

## 90-Minute Mock Examination 1

**Time:** 90 minutes **Total:** 100 marks

### Instructions

- Answer all questions.
- Show your working for calculations.
- For conceptual questions, give concise but technically precise answers.
- When a question asks you to justify an answer, a one-word response is insufficient.
- Assume standard deep-learning notation unless otherwise stated.
- Calculators are permitted.

---

## Section A — Multiple Choice

### 10 questions × 2 marks = 20 marks

### Suggested time: 15 minutes

**Q1.** Which statement best describes the purpose of transfer learning?

A. Training a model exclusively on synthetic data B. Reusing knowledge learned from one task/domain to help another task C. Reducing the number of classes in a dataset D. Replacing all pretrained parameters with random values

---

**Q2.** A model is suffering from overfitting. Which change would most directly increase the model's effective regularisation?

A. Increasing model capacity without changing training B. Removing the validation set C. Increasing an appropriate regularisation mechanism D. Training indefinitely on the same data

---

**Q3.** Why is validation data used during model development?

A. To update the model's weights directly B. To estimate performance during model selection without using the test set for tuning C. To replace the training set D. To guarantee generalisation

---

**Q4.** Which statement about quantisation is correct?

A. It increases the number of bits used per parameter B. It represents parameters using fewer numerical values/bits C. It necessarily removes parameters D. It is identical to pruning

---

**Q5.** A model has 1 billion parameters stored using FP32. Ignoring all overhead, approximately how much memory is required for the weights?

A. 0.125 GB B. 0.5 GB C. 4 GB D. 32 GB

---

**Q6.** Which statement best describes LoRA?

A. It removes all pretrained weights B. It approximates the pretrained weights themselves using SVD C. It freezes pretrained weights and learns a low-rank task-specific update D. It permanently reduces the number of layers in the network

---

**Q7.** If a square weight matrix has dimension (d\\times d), how many trainable parameters does a rank-(r) LoRA update require?

A. (d^2r) B. (2dr) C. (d+r) D. (r^2)

---

**Q8.** Why does freezing 90% of a model's parameters not imply a 90% reduction in inference computation?

A. Frozen parameters disappear during inference B. Frozen layers still perform their forward computations C. Frozen parameters become random D. Inference does not use model parameters

---

**Q9.** Which statement distinguishes structured from unstructured pruning?

A. Structured pruning removes individual arbitrary weights only B. Structured pruning removes regular structures such as channels, filters or neurons C. Unstructured pruning always produces greater hardware speedup D. They are mathematically identical

---

**Q10.** Which statement about LoRA and SVD is most accurate?

A. Both necessarily compress the original pretrained weights B. SVD compresses the existing matrix; LoRA represents the adaptation as a low-rank update C. LoRA requires the original matrix to be low rank D. SVD is a PEFT method by definition

---

# Section B — Short Answer

### 5 questions × 6 marks = 30 marks

### Suggested time: 25 minutes

**Q11. — Model generalisation**

Explain the difference between:

- training performance,
- validation performance,
- and test performance.

Why is repeatedly tuning hyperparameters against the test set problematic?

---

**Q12. — Regularisation**

Explain two different mechanisms that can reduce overfitting in a neural network.

For each mechanism:

1. state what it changes;
2. explain why it may improve generalisation.

---

**Q13. — Quantisation**

Explain the difference between:

- FP32,
- FP16,
- INT8,
- and INT4

in terms of numerical representation and storage.

Then explain why reducing precision can potentially hurt model accuracy.

---

**Q14. — Pruning**

Compare:

- magnitude pruning;
- similarity-based pruning;
- structured pruning;
- unstructured pruning.

Your answer should clearly distinguish **what is removed** and **why practical inference speedups may differ**.

---

**Q15. — PEFT**

Explain why parameter-efficient fine-tuning can reduce training memory even when the complete pretrained model still has to be stored.

Your answer must explicitly discuss:

- gradients;
- optimizer state;
- frozen parameters.

---

# Section C — Calculations and Technical Reasoning

### 3 questions × 10 marks = 30 marks

### Suggested time: 27 minutes

**Q16. — Quantisation calculation**

A model contains 2.5 billion parameters.

Assume the weights are stored without overhead.

### (a)

Calculate the approximate weight storage using FP32.

### (b)

Calculate the approximate weight storage using INT8.

### (c)

Calculate the approximate weight storage using INT4.

### (d)

What are the theoretical storage reduction factors of INT8 and INT4 relative to FP32?

### (e)

Explain why the real deployed model may require more memory than your calculation.

---

**Q17. — LoRA parameter calculation**

A fully connected layer has:

\[ d_{in}=4096,\\qquad d_{out}=4096. \]

### (a)

How many parameters are in the original weight matrix?

### (b)

How many trainable parameters are required for LoRA with:

\[ r=8? \]

### (c)

Repeat for:

\[ r=32. \]

### (d)

Calculate the percentage of the original matrix represented by the rank-8 LoRA update.

### (e)

Explain the trade-off between (r=8) and (r=32).

---

**Q18. — Dense fine-tuning vs LoRA**

Suppose:

\[ W\\in\\mathbb{R}^{d\\times d} \]

and the dense fine-tuning update is:

\[ W'=W+\\Delta W. \]

LoRA instead uses:

\[ W'=W+BA \]

where:

\[ A\\in\\mathbb{R}^{r\\times d}, \\qquad B\\in\\mathbb{R}^{d\\times r}. \]

### (a)

Derive the number of trainable parameters for dense fine-tuning.

### (b)

Derive the number of trainable parameters for LoRA.

### (c)

Derive the ratio:

\[ \\frac{N_{LoRA}}{N_{dense}}. \]

### (d)

Explain why the ratio becomes very small when (r\\ll d).

### (e)

Explain why this does **not** mean the original pretrained matrix (W) is low rank.

---

# Section D — Applied Scenarios

### 2 questions × 10 marks = 20 marks

### Suggested time: 18 minutes

**Q19. — Deployment scenario**

You have a very large neural network that performs accurately, but:

- it does not fit comfortably into device memory;
- latency is important;
- the target hardware has strong INT8 support;
- the model contains substantial parameter redundancy.

You are considering:

1. quantisation;
2. pruning;
3. layer freezing;
4. LoRA.

### (a)

Which technique(s) would you prioritise for the deployment problem?

### (b)

Which technique is primarily intended for cheap **adaptation** rather than direct model compression?

### (c)

Explain why 90% pruning does not automatically imply a 10× inference speedup.

### (d)

Explain why INT8 is particularly attractive in this scenario.

---

**Q20. — Fine-tuning scenario**

A company has one pretrained model and wants to adapt it to 20 different tasks.

They have limited GPU memory and want to avoid maintaining 20 complete copies of the model.

Design an appropriate fine-tuning strategy.

Your answer should discuss:

- freezing;
- LoRA;
- task-specific parameters;
- optimizer state;
- model storage;
- inference/merging.

---

# End of Mock Exam 1

**Recommended time allocation:** Section A: 15 min Section B: 25 min Section C: 27 min Section D: 18 min Review: 5 min

**Total: 90 minutes / 100 marks** :::

# Mock Exam 2 — Application & Reasoning Heavy

:::writing{variant="document" id="72641" title="Advanced Machine Learning — 90-Minute Mock Exam 2"}

# Advanced Machine Learning

## 90-Minute Mock Examination 2

**Time:** 90 minutes **Total:** 100 marks

### Instructions

This paper places greater emphasis on **reasoning and application**. Several questions deliberately contain plausible but incorrect statements. Explain your reasoning clearly.

---

# Section A — Conceptual MCQs

### 8 questions × 2 marks = 16 marks

### Suggested time: 12 minutes

**Q1.** A model achieves 99% training accuracy and 72% validation accuracy. Which conclusion is most justified?

A. The model is necessarily underfitting B. The model may be overfitting C. The training set must contain no useful information D. The validation set should be discarded

---

**Q2.** Which change most directly increases the representational capacity of a neural network?

A. Reducing the number of hidden units B. Reducing the number of trainable parameters C. Increasing suitable layer width/depth D. Quantising FP32 weights to INT4

---

**Q3.** A model is quantised from FP32 to INT8 and its accuracy decreases. Which explanation is most appropriate?

A. Quantisation necessarily removes layers B. Quantisation introduces numerical approximation C. INT8 always changes the dataset D. Quantisation increases parameter count

---

**Q4.** PTQ is attractive primarily because:

A. it always produces higher accuracy than QAT B. it can be applied after training without full quantisation-aware retraining C. it increases numerical precision D. it requires every parameter to be retrained

---

**Q5.** Which technique specifically attempts to remove redundant computational structures such as channels?

A. Structured pruning B. PTQ C. Prefix tuning D. LoRA

---

**Q6.** A frozen parameter:

A. cannot participate in a forward pass B. cannot contribute to the model output C. generally does not receive gradient-based updates D. is automatically deleted from memory

---

**Q7.** If (B=0) at the beginning of LoRA training, then:

\[ BA=? \]

A. (A) B. (B) C. (0) D. (AB)

---

**Q8.** Which statement is strongest?

A. Increasing LoRA rank always increases validation accuracy B. Increasing LoRA rank always decreases training time C. Increasing LoRA rank increases adapter capacity and cost, but the accuracy effect is empirical D. Rank has no effect on LoRA

---

# Section B — Explain the Claim

### 4 questions × 7 marks = 28 marks

### Suggested time: 24 minutes

For each question, determine whether the statement is **correct, incorrect, or incomplete**, then explain.

---

**Q9.**

> "Our model has 80% fewer trainable parameters after freezing layers, therefore inference is 80% cheaper."

---

**Q10.**

> "We pruned 95% of the model's weights, so our application will definitely run approximately 20 times faster."

---

**Q11.**

> "LoRA works because the pretrained weight matrix is low rank."

---

**Q12.**

> "Our LoRA checkpoint is almost the same size as the dense model, so LoRA has failed to reduce parameter efficiency."

---

# Section C — Technical Calculations

### 3 questions × 10 marks = 30 marks

### Suggested time: 27 minutes

**Q13. — Parameter-efficiency analysis**

A model contains three trainable square matrices:

\[ W\_1\\in\\mathbb{R}^{2048\\times2048} \]

\[ W\_2\\in\\mathbb{R}^{4096\\times4096} \]

\[ W\_3\\in\\mathbb{R}^{1024\\times1024}. \]

All three are fully fine-tuned.

### (a)

Calculate the total number of trainable weight parameters.

### (b)

Now suppose all three use LoRA with:

\[ r=16. \]

Calculate the total number of trainable LoRA parameters.

### (c)

Calculate the reduction factor.

### (d)

Explain why the largest matrix receives the largest absolute benefit from LoRA.

---

**Q14. — Quantisation and combined compression**

A model contains:

\[ 8\\times10^9 \]

parameters.

### (a)

Calculate theoretical weight storage at FP32.

### (b)

Calculate theoretical weight storage at INT8.

### (c)

Calculate theoretical weight storage at INT4.

### (d)

Suppose pruning removes 75% of the weights before INT4 quantisation. What is the theoretical storage of the remaining weights?

### (e)

Explain why this combined calculation does not automatically predict actual end-to-end inference speed.

---

**Q15. — LoRA rank sweep**

An experiment produces:

| Rank | Trainable parameters | Validation accuracy |
| --- | --- | --- |
| 2 | 40,000 | 81.1% |
| 4 | 80,000 | 82.3% |
| 8 | 160,000 | 83.0% |
| 16 | 320,000 | 83.2% |
| 32 | 640,000 | 83.3% |

### (a)

Which rank gives the highest validation accuracy?

### (b)

If deployment strongly prioritises parameter efficiency, which rank might you choose instead?

### (c)

Why might choosing the maximum rank purely because it gives the highest accuracy be a poor engineering decision?

### (d)

What additional measurements would help make the final decision?

---

# Section D — Experimental Design

### 2 questions × 13 marks = 26 marks

### Suggested time: 22 minutes

**Q16. — Designing a fair rank experiment**

You need to compare LoRA ranks:

\[ r\\in{2,4,8,16,32}. \]

Describe a scientifically fair experimental procedure.

Your answer should address at least:

- model initialization;
- pretrained weights;
- dataset split;
- training hyperparameters;
- random seeds;
- number of epochs;
- evaluation metric;
- validation versus test data;
- parameter counting;
- memory measurement;
- interpretation of results.

---

**Q17. — Diagnosing an unexpected LoRA experiment**

A student reports:

> "My LoRA model initially performs much worse than the pretrained model, even before training begins. Increasing the rank did not solve the issue."

Give at least **three possible implementation or experimental causes**.

Your answer must include a discussion of LoRA initialization.

---

# End of Mock Exam 2

**Suggested timing:** Section A: 12 min Section B: 24 min Section C: 27 min Section D: 22 min Review: 5 min

**Total: 90 minutes / 100 marks** :::

# Mock Exam 3 — High-Difficulty / Exam Simulation

:::writing{variant="document" id="91583" title="Advanced Machine Learning — 90-Minute Mock Exam 3"}

# Advanced Machine Learning

## 90-Minute Mock Examination 3

**Time:** 90 minutes **Total:** 100 marks

### Instructions

This is the most demanding of the three papers.

The exam is designed to test whether you can **connect concepts rather than simply recall definitions**.

Where appropriate, justify why an alternative method would be inferior.

---

# Section A — Rapid-Fire Questions

### 10 questions × 2 marks = 20 marks

### Suggested time: 12 minutes

Answer each in one or two technically precise sentences.

---

**Q1.** Why can an overparameterised neural network still generalise well?

---

**Q2.** What is the purpose of a validation set?

---

**Q3.** Give one fundamental difference between quantisation and pruning.

---

**Q4.** Why can INT4 reduce memory substantially compared with FP32?

---

**Q5.** Give one reason why PTQ may reduce accuracy.

---

**Q6.** What is the key difference between magnitude pruning and similarity pruning?

---

**Q7.** Why does structured pruning generally have better potential for practical hardware acceleration than arbitrary unstructured pruning?

---

**Q8.** In one equation, express the central LoRA idea.

---

**Q9.** Why does LoRA reduce optimizer-state memory?

---

**Q10.** State the most important conceptual difference between SVD compression and LoRA.

---

# Section B — Integrated Short Answers

### 4 questions × 7 marks = 28 marks

### Suggested time: 24 minutes

**Q11. — Compression hierarchy**

Consider the following progression:

\[ \\text{FP32 dense model} \\rightarrow \\text{INT8 dense model} \\rightarrow \\text{INT8 pruned model} \]

Explain what kind of redundancy is being exploited at each stage.

Then explain why the three stages need not provide proportional improvements in real-world latency.

---

**Q12. — Fine-tuning hierarchy**

Compare the following approaches:

1. full fine-tuning;
2. layer freezing;
3. adapters;
4. LoRA.

For each, identify:

- what remains frozen;
- what is trained;
- the main source of efficiency;
- a potential limitation.

---

**Q13. — Low-dimensionality**

Explain the connection between the following ideas:

- overparameterisation;
- redundancy;
- intrinsic/effective dimensionality;
- low-rank representations;
- LoRA.

Your answer should also explain why these concepts do **not** imply that every weight matrix should simply be replaced by a low-rank approximation.

---

**Q14. — Practical interpretation**

A student says:

> "My LoRA model trains 10 times fewer parameters, therefore it must train 10 times faster and use 10 times less GPU memory."

Critically evaluate this statement.

Separate your answer into:

- trainable parameters;
- gradients;
- optimizer state;
- forward computation;
- complete model storage;
- wall-clock training time.

---

# Section C — Multi-Step Problems

### 3 questions × 12 marks = 36 marks

### Suggested time: 32 minutes

**Q15. — Complete LoRA derivation**

A linear layer has:

\[ d_{in}=3072,\\qquad d_{out}=4096. \]

LoRA rank is:

\[ r=16. \]

### (a)

Calculate the number of parameters in the original weight matrix.

### (b)

Calculate the number of LoRA parameters.

### (c)

Calculate the fraction of trainable parameters represented by LoRA.

### (d)

Assume the original matrix is FP32 and the LoRA parameters are also FP32. Calculate the memory required for the LoRA parameters alone.

### (e)

Explain why this number does not represent the total memory required during LoRA training.

### (f)

Explain what happens if the LoRA weights are merged into the original matrix after training.

---

**Q16. — Designing a memory-efficient adaptation system**

You have a 100-billion-parameter pretrained language model.

You need to adapt it to:

- Task A: classification;
- Task B: summarisation;
- Task C: domain-specific generation;
- Task D: another classification problem.

You have limited GPU memory.

### (a)

Would you choose full fine-tuning or PEFT? Explain.

### (b)

Would quantising the base model and using LoRA be a sensible strategy? Explain.

### (c)

How would you organise storage so that the same base model can support all four tasks?

### (d)

What would be stored separately for each task?

### (e)

How could the adaptations be deployed without requiring an additional LoRA branch at inference?

### (f)

What trade-off might occur when selecting the LoRA rank?

---

**Q17. — Compression decision problem**

You are given four possible deployment strategies:

| Strategy | Weight precision | Sparsity | Adaptation | Hardware |
| --- | --- | --- | --- | --- |
| A | FP32 | Dense | Full FT | General GPU |
| B | INT8 | Dense | Frozen | INT8 accelerator |
| C | INT4 | 75% sparse | LoRA | Sparse-support GPU |
| D | FP16 | Structured pruning | LoRA | General GPU |

The requirements are:

- limited memory;
- minimal training cost;
- good target-task accuracy;
- practical inference speed.

### (a)

Which strategy would you investigate first?

### (b)

Which information from the table makes that strategy attractive?

### (c)

What experimental evidence would you need before claiming it is the best strategy?

### (d)

Explain at least two reasons why a theoretically smaller model might not produce the best real-world latency.

### (e)

If INT4 causes unacceptable accuracy degradation, what alternatives could you investigate?

---

# Section D — Grand Synthesis Question

### 16 marks

### Suggested time: 17 minutes

**Q18. — End-to-end system design**

You are given a large pretrained neural network that already performs well on a source task.

A new organisation wants to deploy it across several target tasks.

Their requirements are:

- limited GPU memory during adaptation;
- minimal storage for task-specific models;
- minimal accuracy loss;
- reasonable training time;
- reasonable inference latency;
- ability to maintain many task-specific versions;
- ability to deploy the final models efficiently.

Design an end-to-end strategy using concepts from the course.

Your answer should explicitly address:

### A. Starting model

How should the pretrained model be used?

### B. Model compression

Would you use quantisation, pruning, both, or neither? Explain.

### C. Adaptation

Would you use full fine-tuning, freezing, adapters, LoRA, or a combination?

### D. Rank selection

How would you choose the LoRA rank?

### E. Memory

Explain what happens to:

- gradients;
- optimizer state;
- model weights.

### F. Storage

Explain the difference between storing:

- the complete adapted model;
- the shared base model plus task-specific adapters.

### G. Inference

Explain how LoRA can be merged.

### H. Experimental validation

Design the experiments required to demonstrate that your proposed solution is actually better than the baseline.

Your answer should not simply list techniques. **Explain why each technique is appropriate for the particular constraint.**

---

# End of Mock Exam 3

**Suggested timing:** Section A: 12 min Section B: 24 min Section C: 32 min Section D: 17 min Review: 5 min

**Total: 90 minutes / 100 marks** :::

These three are intentionally different: **Mock 1 tests broad mastery, Mock 2 tests experimental/application reasoning, and Mock 3 tests synthesis under exam pressure.** I would recommend doing them in that order rather than immediately starting with Mock 3.
