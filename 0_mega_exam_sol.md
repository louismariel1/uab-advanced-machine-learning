# TD2 — Correction Guide, Chunk 1: Q1–Q30

Advanced Machine Learning · Autumn 2026 · Exam-ready model answers

## Topic 1 — Transfer learning and domain adaptation

### Q1. Deployment performance and distribution shift (2 points)

Correct answers: A, B, D.

- A — Correct: A distribution shift between training and deployment can reduce accuracy.
- B — Correct: Overfitting can make a model rely on patterns that do not generalize.
- C — Incorrect: The learning rate is a training hyperparameter and does not normally change during inference.
- D — Correct: A validation set that differs from deployment conditions may overestimate real-world performance.
- E — Incorrect: Insufficient capacity can cause underfitting, but does not necessarily explain a validation-to-deployment gap.

### Q2. Fine-tuning a pretrained model (2 points)

Correct answers: A, C, E.

- A — Correct: Similar source and target domains can provide transferable features.
- B — Incorrect: Some layers can remain frozen.
- C — Correct: The target head must match the target output dimension.
- D — Incorrect: The fine-tuning learning rate can differ from the pretraining rate and is often smaller.
- E — Correct: Freezing early layers and updating later layers is a valid strategy.

### Q3. Freezing and feature extraction (3 points)

(a) (1 point) Freezing the backbone keeps its parameters fixed and trains the new head. Full fine-tuning updates both the backbone and the head.

(b) (1 point) An advantage of freezing is lower computational cost and reduced overfitting risk on small datasets. A disadvantage is that fixed features may not adapt sufficiently to the target domain.

(c) (1 point) A frozen layer still performs its forward computation to produce features needed by subsequent layers. Freezing prevents parameter updates; it does not remove the layer from the model.

### Q4. Domain adaptation scenarios (3 points)

(a) (1 point) This is a domain shift, potentially a covariate shift: the input image distribution changes between sunny and foggy conditions while the task remains the same.

(b) (1 point) Source validation measures performance on source-like images, not necessarily on foggy images. Features learned from sunny images may be unreliable under fog.

(c) (1 point) Possible strategies include fine-tuning on labelled foggy images, unsupervised domain adaptation using unlabelled foggy images, or suitable domain-randomization augmentation during training.

### Q5. Interpreting a transfer-learning experiment (3 points)

(a) (1 point) Absolute improvement:

$$
84\%-74\%=\boxed{10\text{ percentage points}}
$$

(b) (1 point) The pretrained backbone already contains useful visual features, allowing the model to learn effectively from the target data without learning all features from scratch.

(c) (1 point) The frozen-backbone approach trains only 150,000 parameters rather than 2 million, potentially reducing training cost, memory use, and overfitting risk.

## Topic 2 — Curriculum learning and semi-supervised learning

### Q6. Self-paced curriculum learning (4 points)

(a) (1 point) This is self-paced learning, a form of curriculum learning in which the model helps determine which examples to learn from.

(b) (1 point) The per-example loss is used as a proxy for difficulty. Low-loss examples are currently easier for the model to predict correctly.

(c) (1 point) Increasing $\lambda$ gradually admits higher-loss examples, allowing the training process to progress from easier to harder examples.

(d) (1 point) The model may initially select easy but unrepresentative examples, or reject difficult yet informative examples. Incorrect labels and outliers can also distort the loss-based selection.

### Q7. Curriculum learning versus random sampling (2 points)

Correct answers: A, C, E.

- A — Correct: A curriculum can introduce easy examples before difficult ones.
- B — Incorrect: Better generalization is not guaranteed.
- C — Correct: Self-paced learning can use the current model's losses to select examples.
- D — Incorrect: High loss can indicate a difficult example, a rare example, a model weakness, or a mislabeled example.
- E — Correct: Example selection changes the sequence of optimization updates and can affect convergence.

### Q8. Semi-supervised learning (3 points)

(a) (1 point) Semi-supervised learning combines labelled and unlabelled data to improve the target task when labelled examples are limited.

(b) (1 point) In pseudo-labeling, the model predicts labels for unlabelled images. Predictions meeting a confidence threshold are treated as approximate labels for additional training.

(c) (1 point) Incorrect pseudo-labels introduce misleading training targets. The model may reinforce its own mistakes, especially when erroneous predictions are accepted with high confidence.

### Q9. Consistency regularisation and the mean teacher (4 points)

(a) (1 point) This is the Mean Teacher method, which uses consistency regularisation with an exponential-moving-average teacher.

(b) (1 point) The exponential moving average (EMA) smooths the teacher parameters over multiple student updates. This produces more stable targets than copying the student's rapidly changing parameters after every update.

A typical update is

$$
\theta_{\text{teacher}}
\leftarrow
\mu\theta_{\text{teacher}}
+(1-\mu)\theta_{\text{student}},
$$

where $\mu$ is the EMA decay factor.

(c) (1 point) The student is encouraged to produce predictions consistent with the teacher's predictions under different augmentations of the same input. This supplies a training signal even without ground-truth labels.

(d) (1 point) The teacher can still make systematic mistakes, especially early in training or on out-of-distribution examples. EMA stabilizes predictions but does not guarantee correctness.

### Q10. Analyse a semi-supervised experiment (3 points)

(a) (1 point)

$$
73\%-68\%=\boxed{5\text{ percentage points}}
$$

(b) (1 point) A high confidence threshold may accept fewer but more reliable pseudo-labels, while a low threshold may admit more incorrect labels. Either extreme can hurt performance depending on the data and model.

(c) (1 point) Keep the architecture, data split, training budget, and other hyperparameters fixed. Change only the confidence threshold across several runs, then compare validation accuracy and the number and accuracy of accepted pseudo-labels.

## Topic 3 — Self-supervised learning, contrastive learning and augmentation

### Q11. Identify a self-supervised pretext task (2 points)

Correct answers: A, C.

- A — Correct: The applied rotation automatically supplies the training label.
- B — Incorrect: Human-provided object-class labels are unnecessary.
- C — Correct: Predicting rotations can encourage the backbone to learn useful visual structure.
- D — Incorrect: The rotation head can be replaced by a target-task classification head.
- E — Incorrect: Pretext performance does not guarantee downstream performance.

### Q12. The self-supervised learning pipeline (3 points)

(a) (1 point) The model predicts which of four rotations was applied to each unlabelled image. Cross-entropy trains the four-class pretext head and backbone.

(b) (1 point) Retain the pretrained backbone weights. Replace the four-class rotation head with a newly initialized five-class target head, then train the head and optionally fine-tune the backbone.

(c) (1 point) The source task learns general visual representations, whereas the target task uses different class definitions. Transfer learning reuses features, not necessarily the source labels.

### Q13. Contrastive learning and positive pairs (3 points)

(a) (1 point) Two augmented views of the same original image form a positive pair because they should retain the same underlying semantic content.

(b) (1 point) Negative pairs, usually views of different images, encourage the model to separate representations of different examples. The objective pulls positive pairs together and separates negatives.

(c) (1 point) Different images can belong to the same semantic class. Treating all different-image pairs as negatives can push semantically similar examples apart, creating false negatives.

### Q14. Interpreting a contrastive distance matrix (3 points)

- (a) $a$ and $a^*$: LOW. They are views of the same image and form a positive pair.
- (b) $b$ and $b^*$: LOW. They also form a positive pair.
- (c) $a^*$ and $b^*$: HIGH. They are views of different images and are treated as a negative pair under the stated assumption.

These are general optimization targets, not guarantees about every individual distance.

### Q15. Data augmentation and representation quality (3 points)

(a) (1 point) Augmented views encourage the encoder to retain semantic information that remains stable across transformations, improving robustness to irrelevant visual variation.

(b) (1 point) If a crop removes the main object, the two views may no longer share the intended semantic content. Pulling their embeddings together can teach an inappropriate invariance.

(c) (1 point) Augmentations must preserve information relevant to the downstream task. For example, horizontal flips may be inappropriate when left-right orientation changes the label.

## Topic 4 — Model compression and parameter-efficient fine-tuning

### Q16. LoRA fundamentals (2 points)

Correct answers: A, C, E.

- A — Correct: LoRA freezes the original adapted-layer weights and learns a low-rank update.
- B — Incorrect: LoRA does not guarantee higher validation accuracy.
- C — Correct: Low-rank factors can substantially reduce trainable parameter count.
- D — Incorrect: The original pretrained weights are still needed for the complete model, unless supplied separately or merged into another checkpoint.
- E — Correct: LoRA can be applied selectively to chosen linear layers.

### Q17. LoRA factorisation and dimensions (3 points)

(a) (1 point)

$$
A\in\mathbb{R}^{r\times d_{\text{in}}},
\qquad
B\in\mathbb{R}^{d_{\text{out}}\times r}.
$$

(b) (1 point) The product has dimensions

$$
BA:
(d_{\text{out}}\times r)(r\times d_{\text{in}})
=d_{\text{out}}\times d_{\text{in}},
$$

matching $W$.

(c) (1 point) Initializing $B=0$ gives $BA=0$, so the adapted layer initially behaves like the pretrained layer. Random $A$ allows learning to begin through the factorization.

### Q18. LoRA parameter-count calculation (4 points)

Given $d_{\text{in}}=2048$, $d_{\text{out}}=1024$, and $r=8$.

(a) (1 point) Original matrix:

$$
2048\times1024=\boxed{2,097,152}
$$

(b) (1 point) Adapter parameters:

$$
r(d_{\text{in}}+d_{\text{out}})
=8(2048+1024)
=\boxed{24,576}
$$

(c) (1 point) Percentage reduction:

$$
\frac{2,097,152-24,576}{2,097,152}\times100\%
$$

$$
=\boxed{98.83\%\text{ approximately}}
$$

(d) (1 point) Increasing rank makes the adapter more expressive, but also increases trainable parameters, optimizer memory, and computation. Higher rank does not guarantee better validation accuracy.

### Q19. Diagnose a LoRA implementation (3 points)

(a) (1 point) The intended LoRA setup freezes the original adapted-layer weights. If they change, the experiment may be training the base weights as well as the adapters.

(b) (1 point) Any two checks:

- Inspect `requires_grad` on the original weights and adapter factors.
- Compare copies of the original weights before and after training.
- Inspect the optimizer's parameter groups.
- Check that wrapping or model initialization has not re-enabled gradients.

(c) (1 point) Good accuracy measures predictive performance, not implementation correctness. The model could perform well through unintended dense fine-tuning while using more trainable parameters than reported.

### Q20. Model compression and deployment trade-offs (3 points)

(a) (1 point)

$$
\frac{1,000,000-60,000}{1,000,000}\times100\%
=\boxed{94\%}
$$

(b) (1 point) LoRA adds adapter operations and framework overhead. These can outweigh the savings from fewer trainable parameters, so reduced parameter count does not guarantee faster training.

(c) (1 point) A full LoRA checkpoint may store both the original pretrained weights and the adapter weights. An adapter-only checkpoint can be much smaller, but still requires access to the base model.

## Topic 5 — Knowledge distillation, ensembles and mean teacher

### Q21. Knowledge distillation fundamentals (2 points)

Correct answers: A, C, E.

- A — Correct: The student can learn from ground-truth labels and teacher predictions.
- B — Incorrect: The student can have a different architecture and capacity.
- C — Correct: The teacher is normally fixed during student training.
- D — Incorrect: The student is not guaranteed to outperform the teacher.
- E — Correct: The student can inherit teacher biases and weaknesses.

### Q22. Temperature-scaled softmax (3 points)

(a) (1 point) Increasing temperature $T$ softens the distribution, making the probabilities more uniform. Lower temperature makes it sharper.

$$
p_i=\frac{\exp(z_i/T)}{\sum_j\exp(z_j/T)}
$$

(b) (1 point) Soft probabilities reveal relative preferences among classes. For example, a teacher may indicate that an image resembles a wolf more than a truck, even when the correct label is dog.

(c) (1 point) As $T\to\infty$, the probabilities approach a uniform distribution over the finite set of output classes.

### Q23. Distillation loss and optimisation (4 points)

(a) (2 points)

- $L_h$, the ground-truth cross-entropy loss, teaches the student to predict the correct labelled class.
- $L_d$, the distillation loss, teaches the student to match the teacher's temperature-scaled probability distribution, including information about relationships between classes.

(b) (1 point) If $\alpha=1$,

$$
L=L_h.
$$

The student trains using only ground-truth cross-entropy; the distillation term has zero weight.

(c) (1 point) Equal coefficients do not imply equal gradient magnitudes. The two losses may have different numerical scales and gradients, so their actual influence on parameter updates can differ.

### Q24. Knowledge distillation with different output dimensions (3 points)

(a) (1 point) KL divergence compares distributions over the same set of outcomes. A 100-class teacher distribution and a 10-class student distribution do not align directly.

(b) (1 point) One approach is to map the teacher's outputs into the same 10 target classes, provided a meaningful mapping exists. Alternatively, use a shared representation or a carefully designed transfer objective that aligns teacher and student outputs.

(c) (1 point) Distillation matches predictive behaviour or another compatible representation, not layer-by-layer architecture. The student and teacher can therefore have different depths and widths.

### Q25. Ensembles and self-distillation (3 points)

(a) (1 point) Averaging predictions can reduce variance and smooth individual model errors, improving accuracy over a single model.

(b) (1 point) Diversity matters because models that make different errors are more likely to correct one another when their predictions are combined. Highly correlated errors reduce the benefit.

(c) (1 point) A student may learn from the teacher's soft targets, gaining richer class-relation information than hard labels alone provide. Training regularization and optimization can also yield a better generalizing solution. Improvement is possible, not guaranteed.

## Topic 6 — Meta-learning, model merging and hypernetworks

### Q26. Few-shot meta-learning (3 points)

(a) (1 point) Meta-learning aims to learn an initialization, update rule, or representation that enables rapid adaptation to a new task using only a few labelled examples.

(b) (1 point) Ordinary training optimizes performance on one task. Meta-training samples tasks from a task distribution and optimizes the model's ability to adapt across those tasks.

(c) (1 point) Meta-learning may transfer poorly when the new task differs substantially from the meta-training tasks or the task distribution is too narrow.

### Q27. Reptile update rule (3 points)

Given

$$
\theta_M\leftarrow\theta_M+\beta(\theta_j-\theta_M).
$$

(a) (1 point) $\theta_M$ is the current meta-initialization. $\theta_j$ is the model parameter vector after inner-loop training on a sampled task.

(b) (1 point) When $\beta=1$,

$$
\theta_M\leftarrow\theta_j.
$$

The meta-parameters are replaced by the adapted parameters of that sampled task, rather than moving only partway toward them.

(c) (1 point) Reptile moves the initialization toward task-adapted parameters using a first-order update. MAML explicitly differentiates through the inner adaptation steps to optimize post-adaptation performance, typically involving second-order derivatives unless approximated.

### Q28. Model merging and task arithmetic (3 points)

(a) (1 point) A task vector is the difference between a task-fine-tuned model and its shared pretrained initialization:

$$
\tau_A=\theta_A-\theta_0,
\qquad
\tau_B=\theta_B-\theta_0.
$$

(b) (1 point) Task arithmetic combines task vectors with chosen coefficients, for example:

$$
\theta_{\text{merged}}=\theta_0+\tau_A+\tau_B.
$$

(c) (1 point) The task vectors may interfere if the tasks require conflicting parameter changes. Simple addition can also magnify updates or cause forgetting.

### Q29. Hypernetworks (3 points)

(a) (1 point) A hypernetwork learns a mapping from a task representation or conditioning input to the parameters of another network.

(b) (1 point) It can generate task-specific parameters without separately training and storing a complete independently learned model for every task, potentially sharing information across tasks.

(c) (1 point) Challenges include the complexity of learning a reliable parameter-generating mapping, generalizing to unseen tasks, and the computational cost of generating parameters.

### Q30. Integrated question: choosing an adaptation strategy (4 points)

(a) (1 point) LoRA directly addresses parameter-efficient adaptation: it freezes the base weights and learns low-rank updates for each task. A limitation is that a low-rank update may not capture all the changes required by a target task.

(b) (1 point) Meta-learning trains an initialization or adaptation strategy across related tasks, allowing a new task to adapt quickly using only a few labelled examples.

(c) (1 point) Model merging combines task-specific updates or model weights into a shared merged model instead of maintaining entirely separate full models. It can reduce storage, but merging may introduce task interference.

(d) (1 point) A strong recommendation is LoRA combined with meta-learning: meta-learn an initialization or adaptation strategy across related tasks, then use small task-specific LoRA adapters for rapid adaptation while retaining the shared pretrained base. Store adapters rather than complete models. Model merging is an alternative when a single combined model is required, but it can compromise performance on individual tasks.

## Final revision checklist — Q1–Q30

## Revision progress

0 of 12

[ ]Explain domain shift and why deployment accuracy can differ from validation accuracy.

[ ]Distinguish freezing a backbone from full fine-tuning.

[ ]Explain self-paced learning and the role of the threshold λ.

[ ]Describe pseudo-labeling and the Mean Teacher EMA update.

[ ]Explain automatically generated rotation labels and downstream transfer.

[ ]Distinguish positive pairs, negative pairs, and false negatives.

[ ]Derive LoRA factor shapes and count adapter parameters.

[ ]Explain why LoRA can train more slowly despite fewer trainable parameters.

[ ]Describe the roles of hard labels, soft labels, temperature, and α.

[ ]Explain why ensemble diversity and task-vector compatibility matter.

[ ]Distinguish Reptile from MAML.

[ ]Compare LoRA, meta-learning, model merging, and hypernetworks.

Reset checklist

Chunk 1 complete: Q1–Q30. When you're ready, send TD1 Chunk 2 (Q31–Q60), and I'll prepare the matching correction guide with the original numbering preserved.
