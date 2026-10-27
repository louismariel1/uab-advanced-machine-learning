# Correction Guide, Chunk 1: Q1–Q30

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

## Topic 6 — Meta-learning, Model Merging and Hypernetworks

### Q26. Few-shot meta-learning (3 points)

(a) Meta-learning learns how to adapt to new tasks quickly using only a few labelled examples.

(b) Ordinary training optimises a model for one task, whereas meta-learning trains across a distribution of tasks to learn transferable adaptation capabilities.

(c) It may fail if the new task differs significantly from the training tasks.

### Q27. Reptile update rule (3 points)

(a) $\theta_M$ represents the meta-parameters, while $\theta_j$ represents the parameters after adapting to a sampled task.

(b) When $\beta=1$, the meta-parameters are set directly to the adapted task parameters: $\theta_M \leftarrow \theta_j$.

(c) Reptile moves the initial parameters towards adapted task parameters, while MAML explicitly optimises for good post-adaptation performance using gradients through the adaptation process.

### Q28. Model merging and task arithmetic (3 points)

(a) A task vector is the difference between a fine-tuned model's parameters and the shared pretrained parameters:

$$
\tau_A=\theta_A-\theta_0,\qquad
\tau_B=\theta_B-\theta_0
$$

(b) Task arithmetic combines task vectors, for example:

$$
\theta_{\text{merged}}=\theta_0+\tau_A+\tau_B
$$

(c) Task vectors may interfere through conflicting parameter updates, reducing performance on one or both tasks.

### Q29. Hypernetworks (3 points)

(a) A hypernetwork learns to generate the parameters of another neural network from a task representation.

(b) It can generate task-specific parameters on demand, reducing the need to train and store a separate full model for every task.

(c) Generating large parameter sets can be computationally expensive, and the generated models may generalise poorly to unseen tasks.

### Q30. Integrated question: choosing an adaptation strategy (4 points)

(a) LoRA minimises trainable parameters while preserving the pretrained weights by learning low-rank updates. A limitation is that its low-rank updates may not capture all task-specific changes.

(b) Meta-learning trains an initialisation or adaptation strategy across related tasks, enabling rapid adaptation to a new task with few labelled examples.

(c) Model merging combines parameter updates into a shared model, whereas separate task-specific models maintain independent parameters for each task.

(d) Recommendation: Combine LoRA with meta-learning. LoRA reduces trainable parameters per task, while meta-learning enables rapid few-shot adaptation. The shared pretrained model is preserved, and small task-specific adapters can be stored instead of complete models.

# TD1 — Advanced Machine Learning

## Correction Guide: Q31–Q60 (Chunk 2)

Short, accurate, exam-ready answers with essential reasoning and calculations.

## Topic 7 — Continual learning, LwF and EWC

### Q31. Catastrophic forgetting

Correct answers: A, B, D.

- A. Correct — this is catastrophic forgetting.
- B. Correct — learning Task B can modify parameters important for Task A.
- C. Incorrect — forgetting does not prove insufficient model capacity.
- D. Correct — continual learning aims to retain previous knowledge while learning new tasks.
- E. Incorrect — setting the learning rate to zero prevents learning.

### Q32. Learning without Forgetting (LwF)

(a) LwF preserves old-task behaviour by using the original model's predictions as soft targets while training the updated model on the new task.

(b) The original predictions act as distillation targets, encouraging the new model to retain its previous outputs on new-task inputs.

(c) If new-task inputs differ substantially from the old training distribution, the old model's predictions may not reliably preserve performance on Task A.

### Q33. Elastic Weight Consolidation (EWC)

(a) The quadratic penalty discourages changes to parameters important for Task A.

(b) Large $F_i$ values indicate parameters estimated to be important, so moving them away from $\theta_i^*$ incurs a larger penalty.

(c) Increasing $\lambda$ strengthens the constraint on parameter changes, generally reducing forgetting but potentially hindering learning Task B.

(d) EWC may struggle when tasks require conflicting parameter changes or when its importance estimates are inaccurate.

### Q34. Compare continual-learning strategies

(a) LwF: Uses distillation to preserve the previous model's outputs on new-task inputs.

(b) EWC: Uses importance-weighted regularisation to discourage changes to important parameters.

(c) Both can struggle when tasks differ substantially, new tasks conflict with old ones, or the preservation mechanism inadequately represents previous knowledge.

### Q35. Evaluate a continual-learning experiment

(a) Decrease in Task A accuracy:

- Sequential fine-tuning: $91-54=\boxed{37}$ percentage points.
- EWC: $91-78=\boxed{13}$ percentage points.

(b) EWC reduces forgetting by 24 percentage points compared with sequential fine-tuning, while Task B accuracy decreases from 88% to 82%, a trade-off of 6 percentage points.

(c) The results cover only one experiment. Performance depends on the tasks, hyperparameters, model, and EWC importance estimates.

## Topic 8 — Explainability, Grad-CAM and LIME

### Q36. Interpretability versus explainability

Correct answers: A, B, E.

- A. Correct — interpretability concerns how understandable a model's operation or decisions are.
- B. Correct — explainability methods help investigate individual predictions.
- C. Incorrect — explanations do not guarantee causal justification.
- D. Incorrect — high accuracy does not imply interpretability.
- E. Correct — explanations can reveal spurious features or unexpected behaviour.

### Q37. Grad-CAM

(a) Grad-CAM produces a class-specific heatmap highlighting image regions that contribute to a CNN's prediction.

(b) Gradients of the target-class score with respect to feature maps determine their importance; weighted feature maps are combined to produce the heatmap.

(c) The heatmap is a coarse, partial explanation and does not prove that highlighted regions causally determine the prediction.

### Q38. LIME and local surrogate models

(a) A local surrogate is a simpler, interpretable model trained to approximate the original model's behaviour near one particular input.

(b) LIME perturbs the input, observes the resulting predictions, and identifies features associated with changes in the output.

(c) Different perturbation samples, neighbourhood definitions, or hyperparameters can produce different local approximations and explanations.

### Q39. Grad-CAM versus LIME

(a) Grad-CAM: Produces a spatial heatmap highlighting image regions relevant to a target class.

(b) LIME: Perturbs the input and fits a simple, locally weighted surrogate model to approximate the original classifier.

(c) Both provide incomplete, model- and configuration-dependent explanations that may not establish causal feature importance.

### Q40. Detecting a spurious correlation

(a) The model may rely on grass as a shortcut rather than learning features that identify cows and horses.

(b) Evaluate the model on images with the animals and backgrounds independently varied, such as placing the same animals against different backgrounds.

(c) A plausible heatmap does not prove causal reliance; visual explanations may omit relevant features or fail to reflect the model's true decision process.

## Topic 9 — Adversarial machine learning, PGD and robustness

### Q41. Adversarial examples

Correct answers: A, C, D.

- A. Correct — small, carefully designed perturbations can change predictions.
- B. Incorrect — perturbations can be visually imperceptible.
- C. Correct — high clean accuracy does not guarantee adversarial robustness.
- D. Correct — adversarial examples help evaluate robustness.
- E. Incorrect — validation accuracy alone does not establish robustness.

### Q42. Projected Gradient Descent (PGD)

(a) The attack seeks a perturbation that maximises classification loss within a permitted region around the original input.

(b) Repeated gradient updates can find stronger adversarial examples than a single step by iteratively refining the perturbation.

(c) Projection ensures that the perturbation remains within the specified norm bound and any applicable input constraints.

(d) Increasing the perturbation bound can make stronger attacks possible and reduce adversarial accuracy, although the changes may become more visible.

### Q43. FGSM versus PGD

(a) FGSM takes one gradient-sign step; PGD takes multiple gradient-based steps with projection after each update.

(b) PGD can refine the perturbation iteratively, often finding higher-loss adversarial examples.

(c) Attack strength depends on parameters such as perturbation budget, step size, number of iterations, and initialisation.

### Q44. Adversarial training

(a) It trains the model on adversarial examples to improve performance against specified perturbations.

(b) Robustness may improve at the expense of clean accuracy, computational efficiency, or both.

(c) Weak or poorly configured attacks can underestimate vulnerability, making robustness appear greater than it is.

### Q45. Interpret a robustness experiment

(a) Clean-accuracy difference:

$$
92\%-86\%=\boxed{6\text{ percentage points}}
$$

Standard training achieves 6 percentage points higher clean accuracy.

(b) Adversarial training increases PGD accuracy from 18% to 57%, a gain of 39 percentage points, while reducing clean accuracy by 6 percentage points.

(c) Robustness is attack-dependent. Other attacks, perturbation budgets, configurations, or threat models may still expose vulnerabilities.

## Topic 10 — Privacy-preserving ML and federated learning

### Q46. Federated learning fundamentals

Correct answers: A, C, E.

- A. Correct — raw data can remain on individual clients while updates are communicated.
- B. Incorrect — federated learning alone does not guarantee privacy.
- C. Correct — the server can aggregate client updates.
- D. Incorrect — central collection of all training examples is not required.
- E. Correct — it supports learning across distributed devices or organisations.

### Q47. Federated averaging (FedAvg)

(a) The server distributes the current model, clients train locally, send model updates or weights, and the server aggregates them, typically using weighted averaging.

(b) Clients compute updates using their own data locally, so raw training examples need not be transmitted.

(c) Non-identically distributed client data can cause client drift and reduce convergence or global model performance.

### Q48. Differential privacy

(a) Differential privacy limits how much an algorithm's output can reveal about the participation or data of an individual.

(b) Calibrated random noise, relative to the query's sensitivity and privacy parameters, limits the influence of individual records.

(c) Stronger privacy generally requires more noise, which can reduce model accuracy or utility.

(d) Noise alone is insufficient: a formal guarantee requires appropriate sensitivity calibration, a privacy mechanism, and accounting for the privacy budget $(\varepsilon,\delta)$, where applicable.

### Q49. Privacy threats in federated learning

(a) Model updates can leak information about training examples through attacks such as gradient inversion or membership inference.

(b) Differential privacy can reduce leakage by limiting individual contributions, for example through gradient clipping and calibrated noise.

(c) Added noise can reduce model utility, and the privacy guarantee depends on correct calibration and privacy-budget accounting.

### Q50. Compare privacy-preserving approaches

(a) Federated learning: Enables local training at each hospital and aggregation of updates without centrally pooling raw records.

(b) Differential privacy: Provides a formal bound on how much an individual's data can influence the released output.

(c) They can be combined: federated learning decentralises data processing, while differential privacy adds protection against information leakage from updates or released models.

## Topic 11 — Integrated algorithm interpretation and quantitative reasoning

### Q51. Identify the learning algorithm from its update rule

(a) Self-paced learning.

(b) It initially selects easier, lower-loss examples and gradually includes more difficult examples as the threshold increases.

(c) High loss can result from difficult but valid examples, label noise, distribution shifts, or model limitations; it does not automatically imply that an example is useless.

### Q52. Distinguish model efficiency metrics

(a) A model may contain frozen or non-trainable parameters, so fewer trainable parameters do not necessarily mean fewer total stored parameters.

(b) Inference latency also depends on architecture, operation types, memory access, hardware, and implementation.

(c) Examples include inference latency, peak memory usage, FLOPs, throughput, or total checkpoint size.

### Q53. Interpret an optimisation objective

Given:

$$
L=L_{\text{task}}+\lambda L_{\text{regularisation}}
$$

(a) $L_{\text{task}}$ measures the model's error on the target task.

(b) $L_{\text{regularisation}}$ penalises undesirable model properties, such as excessive complexity or large parameter changes.

(c) An excessively large $\lambda$ can over-constrain the model, causing underfitting and poor task performance.

### Q54. Reason about a teacher–student system

(a) Systematic teacher errors can be transferred to the student, reducing its performance on the rare class.

(b) Ground-truth loss supplies correct supervision that can counteract incorrect teacher predictions.

(c) Compare student and baseline performance on a held-out, class-stratified test set, especially the rare class's recall and accuracy.

### Q55. Experimental validity and ablation studies

(a) Multiple simultaneous changes confound the results, so the cause of the improvement cannot be isolated.

(b) An ablation study removes or changes individual components to measure their contributions.

(c) Change one factor at a time while keeping other settings fixed, and use the same data splits and evaluation protocol, ideally across multiple random seeds.

## Topic 12 — Integrated application and exam-style reasoning

### Q56. Choose a strategy for limited labelled data

(a) Self-supervised pretraining: Learn useful visual representations from unlabelled images using a pretext or self-supervised objective.

(b) Semi-supervised learning: Use labelled and unlabelled target-task data together, for example through pseudo-labels or consistency regularisation.

(c) Transfer learning: Reuse representations learned by a pretrained model and adapt them to the five target classes.

(d) Recommended pipeline: use self-supervised pretraining on the large unlabelled dataset if useful pretrained representations are unavailable, then fine-tune on the small labelled target dataset. If a suitable pretrained model already exists, transfer learning is a cheaper starting point. Semi-supervised learning can further exploit unlabelled target-task data.

### Q57. Explain a model's prediction and test robustness

(a) The heatmap may indicate that the model relies on irrelevant regions or spurious correlations rather than relevant anatomy.

(b) Grad-CAM is a coarse, incomplete explanation and does not prove which features causally determine the prediction.

(c) The changed prediction suggests vulnerability to small perturbations and inadequate robustness under the tested attack.

(d) Test performance on an independent, clinically representative dataset and evaluate with multiple appropriately configured adversarial attacks. This helps assess generalisation and robustness.

### Q58. Continual learning under privacy constraints

(a) Federated learning allows hospitals to train locally and share model updates rather than raw patient data.

(b) Continual learning is needed to incorporate new tasks while retaining knowledge from earlier tasks and reducing catastrophic forgetting.

(c) EWC penalises changes to parameters important for previous tasks. Alternatively, LwF uses distillation to preserve previous outputs.

(d) Federated updates can leak information, while differential privacy can reduce leakage at the cost of model utility. Combining privacy protection with continual learning may also make adaptation more difficult.

### Q59. Evaluate a multi-method experiment

(a) Full fine-tuning: highest target accuracy at 88%.

(b) Frozen backbone + head: fewest trainable parameters at 100,000.

(c) Methods trade off accuracy, training time, parameter count, memory, and deployment requirements; the best choice depends on the constraints.

(d) LoRA: achieves 86% accuracy with 120,000 trainable parameters, satisfying the requirement of accuracy above 85% with far fewer trainable parameters than full fine-tuning.

### Q60. Integrated question: understand the algorithm, not just its name

(a) LoRA (Low-Rank Adaptation) is a plausible method because it freezes most pretrained parameters and learns a small number of low-rank adaptation parameters.

(b) Frozen parameters are excluded from optimiser updates but still participate in forward computation and influence predictions.

(c) The complete checkpoint may still include all frozen pretrained weights, so fewer trainable parameters do not necessarily mean a smaller total checkpoint.

(d) Validation accuracy alone ignores training cost, inference latency, memory, checkpoint size, robustness, and generalisation.

(e) Verify that the intended parameters are frozen, only the adaptation parameters receive gradients and updates, and the validation set is held out from training.

End of Chunk 2 — Q31–Q60 complete. All answers retain the original question numbering and include the essential calculations and reasoning.

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
