# Advanced Machine Learning — Partial Exam 1 Practice Pack

Exam date: 27 October 2026

I’ll keep this strictly within the six topics listed for Partial Exam 1. Questions on continual learning, LwF/EWC, Grad-CAM/LIME, adversarial robustness, and privacy-preserving learning are excluded.

The practice exams will follow the official format in your course document:

- 90 minutes, pen and paper.
- A mixture of multiple-choice and free-written answers.
- Conceptual understanding, algorithm interpretation, explanations, and calculations.
- No coding required, although you may need to interpret pseudocode or diagnose an implementation.
- Each exam is worth 50 points, matching the real exam's stated total.

## Scope of all three practice exams

1. Transfer learning and domain adaptation.
2. Curriculum learning and semi-supervised learning.
3. Self-supervised learning, contrastive learning, and data augmentation.
4. Model compression and parameter-efficient fine-tuning, especially LoRA.
5. Knowledge distillation, ensemble learning, and Mean Teacher.
6. Meta-learning, model merging, task arithmetic, and hypernetworks.

Each exam will use different scenarios and numbers to test understanding rather than memorization. The solutions will be provided separately after the questions, so you can attempt each exam under timed conditions first.

# Practice Exam 1 — Foundations and Core Methods

Total marks

# 50 points

Suggested time

# 90 min

Instructions: Attempt all questions. For multiple-choice questions, select all correct options where indicated. Explain your reasoning in written answers. Calculators are unnecessary.

## Questions

### Q1. Transfer learning and domain adaptation — 6 points

(a) (2 points) A classifier achieves 94% accuracy on validation images of sunny roads but only 71% on foggy-road deployment images. Which statements are correct? Select all that apply.

- A. The input distribution may have shifted between validation and deployment.
- B. The learning rate during inference is probably too high.
- C. Fine-tuning on representative labelled foggy-road images may help.
- D. High validation accuracy guarantees high deployment accuracy.
- E. Domain-randomization augmentation may improve robustness to weather variation.

(b) (2 points) Explain the difference between freezing a pretrained backbone and fully fine-tuning it.

(c) (2 points) A pretrained model has 3 million backbone parameters and a new classification head with 120,000 parameters. If the backbone is frozen, how many parameters are trainable? Give one benefit and one limitation of this approach.

### Q2. Curriculum and semi-supervised learning — 7 points

(a) (3 points) Consider the following algorithm:

1. Calculate each training example's loss.
2. Select examples whose loss is below a threshold $\lambda$.
3. Update the model using only the selected examples.
4. Gradually increase $\lambda$ during training.

Identify the method, explain how it works, and state one possible failure mode.

(b) (2 points) A model uses 2,000 labelled images and 20,000 unlabelled images. Explain how pseudo-labeling can exploit the unlabelled images.

(c) (2 points) Why can incorrect pseudo-labels cause a feedback loop? Give one way to reduce this risk.

### Q3. Self-supervised and contrastive learning — 7 points

(a) (2 points) An unlabelled image is randomly rotated by 0°, 90°, 180°, or 270°. A model predicts which rotation was applied. Explain why this is self-supervised learning.

(b) (3 points) Two images $a$ and $b$ are independently sampled. Two augmented views $a^*$ and $b^*$ are created. State whether the following distances should generally be optimized to be low or high:

- $d(a,a^*)$
- $d(b,b^*)$
- $d(a^*,b^*)$, assuming they are treated as negatives.

Explain your answers and identify one potential problem with the negative-pair assumption.

(c) (2 points) Why can an aggressive random crop harm contrastive learning? Give an example of an augmentation that may be unsuitable for a particular downstream task.

### Q4. LoRA and model compression — 8 points

A pretrained linear layer has weight matrix $W\in\mathbb{R}^{1024\times2048}$. LoRA represents its update as $\Delta W=BA$, where the rank is $r=4$.

(a) (2 points) State the shapes of $A$ and $B$.

(b) (2 points) Calculate the number of parameters in the original weight matrix and in the two LoRA factors.

(c) (2 points) Calculate the percentage reduction in trainable parameters when comparing the LoRA factors with the original matrix, assuming biases are excluded.

(d) (2 points) Why might LoRA reduce trainable parameter count without reducing training time proportionally? What should happen to the original weights during standard LoRA training?

### Q5. Knowledge distillation and ensembles — 8 points

(a) (2 points) Explain how a student model can learn from both ground-truth labels and teacher predictions.

(b) (2 points) What is the effect of increasing the softmax temperature $T$ in distillation? Why can softened teacher predictions be useful?

(c) (2 points) Five independently initialized networks each achieve 80% test accuracy. Averaging their predictions achieves 86%. Explain why this improvement is possible and why it is not guaranteed.

(d) (2 points) A teacher predicts a probability distribution over 100 classes, while its student predicts over 10 classes. Explain why their distributions cannot ordinarily be compared directly using KL divergence and describe one possible solution.

### Q6. Meta-learning, merging, and hypernetworks — 7 points

(a) (2 points) In Reptile,

$$
\theta_M\leftarrow\theta_M+\beta(\theta_j-\theta_M),
$$

explain $\theta_M$, $\theta_j$, and the effect of $\beta=1$.

(b) (2 points) Given a shared pretrained initialization $\theta_0$ and two fine-tuned models $\theta_A$ and $\theta_B$, define their task vectors and write a simple model-merging equation.

(c) (3 points) Briefly distinguish meta-learning, model merging, and hypernetworks. Give one limitation of combining task vectors by simple addition.

### Q7. Integrated experiment — 7 points

A team has 500 labelled images and 15,000 unlabelled images from a new target domain. They have a suitable pretrained backbone.

- Method A freezes the backbone and trains a new head.
- Method B fully fine-tunes the model.
- Method C uses LoRA adapters.
- Method D uses pseudo-labels on the unlabelled images.

(a) (2 points) Explain the main trade-off between Methods A and B.

(b) (2 points) Why might Method C be attractive if compute and trainable-parameter budgets are limited?

(c) (2 points) Describe how Method D could be incorporated into the pipeline and identify one risk.

(d) (1 point) State one fair experimental control needed to compare the methods.

# Practice Exam 2 — Algorithm Interpretation and Quantitative Reasoning

Total marks

# 50 points

Suggested time

# 90 min

## Questions

### Q1. Fine-tuning and distribution shift — 7 points

(a) (3 points) A pretrained model is adapted from classifying household objects to classifying five types of industrial defects. Explain why the original classification head may need replacement and why the backbone may still be useful.

(b) (2 points) State two reasons why a smaller fine-tuning learning rate is often used than the original pretraining learning rate.

(c) (2 points) A model has 2.4 million backbone parameters and a 60,000-parameter head. Only the head is trained. Calculate the trainable parameter count and its percentage of the full model, ignoring other parameters.

### Q2. Self-paced learning and Mean Teacher — 8 points

(a) (3 points) A training algorithm selects examples with loss $L_i<\lambda$, then increases $\lambda$ over successive epochs. Explain the selection mechanism, why it may help, and one reason it can fail.

(b) (3 points) In Mean Teacher, the teacher parameters are updated as

$$
\theta_T\leftarrow\mu\theta_T+(1-\mu)\theta_S.
$$

Explain the roles of $\theta_T$, $\theta_S$, and $\mu$. Why can the teacher provide more stable targets than copying the student after every update?

(c) (2 points) Explain consistency regularization and identify a reason why teacher predictions may still be unreliable.

### Q3. Contrastive learning and augmentation — 7 points

(a) (3 points) A contrastive learner produces embeddings for two augmented views of the same image and for a view of a different image. Explain the intended relationship among their embedding distances.

(b) (2 points) What is a false negative in contrastive learning? Explain why it is harmful.

(c) (2 points) A medical image classifier predicts whether a scan contains a left-sided or right-sided abnormality. Explain why horizontal flipping might be unsuitable during pretraining or fine-tuning for this task.

### Q4. LoRA parameter calculations — 9 points

A linear layer has $d_{\text{in}}=1536$, $d_{\text{out}}=768$, and LoRA rank $r=16$.

(a) (2 points) Write the dimensions of $A$ and $B$.

(b) (2 points) Calculate the original matrix parameter count.

(c) (2 points) Calculate the number of LoRA trainable parameters.

(d) (2 points) Calculate the percentage reduction relative to training the full original matrix.

(e) (1 point) Explain why increasing $r$ is not guaranteed to improve validation accuracy.

### Q5. Distillation and self-distillation — 7 points

(a) (2 points) Write the general combined objective using a hard-label loss $L_h$, a distillation loss $L_d$, and a coefficient $\alpha$. What does the objective become when $\alpha=0$, assuming the hard-label term has coefficient $1-\alpha$?

(b) (2 points) Explain why equal coefficients on two loss terms do not guarantee equal contributions to parameter updates.

(c) (3 points) Explain how a student can outperform its teacher even if both have the same architecture and train on the same dataset.

### Q6. Meta-learning and task arithmetic — 6 points

(a) (2 points) State the main difference between Reptile and MAML.

(b) (2 points) Two task vectors are $\tau_A=\theta_A-\theta_0$ and $\tau_B=\theta_B-\theta_0$. Write the merged parameters and explain one reason merging can hurt task performance.

(c) (2 points) Explain what a hypernetwork generates and why it can be useful for multiple tasks.

### Q7. Experimental interpretation — 6 points

Three adaptation methods give the following results:

| Method | Accuracy | Trainable parameters |
| --- | --- | --- |
| Full fine-tuning | 91% | 2,000,000 |
| Frozen backbone + head | 84% | 100,000 |
| LoRA | 89% | 120,000 |

(a) (2 points) Which method has the highest accuracy, and which has the fewest trainable parameters?

(b) (2 points) If the requirements are accuracy above 88% and fewer than 150,000 trainable parameters, which method satisfies both?

(c) (2 points) Explain why these results alone cannot establish which method is fastest to train or cheapest to deploy.

# Practice Exam 3 — Integrated Applications and Exam-Style Reasoning

Total marks

# 50 points

Suggested time

# 90 min

## Questions

### Q1. Transfer learning under limited data — 7 points

A researcher has 300 labelled images from a new industrial domain and access to a pretrained vision model.

(a) (2 points) Recommend an initial transfer-learning strategy and justify it.

(b) (2 points) Explain one advantage and one disadvantage of freezing the backbone.

(c) (3 points) The training images come from a controlled laboratory, but deployment images contain different lighting and backgrounds. Identify the problem and propose two ways to address it.

### Q2. Semi-supervised and self-paced learning — 7 points

(a) (3 points) Describe a pipeline that combines a small labelled dataset with a larger unlabelled dataset using pseudo-labeling. Include a confidence threshold and explain its role.

(b) (2 points) Explain why accepting every pseudo-label can be harmful.

(c) (2 points) Compare pseudo-labeling with self-paced curriculum learning: what determines the selected examples in each method?

### Q3. Contrastive learning — 6 points

(a) (2 points) Describe how two augmented views of one image create a positive pair without requiring a human class label.

(b) (2 points) Explain the purpose of negative pairs and define a false negative.

(c) (2 points) An augmentation removes the object entirely from one view. Explain why this can undermine the training objective.

### Q4. LoRA implementation and deployment — 9 points

A pretrained linear layer has dimensions $W\in\mathbb{R}^{512\times1024}$. LoRA rank is $r=8$.

(a) (2 points) Calculate the original matrix's parameter count.

(b) (2 points) Calculate the LoRA parameter count and percentage reduction relative to the original matrix.

(c) (2 points) Explain why setting $B=0$ at initialization allows the adapted layer initially to behave like the pretrained layer.

(d) (2 points) Name two checks that can confirm the original weights are frozen and the adapter factors are trainable.

(e) (1 point) Why can an adapter-only checkpoint still depend on a much larger pretrained model?

### Q5. Knowledge distillation and ensembles — 7 points

(a) (2 points) Explain the difference between the information conveyed by hard labels and by teacher soft targets.

(b) (2 points) A teacher is very confident but systematically misclassifies a rare class. Explain a possible effect on its student and one way to reduce the risk.

(c) (3 points) Explain why averaging predictions can outperform one model, why diversity matters, and why the same principle can also apply to regressors.

### Q6. Meta-learning, model merging, and hypernetworks — 7 points

(a) (2 points) Explain how meta-learning can help a model adapt to a new task with only a few labelled examples.

(b) (2 points) Explain task vectors and task arithmetic.

(c) (3 points) Distinguish model merging from a hypernetwork approach. Give one limitation of each.

### Q7. Integrated method selection — 7 points

A company needs a vision model for six related tasks. It has a large pretrained model, only a few labelled examples per task, and limited storage for task-specific models.

(a) (2 points) Recommend a parameter-efficient approach and explain why it suits the storage constraint.

(b) (2 points) Explain how meta-learning could improve adaptation to the six tasks.

(c) (2 points) Would merging all six task vectors necessarily preserve performance on every task? Explain.

(d) (1 point) State one important experiment to validate the proposed solution.

# Solutions and marking guide

Use these after attempting the exams. The answers below are model answers, not the only acceptable wording. For written questions, award yourself credit for including the essential concept and reasoning.

# Exam 1 — Solutions

## Q1. Transfer learning and domain adaptation — 6 points

(a) (2 points): A, C, E.

- A: Fog changes the input distribution, potentially reducing generalization.
- C: Labelled target-domain data can help the model adapt to foggy conditions.
- E: Appropriate weather augmentation can improve robustness to changes in lighting and visibility.
- B is incorrect because the learning rate is not normally used during inference.
- D is incorrect because validation performance does not guarantee deployment performance.

(b) (2 points) Freezing the backbone keeps its pretrained parameters fixed and trains the new head. Full fine-tuning updates both the backbone and head, allowing the learned features to adapt to the target task.

(c) (2 points) Trainable parameters: $120{,}000$.

- Benefit: Lower computational and optimizer-memory costs, with reduced overfitting risk on small datasets.
- Limitation: Fixed backbone features may not represent the target domain adequately.

## Q2. Curriculum and semi-supervised learning — 7 points

(a) (3 points) This is self-paced learning, a form of curriculum learning. The model uses per-example loss as a difficulty proxy, initially selecting low-loss examples. Increasing $\lambda$ admits harder examples progressively.

A failure mode is that high-loss examples may be informative but excluded, while low-loss examples may be unrepresentative or incorrectly labelled.

(b) (2 points) In pseudo-labeling, the model predicts labels for the 20,000 unlabelled images. Predictions above a confidence threshold are treated as approximate labels and included in subsequent training alongside the 2,000 labelled examples.

(c) (2 points) Incorrect pseudo-labels can train the model toward its own mistakes, making subsequent incorrect predictions more likely. Possible safeguards include a higher confidence threshold, validation-based threshold selection, or consistency checks.

## Q3. Self-supervised and contrastive learning — 7 points

(a) (2 points) The rotation itself supplies the label automatically. The model learns to predict one of four transformations without human-provided object labels, encouraging useful visual representations.

(b) (3 points)

- $d(a,a^*)$: Low — both views originate from the same image.
- $d(b,b^*)$: Low — both views originate from the same image.
- $d(a^*,b^*)$: High — the views come from different images and are treated as negatives.

A false negative occurs when two different images actually belong to the same semantic class. Separating them may push semantically similar representations apart.

(c) (2 points) An aggressive crop may remove the main object, so the two views no longer preserve the intended shared semantic content. Horizontal flips may be unsuitable for tasks where left-right orientation determines the label.

## Q4. LoRA and model compression — 8 points

(a) (2 points)

$$
A\in\mathbb{R}^{4\times2048},
\qquad
B\in\mathbb{R}^{1024\times4}.
$$

(b) (2 points) Original matrix:

$$
1024\times2048=\boxed{2{,}097{,}152}
$$

LoRA factors:

$$
4(2048+1024)=\boxed{12{,}288}
$$

(c) (2 points)

$$
\frac{2{,}097{,}152-12{,}288}
{2{,}097{,}152}\times100\%
$$

$$
=\boxed{99.414\%\text{ approximately}}
$$

(d) (2 points) LoRA can introduce additional adapter operations and framework overhead, so fewer trainable parameters do not guarantee proportionally faster training. In standard LoRA, the original weight matrix is frozen while the adapter factors are trainable.

## Q5. Knowledge distillation and ensembles — 8 points

(a) (2 points) The hard-label loss teaches the student the ground-truth class. The distillation loss encourages it to match the teacher's probability distribution, conveying relative similarities between classes.

(b) (2 points) Increasing temperature makes the softmax distribution more uniform; lowering temperature makes it sharper. Softer predictions can reveal secondary class preferences that hard labels omit.

(c) (2 points) Averaging can reduce prediction variance and cancel some individual model errors. The benefit depends on the models' error patterns: highly correlated errors provide less improvement, and averaging does not guarantee better accuracy.

(d) (2 points) KL divergence ordinarily compares distributions defined over the same outcomes. A 100-class distribution cannot directly align with a 10-class distribution without a mapping. One solution is to aggregate teacher probabilities into the same 10 target classes, if a meaningful class mapping exists.

## Q6. Meta-learning, merging, and hypernetworks — 7 points

(a) (2 points) $\theta_M$ is the meta-initialization, while $\theta_j$ is the parameter vector after adaptation to a sampled task. With $\beta=1$, the meta-parameters are replaced by the adapted task parameters.

(b) (2 points) Task vectors:

$$
\tau_A=\theta_A-\theta_0,\qquad
\tau_B=\theta_B-\theta_0.
$$

Simple merging:

$$
\theta_{\text{merged}}=\theta_0+\tau_A+\tau_B.
$$

(c) (3 points)

- Meta-learning: Learns an initialization or adaptation strategy that transfers across tasks.
- Model merging: Combines task-specific parameter updates or models into a shared model.
- Hypernetwork: Generates another network's parameters from a task representation or conditioning input.

Simple task-vector addition can cause interference when tasks require conflicting parameter changes.

## Q7. Integrated experiment — 7 points

(a) (2 points) Freezing the backbone reduces training cost and overfitting risk but limits feature adaptation. Full fine-tuning can adapt representations more effectively but uses more computation and can overfit limited labelled data.

(b) (2 points) LoRA trains small low-rank updates rather than all pretrained weights, reducing trainable parameter count and potentially optimizer-state memory.

(c) (2 points) Train a model on labelled images, generate pseudo-labels for unlabelled target-domain images, select sufficiently confident predictions, and retrain using both sources. Incorrect pseudo-labels can reinforce mistakes.

(d) (1 point) Keep the data split, evaluation protocol, and comparable training budget fixed across methods.

# Exam 2 — Solutions

## Q1. Fine-tuning and distribution shift — 7 points

(a) (3 points) Replace the original head because its outputs correspond to the old classes, whereas the target task has five defect classes. Retain the pretrained backbone because its learned visual features may transfer to defect recognition.

(b) (2 points) A smaller learning rate can limit destructive changes to useful pretrained features and reduce overfitting or unstable updates on a small target dataset.

(c) (2 points) Trainable parameters:

$$
60{,}000.
$$

Total parameters:

$$
2{,}400{,}000+60{,}000=2{,}460{,}000.
$$

Percentage trainable:

$$
\frac{60{,}000}{2{,}460{,}000}\times100\%
=\boxed{2.44\%\text{ approximately}}.
$$

## Q2. Self-paced learning and Mean Teacher — 8 points

(a) (3 points) Self-paced learning selects low-loss examples first and gradually includes higher-loss examples by increasing $\lambda$. This can stabilize early optimization. It can fail when high-loss examples are informative or when noisy labels distort the loss-based difficulty estimate.

(b) (3 points)

- $\theta_T$: Teacher parameters.
- $\theta_S$: Student parameters.
- $\mu$: EMA decay factor controlling how strongly the teacher retains its previous parameters.

The EMA smooths teacher updates over multiple student steps, usually producing more stable targets than copying the student's rapidly changing parameters after every update.

(c) (2 points) Consistency regularization encourages student predictions to agree with teacher predictions for differently augmented versions of the same input. The teacher can still make systematic errors, especially early in training or on unfamiliar inputs.

## Q3. Contrastive learning and augmentation — 7 points

(a) (3 points) The two views of the same image form a positive pair and should have low embedding distance. A view from a different image treated as a negative should generally have high distance.

(b) (2 points) A false negative is a pair of different images that share the same semantic class but are treated as negatives. This pushes semantically similar examples apart and may damage representation quality.

(c) (2 points) Horizontal flipping can reverse the anatomical side of an abnormality. If the label depends on left-versus-right orientation, the transformation may alter the correct label and teach an inappropriate invariance.

## Q4. LoRA parameter calculations — 9 points

(a) (2 points)

$$
A\in\mathbb{R}^{16\times1536},
\qquad
B\in\mathbb{R}^{768\times16}.
$$

(b) (2 points)

$$
1536\times768=\boxed{1{,}179{,}648}.
$$

(c) (2 points)

$$
16(1536+768)=\boxed{36{,}864}.
$$

(d) (2 points)

$$
\frac{1{,}179{,}648-36{,}864}
{1{,}179{,}648}\times100\%
=\boxed{96.875\%}.
$$

(e) (1 point) Higher rank increases the adapter's expressive capacity but also its parameter count and computational cost. It may not improve validation accuracy if the additional flexibility is unnecessary or encourages overfitting.

## Q5. Distillation and self-distillation — 7 points

(a) (2 points)

$$
L=(1-\alpha)L_h+\alpha L_d.
$$

When $\alpha=0$,

$$
\boxed{L=L_h}.
$$

Only the ground-truth hard-label loss contributes to the objective.

(b) (2 points) Equal coefficients do not imply equal gradient magnitudes. The losses may have different scales and produce different gradients, so their influence on parameter updates can differ.

(c) (3 points) A student can outperform its teacher by learning useful soft-target information while also benefiting from regularization and optimization. The training process may produce a solution that generalizes better. Improvement is possible but not guaranteed.

## Q6. Meta-learning and task arithmetic — 6 points

(a) (2 points) Reptile moves the meta-initialization toward task-adapted parameters using a first-order update. MAML optimizes post-adaptation performance by differentiating through the inner adaptation steps, often involving second-order derivatives.

(b) (2 points)

$$
\theta_{\text{merged}}=\theta_0+\tau_A+\tau_B.
$$

Merging can hurt when the tasks require conflicting parameter changes, causing interference or degradation on one task.

(c) (2 points) A hypernetwork generates another network's parameters from a task representation or conditioning input. It can share information across tasks and generate task-specific models without independently storing a complete model for every task. Limitations include generation cost and poor generalization to unseen tasks.

## Q7. Experimental interpretation — 6 points

(a) (2 points) Full fine-tuning has the highest accuracy at 91%. The frozen-backbone method has the fewest trainable parameters at 100,000.

(b) (2 points) LoRA satisfies both requirements: its accuracy is 89%, which exceeds 88%, and it has 120,000 trainable parameters, fewer than 150,000.

(c) (2 points) Training speed depends on operations, hardware, memory access, and implementation overhead. Deployment cost also depends on total checkpoint size, inference latency, memory requirements, and whether the pretrained base must be stored.

# Exam 3 — Solutions

## Q1. Transfer learning under limited data — 7 points

(a) (2 points) Start by retaining the pretrained backbone, replacing the classification head, and training the head. This provides useful initial features while limiting the number of trainable parameters and reducing overfitting risk.

(b) (2 points) An advantage of freezing is lower training cost and potentially better generalization on a small dataset. A disadvantage is that the fixed features may not adapt to the new domain.

(c) (3 points) The difference between laboratory and deployment lighting/backgrounds is a domain shift. Possible solutions include fine-tuning on representative deployment-domain images, suitable data augmentation, or unsupervised domain adaptation using unlabelled deployment images.

## Q2. Semi-supervised and self-paced learning — 7 points

(a) (3 points) Train a model on the labelled data, predict labels for unlabelled images, and accept predictions whose confidence exceeds a selected threshold. Retrain on the labelled data plus accepted pseudo-labels. The threshold controls the trade-off between the quantity and reliability of accepted examples.

(b) (2 points) Low-confidence and incorrect predictions can introduce noisy targets, causing the model to reinforce its own mistakes.

(c) (2 points) Pseudo-labeling selects unlabelled examples according to model predictions and their confidence. Self-paced learning typically selects examples according to the current model's loss, starting with easier examples and gradually admitting harder ones.

## Q3. Contrastive learning — 6 points

(a) (2 points) Two independently augmented views of the same original image form a positive pair. The shared origin provides the relationship without needing a human-provided class label.

(b) (2 points) Negative pairs encourage separation between representations of different examples. A false negative occurs when two different images have the same semantic class but are treated as negatives.

(c) (2 points) If one view no longer contains the object, the views may not preserve the same semantic information. Forcing their embeddings together can teach the model an inappropriate invariance.

## Q4. LoRA implementation and deployment — 9 points

(a) (2 points)

$$
512\times1024=\boxed{524{,}288}.
$$

(b) (2 points) LoRA parameters:

$$
8(1024+512)=\boxed{12{,}288}.
$$

Percentage reduction:

$$
\frac{524{,}288-12{,}288}{524{,}288}\times100\%
=\boxed{97.65625\%}.
$$

(c) (2 points) If $B=0$, then $BA=0$. Thus the initial adapted weight is $W+\Delta W=W$, so the layer initially behaves like the pretrained layer.

(d) (2 points) Any two of the following:

- Check that the original weights have `requires_grad=False`.
- Check that adapter factors have `requires_grad=True`.
- Compare copies of the original weights before and after training.
- Inspect optimizer parameter groups to confirm the intended parameters are optimized.

(e) (1 point) The adapter-only checkpoint stores the learned updates, not the complete pretrained model. The base weights are still needed to reconstruct the adapted model.

## Q5. Knowledge distillation and ensembles — 7 points

(a) (2 points) Hard labels identify the correct class. Teacher soft targets also convey relative probabilities among classes, providing information about similarities and alternative predictions.

(b) (2 points) The student may inherit the teacher's systematic rare-class errors through distillation. Combining distillation with ground-truth supervision and evaluating class-specific performance can reduce or reveal this problem.

(c) (3 points) Averaging predictions can reduce variance and smooth individual errors. Diversity matters because models with different errors can compensate for one another; highly correlated errors limit the gain. The principle also applies to regression, where averaging predictions can reduce variance and improve mean squared error when model errors are not perfectly correlated.

## Q6. Meta-learning, model merging, and hypernetworks — 7 points

(a) (2 points) Meta-learning trains an initialization, representation, or adaptation strategy across related tasks. The learned starting point or strategy can then adapt to a new task using only a few labelled examples.

(b) (2 points) A task vector is the parameter difference between a fine-tuned task model and a shared pretrained model:

$$
\tau_A=\theta_A-\theta_0.
$$

Task arithmetic combines such updates, for example:

$$
\theta_{\text{merged}}=\theta_0+\tau_A+\tau_B.
$$

(c) (3 points) Model merging combines existing task-specific updates into a shared parameter set. Its limitation is potential interference between tasks. A hypernetwork generates a model's parameters from task information, allowing task-conditioned parameter generation; its limitations include computational cost and poor generalization to unseen tasks.

## Q7. Integrated method selection — 7 points

(a) (2 points) Use a shared pretrained backbone with small task-specific LoRA adapters. The base model is stored once, and each task requires only its adapter weights, reducing storage compared with keeping six complete fine-tuned models.

(b) (2 points) Meta-learning can learn a shared initialization or adaptation strategy across related tasks, helping the model adapt to each task using few labelled examples.

(c) (2 points) No. Task vectors may interfere when their parameter updates conflict. A merged model must be evaluated on each task because aggregate performance can hide degradation on individual tasks.

(d) (1 point) Evaluate the proposed approach on held-out data for all six tasks, comparing accuracy and resource usage against appropriate baselines, using consistent data splits and evaluation procedures.

## Final revision: high-value facts to memorize

| Topic | Essential exam-ready point |
| --- | --- |
| Transfer learning | Reuse pretrained features; replace the head when target classes differ. |
| Domain adaptation | Source validation does not guarantee deployment performance under distribution shift. |
| Self-paced learning | Select low-loss examples first; gradually increase the loss threshold. |
| Pseudo-labeling | Use confident predictions on unlabelled data; incorrect pseudo-labels can reinforce mistakes. |
| Contrastive learning | Positive pairs are pulled together; negative pairs are pushed apart. |
| Augmentation | Preserve the semantic information relevant to the downstream task. |
| LoRA | $\Delta W=BA$, with $A:r\times d_{\text{in}}$ and $B:d_{\text{out}}\times r$. |
| Distillation | Hard labels provide ground truth; soft targets convey inter-class relationships. |
| Mean Teacher | EMA smooths teacher parameters and provides more stable targets. |
| Ensembles | Averaging can reduce variance; diversity in model errors matters. |
| Meta-learning | Learn how to adapt quickly across a distribution of tasks. |
| Model merging | Combine task vectors, but watch for task interference. |
| Hypernetworks | Generate another network's parameters from task-related inputs. |

Exam strategy: Prioritize the reason behind each method, what it optimizes, its principal trade-off, and the assumptions under which it works. For calculations, practice writing the formula, substituting the values, and reporting the final result with appropriate units or percentages.
