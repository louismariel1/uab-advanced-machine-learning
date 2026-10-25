# TD1 — Advanced Machine Learning Mega Exam Question Pool

## Chunk 1 of 2: Questions Q1–Q30

## Chunk 1 overview

Questions only

Question range

# Q1–Q30

Target coverage

# 6 topics

This is the first half of the master question bank. It contains MCQs, short written answers, calculations, algorithm interpretation, and experimental analysis. No answers or hints are included.

## Topic 1 — Transfer learning and domain adaptation

### Q1. Deployment performance and distribution shift (2 points)

A model achieves 92% accuracy on its validation set but only 71% accuracy after deployment. Which situations could explain this difference? Select all that apply.

- A. The deployment data distribution differs from the training distribution.
- B. The model has overfitted to patterns specific to the training data.
- C. The model uses a smaller learning rate during inference.
- D. The validation set does not adequately represent deployment conditions.
- E. The model has insufficient capacity, which necessarily explains the validation-to-deployment gap.

### Q2. Fine-tuning a pretrained model (2 points)

Which statements about transferring a pretrained model from source task $A$ to target task $B$ are correct? Select all that apply.

- A. Fine-tuning can be more effective when the source and target domains share useful characteristics.
- B. All layers must always be updated during fine-tuning.
- C. The target classification head may need to be replaced if the number of classes changes.
- D. The learning rate must be identical to the one used during pretraining.
- E. Early layers can sometimes be frozen while later layers and the target head are trained.

### Q3. Freezing and feature extraction (3 points)

A pretrained convolutional network is adapted to a small labelled dataset.

(a) Explain the difference between freezing the backbone and fine-tuning the entire backbone. (1 point)

(b) Give one advantage and one disadvantage of freezing the backbone. (1 point)

(c) Explain why a frozen layer still participates in the forward pass. (1 point)

### Q4. Domain adaptation scenarios (3 points)

A model is trained on labelled images collected in sunny weather and deployed on images collected in foggy weather. The target task and class definitions remain the same.

(a) Identify the main distribution-shift problem. (1 point)

(b) Explain why good source-domain validation accuracy may not guarantee good target-domain performance. (1 point)

(c) Describe one strategy that could help adapt the model to the target domain. (1 point)

### Q5. Interpreting a transfer-learning experiment (3 points)

A researcher obtains these results:

| Model | Target validation accuracy | Trainable parameters |
| --- | --- | --- |
| Training from scratch | 74% | 2.0 million |
| Frozen pretrained backbone + new head | 81% | 150,000 |
| Full fine-tuning | 84% | 2.0 million |

(a) Calculate the absolute accuracy improvement of full fine-tuning over training from scratch, in percentage points. (1 point)

(b) Explain one reason why the frozen-backbone model can outperform the model trained from scratch. (1 point)

(c) Give one reason why the researcher might still choose the frozen-backbone model. (1 point)

## Topic 2 — Curriculum learning and semi-supervised learning

### Q6. Self-paced curriculum learning (4 points)

Consider the following algorithm:

1. Initialise a model with parameters $\theta$.
2. At each epoch, calculate a loss for every training example.
3. Select examples whose loss is below a threshold $\lambda$.
4. Update the model using only the selected examples.
5. Gradually increase $\lambda$ during training.

(a) Identify the learning strategy. (1 point)

(b) Explain how the loss is used as an estimate of example difficulty. (1 point)

(c) Explain the effect of gradually increasing $\lambda$. (1 point)

(d) Give one possible failure mode of this strategy. (1 point)

### Q7. Curriculum learning versus random sampling (2 points)

Which statements are correct? Select all that apply.

- A. Curriculum learning can introduce easier examples before more difficult ones.
- B. A curriculum guarantees better generalisation than random sampling.
- C. Self-paced learning can use the current model's losses to determine which examples to train on.
- D. Examples with high loss are always incorrectly labelled.
- E. The choice of curriculum can affect the optimisation trajectory.

### Q8. Semi-supervised learning (3 points)

A dataset contains 1,000 labelled images and 20,000 unlabelled images from the same target domain.

(a) Explain the central idea of semi-supervised learning in this setting. (1 point)

(b) Describe how pseudo-labeling can use the unlabelled examples. (1 point)

(c) Explain why incorrect pseudo-labels can damage training. (1 point)

### Q9. Consistency regularisation and the mean teacher (4 points)

A student model receives an augmented input, while a teacher model provides a target prediction. The teacher parameters are updated as an exponential moving average of the student's parameters.

(a) Identify the learning approach. (1 point)

(b) Explain why the teacher's parameters are updated using an exponential moving average rather than copied directly after every student update. (1 point)

(c) Explain how consistency between predictions can provide a training signal for unlabelled data. (1 point)

(d) State one reason why the teacher's predictions may still be unreliable. (1 point)

### Q10. Analyse a semi-supervised experiment (3 points)

A researcher compares three methods:

| Method | Labelled training examples | Unlabelled training examples | Validation accuracy |
| --- | --- | --- | --- |
| Supervised baseline | 500 | 0 | 68% |
| Pseudo-labeling | 500 | 5,000 | 73% |
| Pseudo-labeling | 500 | 5,000 | 64% |

The two pseudo-labeling runs differ in their confidence thresholds.

(a) State the absolute improvement of the better pseudo-labeling run over the supervised baseline. (1 point)

(b) Explain one possible reason why changing the confidence threshold could produce substantially different results. (1 point)

(c) Propose one controlled experiment to investigate the cause of the performance difference. (1 point)

## Topic 3 — Self-supervised learning, contrastive learning and augmentation

### Q11. Identify a self-supervised pretext task (2 points)

A model receives unlabelled images. Each image is randomly rotated by $0^\circ$, $90^\circ$, $180^\circ$, or $270^\circ$. The model predicts which rotation was applied.

Which statements are correct? Select all that apply.

- A. The rotation labels can be generated automatically.
- B. The model must receive human-provided object-class labels.
- C. The task can encourage the backbone to learn visual features.
- D. The pretext task's four output classes must be retained for every downstream classification task.
- E. Strong performance on the pretext task guarantees strong downstream accuracy.

### Q12. The self-supervised learning pipeline (3 points)

A researcher pretrains a backbone on unlabelled CIFAR-10 images using rotation prediction and then adapts it to a five-class CIFAR-100 classification task.

(a) Describe the pretraining objective. (1 point)

(b) Explain which weights are retained and which output layer is replaced during transfer. (1 point)

(c) Explain why the source and target datasets can use different class labels. (1 point)

### Q13. Contrastive learning and positive pairs (3 points)

A contrastive-learning algorithm creates two independently augmented views of the same image and embeddings $z_i$ and $z_j$.

(a) Explain why the two views are treated as a positive pair. (1 point)

(b) Explain the role of negative pairs in a standard contrastive objective. (1 point)

(c) State one limitation of treating every view from a different image as a negative example. (1 point)

### Q14. Interpreting a contrastive distance matrix (3 points)

Four images are passed through an encoder:

- $a$: original image A
- $a^*$: augmented image A
- $b$: original image B
- $b^*$: augmented image B

Assume A and B are different images and the method uses a standard positive-pair/negative-pair objective.

For each pair below, state whether the model should generally learn a HIGH or LOW embedding distance.

(a) $a$ and $a^*$. (1 point)

(b) $b$ and $b^*$. (1 point)

(c) $a^*$ and $b^*$. (1 point)

### Q15. Data augmentation and representation quality (3 points)

A researcher uses random crops, colour jitter, and horizontal flips during contrastive pretraining.

(a) Explain why two augmented views of the same image can help the model learn robust representations. (1 point)

(b) Explain why an augmentation that removes the main object from an image can harm representation learning. (1 point)

(c) Give one reason why the appropriate augmentation strategy depends on the downstream task. (1 point)

## Topic 4 — Model compression and parameter-efficient fine-tuning

### Q16. LoRA fundamentals (2 points)

Which statements about Low-Rank Adaptation (LoRA) are correct? Select all that apply.

- A. LoRA freezes the original weights of adapted layers and learns a low-rank update.
- B. LoRA guarantees higher validation accuracy than full fine-tuning.
- C. LoRA can substantially reduce the number of trainable parameters.
- D. The complete model no longer needs the original pretrained weights.
- E. LoRA can be applied selectively to some linear layers.

### Q17. LoRA factorisation and dimensions (3 points)

A linear layer has weight matrix

$$
W\in\mathbb{R}^{d_{\text{out}}\times d_{\text{in}}}.
$$

Its effective weight is

$$
W_{\text{eff}}=W+BA,
$$

where $A$ and $B$ are low-rank factors.

(a) State the dimensions of $A$ and $B$ for rank $r$. (1 point)

(b) Explain why $BA$ has the same dimensions as $W$. (1 point)

(c) Explain the purpose of setting $B=0$ at initialisation while initialising $A$ randomly. (1 point)

### Q18. LoRA parameter-count calculation (4 points)

A fully connected layer has 2,048 input features and 1,024 output features. A LoRA adapter uses rank $r=8$. Ignore biases.

(a) Calculate the number of weights in the original matrix. (1 point)

(b) Calculate the number of trainable adapter parameters. (1 point)

(c) Calculate the percentage reduction in trainable parameters compared with training the full matrix. (1 point)

(d) Explain one trade-off associated with increasing the rank. (1 point)

### Q19. Diagnose a LoRA implementation (3 points)

A student intends to use LoRA but observes that the original adapted-layer weights change during training.

(a) Explain why this indicates a potential implementation problem. (1 point)

(b) Name two checks that could help identify the cause. (1 point)

(c) Explain why reasonable validation accuracy does not establish that the intended LoRA experiment was implemented correctly. (1 point)

### Q20. Model compression and deployment trade-offs (3 points)

A researcher compares two adapted models:

| Metric | Dense fine-tuning | LoRA |
| --- | --- | --- |
| Trainable parameters | 1,000,000 | 60,000 |
| Training time | 45 seconds | 58 seconds |
| Full checkpoint size | 4.0 MB | 4.2 MB |
| Validation accuracy | 86% | 85% |

(a) Calculate the percentage reduction in trainable parameters. (1 point)

(b) Explain why LoRA can take longer to train despite using fewer trainable parameters. (1 point)

(c) Explain why the full LoRA checkpoint might be larger than the dense checkpoint. (1 point)

## Topic 5 — Knowledge distillation, ensembles and mean teacher

### Q21. Knowledge distillation fundamentals (2 points)

Which statements about teacher–student knowledge distillation are correct? Select all that apply.

- A. The student can learn from both ground-truth labels and teacher predictions.
- B. The student must have exactly the same architecture as the teacher.
- C. The teacher is normally kept fixed during student training.
- D. The student is guaranteed to outperform the teacher.
- E. The student may inherit biases or weaknesses of the teacher.

### Q22. Temperature-scaled softmax (3 points)

The teacher produces logits

$$
z=[5,3,1].
$$

Its probabilities are computed using softmax with temperature $T$.

(a) Explain the effect of increasing $T$ on the probability distribution. (1 point)

(b) Explain why softer probabilities can provide more information than a one-hot class label. (1 point)

(c) Describe the limiting behaviour of the distribution as $T$ becomes very large. (1 point)

### Q23. Distillation loss and optimisation (4 points)

A student is trained with the objective

$$
L=\alpha L_h+(1-\alpha)L_d,
$$

where $L_h$ is cross-entropy against ground-truth labels and $L_d$ is a temperature-scaled distillation loss.

(a) Explain what information each loss provides. (2 points)

(b) Describe the training objective when $\alpha=1$. (1 point)

(c) Explain why $\alpha=0.5$ does not necessarily mean both losses have equal influence on the parameter updates. (1 point)

### Q24. Knowledge distillation with different output dimensions (3 points)

A teacher produces 100 output logits, while a student produces 10 output logits for a target task.

(a) Explain why directly calculating KL divergence between the teacher and student output distributions is not straightforward. (1 point)

(b) Describe one approach that could make distillation possible. (1 point)

(c) Explain why the teacher and student do not need the same number of layers for distillation to work. (1 point)

### Q25. Ensembles and self-distillation (3 points)

Five neural networks with the same architecture are trained independently on the same task. Their individual test accuracies are around 80%, while averaging their predictions achieves 85%.

(a) Explain how averaging predictions can improve performance. (1 point)

(b) Explain why diversity between model errors matters. (1 point)

(c) Explain how a student with the same architecture as its teacher could theoretically outperform the teacher. (1 point)

## Topic 6 — Meta-learning, model merging and hypernetworks

### Q26. Few-shot meta-learning (3 points)

A model is trained across many related tasks so that it can adapt to a new task using only a few labelled examples.

(a) Explain the purpose of meta-learning in this setting. (1 point)

(b) Distinguish between ordinary training on a single task and training across a distribution of tasks. (1 point)

(c) Give one reason why meta-learning may fail to transfer effectively to a new task. (1 point)

### Q27. Reptile update rule (3 points)

Reptile uses the update

$$
\theta_M\leftarrow\theta_M+\beta(\theta_j-\theta_M),
$$

where $\theta_M$ represents the meta-parameters and $\theta_j$ represents the parameters obtained after adapting to a sampled task.

(a) Explain the roles of $\theta_M$ and $\theta_j$. (1 point)

(b) Describe what happens when $\beta=1$. (1 point)

(c) Explain the main conceptual difference between Reptile and MAML. (1 point)

### Q28. Model merging and task arithmetic (3 points)

Two models share the same architecture and parameter ordering. Model A is fine-tuned on task A, while model B is fine-tuned on task B from the same pretrained initialisation.

(a) Define a task vector for each fine-tuned model. (1 point)

(b) Describe how task arithmetic can combine task vectors to form a merged model. (1 point)

(c) Give one reason why simply adding two task vectors might produce poor results. (1 point)

### Q29. Hypernetworks (3 points)

A hypernetwork receives a task representation and generates the parameters of another neural network.

(a) Explain what the hypernetwork learns to produce. (1 point)

(b) Describe one advantage of generating task-specific parameters rather than training a separate full model for every task. (1 point)

(c) State one limitation or challenge of this approach. (1 point)

### Q30. Integrated question: choosing an adaptation strategy (4 points)

A research team must adapt a pretrained model to several related target tasks. Each task has limited labelled data. The team has the following priorities:

- Preserve the original pretrained model.
- Minimise the number of trainable parameters per task.
- Support rapid adaptation to new tasks.
- Avoid storing a separate complete model for every task.

Compare LoRA, meta-learning, and model merging for this scenario.

(a) Explain which priority LoRA directly addresses and one limitation of LoRA. (1 point)

(b) Explain how meta-learning could support rapid adaptation to new tasks. (1 point)

(c) Explain how model merging differs from maintaining separate task-specific models. (1 point)

(d) Recommend one approach or combination of approaches and justify your choice. (1 point)

## End of TD1 — Chunk 1

## Progress: 30 of 60 questions

Completed: Q1–Q30. Remaining: Q31–Q60.

As requested, I will stop here. No solutions have been revealed.

When you're ready, confirm with “Generate chunk 2”, and I'll continue with Q31–Q60, covering continual learning, explainability, adversarial machine learning, privacy-preserving machine learning, and additional integrated exam questions.
