Absolutely. I’ll structure it as a **GitHub-compatible Markdown file**, with the questions covering the full lecture material and with the **answer and explanation immediately after each question** so it can also be used for revision or teaching.

 # Curriculum Learning & Semi-Supervised Learning — Comprehensive MCQ

 > **Purpose:** Comprehensive exam-preparation questions covering Curriculum Learning, Self-Training, Co-Training, Pseudo-Labels, Consistency Regularisation, UDA, Temporal Ensembling, Mean Teacher, Model Stability, Active Learning, and the relationships between these approaches.
>
>  **Format:** Each question has one best answer, followed by the answer and an explanation.

---

 # Part 1 — Curriculum Learning

 ## Q1. What is the central idea of Curriculum Learning?

 A. Train a model only on the largest available dataset.

 B. Train a model using examples ordered according to their difficulty or quality.

 C. Train a model without labels.

 D. Train multiple models and average their predictions.

 **Answer: B**

 **Explanation:**\
 Curriculum Learning introduces training examples in a strategically chosen order, typically from easier or higher-quality examples to harder or noisier ones. The goal is to improve training dynamics and potentially produce a better final model.

---

 ## Q2. What is the analogy behind Curriculum Learning?

 A. A teacher first gives students the hardest problems.

 B. A teacher randomly selects topics throughout a course.

 C. A teacher typically introduces simpler concepts before more difficult ones.

 D. A teacher removes difficult topics from the curriculum permanently.

 **Answer: C**

 **Explanation:**\
 Curriculum Learning is inspired by the way humans often learn: foundational or easy concepts are introduced first, followed by increasingly difficult material.

---

 ## Q3. What does "sample difficulty" refer to in Curriculum Learning?

 A. The physical size of an input.

 B. How challenging an example is for the model to learn correctly.

 C. The number of classes in the dataset.

 D. The amount of memory required to store an example.

 **Answer: B**

 **Explanation:**\
 Difficulty describes how challenging an example is for the learner. Depending on the problem, difficulty can be estimated from loss, model confidence, prediction uncertainty, complexity of the input, or other measures.

---

 ## Q4. Why might starting with difficult examples be problematic?

 A. The model cannot perform gradient descent.

 B. Difficult examples may provide noisy or unreliable learning signals early in training.

 C. Difficult examples always contain incorrect labels.

 D. Difficult examples cannot be represented by neural networks.

 **Answer: B**

 **Explanation:**\
 At the beginning of training, the model is poorly calibrated and may struggle with difficult examples. Training heavily on them can lead to unstable optimisation or poor representations.

---

 ## Q5. Which statement best describes the relationship between Curriculum Learning and sample quality?

 A. Curriculum Learning can exploit knowledge about the quality of training examples.

 B. Curriculum Learning requires every example to have a perfect label.

 C. Curriculum Learning ignores sample quality.

 D. Curriculum Learning is exclusively an unsupervised technique.

 **Answer: A**

 **Explanation:**\
 Curriculum Learning can use information about both **difficulty** and **quality** to determine an appropriate training order.

---

 # Part 2 — Semi-Supervised Learning

 ## Q6. What is Semi-Supervised Learning?

 A. Learning exclusively from labelled data.

 B. Learning exclusively from unlabelled data.

 C. Learning from a combination of labelled and unlabelled data.

 D. Learning without a training objective.

 **Answer: C**

 **Explanation:**\
 Semi-Supervised Learning (SSL) combines a relatively small labelled dataset with a larger unlabelled dataset.

---

 ## Q7. Why is Semi-Supervised Learning useful in practice?

 A. Unlabelled data is often much easier and cheaper to obtain than labelled data.

 B. Labels are always unnecessary.

 C. Neural networks cannot learn from labelled data.

 D. It eliminates the need for evaluation.

 **Answer: A**

 **Explanation:**\
 Labelling data can be expensive because it often requires human experts. In contrast, collecting unlabelled examples can be comparatively cheap. SSL attempts to extract useful information from those unlabelled examples.

---

 ## Q8. Consider a dataset containing 1,000 labelled images and 100,000 unlabelled images. Which approach is most characteristic of Semi-Supervised Learning?

 A. Ignore the 100,000 unlabelled images.

 B. Use only the 1,000 labelled images.

 C. Use the labelled images for supervised learning and exploit the unlabelled images through an unsupervised objective.

 D. Randomly assign labels to all 100,000 images.

 **Answer: C**

 **Explanation:**\
 The labelled data contributes a supervised loss, while the unlabelled data contributes an additional learning signal such as pseudo-labeling or consistency regularisation.

---

 # Part 3 — The Basic Semi-Supervised Objective

 ## Q9. Suppose labelled data are `(x_i, y_i)` and unlabelled data are `x_j`. Which expression represents the general structure of an SSL objective?

 A. `L = supervised loss only`

 B. `L = unsupervised loss only`

 C. `L = supervised loss + unsupervised loss`

 D. `L = supervised loss - unsupervised loss`

 **Answer: C**

 **Explanation:**\
 A common SSL objective is:

```
L = L_supervised + lambda * L_unsupervised
```

 where `lambda` controls the importance of the unsupervised component.

---

 ## Q10. What is the role of the supervised loss?

 A. It teaches the model using ground-truth labels.

 B. It generates random labels.

 C. It prevents the model from seeing labelled data.

 D. It replaces the model's predictions with noise.

 **Answer: A**

 **Explanation:**\
 For labelled examples, the ground-truth target is known. A typical supervised classification loss is cross-entropy:

```
L_supervised = CE(y_i, f(x_i))
```

---

 ## Q11. What is the role of the unsupervised loss?

 A. It necessarily requires human-generated labels.

 B. It extracts a training signal from unlabelled examples.

 C. It removes all unlabelled examples.

 D. It guarantees perfect predictions.

 **Answer: B**

 **Explanation:**\
 The challenge of SSL is to obtain useful information from unlabelled data. Methods such as self-training and consistency regularisation define different forms of unsupervised loss.

---

 # Part 4 — Self-Training

 ## Q12. What is Self-Training?

 A. A model learns only from human labels.

 B. A model generates predictions for unlabelled data and uses selected predictions as pseudo-labels.

 C. Two models are trained independently without interaction.

 D. A model discards its own predictions.

 **Answer: B**

 **Explanation:**\
 Self-training allows a model to act as its own teacher. It predicts labels for unlabelled examples, and those predictions can subsequently be used as training targets.

---

 ## Q13. What is a pseudo-label?

 A. A human-verified label.

 B. A randomly generated label.

 C. A model-generated label assigned to an unlabelled example.

 D. A label that has been removed from the dataset.

 **Answer: C**

 **Explanation:**\
 A pseudo-label is a model's prediction that is treated as a target for an originally unlabelled example.

---

 ## Q14. Why should pseudo-labels often be selected based on confidence?

 A. Low-confidence predictions are always correct.

 B. High-confidence predictions are generally more likely to be correct.

 C. Confidence has no relationship to prediction quality.

 D. Confidence prevents gradient descent.

 **Answer: B**

 **Explanation:**\
 A model's incorrect predictions can reinforce themselves if they are used as labels. Selecting high-confidence predictions reduces the risk of propagating incorrect pseudo-labels.

---

 ## Q15. What is the main danger of naive self-training?

 A. It cannot use unlabelled data.

 B. It can reinforce its own mistakes.

 C. It always requires two models.

 D. It cannot perform classification.

 **Answer: B**

 **Explanation:**\
 This is the **confirmation bias** problem. If the model makes an incorrect prediction and then trains on that prediction as if it were true, the model may become increasingly confident in the mistake.

---

 ## Q16. Which sequence best describes self-training?

 A. Label → discard → predict

 B. Predict → select pseudo-labels → train using pseudo-labels

 C. Randomise → delete → train

 D. Train teacher → never use predictions

 **Answer: B**

 **Explanation:**\
 The basic loop is:

```
Unlabelled example
       ↓
Model prediction
       ↓
Select reliable prediction
       ↓
Pseudo-label
       ↓
Train model
       ↓
Improved predictions
```

---

 # Part 5 — Online Self-Training

 ## Q17. What does "online self-training" mean?

 A. Pseudo-labels are generated dynamically during training.

 B. Pseudo-labels are generated only once before training.

 C. No predictions are generated.

 D. Only labelled data is processed online.

 **Answer: A**

 **Explanation:**\
 In online self-training, pseudo-labels can be generated from the current model during training rather than being permanently generated beforehand.

---

 ## Q18. For an unlabelled example `x_j`, how can a pseudo-label be constructed?

 A. By choosing the class with the highest predicted probability.

 B. By choosing a random class.

 C. By choosing the least probable class.

 D. By averaging the input pixels.

 **Answer: A**

 **Explanation:**\
 A common pseudo-label is:

```
y_j = argmax_k f_k(x_j)
```

 where `f_k(x_j)` is the predicted probability for class `k`.

---

 # Part 6 — Co-Training

 ## Q19. What is the key idea behind Co-Training?

 A. One model teaches itself exclusively.

 B. Two models train each other using their predictions.

 C. No model uses unlabelled data.

 D. The same model is copied without modification.

 **Answer: B**

 **Explanation:**\
 Co-Training uses two models, often based on different views or feature subsets of the data. Each model can provide pseudo-labels for the other.

---

 ## Q20. Why can two models be more useful than one in Co-Training?

 A. They are guaranteed to have identical errors.

 B. Their different views can lead to different and potentially independent errors.

 C. They never make mistakes.

 D. They eliminate the need for labels.

 **Answer: B**

 **Explanation:**\
 A key assumption is that the models can provide complementary information. If they make sufficiently different errors, one model's confident predictions can help train the other.

---

 ## Q21. What is an important stability advantage of Co-Training?

 A. It can reduce dependence on a single model's potentially incorrect predictions.

 B. It guarantees perfect pseudo-labels.

 C. It eliminates model uncertainty.

 D. It prevents all forms of overfitting.

 **Answer: A**

 **Explanation:**\
 Naive self-training can suffer from confirmation bias. Co-Training introduces another model that can provide an alternative source of pseudo-labels.

---

 # Part 7 — Consistency Regularisation

 ## Q22. What is the central idea of Consistency Regularisation?

 A. The model should produce completely different predictions for augmented versions of the same input.

 B. The model should produce similar predictions for the same input under perturbations or augmentations.

 C. The model should ignore unlabelled data.

 D. The model should memorise every augmentation.

 **Answer: B**

 **Explanation:**\
 If two inputs are different views of the same underlying example, their predictions should ideally be consistent.

---

 ## Q23. Why is consistency useful for unlabelled data?

 A. It provides a learning signal without requiring a ground-truth label.

 B. It creates human annotations automatically.

 C. It guarantees that every prediction is correct.

 D. It removes the need for a model.

 **Answer: A**

 **Explanation:**\
 For an unlabelled example `x`, we may create a perturbed version `x + eta` and require:

```
f(x) ≈ f(x + eta)
```

 This provides an unsupervised constraint.

---

 ## Q24. Which assumption is fundamental to Consistency Regularisation?

 A. Small meaningful perturbations should not change the semantic class.

 B. Every augmentation should change the class.

 C. Noise should always be amplified.

 D. Labels should be randomised.

 **Answer: A**

 **Explanation:**\
 The method assumes that the perturbation preserves the underlying meaning of the example. For example, a small image transformation should generally not change the object's class.

---

 ## Q25. What problem does consistency regularisation address in Semi-Supervised Learning?

 A. It provides stability against perturbations and helps exploit unlabelled data.

 B. It removes all model parameters.

 C. It replaces the supervised objective entirely.

 D. It guarantees zero training loss.

 **Answer: A**

 **Explanation:**\
 Consistency regularisation encourages smooth, stable predictions around the observed data and provides an unsupervised learning signal.

---

 # Part 8 — The Π-Model

 ## Q26. What is the Π-model?

 A. An early implementation of consistency regularisation.

 B. A supervised-only algorithm.

 C. An optimisation algorithm unrelated to SSL.

 D. A method that removes augmentation.

 **Answer: A**

 **Explanation:**\
 The Π-model, associated with Laine and Aila (2016), is an early consistency-based semi-supervised learning method.

---

 ## Q27. What happens to an input in the Π-model?

 A. It is always discarded.

 B. It can be passed through the model with stochastic perturbations, and predictions are encouraged to agree.

 C. It receives a random label.

 D. It is always manually labelled.

 **Answer: B**

 **Explanation:**\
 The model uses stochasticity or noise to obtain different predictions for the same underlying example and encourages consistency between them.

---

 ## Q28. How are labelled examples handled in a consistency-based SSL method?

 A. Only the consistency loss is used.

 B. They can receive the supervised loss based on their ground-truth labels.

 C. They are ignored.

 D. Their labels are replaced by random pseudo-labels.

 **Answer: B**

 **Explanation:**\
 Labelled examples can contribute the normal supervised cross-entropy loss. Unlabelled examples contribute the consistency-based unsupervised loss.

---

 # Part 9 — Unsupervised Data Augmentation (UDA)

 ## Q29. What does UDA stand for?

 A. Universal Data Averaging

 B. Unsupervised Data Augmentation

 C. Unlabelled Dataset Analysis

 D. Unified Deep Architecture

 **Answer: B**

 **Explanation:**\
 UDA stands for **Unsupervised Data Augmentation**.

---

 ## Q30. What is the basic idea of UDA?

 A. Apply different augmentations and encourage consistent predictions.

 B. Remove all augmentation.

 C. Use only labelled examples.

 D. Generate random ground-truth labels.

 **Answer: A**

 **Explanation:**\
 UDA performs prediction on an original or weakly perturbed example and on a heavily augmented version, then encourages their predictions to agree.

---

 ## Q31. Why can strong augmentation be useful in UDA?

 A. It forces the model to learn features that are robust to meaningful transformations.

 B. It guarantees that every prediction is correct.

 C. It eliminates the need for optimisation.

 D. It ensures that augmented images belong to different classes.

 **Answer: A**

 **Explanation:**\
 Strong augmentation creates challenging views of an example. If the semantic class remains unchanged, the model should maintain a consistent prediction.

---

 ## Q32. How can UDA use confidence on unlabelled examples?

 A. It can apply the consistency loss only when the prediction on the original example is sufficiently confident.

 B. It always uses the least confident prediction.

 C. It ignores prediction confidence.

 D. It replaces the model with a random classifier.

 **Answer: A**

 **Explanation:**\
 Confidence filtering reduces the risk of forcing the model to match unreliable targets.

---

 ## Q33. Why is confidence filtering important?

 A. An unreliable prediction can become an unreliable training target.

 B. Confidence always reduces accuracy.

 C. It prevents the use of labelled data.

 D. It guarantees perfect calibration.

 **Answer: A**

 **Explanation:**\
 Consistency training depends on one prediction serving as a target. If that target is highly uncertain or wrong, the model may reinforce an incorrect belief.

---

 # Part 10 — Model Stability

 ## Q34. Why is stability important in Semi-Supervised Learning?

 A. Because the model can otherwise reinforce errors through its own predictions.

 B. Because neural networks cannot process unlabelled data.

 C. Because supervised learning is impossible.

 D. Because all pseudo-labels are guaranteed to be wrong.

 **Answer: A**

 **Explanation:**\
 The central difficulty is that the model generates part of its own training signal. Without stabilisation, errors can propagate and amplify.

---

 ## Q35. What is confirmation bias in Self-Training?

 A. The model becomes increasingly confident in predictions that it generated itself, even if those predictions are wrong.

 B. The model always predicts the correct class.

 C. The model ignores its predictions.

 D. The model only trains on human labels.

 **Answer: A**

 **Explanation:**\
 A model can repeatedly train on its own mistakes. This creates a feedback loop:

```
wrong prediction
      ↓
pseudo-label
      ↓
training
      ↓
stronger wrong prediction
      ↓
more confident pseudo-label
```

---

 ## Q36. What is mode collapse in the context of naive self-training?

 A. The model may converge toward predicting a small subset of classes or a degenerate solution.

 B. The model uses too many classes.

 C. The model becomes perfectly calibrated.

 D. The model stops using gradient descent.

 **Answer: A**

 **Explanation:**\
 If pseudo-labels become systematically biased, the model can reinforce that bias and collapse toward a poor prediction distribution.

---

 ## Q37. Which of the following is a stability mechanism?

 A. Co-Training

 B. Consistency Regularisation

 C. Temporal Ensembling

 D. Mean Teacher

 E. All of the above

 **Answer: E**

 **Explanation:**\
 All four approaches introduce mechanisms intended to reduce the instability and confirmation bias associated with naive self-training.

---

 # Part 11 — Temporal Ensembling

 ## Q38. What is Temporal Ensembling?

 A. Averaging predictions for an example over previous training times.

 B. Averaging all input features.

 C. Training only once.

 D. Removing historical predictions.

 **Answer: A**

 **Explanation:**\
 Temporal Ensembling maintains an averaged prediction for each training example using predictions obtained at earlier points in training.

---

 ## Q39. What problem does Temporal Ensembling attempt to solve?

 A. The current model prediction may be noisy or unreliable.

 B. The dataset contains too many labels.

 C. Neural networks cannot calculate probabilities.

 D. The model cannot perform augmentation.

 **Answer: A**

 **Explanation:**\
 Instead of trusting one potentially noisy prediction, Temporal Ensembling uses a smoothed history of predictions.

---

 ## Q40. What is the update rule for the temporal ensemble?

 A.

```
z_tilde_i = alpha * z_tilde_(t-1) + (1-alpha) * z_t
```

 B.

```
z_tilde_i = z_t + z_(t-1)
```

 C.

```
z_tilde_i = z_t / alpha
```

 D.

```
z_tilde_i = alpha - z_t
```

 **Answer: A**

 **Explanation:**\
 The exponential moving average (EMA) update is:

```
z_tilde_i(t) =
    alpha * z_tilde_i(t-1)
    + (1-alpha) * z_i(t)
```

 The old estimate receives weight `alpha`, while the current prediction receives weight `1-alpha`.

---

 ## Q41. What does EMA stand for?

 A. Estimated Model Accuracy

 B. Exponential Moving Average

 C. Ensemble Model Architecture

 D. Expected Model Activation

 **Answer: B**

 **Explanation:**\
 EMA means **Exponential Moving Average**.

---

 ## Q42. Why is the average called "exponential"?

 A. Older observations receive exponentially decreasing effective weights.

 B. The number of parameters grows exponentially.

 C. The input is exponentiated.

 D. The model has an exponential activation function.

 **Answer: A**

 **Explanation:**\
 Repeatedly applying the EMA gives exponentially decreasing influence to older observations.

---

 ## Q43. What is the effect of increasing `alpha` in the EMA?

 A. The moving average relies more heavily on its previous value and changes more slowly.

 B. The moving average becomes completely random.

 C. The current prediction gets all the weight.

 D. The model stops learning.

 **Answer: A**

 **Explanation:**\
 A larger `alpha` means stronger smoothing:

```
new_average = alpha * old_average
            + (1-alpha) * current_value
```

 Therefore, the historical estimate has more influence.

---

 ## Q44. Why does Temporal Ensembling improve stability?

 A. It smooths noisy predictions over time.

 B. It removes all unlabelled data.

 C. It makes every prediction independent.

 D. It prevents all optimisation.

 **Answer: A**

 **Explanation:**\
 Noise in individual predictions is reduced by averaging across multiple training iterations.

---

 # Part 12 — Limitations of Temporal Ensembling

 ## Q45. What is a major scalability problem with Temporal Ensembling?

 A. It requires storing an averaged prediction for every training example.

 B. It requires no memory.

 C. It cannot process labelled examples.

 D. It requires a separate model for every example.

 **Answer: A**

 **Explanation:**\
 For every dataset example, the method needs to maintain a moving-average prediction. This becomes increasingly expensive for very large datasets.

---

 ## Q46. How often does the stored prediction in basic Temporal Ensembling update?

 A. Every example, with no storage.

 B. Once per epoch for each example.

 C. Never.

 D. Only during testing.

 **Answer: B**

 **Explanation:**\
 The stored prediction is updated when the corresponding example is encountered, which in the standard setup means approximately once per epoch.

---

 ## Q47. Why can the once-per-epoch update be problematic?

 A. The target can lag behind the rapidly changing model.

 B. It makes the model infinitely fast.

 C. It eliminates smoothing.

 D. It prevents the use of labels.

 **Answer: A**

 **Explanation:**\
 If the model changes significantly between epochs, the stored target may become outdated.

---

 # Part 13 — Mean Teacher

 ## Q48. What is the key idea of Mean Teacher?

 A. Maintain an EMA of the model parameters to create a teacher model.

 B. Maintain an EMA prediction for every training example.

 C. Train only the teacher.

 D. Randomly initialise a teacher for every example.

 **Answer: A**

 **Explanation:**\
 Mean Teacher replaces the per-example prediction storage of Temporal Ensembling with a **teacher model whose parameters are an EMA of the student model's parameters**.

---

 ## Q49. What are the student and teacher models?

 A. The student is the model being optimised directly; the teacher is an EMA version of the student.

 B. The teacher is trained from scratch independently.

 C. The student never receives gradients.

 D. Both models are always identical.

 **Answer: A**

 **Explanation:**\
 The student parameters another suitable distance between are updated through gradient descent. The teacher parameters are updated through an EMA of the student parameters.

---

 ## Q50. Which expression best represents the Mean Teacher teacher update?

 A.

```
theta_teacher = alpha * theta_teacher_old
              + (1-alpha) * theta_student
```

 B.

```
theta_teacher = theta_student - alpha
```

 C.

```
theta_teacher = theta_student + random_noise
```

 D.

```
theta_teacher = theta_student / alpha
```

 **Answer: A**

 **Explanation:**\
 The teacher is a smoothed version of the student's historical parameters.

---

 ## Q51. Why is Mean Teacher more scalable than Temporal Ensembling?

 A. It stores one teacher model instead of one prediction vector for every dataset example.

 B. It removes all model parameters.

 C. It does not use unlabelled data.

 D. It stores more information for every example.

 **Answer: A**

 **Explanation:**\
 Temporal Ensembling requires per-example state. Mean Teacher requires only the teacher model parameters, making it much more practical for large datasets.

---

 ## Q52. What is the consistency loss in Mean Teacher?

 A.

```
MSE(
    f_student(x + eta),
    f_teacher(x + eta_prime)
)
```

 B.

```
CE(
    random_label,
    f_student(x)
)
```

 C.

```
MSE(x, y)
```

 D.

```
CE(student, teacher_parameters)
```

 **Answer: A**

 **Explanation:**\
 The student and teacher receive potentially different perturbations, and their predictions are encouraged to agree.

---

 ## Q53. Why can the teacher provide a more stable target than the student itself?

 A. The teacher averages information across many historical student states.

 B. The teacher changes more rapidly than the student.

 C. The teacher is randomly initialised every iteration.

 D. The teacher never changes.

 **Answer: A**

 **Explanation:**\
 The EMA acts as a low-pass filter on model updates. Short-term fluctuations in the student have less influence on the teacher.

---

 # Part 14 — Comparing SSL Methods

 ## Q54. Which method uses the model's own prediction as a pseudo-label?

 A. Self-Training

 B. Temporal Ensembling

 C. Mean Teacher

 D. All of the above

 **Answer: D**

 **Explanation:**\
 All these methods ultimately exploit model predictions as training targets, although they stabilise those targets differently.

---

 ## Q55. Which method explicitly uses two models with different views?

 A. Self-Training

 B. Co-Training

 C. UDA

 D. Temporal Ensembling

 **Answer: B**

 **Explanation:**\
 Co-Training uses multiple learners, traditionally with different views of the data, so that one model can provide information to the other.

---

 ## Q56. Which method uses strong data augmentation as a central component?

 A. UDA

 B. Temporal Ensembling

 C. Co-Training

 D. Basic Self-Training

 **Answer: A**

 **Explanation:**\
 UDA explicitly uses augmented versions of unlabelled examples and encourages consistent predictions.

---

 ## Q57. Which method stores a historical prediction for each training example?

 A. Temporal Ensembling

 B. Mean Teacher

 C. Co-Training

 D. UDA

 **Answer: A**

 **Explanation:**\
 Temporal Ensembling maintains a moving-average prediction for each example.

---

 ## Q58. Which method stores an EMA of model parameters?

 A. Mean Teacher

 B. Temporal Ensembling

 C. Basic Self-Training

 D. UDA

 **Answer: A**

 **Explanation:**\
 Mean Teacher maintains a teacher network whose weights are an EMA of the student network's weights.

---

 ## Q59. Which progression best captures the evolution of these ideas?

 A. Self-training → consistency → temporal averaging → model averaging

 B. Mean Teacher → random labels → supervised learning

 C. Co-Training → remove labels → random training

 D. UDA → remove augmentation → self-training

 **Answer: A**

 **Explanation:**\
 The methods can be understood as progressively improving the stability of pseudo-targets:

```
Self-Training
      ↓
Consistency Regularisation
      ↓
Temporal Ensembling
      ↓
Mean Teacher
```

---

 # Part 15 — The SSL Timeline

 ## Q60. What is the basic loss in Self-Training?

 A.

```
L = CE(y_i, f(x_i))
  + CE(y_j, f(x_j))
```

 B.

```
L = MSE(x_i, x_j)
```

 C.

```
L = CE(y_i, x_j)
```

 D.

```
L = 0
```

 **Answer: A**

 **Explanation:**\
 The labelled example contributes supervised cross-entropy. The pseudo-labelled unlabelled example contributes an additional cross-entropy term.

---

 ## Q61. What changes when moving from Self-Training to Consistency Regularisation?

 A. The target becomes a prediction from another perturbed version of the same input rather than necessarily a hard pseudo-label.

 B. Labels become mandatory for every example.

 C. Unlabelled examples are discarded.

 D. The model no longer predicts.

 **Answer: A**

 **Explanation:**\
 Consistency methods can compare predictions directly:

```
CE(
    f(x_j),
    f(x_j + eta)
)
```

 or use another suitable distance between predictions.

---

 ## Q62. What changes when moving from Consistency Regularisation to Temporal Ensembling?

 A. The target can become an EMA of previous predictions.

 B. The target becomes random.

 C. The model stops using augmentation.

 D. The model becomes supervised-only.

 **Answer: A**

 **Explanation:**\
 Temporal Ensembling stabilises the target by averaging predictions from previous training times.

---

 ## Q63. What changes when moving from Temporal Ensembling to Mean Teacher?

 A. Per-example prediction averages are replaced by an EMA teacher model.

 B. The model stops using consistency.

 C. All labels are removed.

 D. The teacher is trained independently from scratch.

 **Answer: A**

 **Explanation:**\
 Mean Teacher transfers the temporal averaging mechanism from predictions to model parameters.

---

 # Part 16 — Mathematical Understanding

 ## Q64. Consider the loss:

```
L = CE(y_i, f_theta(x_i))
  + MSE(f_theta(x_j + eta), f_theta_prime(x_j + eta_prime))
```

 What does the first term represent?

 A. Unsupervised consistency loss

 B. Supervised loss

 C. Teacher update

 D. Data augmentation

 **Answer: B**

 **Explanation:**\
 `y_i` is the ground-truth label, so the first term is the supervised cross-entropy loss.

---

 ## Q65. What does the second term represent?

 A. Supervised classification loss

 B. Unsupervised consistency loss

 C. Data preprocessing

 D. Parameter initialisation

 **Answer: B**

 **Explanation:**\
 The second term forces the student and teacher predictions to be similar for an unlabelled example.

---

 ## Q66. In the Mean Teacher equation, what does `theta` represent?

 A. The input

 B. The ground-truth label

 C. Student model parameters

 D. The learning rate

 **Answer: C**

 **Explanation:**\
 `theta` denotes the parameters of the student model, while `theta'` can denote the EMA teacher parameters.

---

 ## Q67. What does `eta` represent in the consistency objective?

 A. A perturbation or noise applied to an input.

 B. A ground-truth class.

 C. A model parameter.

 D. The dataset size.

 **Answer: A**

 **Explanation:**\
 `eta` denotes a perturbation/noise term. The goal is to make predictions robust to such perturbations.

---

 ## Q68. Why might the student and teacher receive different perturbations?

 A. To encourage invariance across different views of the same underlying example.

 B. To guarantee different class predictions.

 C. To make training random without purpose.

 D. To remove the need for labels.

 **Answer: A**

 **Explanation:**\
 If both perturbed inputs represent the same underlying example, their predictions should agree even though the exact input views differ.

---

 # Part 17 — Confidence and Pseudo-Labels

 ## Q69. A model predicts the following probabilities for an unlabelled image:

```
Cat       0.96
Dog       0.02
Horse     0.02
```

 Would this generally be a good is highly confident in the Cat prediction. Confidence-based pseudo-labeling would typically consider this prediction71\. Suppose a model incorrectly predicts that an unlabelled image is a dog with high confidence. The model uses "dog" as a pseudo-label and trains on80\. A researcher has 10,000 labelled examples and 1 million unlabelled examples. Which approach is most directly motivated by is trained on unlabelled images. For every image, two augmented versions are generated, and the model is encouraged to image twice: once with weak augmentation and once with strong augmentation. The loss penalises disagreement. can consistency regularisation be considered a form of induct labels, while Temporal En sample ground-truth labels while avoiding unreliable valuable structure, but exploiting it requires assumptions or mechanisms that prevent noisy model-generated is the central trade-off in pseudo-label-based gives fewer but potentially cleaner pseudo-labels. A lower threshold gives more pseudo-labels but central conceptual lesson. The methods progressively seek more reliable learning targets from story is about making use of model-generated Revision — The 15 Questions You Must according to properties such as difficulty or quality, often progressing from easier/higher-quality examples to harder110\. Why does Temporal Ensembling improve stability information candidate for confidence-based pseudo-labeling?

 A. Yes

 B. No

 C. Only if Cat is the least probable class

 D. It cannot be used because it is unlabelled

 **Answer: A**

 **Explanation:**\
 The model is highly confident in the Cat prediction. Confidence-based pseudo-labeling would typically consider this prediction reliable enough to use, depending on the chosen threshold.

---

 ## Q70. A model predicts:

```
Cat       0.36
Dog       0.34
Horse     0.30
```

 Why might this example be excluded from pseudo-label training?

 A. The model is uncertain.

 B. The model is perfectly confident.

 C. The example has a human label.

 D. The probabilities sum to more than one.

 **Answer: A**

 **Explanation:**\
 The highest probability is only 0.36, indicating substantial uncertainty. Using this prediction as a target could introduce significant label noise.

---

 # Part 18 — Stability and Confirmation Bias

 ## Q71. Suppose a model incorrectly predicts that an unlabelled image is a dog with high confidence. The model uses "dog" as a pseudo-label and trains on it repeatedly. What can happen?

 A. The model may reinforce the incorrect prediction.

 B. The model is guaranteed to discover the correct label.

 C. The image automatically becomes labelled by a human.

 D. The model stops learning.

 **Answer: A**

 **Explanation:**\
 This is the classic confirmation-bias problem in self-training.

---

 ## Q72. Which strategy directly attempts to make the target more stable by averaging historical model states?

 A. Mean Teacher

 B. Random labelling

 C. Basic Self-Training

 D. Hard negative mining

 **Answer: A**

 **Explanation:**\
 Mean Teacher averages the student's parameters over time to produce a more stable teacher.

---

 ## Q73. Why is "stability" especially important in SSL compared with ordinary supervised learning?

 A. Because the model itself may generate the targets used to train it.

 B. Because supervised models never use gradients.

 C. Because SSL has no objective function.

 D. Because labels cannot be represented numerically.

 **Answer: A**

 **Explanation:**\
 In supervised learning, the target is externally provided. In SSL, targets can be generated by the model itself, creating the possibility of feedback loops.

---

 # Part 19 — Active Learning

 ## Q74. What is Active Learning?

 A. The model strategically selects informative unlabelled examples to be labelled.

 B. The model labels every example randomly.

 C. The model ignores human annotation.

 D. The model trains only on synthetic data.

 **Answer: A**

 **Explanation:**\
 Active Learning aims to reduce labelling costs by selecting the most useful examples for human annotation.

---

 ## Q75. What is the key difference between Semi-Supervised Learning and Active Learning?

 A. SSL exploits unlabelled data directly, whereas Active Learning selects examples to obtain labels.

 B. SSL never uses labels.

 C. Active Learning never uses humans.

 D. They are exactly the same technique.

 **Answer: A**

 **Explanation:**

 **Semi-Supervised Learning:**

```
unlabelled data
      ↓
learn from it without obtaining labels
```

 **Active Learning:**

```
unlabelled data
      ↓
select informative examples
      ↓
ask for human labels
      ↓
train again
```

---

 ## Q76. Why is Active Learning useful when labels are expensive?

 A. It attempts to spend the labelling budget on the most informative examples.

 B. It makes labels unnecessary.

 C. It guarantees zero annotation cost.

 D. It prevents humans from participating.

 **Answer: A**

 **Explanation:**\
 If annotation is expensive, it is useful to ask humans to label examples that are expected to provide the greatest benefit.

---

 # Part 20 — Curriculum vs Semi-Supervised vs Active Learning

 ## Q77. Which statement best describes Curriculum Learning?

 A. It exploits known sample difficulty or quality to influence training.

 B. It selects samples for human annotation.

 C. It exclusively uses unlabelled data.

 D. It requires two models.

 **Answer: A**

 **Explanation:**\
 Curriculum Learning concerns the **order or schedule in which training examples are presented**, often using difficulty or quality.

---

 ## Q78. Which statement best describes Semi-Supervised Learning?

 A. It exploits both labelled and unlabelled data.

 B. It only orders examples by difficulty.

 C. It exclusively requests human labels.

 D. It requires an ensemble of ten models.

 **Answer: A**

 **Explanation:**\
 SSL attempts to extract useful learning signals from both labelled and unlabelled examples.

---

 ## Q79. Which statement best describes Active Learning?

 A. It chooses which unlabelled examples should receive human labels.

 B. It always avoids human labels.

 C. It only changes the order of examples.

 D. It requires pseudo-labels for every example.

 **Answer: A**

 **Explanation:**\
 Active Learning treats labelling as a limited resource and strategically selects examples for annotation.

---

 # Part 21 — Integrated Conceptual Questions

 ## Q80. A researcher has 10,000 labelled examples and 1 million unlabelled examples. Which approach is most directly motivated by this situation?

 A. Semi-Supervised Learning

 B. Active Learning only

 C. Curriculum Learning only

 D. No learning is possible

 **Answer: A**

 **Explanation:**\
 The large amount of unlabelled data makes Semi-Supervised Learning particularly attractive.

---

 ## Q81. A researcher has 100,000 unlabelled examples but can afford to have only 1,000 labelled by humans. Which approach may help decide which examples should receive those labels?

 A. Active Learning

 B. Temporal Ensembling

 C. Mean Teacher

 D. UDA

 **Answer: A**

 **Explanation:**\
 Active Learning selects informative examples for annotation, helping maximise the value of a limited labelling budget.

---

 ## Q82. A researcher knows that some training examples are easy and reliable while others are difficult and noisy. Which approach can exploit this knowledge?

 A. Curriculum Learning

 B. Mean Teacher

 C. UDA only

 D. Co-Training only

 **Answer: A**

 **Explanation:**\
 Curriculum Learning can order examples according to difficulty or quality.

---

 ## Q83. A model is trained on unlabelled images. For every image, two augmented versions are generated, and the model is encouraged to produce the same prediction for both. Which principle is being used?

 A. Consistency Regularisation

 B. Active Learning

 C. Curriculum Learning

 D. Random sampling

 **Answer: A**

 **Explanation:**\
 The method explicitly enforces prediction consistency across perturbations or augmentations.

---

 ## Q84. A model produces pseudo-labels, but the pseudo-labels are becoming increasingly biased toward one class. What problem may be occurring?

 A. Confirmation bias or mode collapse

 B. Perfect calibration

 C. Successful curriculum learning

 D. Active learning

 **Answer: A**

 **Explanation:**\
 Self-training can reinforce its own errors, causing the model to become increasingly biased toward particular predictions.

---

 ## Q85. Which method is particularly designed to stabilise consistency targets using an EMA of model parameters?

 A. Mean Teacher

 B. Basic Self-Training

 C. Active Learning

 D. Curriculum Learning

 **Answer: A**

 **Explanation:**\
 Mean Teacher creates a teacher network using an EMA of the student's parameters.

---

 # Part 22 — Scenario-Based Questions

 ## Q86. Imagine a student learning mathematics. The teacher first gives simple addition problems, then multiplication, then algebra, and finally calculus. Which ML idea does this resemble?

 A. Curriculum Learning

 B. Self-Training

 C. Co-Training

 D. Active Learning

 **Answer: A**

 **Explanation:**\
 The material is ordered from easier concepts toward harder concepts, analogous to Curriculum Learning.

---

 ## Q87. Imagine a classifier sees an unlabelled image and predicts "cat" with 99% confidence. The system adds "cat" as a training target. What technique is this?

 A. Pseudo-labeling / Self-Training

 B. Active Learning

 C. Curriculum Learning

 D. Co-Training

 **Answer: A**

 **Explanation:**\
 The model generates a label for an originally unlabelled example and uses it for training.

---

 ## Q88. Two classifiers look at the same unlabelled data using different feature views. Classifier A confidently labels examples for classifier B, while B does the same for A. What is this?

 A. Co-Training

 B. Temporal Ensembling

 C. UDA

 D. Mean Teacher

 **Answer: A**

 **Explanation:**\
 The defining characteristic is mutual teaching between different learners.

---

 ## Q89. A model predicts an image twice: once with weak augmentation and once with strong augmentation. The loss penalises disagreement. What method does this resemble?

 A. UDA / Consistency Regularisation

 B. Active Learning

 C. Curriculum Learning

 D. Standard supervised learning only

 **Answer: A**

 **Explanation:**\
 UDA is based on enforcing consistency between predictions on original/weakly transformed and strongly augmented inputs.

---

 ## Q90. A teacher model is updated using:

```
theta_teacher =
    alpha * theta_teacher
    + (1-alpha) * theta_student
```

 What is the purpose?

 A. To create a stable moving-average teacher.

 B. To randomly initialise the student.

 C. To remove the teacher.

 D. To create a human label.

 **Answer: A**

 **Explanation:**\
 The teacher changes more smoothly than the student because its parameters are updated through an EMA.

---

 # Part 23 — High-Level Exam Questions

 ## Q91. What is the fundamental difference between pseudo-labeling and consistency regularisation?

 A. Pseudo-labeling treats model predictions as labels, while consistency regularisation encourages predictions to remain stable across perturbations.

 B. They are completely unrelated.

 C. Pseudo-labeling never uses model predictions.

 D. Consistency regularisation requires every example to have a human label.

 **Answer: A**

 **Explanation:**\
 Both exploit model predictions, but they formulate the learning signal differently.

 Pseudo-labeling:

```
prediction → pseudo-label → classification loss
```

 Consistency:

```
prediction on one view ≈ prediction on another view
```

---

 ## Q92. Why can consistency regularisation be considered a form of inductive bias?

 A. It assumes that small or appropriate changes to an input should not change its semantic prediction.

 B. It assumes every input has a different class.

 C. It assumes labels are random.

 D. It makes no assumptions about the data.

 **Answer: A**

 **Explanation:**\
 Consistency regularisation introduces the assumption that semantically equivalent views should produce similar predictions.

---

 ## Q93. Why can an EMA provide a useful stabilising effect?

 A. It reduces the influence of short-term fluctuations.

 B. It amplifies every update.

 C. It removes historical information.

 D. It forces parameters to zero.

 **Answer: A**

 **Explanation:**\
 EMA behaves like a low-pass filter: rapid fluctuations have less influence, while persistent trends remain.

---

 ## Q94. Which statement best describes the relationship between Temporal Ensembling and Mean Teacher?

 A. Both use temporal averaging for stability, but Temporal Ensembling averages predictions while Mean Teacher averages model parameters.

 B. They use completely unrelated ideas.

 C. Mean Teacher averages labels, while Temporal Ensembling averages images.

 D. Neither uses EMA.

 **Answer: A**

 **Explanation:**\
 This is one of the most important conceptual comparisons in the lecture.

```
Temporal Ensembling:
average predictions over time

Mean Teacher:
average model parameters over time
```

---

 ## Q95. Why does Mean Teacher scale better than Temporal Ensembling?

 A. It avoids maintaining a separate prediction history for every dataset example.

 B. It uses no parameters.

 C. It does not need a model.

 D. It removes all unlabelled examples.

 **Answer: A**

 **Explanation:**\
 Mean Teacher stores one additional teacher model rather than one state vector per training example.

---

 # Part 24 — Exam-Level Comparison Table Questions

 ## Q96. Which pairing is correct?

 A. Self-Training → pseudo-labels

 B. Co-Training → two mutually teaching models

 C. UDA → data augmentation and consistency

 D. Mean Teacher → EMA teacher model

 E. All of the above

 **Answer: E**

 **Explanation:**\
 Every pairing correctly captures a defining idea.

---

 ## Q97. Which pairing is INCORRECT?

 A. Temporal Ensembling → EMA predictions

 B. Mean Teacher → EMA model parameters

 C. Active Learning → strategic selection for labelling

 D. Curriculum Learning → exploit sample difficulty

 E. UDA → requires every unlabelled example to have a human label

 **Answer: E**

 **Explanation:**\
 UDA is specifically designed to exploit unlabelled data. It does not require human labels for every unlabelled example.

---

 # Part 25 — Deep Understanding

 ## Q98. Why is Semi-Supervised Learning fundamentally challenging?

 A. Because the model must extract useful information from data without ground-truth labels while avoiding unreliable self-generated targets.

 B. Because neural networks cannot process unlabelled inputs.

 C. Because supervised learning is impossible.

 D. Because unlabelled data has no statistical information.

 **Answer: A**

 **Explanation:**\
 Unlabelled data contains potentially valuable structure, but exploiting it requires assumptions or mechanisms that prevent noisy model-generated signals from becoming harmful.

---

 ## Q99. What is the central trade-off in pseudo-label-based SSL?

 A. More pseudo-labels provide more training data but may introduce more label noise.

 B. Fewer pseudo-labels always produce worse models.

 C. More pseudo-labels always produce perfect models.

 D. Pseudo-labels have no effect on training.

 **Answer: A**

 **Explanation:**\
 A high confidence threshold gives fewer but potentially cleaner pseudo-labels. A lower threshold gives more pseudo-labels but risks introducing more errors.

---

 ## Q100. What is the central conceptual lesson connecting Self-Training, Consistency Regularisation, Temporal Ensembling, and Mean Teacher?

 A. The methods progressively seek more reliable learning targets from unlabelled data.

 B. They all discard unlabelled data.

 C. They are all purely supervised methods.

 D. They have no relationship.

 **Answer: A**

 **Explanation:**\
 The overall story is about making use of model-generated information while controlling its unreliability:

```
Self-Training
    ↓
model generates pseudo-labels

Consistency Regularisation
    ↓
model must agree across perturbations

Temporal Ensembling
    ↓
average predictions over time

Mean Teacher
    ↓
average model parameters to create a stable teacher
```

 The common objective is to exploit unlabelled data **without allowing unstable or incorrect predictions to dominate learning**.

---

 # Final Revision — The 15 Questions You Must Be Able to Answer

 If you are preparing for an exam, make sure you can answer these questions without looking at the notes.

 ## Q101. What is Curriculum Learning?

 **Answer:** Training examples are organised according to properties such as difficulty or quality, often progressing from easier/higher-quality examples to harder/noisier ones.

---

 ## Q102. What is Semi-Supervised Learning?

 **Answer:** Learning using both labelled and unlabelled data.

---

 ## Q103. What is Self-Training?

 **Answer:** A model predicts labels for unlabelled examples and uses selected predictions as pseudo-labels for further training.

---

 ## Q104. What is the main problem with naive Self-Training?

 **Answer:** Confirmation bias: the model can reinforce its own incorrect predictions.

---

 ## Q105. What is Co-Training?

 **Answer:** Two models, typically using different views of the data, teach each other using confident predictions.

---

 ## Q106. What is Consistency Regularisation?

 **Answer:** It encourages the model to produce similar predictions for different perturbations or augmentations of the same input.

---

 ## Q107. What is UDA?

 **Answer:** Unsupervised Data Augmentation: use augmented unlabelled examples and enforce prediction consistency, often using confidence filtering.

---

 ## Q108. What is Temporal Ensembling?

 **Answer:** Maintain an EMA of each example's predictions over training time and use the averaged prediction as a more stable target.

---

 ## Q109. What is the EMA equation?

 **Answer:**

```
z_tilde_t = alpha * z_tilde_(t-1)
          + (1-alpha) * z_t
```

---

 ## Q110. Why does Temporal Ensembling improve stability?

 **Answer:** It smooths noisy predictions by averaging information from previous training iterations.

---

 ## Q111. What are the two main problems with Temporal Ensembling?

 **Answer:**

 1. It requires storing a prediction for every example.
2. Predictions update only when examples are revisited, causing the target to lag behind the current model.

---

 ## Q112. What is Mean Teacher?

 **Answer:** A consistency-based SSL method in which the teacher model is an EMA of the student model's parameters.

---

 ## Q113. What is the key difference between Temporal Ensembling and Mean Teacher?

 **Answer:**

```
Temporal Ensembling:
EMA of predictions

Mean Teacher:
EMA of model parameters
```

---

 ## Q114. Why is Mean Teacher more scalable?

 **Answer:** It stores one teacher model rather than a historical prediction for every training example.

---

 ## Q115. What is Active Learning?

 **Answer:** Strategically selecting informative unlabelled examples and obtaining human labels for them.

---

 # One-Minute Exam Summary

 The entire topic can be remembered as the following chain:

```
CURRICULUM LEARNING
        |
        | organise examples by difficulty/quality
        v
SEMI-SUPERVISED LEARNING
        |
        | use labelled + unlabelled data
        v
SELF-TRAINING
        |
        | model generates pseudo-labels
        v
PROBLEM: CONFIRMATION BIAS
        |
        | model can reinforce its own mistakes
        v
STABILITY MECHANISMS
        |
        +------------------+
        |                  |
        v                  v
CO-TRAINING       CONSISTENCY REGULARISATION
two models        predictions should agree
teach each other  under perturbations
                           |
                           v
                  TEMPORAL ENSEMBLING
                  EMA of predictions
                           |
                           | scalability problem
                           v
                     MEAN TEACHER
                     EMA of model
                     parameters
                           |
                           v
                  stable teacher target
```

 The overarching exam principle is:

 > **Semi-Supervised Learning tries to exploit unlabelled data, but because the model often has to generate its own learning targets, stability is crucial.**

 The major stabilisation progression is:

```
Self-Training
    ↓
Consistency Regularisation
    ↓
Temporal Ensembling
    ↓
Mean Teacher
```

 And the three related learning paradigms are:

```
Curriculum Learning
    → When/how should examples be presented?

Semi-Supervised Learning
    → How can we learn from labelled + unlabelled data?

Active Learning
    → Which unlabelled examples should we pay to label?
```

---

 # Answer Key

 | Question | Answer | Question | Answer |
| --- | --- | --- | --- |
| 1 | B | 51 | A |
| 2 | C | 52 | A |
| 3 | B | 53 | A |
| 4 | B | 54 | D |
| 5 | A | 55 | B |
| 6 | C | 56 | A |
| 7 | A | 57 | A |
| 8 | C | 58 | A |
| 9 | C | 59 | A |
| 10 | A | 60 | A |
| 11 | B | 61 | A |
| 12 | B | 62 | A |
| 13 | C | 63 | A |
| 14 | B | 64 | B |
| 15 | B | 65 | B |
| 16 | B | 66 | C |
| 17 | A | 67 | A |
| 18 | A | 68 | A |
| 19 | B | 69 | A |
| 20 | B | 70 | A |
| 21 | A | 71 | A |
| 22 | B | 72 | A |
| 23 | A | 73 | A |
| 24 | A | 74 | A |
| 25 | A | 75 | A |
| 26 | A | 76 | A |
| 27 | B | 77 | A |
| 28 | B | 78 | A |
| 29 | B | 79 | A |
| 30 | A | 80 | A |
| 31 | A | 81 | A |
| 32 | A | 82 | A |
| 33 | A | 83 | A |
| 34 | A | 84 | A |
| 35 | A | 85 | A |
| 36 | A | 86 | A |
| 37 | E | 87 | A |
| 38 | A | 88 | A |
| 39 | A | 89 | A |
| 40 | A | 90 | A |
| 41 | B | 91 | A |
| 42 | A | 92 | A |
| 43 | A | 93 | A |
| 44 | A | 94 | A |
| 45 | A | 95 | A |
| 46 | B | 96 | E |
| 47 | A | 97 | E |
| 48 | A | 98 | A |
| 49 | A | 99 | A |
| 50 | A | 100 | A |
| 101 | — | 111 | — |
| 102 | — | 112 | — |
| 103 | — | 113 | — |
| 104 | — | 114 | — |
| 105 | — | 115 | — |

This version is deliberately **GitHub-safe**: the mathematical notation uses GitHub's supported `$...$` / `$$...$$` style rather than LaTeX commands such as `\operatorname`, which should avoid the `operatorname` error you encountered.
