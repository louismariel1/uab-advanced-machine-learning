Below is a **full model solution and marking guide** for the 100-point mock exam. I’ll answer at the level I would recommend for a demanding UAB Advanced Machine Learning exam: not just definitions, but **reason → method → trade-off → application**.

# Advanced Machine Learning — Mock Exam 1

## Comprehensive Solutions + Marking Scheme

---

# Part A — Conceptual Questions

## 1\. Transfer learning — 6 points

### a) Why is transfer learning useful? — 2 points

Transfer learning is useful because the model has already learned general visual representations from a large dataset such as ImageNet. Early CNN layers typically learn relatively general features such as:

- edges;
- textures;
- shapes;
- local patterns.

With only **2,000 labelled medical X-rays**, training a CNN completely from scratch would have a high risk of overfitting.

Therefore, we can start from pretrained parameters:

\[ \\theta\_{\\text{ImageNet}} \]

and adapt them to the medical task rather than learning everything from random initialization.

The key idea is:

\[ \\boxed{\\text{large source dataset} \\rightarrow \\text{learn reusable representation} \\rightarrow \\text{small target dataset}} \]

**Exam-quality answer:**

> Transfer learning reduces the amount of labelled target-domain data required because the pretrained model already contains useful representations. This can improve generalisation and reduce training time compared with training a network from scratch.

---

### b) Freeze backbone vs fine-tune entire network — 2 points

#### Freeze backbone + train classification head

The pretrained feature extractor remains fixed:

\[ \\theta\_{\\text{backbone}}=\\text{constant} \]

while only the classifier is trained.

Advantages:

- fewer trainable parameters;
- lower computational cost;
- lower memory requirements;
- reduced risk of overfitting with only 2,000 examples.

Disadvantage:

- the pretrained representation may not be sufficiently suitable for medical images.

#### Fine-tune entire network

All or most parameters are updated:

\[ \\theta \\rightarrow \\theta' \]

Advantages:

- allows the representation to adapt to the medical domain;
- potentially higher final performance if the source and target domains differ substantially.

Disadvantages:

- more computation and memory;
- greater risk of overfitting;
- potentially destroys useful pretrained features.

A good practical strategy would often be:

\[ \\boxed{\\text{freeze initially} \\rightarrow \\text{then gradually fine-tune}} \]

---

### c) When could full fine-tuning hurt? — 2 points

Full fine-tuning could hurt when the target dataset is very small.

With only 2,000 X-rays, updating millions of parameters can cause:

\[ \\boxed{\\text{overfitting}} \]

It can also cause **catastrophic forgetting of useful pretrained representations**, particularly if the learning rate is too large.

Therefore, if the target dataset is small, freezing most layers or using PEFT can be safer.

### Full-mark answer

> Full fine-tuning can hurt when the target dataset is small because the large number of trainable parameters allows the model to overfit. It can also overwrite useful pretrained representations, especially with an excessively large learning rate.

---

# 2\. Parameter-efficient learning — 6 points

## Motivation

Full fine-tuning updates all:

\[ 100\\text{ million} \]

parameters.

This requires substantial:

- gradient memory;
- optimizer memory;
- training computation;
- storage for task-specific models.

PEFT instead asks:

> Can we adapt the pretrained model while modifying only a small number of parameters?

Thus:

\[ \\boxed{\\text{large frozen base model}+\\text{small trainable adaptation}} \]

---

## Full fine-tuning

All model parameters are trainable:

\[ W \\rightarrow W+\\Delta W \]

where (\\Delta W) can have the same dimensionality as (W).

Advantages:

- maximum flexibility;
- unrestricted adaptation.

Disadvantages:

- expensive;
- large optimizer state;
- separate full model may need to be stored for each task.

---

## Adapters

Small trainable modules are inserted into the pretrained network.

Conceptually:

\[ \\text{frozen layer} \\rightarrow \\text{small trainable adapter} \\rightarrow \\text{next frozen layer} \]

Only the adapter parameters are updated.

Therefore:

\[ \\boxed{\\text{many base parameters frozen}+\\text{few trainable adapter parameters}} \]

---

## LoRA

LoRA freezes (W) and approximates the task-specific update as:

\[ \\boxed{\\Delta W\\approx BA} \]

where:

\[ A\\in\\mathbb{R}^{r\\times d} \]

and

\[ B\\in\\mathbb{R}^{d\\times r} \]

with:

\[ r\\ll d \]

Therefore:

\[ W'=W+BA \]

For a (d\\times d) matrix:

\[ d^2 \]

parameters would be needed for a full update, while LoRA uses:

\[ 2dr \]

parameters.

The ratio is:

\[ \\frac{2dr}{d^2} = \\frac{2r}{d} \]

which can be extremely small.

---

## Why PEFT helps with limited resources

PEFT reduces:

\[ \\boxed{\\text{trainable parameters}} \]

which reduces:

- gradients;
- optimizer state;
- training memory;
- adaptation storage.

A particularly important distinction is that PEFT **does not necessarily compress the entire pretrained model**.

### Excellent exam answer

> PEFT is useful when resources are limited because the pretrained model can remain frozen while only a small number of task-specific parameters are trained. This greatly reduces gradient and optimizer memory and allows many task-specific adaptations to share the same base model.

---

# 3\. Knowledge distillation — 6 points

Knowledge distillation transfers information from a large **teacher** to a smaller **student**.

The teacher produces a probability distribution rather than merely a hard class label.

For example, instead of:

\[ \[0,0,1,0\] \]

the teacher might produce:

\[ \[0.01,0.10,0.80,0.09\] \]

This contains information about relationships between classes.

---

## Hard labels

The ordinary supervised loss compares the student prediction with the ground-truth class:

\[ L_{\\text{hard}} = CE(y,p_S) \]

where (y) is the one-hot label.

---

## Teacher predictions

The teacher produces softened probabilities:

\[ p_T^{(T)} = \\operatorname{softmax} \\left( \\frac{z_T}{T} \\right) \]

and the student similarly produces:

\[ p_S^{(T)} = \\operatorname{softmax} \\left( \\frac{z_S}{T} \\right) \]

The student is encouraged to imitate the teacher.

---

## Temperature

The temperature (T) controls how soft the distribution becomes.

At:

\[ T=1 \]

we obtain the ordinary softmax.

As:

\[ T\\uparrow \]

the probabilities become more uniform.

This exposes information about the relative likelihood of non-winning classes.

---

## Distillation loss

A common formulation is:

\[ \\boxed{ L = \\alpha L_{\\text{hard}} + (1-\\alpha)T^2 KL(p_T^{(T)}|p\_S^{(T)}) } \]

where:

- (L\_{\\text{hard}}) uses the ground-truth labels;
- the KL term encourages the student to imitate the teacher;
- (T) controls softness;
- (\\alpha) controls the balance.

The (T^2) factor is commonly used to compensate for the gradient scaling introduced by temperature.

### Key exam insight

The teacher's distribution contains **dark knowledge**: information about similarities between classes that is lost in a one-hot label.

For example, if the correct class is "wolf", the teacher might assign:

\[ 0.70\\text{ wolf},\\quad 0.20\\text{ dog},\\quad 0.08\\text{ fox} \]

This tells the student that dog and fox are more visually similar to wolf than an unrelated class.

---

# 4\. Continual learning — 6 points

### a) What phenomenon? — 1 point

This is:

\[ \\boxed{\\text{catastrophic forgetting}} \]

---

### b) Why does it happen? — 2 points

When the model is trained on the new car/bicycle data, its parameters are updated to optimise the new task/data distribution.

Those parameter changes can overwrite representations that were useful for:

- cats;
- dogs;
- horses.

If old data is unavailable, the model has no direct training signal reminding it to preserve old behaviour.

Thus:

\[ \\boxed{ \\text{learning new tasks} \\rightarrow \\text{parameter changes} \\rightarrow \\text{loss of old knowledge} } \]

---

### c) Two mitigation strategies — 3 points

Several valid answers exist.

### Strategy 1 — Replay/rehearsal

Store some examples from previous tasks and train on:

\[ \\text{new data}+\\text{old examples} \]

This reminds the model of previous knowledge.

### Strategy 2 — Regularisation

Penalise large changes to parameters that are important for previous tasks.

Conceptually:

\[ L=L_{\\text{new}}+\\lambda L_{\\text{preserve}} \]

Methods such as **EWC** can assign larger penalties to changing important parameters.

### Other acceptable strategies

- knowledge distillation from the old model;
- parameter isolation;
- task-specific modules;
- rehearsal buffers;
- freezing selected parameters.

### Strong answer

> Catastrophic forgetting occurs because optimisation on the new task changes parameters that also encode knowledge required for previous tasks. Replay reduces the problem by exposing the model to old examples, while regularisation or parameter isolation reduces destructive changes to previously important parameters.

---

# 5\. Explainability — 6 points

## Interpretability

Interpretability generally refers to how inherently understandable the model's decision process is.

For example, a simple linear model:

\[ y=w_1x_1+w_2x_2+\\cdots \]

can be relatively easy to interpret because the contribution of each feature is explicit.

An inherently interpretable model is often called a **glass-box** model.

---

## Post-hoc explainability

Post-hoc explainability attempts to explain the predictions of an already-trained complex model.

For example:

- SHAP;
- LIME;
- saliency maps;
- Grad-CAM.

The original model can remain a complex black box.

---

## Example: Grad-CAM

For an image classifier, Grad-CAM can generate a heatmap showing image regions that contributed strongly to a prediction.

For example:

\[ \\text{X-ray} \\rightarrow \\text{model} \\rightarrow \\text{disease prediction} + \\text{heatmap} \]

A doctor might see whether the model focused on a clinically relevant region.

---

## Limitation

An explanation is **not necessarily a faithful description of the model's actual reasoning**.

A post-hoc explanation may be:

- approximate;
- unstable;
- sensitive to the explanation method;
- misleading;
- correlated with the prediction without proving causality.

### Excellent exam sentence

> A post-hoc explanation should not automatically be interpreted as proof of causal reasoning by the model.

In a safety-critical medical setting, this distinction is particularly important.

---

# Part B — Mathematical / Problem Solving

# 6\. Knowledge distillation calculation — 10 points

Given:

\[ z\_T=\[2,1,0\] \]

\[ z\_S=\[1,0.5,0\] \]

and:

\[ T=2 \]

We calculate:

\[ p_i=\\frac{e^{z_i/T}} {\\sum_j e^{z_j/T}} \]

---

## a) Teacher distribution — 4 points

Divide teacher logits by (T=2):

\[ \\frac{z\_T}{T} = \[1,0.5,0\] \]

Exponentiate:

\[ e^1\\approx2.7183 \]

\[ e^{0.5}\\approx1.6487 \]

\[ e^0=1 \]

Sum:

\[ 2.7183+1.6487+1 = 5.3670 \]

Therefore:

\[ p\_T = \\left\[ \\frac{2.7183}{5.3670}, \\frac{1.6487}{5.3670}, \\frac{1}{5.3670} \\right\] \]

giving approximately:

\[ \\boxed{ p\_T\\approx\[0.5065,;0.3072,;0.1863\] } \]

---

## b) Student distribution — 4 points

Divide student logits by (T=2):

\[ \\frac{z\_S}{T} = \[0.5,0.25,0\] \]

Exponentiate:

\[ e^{0.5}\\approx1.6487 \]

\[ e^{0.25}\\approx1.2840 \]

\[ e^0=1 \]

Sum:

\[ 1.6487+1.2840+1 = 3.9327 \]

Therefore:

\[ p\_S = \\left\[ \\frac{1.6487}{3.9327}, \\frac{1.2840}{3.9327}, \\frac{1}{3.9327} \\right\] \]

giving approximately:

\[ \\boxed{ p\_S\\approx\[0.4192,;0.3265,;0.2543\] } \]

---

## c) What happens as (T) increases? — 1 point

Increasing (T) divides the logits by a larger number, reducing their differences.

Therefore the softmax distribution becomes flatter:

\[ \\boxed{T\\uparrow\\Rightarrow\\text{distribution becomes softer/more uniform}} \]

Conversely:

\[ T\\rightarrow0 \]

makes the distribution increasingly concentrated around the largest logit.

---

## d) Why is the teacher distribution useful? — 1 point

A one-hot label only says:

\[ \\text{class A is correct} \]

The teacher distribution additionally says something like:

\[ P(A)\>P(B)\>P(C) \]

Thus it contains information about **inter-class similarity and relative confidence**.

This is the "dark knowledge" transferred during distillation.

---

# 7\. Domain adaptation — 10 points

We have:

- 50,000 labelled source images;
- 500 labelled target images;
- 20,000 unlabelled target images.

The target domain is low-light photography.

A strong strategy should exploit **all three sources of information**.

---

## 1\. Pretraining — 2 points

Start with the 50,000 labelled source images.

Train a strong source-domain model:

\[ D_S=(X_S,Y\_S) \]

to learn a useful visual representation.

Then transfer the model to the target domain.

\[ \\boxed{ D\_S \\rightarrow \\text{pretrained model} } \]

---

## 2\. Freeze or fine-tune? — 2 points

Because only 500 labelled target examples exist, I would initially freeze most of the backbone and adapt the classifier.

Then, if necessary, gradually unfreeze higher layers using a **small learning rate**.

A reasonable strategy is:

\[ \\boxed{ \\text{source pretraining} \\rightarrow \\text{freeze backbone} \\rightarrow \\text{target adaptation} \\rightarrow \\text{careful fine-tuning} } \]

This reduces overfitting risk.

---

## 3\. Use unlabelled target data — 2 points

The 20,000 unlabelled target images are extremely valuable because they reveal the target-domain distribution.

Possible approaches include:

### Self-supervised pretraining

Learn representations from the unlabelled target images.

For example:

\[ \\text{unlabelled target images} \\rightarrow \\text{self-supervised objective} \\rightarrow \\text{target-adapted representation} \]

Then use the 500 labelled images for supervised fine-tuning.

### Semi-supervised learning

Use pseudo-labeling or consistency regularisation.

For example:

\[ x_u \\rightarrow \\text{model} \\rightarrow \\hat y_u \]

and use confident pseudo-labels as additional training data.

---

## 4\. Reduce domain shift — 2 points

The source and target domains differ because the target images are low-light.

One possible method is **domain-adversarial training**.

The feature extractor learns representations that are useful for the task but less informative about which domain generated the image.

Conceptually:

\[ \\boxed{ \\text{task information}\\uparrow \\qquad \\text{domain-specific information}\\downarrow } \]

Other acceptable methods include:

- image-level style/illumination augmentation;
- histogram/brightness adaptation;
- feature alignment;
- domain-adversarial learning;
- CORAL/MMD-style distribution alignment.

---

## 5\. Evaluate adaptation — 2 points

Evaluate on a **held-out target-domain test set** that is not used during adaptation.

Compare:

1. source-only model;
2. transfer learning model;
3. adapted model.

For example:

\[ \\boxed{ \\text{Target accuracy/F1 before adaptation} \\quad\\text{vs}\\quad \\text{after adaptation} } \]

If the adapted model performs better on the target test distribution, there is evidence that adaptation helped.

It is also useful to measure calibration and performance across relevant target subgroups.

### Full strategy

\[ \\boxed{ \\text{50k labelled source} \\rightarrow \\text{pretraining} } \]

\[ \+ \]

\[ \\boxed{ \\text{20k unlabelled target} \\rightarrow \\text{self/semi-supervised adaptation} } \]

\[ \+ \]

\[ \\boxed{ \\text{500 labelled target} \\rightarrow \\text{supervised fine-tuning/evaluation} } \]

---

# 8\. Federated learning — 10 points

## Basic mechanism — 4 points

Federated learning allows each hospital to keep its patient data locally.

Suppose there are hospitals:

\[ H_1,H_2,H\_3 \]

The server starts with model:

\[ W\_0 \]

Each hospital receives the current model and trains locally:

\[ W_0 \\rightarrow W_1^{(H\_1)} \]

\[ W_0 \\rightarrow W_1^{(H\_2)} \]

\[ W_0 \\rightarrow W_1^{(H\_3)} \]

The hospitals send **model updates**, rather than raw patient records.

The server aggregates them, commonly using a method such as **FedAvg**:

\[ W_{t+1} = \\sum_k \\frac{n_k}{N} W_{t+1}^{(k)} \]

where (n\_k) is the amount of training data at hospital (k).

Then the new global model is redistributed.

Thus:

\[ \\boxed{ \\text{local training} \\rightarrow \\text{update sharing} \\rightarrow \\text{aggregation} \\rightarrow \\text{new global model} } \]

---

# Privacy/security risks — 4 points

The fact that raw data never leaves the hospitals does **not** mean the system is perfectly private.

## Risk 1 — Gradient/update leakage

Model updates can potentially reveal information about the underlying training examples.

An attacker may use updates to infer sensitive information.

---

## Risk 2 — Malicious or compromised clients

A malicious hospital/client could send manipulated updates.

This can produce:

- poisoning attacks;
- backdoors;
- degraded global performance.

Another acceptable risk is server-side inference or membership inference.

---

# Mitigation — 2 points

One important technique is **secure aggregation**.

The server can obtain the aggregate of updates without seeing each hospital's individual update.

Conceptually:

\[ \\boxed{ u_1,u_2,u_3 \\rightarrow u_1+u_2+u_3 } \]

without exposing each (u\_i) individually.

For stronger privacy, **differential privacy** can add carefully calibrated noise to updates.

For example:

\[ u_i' = u_i+\\epsilon \]

where the noise is selected according to a formal privacy mechanism.

### Excellent answer

> Federated learning reduces the need to centralise sensitive data, but it does not automatically guarantee privacy. Updates may leak information and malicious clients may poison the model. Secure aggregation can hide individual client updates, while differential privacy can provide a formal privacy guarantee.

---

# 9\. Adversarial robustness — 10 points

## a) What type of attack? — 1 point

This is an:

\[ \\boxed{\\text{adversarial example}} \]

The attacker adds a small perturbation designed to cause misclassification.

---

## b) Why can a tiny perturbation have a large effect? — 2 points

Neural networks can be highly sensitive to changes in input space.

An attacker can optimise the perturbation specifically against the model's decision boundary.

For example:

\[ x' = x+\\delta \]

where:

\[ |\\delta|\\text{ is small} \]

but:

\[ f(x')\\neq f(x) \]

The perturbation does not need to be visually large if it is carefully aligned with directions in which the model's loss changes strongly.

In high-dimensional spaces, even small changes in many dimensions can produce a significant change in the model's score.

---

## c) Targeted vs untargeted — 2 points

### Untargeted attack

The attacker only wants the model to make **any incorrect prediction**:

\[ f(x+\\delta)\\neq y \]

### Targeted attack

The attacker wants a **specific incorrect class**:

\[ f(x+\\delta)=y\_{\\text{target}} \]

In the example:

\[ \\text{panda}\\rightarrow\\text{gibbon} \]

could be a targeted attack if the attacker specifically wanted "gibbon."

---

## d) Two defences — 3 points

### Adversarial training

Generate adversarial examples during training and train the model to classify them correctly:

\[ \\boxed{ \\text{clean examples} + \\text{adversarial examples} \\rightarrow \\text{robust model} } \]

### Input preprocessing / robust architectures

Potential approaches include:

- carefully designed input transformations;
- robust feature representations;
- certified defenses;
- adversarial detection.

A strong answer should explain that no generic preprocessing method guarantees complete robustness.

---

## e) Trade-off — 2 points

Adversarial training can reduce clean-data accuracy.

The model may be optimised to perform well against perturbations, which can create a trade-off:

\[ \\boxed{ \\text{robustness}\\uparrow \\quad\\leftrightarrow\\quad \\text{clean accuracy potentially}\\downarrow } \]

It also increases training cost because adversarial examples must be generated.

For a safety-critical application, however, this trade-off may be justified.

---

# Part C — Integrated Case Study

# 10\. Designing an advanced ML system — 30 points

This is the most important question in the exam because it tests **synthesis**.

The key is not to list six definitions. We need to construct a coherent system.

---

# Proposed architecture

A strong overall pipeline would be:

\[ \\boxed{ \\text{self-supervised pretraining} \\rightarrow \\text{transfer learning} \\rightarrow \\text{semi-supervised adaptation} \\rightarrow \\text{robust training} \\rightarrow \\text{compression/distillation} \\rightarrow \\text{federated continual deployment} } \]

with explainability and monitoring throughout.

Let's justify each component.

---

## 1\. Transfer learning — 3 points

Only:

\[ 5,000 \]

labelled examples are available.

Training a large image model from scratch would risk overfitting.

Therefore start from a pretrained vision model.

\[ \\boxed{ \\text{pretrained model} \\rightarrow \\text{adapt to target task} } \]

The pretrained representation provides useful generic visual features.

If necessary, fine-tune only selected layers or use PEFT.

---

# 2\. Self-supervised learning — 3 points

The problem gives us:

> large amounts of unlabelled data.

This should not be ignored.

Self-supervised learning can exploit the unlabelled images to learn representations without manual labels.

For example:

\[ x \\rightarrow \\text{augmentation} \\rightarrow \\text{representation learning} \]

Then the resulting model can be fine-tuned using the 5,000 labelled examples.

This is particularly appropriate because labels are scarce but raw images are abundant.

---

# 3\. Semi-supervised learning — 3 points

After supervised training on the 5,000 labelled images, we can use the large unlabelled dataset with methods such as:

- pseudo-labeling;
- consistency regularisation.

For example:

\[ x_u \\rightarrow f(x_u) \\rightarrow \\hat y\_u \]

Only sufficiently confident pseudo-labels should be used.

The objective becomes something like:

\[ L = L_{\\text{supervised}} + \\lambda L_{\\text{unsupervised}} \]

This increases the effective amount of training information without requiring manual labels for every image.

---

# 4\. Model compression — 3 points

Mobile phones have:

- limited memory;
- limited computation;
- limited battery.

Therefore the final model should be compressed.

Possible techniques include:

### Quantisation

For example:

\[ FP32\\rightarrow INT8 \]

which reduces weight storage substantially.

### Pruning

Remove unnecessary parameters.

Structured pruning is particularly attractive because it can produce hardware-friendly smaller networks.

Thus:

\[ \\boxed{ \\text{compression} \\rightarrow \\text{lower memory} + \\text{potentially lower inference cost} } \]

---

# 5\. Knowledge distillation — 3 points

A large accurate teacher can be trained offline.

Then a smaller mobile student learns from:

- hard labels;
- teacher predictions.

The objective can be:

\[ L = \\alpha L_{\\text{hard}} + (1-\\alpha)T^2 KL(p_T^T|p\_S^T) \]

This allows the student to inherit some of the teacher's knowledge while remaining computationally small.

Therefore:

\[ \\boxed{ \\text{large accurate teacher} \\rightarrow \\text{small mobile student} } \]

This is particularly appropriate for deployment on phones.

---

# 6\. PEFT — 3 points

If the deployed model must continually adapt to new domains or tasks, full fine-tuning may be too expensive.

Instead, use a PEFT technique such as LoRA:

\[ W'=W+BA \]

where (W) is frozen.

This allows task-specific adaptation with a small number of trainable parameters.

For example:

\[ \\boxed{ \\text{shared base model} + \\text{small task-specific adaptation} } \]

This is particularly useful if many personalised or task-specific models are required.

---

# 7\. Continual learning — 3 points

The users continuously generate new types of images.

Therefore the data distribution changes over time:

\[ P_t(x)\\neq P_{t+1}(x) \]

If we simply train on new data, the model may suffer:

\[ \\boxed{\\text{catastrophic forgetting}} \]

A continual-learning strategy could use:

- replay buffers;
- regularisation;
- knowledge distillation from the previous model;
- parameter isolation.

For example:

\[ \\text{old examples} + \\text{new examples} \\rightarrow \\text{updated model} \]

This allows adaptation while preserving old capabilities.

---

# 8\. Federated learning — 3 points

The company wants to protect user privacy.

Instead of sending raw images to a central server:

\[ \\boxed{ \\text{data remains on phone} } \]

The phone performs local training and sends model updates.

The server aggregates updates.

This reduces the need to centralise sensitive user data.

However, federated learning should be combined with privacy/security mechanisms because model updates themselves can leak information.

---

# 9\. Privacy — 2 points

I would combine federated learning with:

### Secure aggregation

The server receives an aggregate rather than individual client updates.

And potentially:

### Differential privacy

Add carefully calibrated noise to updates to reduce information leakage.

Therefore:

\[ \\boxed{ \\text{Federated learning} + \\text{secure aggregation} + \\text{differential privacy} } \]

provides a substantially stronger privacy architecture than federated learning alone.

---

# 10\. Adversarial robustness — 2 points

The application is explicitly:

> safety-critical.

Therefore clean accuracy alone is insufficient.

The system should be evaluated against adversarial perturbations.

Possible defence:

\[ \\boxed{\\text{adversarial training}} \]

The model should be tested using both:

- ordinary test data;
- adversarial examples.

The relevant objective is therefore:

\[ \\boxed{ \\text{accuracy} + \\text{robustness} } \]

rather than accuracy alone.

---

# 11\. Explainability — 2 points

For a safety-critical application, users may need to understand why a prediction was made.

For an image model, techniques such as:

- Grad-CAM;
- saliency methods;
- feature attribution

could provide visual explanations.

However, these should be validated because:

\[ \\boxed{ \\text{post-hoc explanation} \\neq \\text{guaranteed faithful reasoning} } \]

The explanation should therefore be treated as a supporting diagnostic tool rather than unquestionable proof of model reasoning.

---

# 12\. Overall system

A strong final architecture would look like:

```
                 LARGE UNLABELLED DATA
                         │
                         ▼
              Self-supervised learning
                         │
                         ▼
                 Pretrained model
                         │
                ┌────────┴────────┐
                │                 │
          5,000 labelled      Semi-supervised
             examples           learning
                │                 │
                └────────┬────────┘
                         ▼
                  Accurate teacher
                         │
                         ▼
                 Knowledge distillation
                         │
                         ▼
                Small mobile student
                         │
              ┌──────────┼──────────┐
              │          │          │
         Quantisation  Pruning   Robustness
              │          │          │
              └──────────┼──────────┘
                         ▼
                  Mobile deployment
                         │
                         ▼
                 Continual learning
                         │
                         ▼
              Federated local updates
                         │
                 Secure aggregation
                         │
                         ▼
                  Updated global model
```

---

# Ideal integrated answer

If you had limited time in the exam, a high-quality condensed answer would be:

> I would begin with a pretrained vision model because only 5,000 labelled examples are available. The large unlabelled dataset can be exploited using self-supervised and/or semi-supervised learning to improve the representation without requiring additional manual labels. For deployment on phones, I would train an accurate teacher and use knowledge distillation to obtain a smaller student, followed by quantisation and potentially structured pruning to reduce memory, computation and battery consumption. If the model needs task-specific adaptation, PEFT such as LoRA can reduce the number of trainable parameters. Since users continuously generate new data, continual-learning methods such as replay or regularisation should be used to reduce catastrophic forgetting. Because user data is sensitive, federated learning can keep raw images on-device, while secure aggregation and potentially differential privacy reduce risks from model updates. Finally, because the application is safety-critical, I would evaluate and improve adversarial robustness and provide explanations such as Grad-CAM, while recognising that post-hoc explanations are not guaranteed to be faithful. The system should therefore be evaluated not only on accuracy, but also robustness, privacy, computational cost and explainability.

That would be a **very strong integrated answer**.

---

# Bonus — "Higher test accuracy is not necessarily better"

The statement is correct.

A model should be evaluated across multiple dimensions.

## Fairness

A model could achieve high average accuracy while performing poorly for a particular demographic or subgroup.

For example:

\[ \\text{overall accuracy}=97% \]

could hide:

\[ \\text{Group A}=99% \]

but:

\[ \\text{Group B}=82% \]

Therefore overall accuracy is not sufficient.

---

## Robustness

A model may achieve excellent clean test accuracy but be extremely vulnerable to adversarial perturbations.

Thus:

\[ \\boxed{ \\text{high clean accuracy} \\not\\Rightarrow \\text{high robustness} } \]

For safety-critical applications, robustness can be more important than a small improvement in clean accuracy.

---

## Explainability

A slightly less accurate model may be preferable if its decisions are substantially more interpretable in a high-stakes environment.

For example, doctors may need to understand the evidence supporting a medical prediction.

---

## Privacy

A model trained using inappropriate data collection or a privacy-invasive architecture may have excellent accuracy but be unacceptable in deployment.

Federated learning, secure aggregation and differential privacy can impose some utility costs while improving privacy.

---

## Computational cost

Suppose:

| Model | Accuracy | Inference |
| --- | --- | --- |
| A | 95% | 5 ms |
| B | 96% | 500 ms |

For a resource-constrained mobile application, Model A may be preferable.

Similarly:

\[ \\text{FP32 model} \]

might achieve slightly higher accuracy than:

\[ \\text{INT8 model} \]

but the INT8 model may be substantially cheaper to deploy.

---

# Final high-yield marking map

For this particular mock, I would think about the marks like this:

| Question | Main skill being tested | Key concept |
| --- | --- | --- |
| Q1 | Transfer-learning reasoning | Freeze vs fine-tune |
| Q2 | PEFT understanding | Adapters vs LoRA |
| Q3 | Concept + loss | Teacher/student + temperature |
| Q4 | Continual learning | Catastrophic forgetting |
| Q5 | Critical understanding | Interpretability vs post-hoc explanation |
| Q6 | Calculation | Temperature-softened softmax |
| Q7 | System design | Domain adaptation + unlabelled data |
| Q8 | System design + security | Federated learning + privacy |
| Q9 | Adversarial reasoning | Attacks + defences |
| Q10 | Synthesis | Combining multiple ML techniques |

## The formulas I would absolutely memorise

### Knowledge distillation

\[ \\boxed{ p_i^{(T)} = \\frac{e^{z_i/T}} {\\sum_j e^{z_j/T}} } \]

and commonly:

\[ \\boxed{ L= \\alpha L_{\\text{hard}} + (1-\\alpha)T^2 KL(p_T^{(T)}|p\_S^{(T)}) } \]

### LoRA

\[ \\boxed{W'=W+BA} \]

\[ \\boxed{ N_{\\text{LoRA}} = r(d_{\\text{in}}+d\_{\\text{out}}) } \]

For a square matrix:

\[ \\boxed{ N\_{\\text{LoRA}}=2dr } \]

### Quantisation

\[ \\boxed{ \\text{storage}\\propto \\text{parameters}\\times\\text{bits/parameter} } \]

### Continual learning

\[ \\boxed{ \\text{new-task training} \\rightarrow \\text{catastrophic forgetting} } \]

### Federated learning

\[ \\boxed{ \\text{local training} \\rightarrow \\text{update aggregation} \\rightarrow \\text{global model} } \]

### Adversarial attack

\[ \\boxed{ x'=x+\\delta,\\qquad |\\delta|\\text{ small} } \]

while attempting to change:

\[ \\boxed{f(x')\\neq f(x)} \]

---

# What would distinguish a 50% answer from a 90%+ answer?

For this exam, the biggest difference is **not memorisation**.

A weaker answer says:

> "LoRA is a PEFT method that uses low-rank matrices."

A strong answer says:

> "LoRA freezes (W) and represents the task-specific update as (BA), reducing a (d^2)-parameter update to (2dr). This reduces gradients and optimizer state, but does not necessarily reduce the complete model state dict, and it is not guaranteed to reduce wall-clock training time because the adapter adds forward computation."

That pattern applies to essentially every topic:

\[ \\boxed{ \\text{Definition} \\rightarrow \\text{mechanism} \\rightarrow \\text{equation/implementation} \\rightarrow \\text{benefit} \\rightarrow \\text{limitation} \\rightarrow \\text{when to use} } \]

**That is the level I would aim for in the actual 90-minute exam.**
