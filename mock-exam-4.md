Yes. I checked the official UAB guide for **2026/2027**. Course **106575 – Advanced Machine Learning** is a 6-credit, 3rd-year Bachelor in Artificial Intelligence course, taught in English in the first semester.  UAB Apps

The official syllabus covers **transfer/domain adaptation, parameter-efficient learning and compression, semi/self-supervised learning, ensembles and knowledge distillation, continual/online learning, multi-task/meta-learning, explainability, adversarial robustness/security, and federated learning/privacy**.  UAB Apps

Since the written exams account for **50%** of the final mark and are explicitly intended to test both conceptual understanding and problem solving, I'd practice with an exam like this.  UAB Apps

# Practice Exam — Advanced Machine Learning (106575)

**Suggested time:** 90 minutes **Total:** 100 points **Difficulty:** Similar to what I would expect for a university-level theory/problem-solving exam **No internet / no AI assistance**

## Part A — Conceptual questions (30 points)

### 1\. Transfer learning — 6 points

You have a CNN trained on ImageNet and want to classify medical X-ray images, but you only have 2,000 labelled X-rays.

**a)** Explain why transfer learning is likely to be useful.

**b)** Compare these two approaches:

- freezing the backbone and training only a new classification head;
- fine-tuning the entire network.

**c)** Give one situation where fine-tuning the entire network could actually hurt performance.

---

### 2\. Parameter-efficient learning — 6 points

A transformer contains **100 million parameters**, but you only have a small domain-specific dataset.

Explain the motivation behind parameter-efficient fine-tuning (PEFT). Compare the basic idea of:

- full fine-tuning;
- adapters;
- low-rank adaptation (LoRA).

Why can PEFT be particularly useful when computational resources are limited?

---

### 3\. Knowledge distillation — 6 points

A large neural network ("teacher") achieves 94% accuracy, while a much smaller network ("student") achieves only 86%.

Explain how knowledge distillation could be used to improve the student.

Your answer should explain the role of:

- hard labels;
- teacher predictions;
- temperature;
- the distillation loss.

---

### 4\. Continual learning — 6 points

A model initially learns to recognize cats, dogs and horses. Six months later it is retrained on a new dataset containing cars and bicycles.

After training on the new data, its performance on cats and dogs drops dramatically.

**a)** What phenomenon has occurred?

**b)** Explain why it happens.

**c)** Describe **two strategies** for mitigating the problem.

---

### 5\. Explainability — 6 points

A hospital uses a neural network to predict whether a patient has a disease.

The model has 97% accuracy, but doctors do not trust it because its decisions are difficult to interpret.

Explain the difference between:

- **model interpretability**;
- **post-hoc explainability**.

Give one example of an explainability technique and discuss one potential limitation of using explanations from a complex neural network.

---

# Part B — Mathematical/problem-solving questions (40 points)

### 6\. Knowledge distillation calculation — 10 points

A teacher network produces logits:

\[ z\_T=\[2.0,;1.0,;0.0\] \]

and a student produces:

\[ z\_S=\[1.0,;0.5,;0.0\]. \]

The distillation temperature is:

\[ T=2. \]

The softmax function is:

\[ p_i=\\frac{e^{z_i/T}} {\\sum_j e^{z_j/T}}. \]

**a)** Calculate the teacher's softened probability distribution.

**b)** Calculate the student's softened probability distribution.

**c)** Explain qualitatively what happens to the distributions as (T) increases.

**d)** Why might the softened teacher distribution contain useful information that is absent from the one-hot training label?

---

### 7\. Domain adaptation — 10 points

You have:

- 50,000 labelled images from a **source domain**;
- 500 labelled images from a **target domain**;
- 20,000 unlabelled target-domain images.

The source images are normal photographs, while the target images are low-light photographs.

Design a training strategy that makes use of the available data.

Your answer should specify:

1. what data you would use for pretraining;
2. whether you would freeze or fine-tune the model;
3. how you would use the unlabelled target data;
4. one method for reducing the domain shift;
5. how you would evaluate whether adaptation actually helped.

---

### 8\. Federated learning — 10 points

Three hospitals want to train a common disease-prediction model.

They cannot send patient records to a central server.

Explain how **federated learning** could solve this problem.

Then discuss **two privacy/security risks** that remain even though the raw patient data never leaves the hospitals.

Finally, explain one technique that could mitigate one of these risks.

---

### 9\. Adversarial robustness — 10 points

A classifier correctly identifies an image as:

> "panda — 98%"

After adding a very small perturbation that is almost invisible to humans, the model predicts:

> "gibbon — 97%"

**a)** What type of attack is this?

**b)** Why can a tiny perturbation have such a large effect on a neural network?

**c)** Distinguish between a targeted and an untargeted attack.

**d)** Give two possible defences against adversarial examples.

**e)** What trade-off can adversarial training introduce?

---

# Part C — Integrated case study (30 points)

### 10\. Designing an advanced ML system — 30 points

A company wants to deploy an image-recognition model on **mobile phones**.

The constraints are:

- only 5,000 labelled training examples;
- large amounts of unlabelled data;
- limited phone memory;
- limited battery/computation;
- users continuously generate new types of images;
- the company wants to protect user privacy;
- the model will be used in a safety-critical application.

Design an overall machine-learning strategy.

Your answer should discuss **at least six** of the following:

- transfer learning;
- self-supervised learning;
- semi-supervised learning;
- model compression;
- knowledge distillation;
- parameter-efficient learning;
- continual learning;
- explainability;
- adversarial robustness;
- federated learning;
- privacy;
- computational efficiency.

For each technique you choose, explain **why it is appropriate for this particular problem** rather than merely defining it.

---

# Bonus — Exam-style comparison question

If you have extra time, try this without notes:

> **"A model with higher test accuracy is not necessarily the better model."**

Discuss this statement in the context of **fairness, robustness, explainability, privacy and computational cost**.

---

## What I would prioritise for revision

Based specifically on the 2026/27 UAB syllabus, I'd rank the topics like this:

| Priority | Topic | What you should be able to do |
| --- | --- | --- |
| 🔴 Very high | Transfer learning / domain adaptation | Design and justify a transfer strategy |
| 🔴 Very high | PEFT & model compression | Explain methods and resource trade-offs |
| 🔴 Very high | Knowledge distillation | Understand loss functions and teacher/student training |
| 🔴 Very high | Adversarial ML | Explain attacks, vulnerabilities and defences |
| 🔴 Very high | Federated learning & privacy | Design a privacy-aware training setup |
| 🟠 High | Continual/online learning | Explain catastrophic forgetting and solutions |
| 🟠 High | Self/semi-supervised learning | Select an appropriate approach with limited labels |
| 🟠 High | Explainability | Compare techniques and their limitations |
| 🟡 Medium-high | Multi-task & meta-learning | Explain when/why they are useful |
| 🟡 Medium-high | Ensemble learning | Compare ensemble strategies and their benefits |

These priorities follow the actual content listed in the UAB guide.  UAB Apps

### One important exam tip

Don't prepare this course as **"memorise definitions."** The UAB guide says the written exams assess both the ability to **solve problems using the techniques** and the student's **conceptual understanding** of those techniques.  UAB Apps

So for every topic, practise answering:

> **What is it? → Why does it work? → When should I use it? → What are its limitations? → How would I implement it?**

If you want, I can also make you a **full 2026/27-style mock exam with 20 questions + complete solutions/marking scheme**, specifically designed to resemble the difficulty of the UAB 106575 exams.  UAB Apps
