# Advanced Machine Learning — Transfer Learning: Theory Summary

 This lecture introduces **transfer learning** as a way to reuse knowledge learned from one dataset/task to solve another, especially when the target problem has limited data, different data characteristics, or limited computational resources.

 ## 1\. Why Transfer Learning?

 In ordinary supervised learning, we:

 > **Train a model on some data → deploy it on new data.**

 This works well when:

 - There is enough training data.
- The model has enough capacity.
- Training and deployment data come from the same or sufficiently similar distribution.

 When these assumptions fail, training a new model from scratch can be inefficient or impossible.

 ### Main motivation

 Transfer learning can provide:

 - **Data efficiency:** fewer labelled target examples are needed.
- **Compute efficiency:** expensive pre-training can be reused.
- **Faster training:** the model starts from useful learned representations rather than random initialization.

 The fundamental idea is:

 > **Do not learn everything from scratch if useful knowledge has already been learned elsewhere.**

---

 # 2\. What Is Transfer Learning?

 **Transfer learning** means learning to perform a target task using knowledge obtained from a different source task/domain.

 There are two important dimensions:

 ### Domain

 A **domain** describes the characteristics/statistical distribution of the data.

 Examples:

 - City roads vs. country roads
- Photographs vs. medical scans
- Social-media images from different platforms
- Handwritten digits vs. photographs of house numbers

 ### Task

 A **task** describes the problem the model is solving, often characterized by the predictive objective/loss.

 Examples:

 - Classification
- Object detection
- Image segmentation
- Classifying cats vs. dogs
- Classifying dog breeds

---

 # 3\. Formal Definition

 Consider:

 - Source domain: $D_S$
- Source task: $T_S$
- Target domain: $D_T$
- Target task: $T_T$

 Transfer learning aims to improve learning of the target predictive function

 $$
f_T(\cdot)
$$

 in the target domain $D_T$ for task $T_T$, using knowledge from $D_S$ and $T_S$.

 At least one of the following differs:

 $$
D_S \neq D_T
$$

 or

 $$
T_S \neq T_T.
$$

 The terminology is:

 - **Source:** where the existing knowledge comes from.
- **Target:** the new domain/task we actually want to solve.

---

 # 4\. Main Transfer Scenarios

 The lecture distinguishes two important cases.

 ## Inductive transfer learning

 The **domain stays the same**, but the **task changes**:

 $$
D_S = D_T,\qquad T_S \neq T_T.
$$

 ### Example

 Train on:

 > Vehicle image classification

 Then transfer the learned knowledge to:

 > Road-image segmentation.

 The images may come from the same general domain, but the desired prediction is different.

---

 ## Transductive transfer learning

 The **task stays the same or is similar**, but the **domain changes**:

 $$
D_S \neq D_T,\qquad T_S \approx T_T.
$$

 ### Example

 Train an image classifier using ordinary photographs and apply it to images from another domain.

 The classification objective remains similar, but the input distribution changes.

 This is closely related to **domain adaptation**.

---

 # 5\. Model-Centric Transfer: Fine-Tuning

 The simplest and most common transfer-learning strategy is **fine-tuning**.

 The process is:

 1. **Pre-train** a model on a source dataset/task.
2. Modify the architecture if necessary for the target task.
   - For example, replace the classification head.
3. Continue training using the target data.

 Conceptually:

 $$
\text{Source data}
\rightarrow
\text{Pre-trained model}
\rightarrow
\text{Target training}
\rightarrow
\text{Target model}.
$$

 ### Why does fine-tuning work?

 Different neural networks trained on different datasets/tasks often learn **similar low-level and mid-level features**.

 For vision networks, early layers may learn things such as:

 - edges
- textures
- shapes
- simple visual patterns

 These features can remain useful even when the final task changes.

 Therefore:

 > **Pre-training learns a useful feature-extraction backbone; fine-tuning teaches the model how to use those features for the new task.**

---

 # 6. Feature Extraction and Freezing Layers

 You do **not necessarily need to retrain the whole network**.

 A common approach is to **freeze** some layers:

 - Their weights remain fixed.
- Only later layers are trained on the target data.

 The basic intuition is:

 $$
\boxed{\text{Early layers → generic features}}
$$

 $$
\boxed{\text{Later layers → task-specific features}}
$$

 Therefore, early layers can often be reused directly.

 ### Practical trade-off

 Which layers should be frozen depends on:

 - the architecture,
- similarity between source and target domains,
- amount of target data,
- similarity between source and target tasks.

 There is no universal optimal freezing strategy; it generally requires experimentation and monitoring.

---

 # 7\. Domain Shift

 A central problem in transfer learning is **domain shift**.

 ### Definition

 Domain shift occurs when the training and deployment/test distributions differ:

 $$
P_S(X) \neq P_T(X).
$$

 This difference can create a **generalisation gap**.

 For example, suppose a model is trained mostly on bright images but deployed on darker images.

 The underlying relationships may still be similar, but the statistics of the inputs have changed.

 ### Important idea

 The model may already know the relevant concept, but the **distribution of the inputs is different**.

 Fine-tuning on target-domain data can address this, but the lecture describes this as a relatively brute-force solution.

---

 # 8\. Domain Adaptation

 **Domain adaptation** focuses specifically on dealing with differences between source and target domains.

 The goal is essentially:

 $$
\text{Source domain}
\longrightarrow
\text{Target-compatible representation/data}.
$$

 Instead of simply retraining the model, we can explicitly try to compensate for the domain shift.

 There are two major strategies discussed:

 1. **Input-space alignment**
2. **Feature-space alignment**

---

 # 9\. Input-Space Alignment

 Here, we modify the **data itself** so that the source and target distributions become more similar.

 ## Correlation Alignment — CORAL

 One example is **CORAL** (Correlation Alignment).

 The basic idea is to align statistical properties of the two domains, particularly:

 - first-order statistics
- second-order statistics

 So instead of changing the model's representation, we transform the input distributions.

---

 ## Domain Translation

 Another name you may encounter for input-space alignment is **domain translation**.

 The core idea is to learn a mapping

 $$
g:X_S\rightarrow X_T
$$

 that converts source-domain samples into target-domain-like samples.

 Importantly, this can potentially be done **without target labels**, because the method can operate on the input space itself.

---

 # 10\. Generative Domain Translation

 More sophisticated transformations can use **generative models**.

 These can learn complex mappings between domains where the shift is not simply:

 - brightness,
- contrast,
- colour,
- or another simple statistical transformation.

 This is useful when source and target domains have substantially different appearances but still represent related underlying concepts.

---

 # 11\. Sim2Real Transfer

 A particularly important application is **simulation-to-reality (sim2real)** transfer.

 This occurs when:

 $$
\text{Simulation} \rightarrow \text{Real world}.
$$

 Examples include:

 - robotics
- autonomous driving

 A simulator can generate huge amounts of inexpensive training data, but simulated environments rarely look exactly like reality.

 Therefore:

 $$
P_{\text{simulation}}(X)
\neq
P_{\text{real}}(X).
$$

 This creates a domain-shift problem.

---

 # 12\. Domain Randomisation

 Instead of trying to make the simulation look exactly like reality, another strategy is **domain randomisation**.

 ### Basic idea

 Rather than carefully matching the source domain to the target:

 > Make the source domain extremely diverse.

 For example, randomly vary:

 - lighting
- colours
- textures
- object appearances
- camera properties
- environmental conditions

 The objective is to create a source distribution broad enough to **cover the target domain**.

 Conceptually:

 ### Domain alignment

 $$
\text{Source} \rightarrow \boxed{\text{Target}}
$$

 ### Domain randomisation

 $$
\boxed{\text{Large/diverse source distribution}}
\supseteq
\text{Target}.
$$

 Thus the target becomes less "out-of-distribution" because it is effectively just another possible case within the expanded training distribution.

 ### Key insight

 > **Don't necessarily transform the source to look like the target; make the source diverse enough that the target is included within it.**

 This is especially useful for robotics and sim2real problems.

---

 # 13\. Feature-Space Alignment

 Instead of modifying the input data, we can align the domains **inside the neural network's representation space**.

 The intuition is:

 $$
\text{Domain shift}
\rightarrow
\text{different features}
\rightarrow
\text{classification failure}.
$$

 Therefore:

 > Make the source and target domains produce similar feature representations.

---

 # 14\. Domain Confusion

 One method is **domain confusion**.

 During training:

 1. Take a batch from the source domain.
2. Take a batch from the target domain.
3. Pass both through the network.
4. Compute the normal task/classification loss:

 $$
L_{\text{cls}}
$$

 5. Compute a loss measuring the difference between source and target feature distributions:

 $$
L_{\text{dom}}.
$$

 6. Optimise a combined objective:

 $$
\boxed{
L = L_{\text{cls}} + L_{\text{dom}}
}
$$

 The result is intended to make the learned feature representation more **domain-invariant**.

---

 # 15\. Domain-Adversarial Neural Networks — DANN

 A related approach is **Domain-Adversarial Neural Networks (DANN)**.

 Instead of explicitly minimising a distance between source and target representations, the system introduces a **domain classifier**.

 The goal is essentially:

 > Learn features that are useful for the task but make it difficult to determine which domain the features came from.

 Thus the representation should contain:

 - information useful for the target task,
- less information that identifies the source vs. target domain.

 This produces a domain-invariant representation through **adversarial training**.

---

 # 16\. Input-Space vs Feature-Space Alignment

 This is an important distinction.

 | Approach | What is aligned? | Basic idea |
| --- | --- | --- |
| **Input-space alignment** | Raw data | Transform source data toward target |
| **Domain translation** | Raw data | Learn a source → target mapping |
| **Domain randomisation** | Source distribution | Make source sufficiently diverse |
| **Feature-space alignment** | Learned representations | Make source/target features similar |
| **Domain confusion** | Feature distributions | Minimise feature-domain discrepancy |
| **DANN** | Feature representations | Adversarially prevent domain identification |

A useful exam distinction is:

 > **Input-space methods change the data; feature-space methods change how the model represents the data.**

---

 # 17\. Pre-Trained Models

 In practice, you often **do not perform the expensive pre-training yourself**.

 For common tasks and datasets, someone has already trained a model and released its weights.

 The practical workflow becomes:

 1. Choose an architecture.
2. Choose a pre-trained checkpoint.
3. Download the weights.
4. Modify the necessary layers.
5. Train/fine-tune using your target dataset.

 For example:

 $$
\text{Pre-trained MobileNet/BERT}
\rightarrow
\text{modify output}
\rightarrow
\text{fine-tune}.
$$

 The lecture gives examples from:

 - **PyTorch/Torchvision** for vision models.
- **Hugging Face Transformers** for language models such as BERT.

---

 # 18\. Pre-Training as Good Initialisation

 An important conceptual point from the lecture is that, when using an existing pre-trained model:

 > **Pre-training can be viewed less as a step you personally perform and more as an extremely good model initialisation.**

 Instead of:

 $$
\text{Random initialisation}
\rightarrow
\text{train from scratch},
$$

 you begin with:

 $$
\boxed{\text{Pre-trained weights}}
\rightarrow
\text{fine-tuning}.
$$

 This saves substantial:

 - training time,
- labelled data,
- computational resources.

---

 # 19\. Foundation Models

 Modern **foundation models** take transfer learning to a much larger scale.

 They are:

 - very large,
- trained on enormous datasets,
- often general-purpose,
- capable of learning representations useful for many downstream tasks.

 Their huge parameter counts make training them from scratch extremely expensive.

 But that expense is amortised because the resulting models can be reused across many tasks.

 ### General pattern

 $$
\text{Massive pre-training}
\rightarrow
\boxed{\text{Foundation model}}
\rightarrow
\begin{cases}
\text{Task A}\\
\text{Task B}\\
\text{Task C}\\
\text{Task D}
\end{cases}
$$

 So the larger the reusable knowledge base, the more downstream applications can benefit from transfer.

---

 # 20\. Combining Pre-Trained Backbones

 A modern architecture can also combine multiple pre-trained components/backbones.

 This is particularly relevant for **multimodal learning**, where different modalities can have their own specialised representations.

 For example:

 $$
\text{Vision backbone}
+
\text{Language backbone}
\rightarrow
\text{Multimodal model}.
$$

 The general principle remains the same:

 > Reuse previously learned representations rather than learning every component from scratch.

---

 # 21\. The Big Picture

 The entire lecture can be understood as a progression:

```
TRAINING FROM SCRATCH
        │
        │ insufficient data / compute
        ▼
TRANSFER LEARNING
        │
        ├── Model-centric
        │      └── Fine-tuning
        │             └── Freeze/retrain layers
        │
        └── Data-centric
               │
               └── Domain Adaptation
                      │
                      ├── Input-space alignment
                      │      ├── CORAL
                      │      ├── Domain translation
                      │      ├── Sim2real
                      │      └── Domain randomisation
                      │
                      └── Feature-space alignment
                             ├── Domain confusion
                             └── DANN
```

---

 # 22\. Most Important Concepts to Remember

 If this lecture is being studied for an exam, these are the concepts I would prioritise:

 ### 1\. Transfer learning

 Using knowledge from a **source domain/task** to improve learning on a **target domain/task**.

 ### 2\. Domain vs task

 - **Domain:** characteristics/distribution of the data.
- **Task:** what the model is trying to predict/solve.

 ### 3\. Inductive vs transductive transfer

 $$
\boxed{\text{Inductive: same domain, different task}}
$$

 $$
\boxed{\text{Transductive: different domain, same/similar task}}
$$

 ### 4\. Fine-tuning

 Start from a pre-trained model and continue training it on the target task.

 ### 5\. Feature extraction

 Freeze some pre-trained layers and train only selected parts, typically later layers.

 ### 6\. Domain shift

 $$
P_S(X)\neq P_T(X)
$$

 Training and target distributions differ, causing a generalisation problem.

 ### 7\. Domain adaptation

 Explicitly compensate for domain shift.

 ### 8\. Input-space alignment

 Modify the data/distribution.

 Examples:

 - CORAL
- domain translation
- sim2real transformations

 ### 9\. Domain randomisation

 Instead of matching source → target, **expand source variability so that the target is covered**.

 ### 10\. Feature-space alignment

 Make source and target produce similar internal representations.

 ### 11\. Domain confusion

 Optimise task performance while reducing differences between source and target feature distributions:

 $$
L=L_{\text{cls}}+L_{\text{dom}}.
$$

 ### 12\. DANN

 Use an adversarial domain classifier to encourage domain-invariant features.

 ### 13\. Pre-trained models

 Reuse publicly available weights rather than performing pre-training yourself.

 ### 14\. Foundation models

 Very large pre-trained models that provide highly reusable representations across many downstream tasks.

---

 # 23\. One Unifying Mental Model

 The most useful way to think about the lecture is:

 > **Transfer learning asks: "What useful information have I already learned that I can reuse?"**

 There are two major places where that knowledge can be reused:

 **Reuse the model's knowledge:**

 $$
\text{Pre-trained model}
\rightarrow
\text{Fine-tuning}
$$

 or **adapt the data/domain:**

 $$
\text{Source domain}
\rightarrow
\text{Domain adaptation}
\rightarrow
\text{Target-compatible representation}.
$$

 And domain adaptation itself can happen either **before/at the input level** or **inside the learned feature space**.

 That distinction—**fine-tuning vs. domain adaptation, and input-space vs. feature-space alignment**—is the central conceptual structure of this lecture.
