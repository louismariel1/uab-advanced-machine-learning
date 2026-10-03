 # Advanced Machine Learning — Transfer Learning MCQ Bank

 ## Part A — Foundations of Transfer Learning

 ### Q1. What is the central idea of transfer learning?

 A. Training a larger model from scratch\
 B. Using knowledge learned from one task/domain to help solve another\
 C. Removing all pre-trained parameters from a model\
 D. Using only unlabeled data

 **Answer: B**

 **Explanation:** Transfer learning reuses knowledge learned from a **source** domain/task to improve learning on a **target** domain/task.

---

 ### Q2. Which situation most strongly motivates transfer learning?

 A. You have unlimited labeled target data and unlimited compute\
 B. The target task is identical to the source task and dataset\
 C. You have limited target data or compute and can exploit previously learned knowledge\
 D. The model has no parameters

 **Answer: C**

 **Explanation:** Transfer learning is particularly valuable when **data and/or computational resources are bottlenecks**.

---

 ### Q3. Which of the following is NOT one of the main benefits of transfer learning?

 A. Data efficiency\
 B. Compute efficiency\
 C. Faster training\
 D. Guaranteed perfect generalisation

 **Answer: D**

 **Explanation:** Transfer learning can improve efficiency and performance, but it does **not guarantee** perfect generalisation.

---

 ### Q4. In the lecture, a "domain" primarily refers to:

 A. The neural-network architecture\
 B. The feature/data distribution\
 C. The loss function only\
 D. The number of model parameters

 **Answer: B**

 **Explanation:** A domain refers to the **feature distribution/statistical characteristics of a dataset**.

---

 ### Q5. Which is an example of a domain difference?

 A. Classification vs. segmentation\
 B. Cross-entropy vs. MSE\
 C. City roads vs. country roads\
 D. ResNet vs. MobileNet

 **Answer: C**

 **Explanation:** City roads and country roads can represent different **data domains**, even if the underlying task is the same.

---

 ### Q6. What does a "task" refer to?

 A. Where the data was collected\
 B. The problem the model is solving\
 C. The number of training samples\
 D. The hardware used for training

 **Answer: B**

 **Explanation:** A task describes the **learning problem**, such as classification, segmentation, object detection, or a particular classification objective.

---

 ### Q7. Which pair correctly distinguishes domain and task?

 A. Domain = objective; task = data distribution\
 B. Domain = architecture; task = optimizer\
 C. Domain = data characteristics/distribution; task = learning problem\
 D. Domain = labels; task = images

 **Answer: C**

 **Explanation:** This distinction is fundamental to understanding the different transfer-learning scenarios.

---

 ## Part B — Formal Definition

 ### Q8. In the lecture notation, $D_S$ represents:

 A. Target domain\
 B. Source domain\
 C. Source task\
 D. Target task

 **Answer: B**

---

 ### Q9. In the lecture notation, $T_T$ represents:

 A. Source task\
 B. Target task\
 C. Source domain\
 D. Target domain

 **Answer: B**

---

 ### Q10. Transfer learning attempts to improve learning of:

 A. $f_S$ in $D_S$\
 B. $f_T$ in $D_T$\
 C. $D_T$ using no source information\
 D. $T_S$ using random initialization

 **Answer: B**

 **Explanation:** The objective is to improve the target predictive function $f_T(\cdot)$ in the target domain $D_T$, using knowledge from the source.

---

 ### Q11. Which condition indicates that source and target domains are different?

 A. $D_S = D_T$\
 B. $D_S \neq D_T$\
 C. $T_S = T_T$\
 D. $T_S \neq T_T$

 **Answer: B**

---

 ### Q12. Which condition indicates that source and target tasks are different?

 A. $D_S \neq D_T$\
 B. $D_S = D_T$\
 C. $T_S \neq T_T$\
 D. $D_T = T_T$

 **Answer: C**

---

 # Part C — Transfer Scenarios

 ### Q13. Inductive transfer learning is characterised by:

 A. Same domain, different task\
 B. Different domain, same task\
 C. Same domain, same task\
 D. Different domain, different model architecture only

 **Answer: A**

 **Explanation:**

 $$
D_S=D_T,\qquad T_S\neq T_T.
$$

 The domain remains the same while the task changes.

---

 ### Q14. Which is an example of inductive transfer?

 A. Training on photographs and testing on medical scans for the same classification task\
 B. Training on vehicle classification and transferring knowledge to road segmentation within the same image domain\
 C. Training and testing on exactly the same dataset\
 D. Changing the optimizer from SGD to Adam

 **Answer: B**

 **Explanation:** The domain is broadly the same, but **classification → segmentation** changes the task.

---

 ### Q15. Transductive transfer learning involves:

 A. Same domain and different task\
 B. Different domains and same or similar task\
 C. No source data\
 D. Different optimizers

 **Answer: B**

 **Explanation:**

 $$
D_S\neq D_T,\qquad T_S\approx T_T.
$$

---

 ### Q16. A model is trained to classify ordinary photographs and then adapted to classify medical images. Which transfer scenario is this most closely associated with?

 A. Inductive transfer\
 B. Transductive transfer\
 C. No transfer\
 D. Meta-learning only

 **Answer: B**

 **Explanation:** The **task remains classification**, but the domain changes substantially.

---

 ### Q17. Why can transfer learning work even when both domain and task differ?

 A. Neural networks do not depend on data\
 B. Useful low-level and mid-level features can be shared across domains/tasks\
 C. The target labels are unnecessary in every case\
 D. The source and target distributions are always identical

 **Answer: B**

 **Explanation:** Learned representations such as edges, textures, shapes and other patterns can remain useful even when the precise task or domain changes.

---

 # Part D — Fine-Tuning

 ### Q18. What is the typical first step in model-centric fine-tuning?

 A. Randomly initialise the entire target model\
 B. Pre-train a model on the source data/task\
 C. Delete the model's feature extractor\
 D. Remove the target dataset

 **Answer: B**

---

 ### Q19. After pre-training, what may need to be changed before training on the target task?

 A. The entire dataset must be discarded\
 B. The architecture/output layers may need modification\
 C. All weights must be reset\
 D. The optimizer must be removed

 **Answer: B**

 **Explanation:** For example, a classification head may need to be replaced when the target has different classes.

---

 ### Q20. What is the final major step of ordinary fine-tuning?

 A. Continue training on target data\
 B. Delete the source weights\
 C. Train only on random noise\
 D. Increase the source dataset

 **Answer: A**

---

 ### Q21. Why are early layers often frozen during fine-tuning?

 A. Early layers contain only random values\
 B. Early layers tend to learn more generic features\
 C. Early layers cannot be trained mathematically\
 D. Later layers contain no useful information

 **Answer: B**

 **Explanation:** Early layers often learn relatively generic visual features, while later layers tend to become more task-specific.

---

 ### Q22. Which statement best describes feature extraction?

 A. Train every layer from scratch\
 B. Freeze some pre-trained layers and use them to extract useful representations\
 C. Remove all convolutional layers\
 D. Replace all source data with noise

 **Answer: B**

---

 ### Q23. Which statement about freezing layers is correct?

 A. You must always freeze exactly half the network\
 B. You must always freeze the first layer only\
 C. The optimal choice depends on the model, data and task\
 D. Freezing layers always produces worse results

 **Answer: C**

 **Explanation:** The lecture explicitly emphasises that deciding what to keep and retrain is a **design/hyperparameter choice** requiring experimentation.

---

 ### Q24. Which statement best captures the relationship between pre-training and fine-tuning?

 A. Pre-training learns a useful representation; fine-tuning adapts it to a new task\
 B. Pre-training always destroys useful features\
 C. Fine-tuning requires random initialisation\
 D. They are unrelated procedures

 **Answer: A**

---

 ### Q25. A model is pre-trained on ImageNet and then its final classification layer is replaced to classify 5 medical categories. What is happening?

 A. Domain randomisation\
 B. Fine-tuning/transfer learning\
 C. DANN\
 D. CORAL

 **Answer: B**

---

 # Part E — Domain Shift

 ### Q26. What is domain shift?

 A. The model architecture changes during training\
 B. Training and target/test data come from different distributions\
 C. The loss function becomes zero\
 D. The number of classes increases

 **Answer: B**

---

 ### Q27. Domain shift can be represented conceptually as:

 A. $P_S(X)=P_T(X)$\
 B. $P_S(X)\neq P_T(X)$\
 C. $T_S=T_T$ only\
 D. $D_S=D_T$ only

 **Answer: B**

---

 ### Q28. Why can domain shift cause generalisation failure?

 A. The model encounters input statistics that differ from those seen during training\
 B. Neural networks cannot process images\
 C. Labels automatically disappear\
 D. The source model becomes smaller

 **Answer: A**

---

 ### Q29. Suppose the source and target images depict the same types of objects, but all target images are significantly darker. What is this primarily an example of?

 A. Task shift\
 B. Domain shift\
 C. Architecture shift\
 D. Label encoding

 **Answer: B**

---

 ### Q30. Fine-tuning on target-domain data can address domain shift, but the lecture describes this approach as:

 A. Impossible\
 B. A brute-force approach\
 C. Unsupervised only\
 D. A form of randomisation

 **Answer: B**

---

 # Part F — Domain Adaptation

 ### Q31. What is the main purpose of domain adaptation?

 A. Increase model size\
 B. Explicitly compensate for differences between source and target domains\
 C. Eliminate all source data\
 D. Change classification into regression

 **Answer: B**

---

 ### Q32. Domain adaptation can be considered the:

 A. Data-centric side of transfer learning\
 B. Opposite of machine learning\
 C. Same thing as random initialisation\
 D. Optimizer-selection process

 **Answer: A**

---

 ### Q33. Which two broad domain-adaptation strategies are discussed?

 A. Parameter pruning and quantisation\
 B. Input-space alignment and feature-space alignment\
 C. Regression and classification\
 D. SGD and Adam

 **Answer: B**

---

 # Part G — Input-Space Alignment

 ### Q34. What does input-space alignment attempt to do?

 A. Change the raw data so source and target distributions become more similar\
 B. Change only the optimizer\
 C. Freeze every model layer\
 D. Remove the target domain

 **Answer: A**

---

 ### Q35. CORAL stands for an approach based on:

 A. Correlation alignment\
 B. Classification randomisation\
 C. Convolutional representation learning\
 D. Contrastive regression

 **Answer: A**

---

 ### Q36. What does CORAL attempt to align?

 A. Model architectures\
 B. First- and second-order statistics of data domains\
 C. Only class labels\
 D. Optimizers

 **Answer: B**

 **Explanation:** CORAL explicitly aligns statistical properties, particularly first- and second-order statistics.

---

 ### Q37. Which statement best describes domain translation?

 A. Learning a mapping from source-domain data to target-domain-like data\
 B. Changing the model optimizer\
 C. Replacing labels with random values\
 D. Increasing the learning rate

 **Answer: A**

---

 ### Q38. Why can input-space alignment sometimes be unsupervised?

 A. It only requires information from the input spaces rather than target labels\
 B. It requires no data\
 C. It uses random labels\
 D. It eliminates the need for a model

 **Answer: A**

---

 ### Q39. Which is an example of a situation where domain translation can be particularly useful?

 A. Converting one visual domain into another related visual domain\
 B. Choosing between Adam and SGD\
 C. Increasing batch size\
 D. Reducing model parameters

 **Answer: A**

---

 # Part H — Sim2Real

 ### Q40. What does "sim2real" mean?

 A. Simulation-to-reality transfer\
 B. Simple-to-regression transfer\
 C. Similarity-to-random transfer\
 D. Simulation-to-regression transfer

 **Answer: A**

---

 ### Q41. Why is sim2real transfer challenging?

 A. Simulated environments are always more complex than reality\
 B. Simulated and real-world data distributions can differ substantially\
 C. Simulators cannot generate data\
 D. Real-world data has no variation

 **Answer: B**

---

 ### Q42. Which application is especially associated with sim2real transfer?

 A. Robotics\
 B. Spreadsheet processing\
 C. Database indexing\
 D. Sorting algorithms

 **Answer: A**

 **Explanation:** Robotics and autonomous driving are highlighted as important applications.

---

 # Part I — Domain Randomisation

 ### Q43. What is the central idea of domain randomisation?

 A. Make the source domain identical to the target\
 B. Make the source domain highly varied so it covers the target domain\
 C. Remove source-domain variability\
 D. Train only on target labels

 **Answer: B**

---

 ### Q44. Why can domain randomisation be more robust than carefully matching the source to the target?

 A. It exposes the model to many possible variations\
 B. It guarantees that no domain shift exists\
 C. It removes the need for training\
 D. It reduces the source dataset to one example

 **Answer: A**

---

 ### Q45. Which statement best captures the key intuition behind domain randomisation?

 A. "Make source look exactly like target."\
 B. "Make target look exactly like source."\
 C. "Expand the source distribution so the target becomes a possible case."\
 D. "Ignore the source distribution."

 **Answer: C**

---

 ### Q46. Why does domain randomisation help with out-of-distribution generalisation?

 A. The target is made into a special case within a broader training distribution\
 B. The target data is deleted\
 C. The model becomes deterministic\
 D. The source model is randomly initialised

 **Answer: A**

---

 ### Q47. A robotics simulator randomly changes lighting, textures, object colours and camera parameters during training. What technique is this?

 A. CORAL\
 B. DANN\
 C. Domain randomisation\
 D. Feature extraction

 **Answer: C**

---

 # Part J — Feature-Space Alignment

 ### Q48. Feature-space alignment operates primarily on:

 A. Raw pixels only\
 B. Internal representations learned by the model\
 C. Optimizer parameters\
 D. Ground-truth labels only

 **Answer: B**

---

 ### Q49. What is the intuition behind feature-space alignment?

 A. Source and target data should produce similar useful representations\
 B. Source and target must have different features\
 C. Features should be random\
 D. Classification heads should be removed

 **Answer: A**

---

 ### Q50. Why can domain shift cause problems for a classification head?

 A. The classifier may receive feature representations unlike those it encountered during training\
 B. The classifier cannot process features\
 C. The classifier always changes the domain\
 D. The feature extractor becomes smaller

 **Answer: A**

---

 # Part K — Domain Confusion

 ### Q51. In the domain-confusion approach, what is passed through the model?

 A. Only source data\
 B. Only target data\
 C. Source and target batches\
 D. Random noise only

 **Answer: C**

---

 ### Q52. What losses are combined in the lecture's domain-confusion formulation?

 A. $L_{\text{cls}}$ and $L_{\text{dom}}$\
 B. $L_{\text{MSE}}$ and $L_{\text{CE}}$ only\
 C. $L_{\text{train}}$ and $L_{\text{test}}$\
 D. $L_{\text{source}}$ and $L_{\text{target}}$

 **Answer: A**

---

 ### Q53. The combined objective presented in the lecture is:

 A. $L=L_{\text{cls}}-L_{\text{dom}}$\
 B. $L=L_{\text{cls}}+L_{\text{dom}}$\
 C. $L=L_{\text{cls}}L_{\text{dom}}$\
 D. $L=L_{\text{dom}}$ only

 **Answer: B**

---

 ### Q54. What does $L_{\text{cls}}$ represent?

 A. Classification/task loss\
 B. Domain distance only\
 C. Model size\
 D. Data augmentation strength

 **Answer: A**

---

 ### Q55. What is the purpose of $L_{\text{dom}}$?

 A. Encourage source and target feature distributions to become similar\
 B. Increase the number of classes\
 C. Increase the image resolution\
 D. Randomise model parameters

 **Answer: A**

---

 ### Q56\. What is the desired result of domain confusion?

 A. A representation in which source and target domains are more aligned\
 B. Completely different representations for source and target\
 C. No representation at all\
 D. Random features

 **Answer: A**

---

 # Part L — DANN

 ### Q57. What does DANN stand for?

 A. Deep Adaptive Neural Network\
 B. Domain-Adversarial Neural Network\
 C. Distributed Artificial Neural Network\
 D. Domain-Aligned Normalised Network

 **Answer: B**

---

 ### Q58. How does DANN differ conceptually from direct feature-distance minimisation?

 A. It uses adversarially trained domain classification\
 B. It removes the feature extractor\
 C. It only uses source labels\
 D. It works only with regression

 **Answer: A**

---

 ### Q59. In DANN, what is the desired behaviour of the learned features with respect to the domain classifier?

 A. Make the domain easy to identify\
 B. Make the domain difficult to identify\
 C. Remove all task information\
 D. Make source and target labels identical

 **Answer: B**

 **Explanation:** The representation should be **domain-invariant** while remaining useful for the actual task.

---

 ### Q60. Which statement best summarises DANN?

 A. Learn task-relevant features while adversarially discouraging domain-specific information\
 B. Learn domain-specific features and ignore the task\
 C. Randomise every feature\
 D. Train exclusively on the target domain

 **Answer: A**

---

 # Part M — Comparing Domain Adaptation Methods

 ### Q61. Which method directly modifies the input distributions?

 A. Input-space alignment\
 B. Feature-space alignment\
 C. DANN\
 D. Feature extraction

 **Answer: A**

---

 ### Q62. Which method attempts to make source and target internal representations similar?

 A. Feature-space alignment\
 B. Domain translation only\
 C. Random initialisation\
 D. Pre-training

 **Answer: A**

---

 ### Q63. Which pairing is correct?

 A. CORAL → feature-space adversarial classifier\
 B. DANN → domain-adversarial feature alignment\
 C. Domain randomisation → freezing neural-network layers\
 D. Fine-tuning → input distribution transformation only

 **Answer: B**

---

 ### Q64. Which pairing is correct?

 A. Domain randomisation → broaden source-domain variability\
 B. CORAL → randomly initialise the target model\
 C. Fine-tuning → align only covariance matrices\
 D. DANN → modify image brightness only

 **Answer: A**

---

 ### Q65. Which statement correctly compares domain translation and domain randomisation?

 A. Both require exactly the same transformation\
 B. Domain translation attempts to map domains toward each other, while randomisation broadens the source distribution\
 C. Domain randomisation changes model weights, while domain translation changes the optimizer\
 D. They are identical

 **Answer: B**

---

 # Part N — Pre-Trained Models

 ### Q66. Why are pre-trained models useful?

 A. They provide learned representations that can be reused for downstream tasks\
 B. They eliminate all need for target data\
 C. They always outperform every model\
 D. They require training from scratch

 **Answer: A**

---

 ### Q67. Which sequence best describes using a pre-trained model?

 A. Randomly initialise → delete weights → train\
 B. Choose architecture → choose checkpoint → download weights → modify layers → train on target data\
 C. Train target model → find source data afterward\
 D. Remove all learned features → fine-tune

 **Answer: B**

---

 ### Q68. What is a checkpoint?

 A. A stored set of model parameters/weights\
 B. A type of loss function\
 C. A dataset label\
 D. A domain-randomisation parameter

 **Answer: A**

---

 ### Q69. What is the main advantage of starting with a pre-trained checkpoint?

 A. It provides a strong initialisation based on previously learned knowledge\
 B. It guarantees no domain shift\
 C. It eliminates model training completely\
 D. It makes the target task identical to the source task

 **Answer: A**

---

 ### Q70. Which statement from the lecture best captures the role of pre-training when using an existing model?

 A. It is essentially a very good initialisation\
 B. It is unnecessary in all circumstances\
 C. It replaces the target task\
 D. It guarantees zero training error

 **Answer: A**

---

 # Part O — Foundation Models

 ### Q71. Why are modern frontier/foundation models expensive to train?

 A. They have enormous parameter counts and require huge computational resources\
 B. They contain no parameters\
 C. They use only tiny datasets\
 D. They cannot perform transfer learning

 **Answer: A**

---

 ### Q72. Why are foundation models particularly useful for transfer learning?

 A. They learn representations useful across a wide variety of downstream tasks\
 B. They only solve one specific task\
 C. They cannot be fine-tuned\
 D. They require every downstream task to start from scratch

 **Answer: A**

---

 ### Q73. What is the key trade-off associated with foundation models?

 A. Expensive large-scale pre-training enables inexpensive reuse across many downstream applications\
 B. Cheap pre-training requires expensive downstream training\
 C. Small models always require more compute\
 D. Foundation models eliminate all computation

 **Answer: A**

---

 ### Q74. What is a multimodal model in the context discussed?

 A. A model combining knowledge/representations from multiple modalities, potentially using multiple pre-trained backbones\
 B. A model trained without data\
 C. A model containing no feature extractor\
 D. A model that only processes images

 **Answer: A**

---

 # Part P — Integrated/Exam-Style Questions

 ### Q75. A model trained on sunny outdoor images performs poorly on nighttime images. The task and object categories remain unchanged. What is the primary issue?

 A. Task transfer\
 B. Domain shift\
 C. Model compression\
 D. Multi-task learning

 **Answer: B**

---

 ### Q76. You have a model trained on photographs and want to classify medical scans. You replace the final classification layer and fine-tune on a small medical dataset. What technique are you primarily using?

 A. Fine-tuning\
 B. Domain randomisation\
 C. DANN\
 D. CORAL

 **Answer: A**

---

 ### Q77. A model is trained to classify cats and dogs and later used to perform pixel-level segmentation of animals. What changed?

 A. Only the domain\
 B. The task\
 C. Only the optimizer\
 D. Nothing

 **Answer: B**

---

 ### Q78\. A robot learns entirely in simulation, but the real-world appearance differs substantially from the simulation. Which transfer problem is most directly relevant?

 A. Sim2real domain shift\
 B. Task compression\
 C. Model pruning\
 D. Classification calibration

 **Answer: A**

---

 ### Q79. A researcher responds to sim2real problems by generating training images with random lighting, textures and object appearances. Which technique is being used?

 A. CORAL\
 B. Domain randomisation\
 C. DANN\
 D. Fine-tuning

 **Answer: B**

---

 ### Q80. A researcher transforms source images so their statistical properties match those of target images. Which approach does this represent?

 A. Input-space alignment\
 B. Feature extraction\
 C. Foundation modelling\
 D. Task transfer

 **Answer: A**

---

 ### Q81. A researcher explicitly aligns first- and second-order statistics between source and target domains. Which method is most associated with this?

 A. CORAL\
 B. DANN\
 C. Fine-tuning\
 D. Domain randomisation

 **Answer: A**

---

 ### Q82. A researcher trains a network so that a classifier cannot reliably determine whether a feature representation came from the source or target domain. What approach is this?

 A. DANN\
 B. CORAL\
 C. Feature extraction\
 D. Domain translation

 **Answer: A**

---

 ### Q83. Which strategy changes the model's internal representation rather than directly transforming the input?

 A. Feature-space alignment\
 B. Input-space alignment\
 C. Domain translation\
 D. Image preprocessing only

 **Answer: A**

---

 ### Q84. Which strategy deliberately makes the source distribution broader instead of trying to make it identical to the target?

 A. Domain randomisation\
 B. CORAL\
 C. DANN\
 D. Fine-tuning

 **Answer: A**

---

 ### Q85. You have only 500 labeled target images but access to a model trained on millions of related images. What is the most obvious strategy to consider?

 A. Discard the pre-trained model and train from scratch\
 B. Transfer learning/fine-tuning\
 C. Remove all target labels\
 D. Randomise all weights

 **Answer: B**

---

 # Part Q — Higher-Level Reasoning Questions

 ### Q86. Why can a model trained on a completely different dataset still provide useful features?

 A. Some fundamental patterns/features are shared across datasets\
 B. Neural networks memorise nothing\
 C. All datasets have identical distributions\
 D. Classification tasks are always identical

 **Answer: A**

 **Explanation:** Particularly in vision, lower-level patterns such as edges, textures and shapes can be useful across many datasets.

---

 ### Q87. Which scenario would make transfer learning least necessary?

 A. Very little target data\
 B. Very expensive training\
 C. Large target dataset closely matching the training distribution\
 D. Large domain shift

 **Answer: C**

 **Explanation:** If you already have abundant representative target data and sufficient compute, training from scratch may be perfectly reasonable.

---

 ### Q88. Why might freezing too many layers hurt performance?

 A. The model may not have enough flexibility to adapt useful representations to the target problem\
 B. Frozen layers become random\
 C. The dataset disappears\
 D. The optimizer stops existing

 **Answer: A**

---

 ### Q89. Why might freezing too few layers be undesirable when target data is scarce?

 A. The model may unnecessarily modify useful pre-trained representations and overfit\
 B. The model becomes incapable of training\
 C. Domain shift becomes mathematically impossible\
 D. The source model has no features

 **Answer: A**

---

 ### Q90. Which statement best captures the relationship between transfer learning and data efficiency?

 A. Transfer learning allows useful information from another dataset/task to reduce how much target data must be learned from scratch\
 B. Transfer learning always increases the amount of target data required\
 C. Transfer learning requires no data whatsoever\
 D. Transfer learning makes labels unnecessary in every situation

 **Answer: A**

---

 ### Q91. Consider:

 $$
D_S\neq D_T,\qquad T_S\approx T_T.
$$

 Which scenario does this most closely describe?

 A. Inductive transfer\
 B. Transductive transfer/domain adaptation\
 C. No transfer\
 D. Multi-task learning

 **Answer: B**

---

 ### Q92. Consider:

 $$
D_S=D_T,\qquad T_S\neq T_T.
$$

 Which scenario does this describe?

 A. Inductive transfer\
 B. Transductive transfer\
 C. Domain randomisation\
 D. DANN

 **Answer: A**

---

 ### Q93\. Which approach would be most appropriate if your main problem is that the target images have different low-level statistics but similar underlying semantic relationships?

 A. Domain adaptation\
 B. Completely unrelated task learning\
 C. Randomly initialise the entire model\
 D. Remove the feature extractor

 **Answer: A**

---

 ### Q94. Which approach is most directly intended to learn representations that are useful for the task while being insensitive to domain identity?

 A. DANN\
 B. CORAL only\
 C. Domain randomisation only\
 D. Random initialisation

 **Answer: A**

---

 ### Q95. What is the fundamental difference between ordinary fine-tuning and domain adaptation?

 A. Fine-tuning primarily reuses/adapts model parameters, while domain adaptation explicitly addresses differences between source and target domains\
 B. Fine-tuning never uses pre-trained models\
 C. Domain adaptation never uses data\
 D. They are exactly identical concepts

 **Answer: A**

---

 # Part R — Challenging Mixed Questions

 ### Q96. Suppose you have:

 - Source: synthetic driving images
- Target: real driving images
- Same object-detection task

 Which statement is most accurate?

 A. $D_S=D_T$, $T_S\neq T_T$\
 B. $D_S\neq D_T$, $T_S\approx T_T$\
 C. $D_S=D_T$, $T_S=T_T$\
 D. There is no transfer problem

 **Answer: B**

 **Explanation:** Synthetic and real images are different domains, while object detection remains the same/similar task. This is a classic **sim2real/domain-adaptation** scenario.

---

 ### Q97. Suppose a model learns useful edge detectors in its first layers on natural images. You then train it on a small medical-image dataset. Why might freezing early layers be reasonable?

 A. Edges and other low-level patterns can remain useful across domains\
 B. Medical images contain no edges\
 C. Frozen parameters automatically adapt\
 D. Early layers only contain labels

 **Answer: A**

---

 ### Q98. You have source and target data but no target labels. Which approach from the lecture could potentially operate without target labels?

 A. Input-space domain alignment\
 B. Standard supervised target fine-tuning\
 C. Target classification loss requiring labels\
 D. None

 **Answer: A**

 **Explanation:** Input-space alignment can work using the input distributions themselves. The lecture specifically notes that domain translation can work in an **unsupervised** setting because only the input space is needed.

---

 ### Q99. Why might domain randomisation be preferable to attempting an exact source-to-target transformation?

 A. Exact alignment may be difficult, while broadening the source distribution can make the target a covered case\
 B. Randomisation always requires fewer training examples\
 C. It eliminates the need for a model\
 D. It guarantees identical distributions

 **Answer: A**

---

 ### Q100. Which sequence best represents the evolution of the lecture's main ideas?

 A. Foundation models → randomisation → classification → regression\
 B. Supervised learning limitations → transfer learning → fine-tuning → domain shift → domain adaptation → modern pre-trained/foundation models\
 C. Classification → pruning → quantisation → regression\
 D. DANN → random initialisation → source training → domain definition

 **Answer: B**

 **Explanation:** This captures the conceptual flow of the lecture.

---

 # Rapid-Fire Revision Quiz

 Try answering these **without looking back at the explanations**.

 ### Q101. Domain means:

 A. Data distribution\
 B. Loss function\
 C. Optimizer\
 D. Architecture

 **Answer: A**

 ### Q102. Task means:

 A. Data source\
 B. Problem being solved\
 C. Hardware\
 D. Dataset size

 **Answer: B**

 ### Q103. Same domain \+ different task:

 A. Inductive transfer\
 B. Transductive transfer

 **Answer: A**

 ### Q104. Different domain \+ similar task:

 A. Inductive transfer\
 B. Transductive transfer

 **Answer: B**

 ### Q105. Training a pre-trained network further on target data:

 A. Fine-tuning\
 B. Domain randomisation

 **Answer: A**

 ### Q106. Freezing early layers is motivated by:

 A. Generic early features\
 B. Random early features

 **Answer: A**

 ### Q107. Training and target distributions differ:

 A. Domain shift\
 B. Task collapse

 **Answer: A**

 ### Q108. CORAL aligns:

 A. First- and second-order statistics\
 B. Model architectures

 **Answer: A**

 ### Q109. Mapping source data toward target data:

 A. Domain translation\
 B. DANN

 **Answer: A**

 ### Q110. Making source data highly diverse:

 A. Domain randomisation\
 B. CORAL

 **Answer: A**

 ### Q111. Aligning learned representations:

 A. Feature-space alignment\
 B. Input-space alignment

 **Answer: A**

 ### Q112. Objective involving $L_{\text{cls}}+L_{\text{dom}}$:

 A. Domain confusion\
 B. Random initialisation

 **Answer: A**

 ### Q113. Adversarially confusing a domain classifier:

 A. DANN\
 B. CORAL

 **Answer: A**

 ### Q114. Reusing publicly available learned weights:

 A. Pre-trained model\
 B. Domain shift

 **Answer: A**

 ### Q115. Very large general-purpose pre-trained models:

 A. Foundation models\
 B. Domain classifiers

 **Answer: A**

---

 # Final Concept Map

 For exam revision, memorise this structure:

```
                         TRANSFER LEARNING
                                │
             ┌──────────────────┴──────────────────┐
             │                                     │
       MODEL-CENTRIC                         DATA-CENTRIC
             │                                     │
       Fine-tuning                           Domain Adaptation
             │                                     │
       ┌─────┴─────┐                    ┌──────────┴──────────┐
       │           │                    │                     │
    Retrain      Freeze             Input-space          Feature-space
    layers       layers              alignment             alignment
       │                                  │                     │
       │                         ┌────────┼────────┐       ┌────┴────┐
       │                         │        │        │       │         │
       │                       CORAL   Domain   Domain   Domain    DANN
       │                              translation randomisation confusion
       │
       └─────────────── Pre-trained models
                              │
                              ▼
                       Foundation models
```

 And the **four equations/relationships worth knowing especially well** are:

 $$
\boxed{D_S=D_T,\quad T_S\neq T_T
\quad\Rightarrow\quad \text{Inductive transfer}}
$$

 $$
\boxed{D_S\neq D_T,\quad T_S\approx T_T
\quad\Rightarrow\quad \text{Transductive transfer}}
$$

 $$
\boxed{P_S(X)\neq P_T(X)
\quad\Rightarrow\quad \text{Domain shift}}
$$

 $$
\boxed{L=L_{\mathrm{cls}}+L_{\mathrm{dom}}
\quad\Rightarrow\quad \text{Domain-confusion objective}}
$$

 ### The single most important distinction

 If you remember only one thing from this lecture, remember:

 > **Fine-tuning transfers knowledge through the model; domain adaptation tries to deal explicitly with differences between the source and target domains.**

 And within domain adaptation:

 > **Input-space alignment changes the data; feature-space alignment changes/aligns the representations.**
