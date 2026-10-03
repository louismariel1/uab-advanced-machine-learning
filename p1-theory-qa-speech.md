 ## Complete Lecturer Narrative + Interactive Question-and-Answer Script

---

 # Part 1 — Opening the Lecture

 Good morning everyone.

 Today we're going to look at an important idea in modern machine learning called **transfer learning**.

 The central question I want you to keep in your mind throughout this lecture is very simple:

 > **If a machine-learning model has already learned something useful, why should we make it learn everything again from scratch?**

 Imagine that I have a model that has already seen millions of images.

 It has learned about edges, textures, shapes, colours, objects and visual patterns.

 Now I give you a new problem where I only have a small amount of labelled data.

 One option is to throw away everything the model has learned and start again with random weights.

 But that seems wasteful.

 Instead, we can ask:

 > **What knowledge from the previous problem can we reuse for the new problem?**

 That is the fundamental idea behind **transfer learning**.

---

 # Part 2 — Why Do We Need Transfer Learning?

 Let's begin with ordinary supervised learning.

 Normally, our workflow looks something like this:

 We collect training data.

 We train a model.

 The model learns a relationship between inputs and outputs.

 Then we deploy it on new data.

 This works particularly well when we have enough training data, enough computational resources, and when our training and deployment data are reasonably similar.

 But what happens when these assumptions don't hold?

 Suppose I only have 500 labelled examples for my target problem.

 Training a large neural network from scratch on 500 examples may be difficult.

 Or perhaps training the original model required enormous computational resources.

 Or perhaps my new dataset comes from a somewhat different environment.

 This is where transfer learning becomes useful.

 There are three important benefits I want you to remember.

 First:

 **Data efficiency.**

 We may need fewer labelled examples in the target domain.

 Second:

 **Compute efficiency.**

 We can reuse expensive pre-training rather than repeating it.

 Third:

 **Faster training.**

 The model doesn't start from completely random knowledge. It starts from a useful representation.

 So the general philosophy is:

 > **Don't learn everything from scratch if useful knowledge has already been learned somewhere else.**

---

 # Part 3 — What Exactly Is Being Transferred?

 Now we need to be precise about what we mean by "somewhere else."

 Transfer learning involves a **source** and a **target**.

 The source is where our existing knowledge comes from.

 The target is the new problem we actually want to solve.

 We therefore talk about:

 - source domain,
- source task,
- target domain,
- target task.

 This leads us to two concepts that you absolutely need to distinguish:

 **domain** and **task**.

---

 # Part 4 — Domain Versus Task

 Let's first define a domain.

 A **domain** describes the characteristics or statistical distribution of the data.

 For example:

 City roads and country roads could represent different domains.

 Photographs and medical scans could represent different domains.

 Images from one social-media platform and images from another platform could potentially represent different domains.

 The important point is that the domain is about the **data**.

 Now compare that with a **task**.

 A task describes what we want the model to do.

 For example:

 - classification,
- object detection,
- image segmentation,
- classifying cats versus dogs,
- classifying different vehicle types.

 So remember:

 > **Domain describes the data. Task describes the problem being solved.**

 This distinction is going to become extremely important when we discuss inductive and transductive transfer learning.

---

 # Part 5 — Formal Definition

 Let's introduce the notation.

 Let:

 - $D_S$ be the source domain.
- $T_S$ be the source task.
- $D_T$ be the target domain.
- $T_T$ be the target task.

 Our goal is to improve the target predictive function:

 $$
f_T(\cdot)
$$

 for the target task in the target domain by using knowledge learned from the source.

 At least one aspect of the problem is different.

 We may have:

 $$
D_S \neq D_T
$$

 meaning the domains are different.

 Or:

 $$
T_S \neq T_T
$$

 meaning the tasks are different.

 The important idea is that we are transferring knowledge because the source and target are not simply the exact same learning problem.

---

 # Part 6 — Inductive and Transductive Transfer

 Now let's distinguish two important scenarios.

 ## Inductive transfer learning

 In **inductive transfer**, the domain stays the same but the task changes.

 So:

 $$
D_S = D_T
$$

 but:

 $$
T_S \neq T_T.
$$

 For example, imagine we train a model to classify vehicle images.

 Later, we want to use knowledge from that model for road-image segmentation.

 The general image domain is similar, but the task has changed.

 We have gone from:

 > "What class does this image belong to?"

 to:

 > "Which pixels belong to which objects?"

 So that's **inductive transfer**.

---

 Now let's consider the opposite situation.

 ## Transductive transfer learning

 Here, the task stays the same or is very similar, but the domain changes.

 Conceptually:

 $$
D_S \neq D_T
$$

 while:

 $$
T_S \approx T_T.
$$

 For example, suppose I train an image classifier using ordinary photographs.

 I then want to classify medical images.

 The task is still classification, but the input domain is substantially different.

 This is closely related to **domain adaptation**.

 So here's the easiest way to remember the distinction:

 > **Inductive: same domain, different task.**

 > **Transductive: different domain, same or similar task.**

---

 # Part 7 — Let's Test That Understanding

 Let me ask you a question.

 ### Question

 Suppose I train a model on vehicle classification and then transfer it to road-image segmentation.

 Did the domain change, or did the task change?

 Pause here and think about it.

 The answer is:

 **The task changed.**

 We're still dealing with a broadly similar image domain, but we changed from classification to segmentation.

 Therefore:

 $$
D_S = D_T,\qquad T_S \neq T_T
$$

 and this is **inductive transfer**.

---

 Now another question.

 Suppose I train an object detector using synthetic driving images and then use the knowledge on real driving images.

 What has changed?

 The task is still object detection.

 But the data has changed from synthetic to real.

 Therefore:

 $$
D_S \neq D_T,\qquad T_S \approx T_T.
$$

 That is **transductive transfer**, and it is a classic domain-adaptation situation.

---

 # Part 8 — Model-Centric Transfer Learning

 Now let's move to the most common practical form of transfer learning:

 **fine-tuning**.

 The idea is very straightforward.

 First, we train a model on a source dataset.

 This produces a pre-trained model.

 Then we take that model and adapt it to our target problem.

 We may modify some layers, particularly the output layer.

 Then we continue training using target data.

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

 Why does this work?

 Because neural networks often learn representations at different levels.

 In a vision network, early layers may learn:

 - edges,
- textures,
- simple shapes,
- local patterns.

 These features can be useful across many different image datasets.

 Later layers tend to become more specific to the particular task.

 So we can think of the network roughly as:

 $$
\boxed{\text{Early layers} \rightarrow \text{generic features}}
$$

 and:

 $$
\boxed{\text{Later layers} \rightarrow \text{task-specific features}}.
$$

 Therefore, a pre-trained model gives us a useful **feature-extraction backbone**.

 Fine-tuning then adapts those learned features to the new problem.

---

 # Part 9 — Freezing Layers

 Do we always have to retrain the entire network?

 No.

 We can **freeze** some layers.

 When a layer is frozen, its weights are kept fixed during target training.

 We might freeze early layers because they contain useful generic representations and only train later layers.

 This can be particularly useful when the target dataset is small.

 But there is an important qualification.

 There is no universal rule saying:

 > "Always freeze exactly these layers."

 The appropriate strategy depends on:

 - the architecture,
- the similarity between source and target domains,
- the amount of target data,
- the similarity between source and target tasks.

 So freezing layers is a design decision and often requires experimentation.

---

 # Part 10 — Interactive Question: Fine-Tuning

 Let's test this.

 ### Question

 Suppose I have a model trained on ImageNet.

 I replace its final classification layer so that it predicts five medical categories.

 Then I continue training using a small medical dataset.

 What am I doing?

 The answer is:

 **Fine-tuning / transfer learning.**

 I'm reusing the pre-trained model and adapting it to the target problem.

---

 Another question.

 Why might I freeze the early layers?

 Because the early layers often learn generic features such as edges and textures.

 But here's another important question:

 ### Could freezing too many layers be a problem?

 Yes.

 If I freeze too much of the network, the model may not have enough flexibility to adapt its representations to the new target problem.

 And what if I freeze too few layers?

 When target data is scarce, I might modify too much of the useful pre-trained representation and potentially overfit.

 So there is a trade-off.

---

 # Part 11 — Domain Shift

 Now let's move to one of the most important ideas in this lecture:

 **domain shift**.

 Suppose I train a model using bright daytime images.

 Then I deploy it on dark nighttime images.

 The objects may be the same.

 The task may be exactly the same.

 But the statistical characteristics of the input data have changed.

 Conceptually:

 $$
P_S(X) \neq P_T(X).
$$

 This is **domain shift**.

 The model has learned from one input distribution, but deployment presents it with another.

 This can produce a generalisation gap.

 The important point is:

 > The model may understand the underlying concept, but the input statistics have changed.

---

 # Part 12 — Domain Shift Question

 Let's test that.

 ### Question

 A model is trained on sunny outdoor images and performs poorly on nighttime images.

 The object categories and classification task remain unchanged.

 What is the main problem?

 The answer is:

 **Domain shift.**

 The task hasn't changed.

 The input distribution has changed.

 So:

 $$
P_S(X)\neq P_T(X).
$$

---

 # Part 13 — Domain Adaptation

 Now, if domain shift is the problem, we can ask:

 > How can we explicitly deal with it?

 This leads us to **domain adaptation**.

 Domain adaptation focuses specifically on compensating for differences between the source and target domains.

 There are two broad strategies that I want you to distinguish.

 The first is:

 > **Input-space alignment.**

 The second is:

 > **Feature-space alignment.**

 This distinction is extremely important.

 I want you to remember:

 > **Input-space methods change the data.**

 > **Feature-space methods change or align the internal representation.**

 Let's examine both.

---

 # Part 14 — Input-Space Alignment

 Suppose I have source images and target images.

 Instead of changing the neural network's internal representation, I could modify the data so that the source and target distributions become more similar.

 That's input-space alignment.

 One example is **CORAL**.

 CORAL stands for **Correlation Alignment**.

 It attempts to align statistical properties of the source and target domains, particularly first-order and second-order statistics.

 So if an exam question asks:

 > What does CORAL align?

 The answer is:

 > **First- and second-order statistics.**

---

 # Part 15 — Domain Translation

 Another input-space approach is **domain translation**.

 Here we learn a mapping such as:

 $$
g:X_S\rightarrow X_T.
$$

 The idea is to transform source-domain examples into target-domain-like examples.

 For example, imagine that our source images are generated in one visual style and our target images have another appearance.

 We can learn a transformation that makes source samples look more like the target domain.

 An important point is that this can potentially be done without target labels because we can operate directly on the input distributions.

 So if you see:

 > "Map source-domain data toward target-domain data"

 think:

 **Domain translation.**

---

 # Part 16 — Generative Domain Translation

 Sometimes the difference between two domains is more complicated.

 Maybe the difference isn't just brightness or contrast.

 Perhaps the entire visual appearance is different.

 In those cases, generative models can be used to learn more sophisticated domain mappings.

 The important conceptual point is not the particular generative architecture.

 The important point is:

 > We are changing the input representation so that source and target domains become more compatible.

---

 # Part 17 — Sim2Real

 This leads naturally to an important application:

 **simulation-to-reality**, or **sim2real**.

 Imagine that I'm developing a robot.

 Collecting millions of real-world training examples could be expensive and time-consuming.

 Instead, I can generate huge amounts of training data in a simulator.

 That's very attractive.

 But there's a problem.

 Simulation is not reality.

 The simulated images may have different:

 - lighting,
- textures,
- object appearances,
- camera characteristics,
- environmental effects.

 Therefore:

 $$
P_{\text{simulation}}(X)
\neq
P_{\text{real}}(X).
$$

 We have a domain shift.

 This is why sim2real transfer is important in robotics and autonomous systems.

---

 # Part 18 — Domain Randomisation

 Now here's an interesting alternative.

 Instead of trying to make simulation look exactly like reality, we can do something different.

 We can make the simulation **extremely diverse**.

 This is called **domain randomisation**.

 For example, during training we randomly change:

 - lighting,
- colours,
- textures,
- camera properties,
- object appearances,
- environmental conditions.

 The philosophy is:

 > Don't necessarily make the source look exactly like the target.

 Instead:

 > **Make the source distribution broad enough that the target becomes one of the possible cases.**

 Conceptually:

 $$
\boxed{\text{Large/diverse source distribution}}
\supseteq
\text{Target}.
$$

 This is fundamentally different from domain alignment.

 With domain alignment, we're thinking:

 $$
\text{Source}\rightarrow\text{Target}.
$$

 With domain randomisation, we're thinking:

 $$
\text{Make Source much broader}.
$$

---

 # Part 19 — Domain Randomisation Question

 Suppose a robotics simulator randomly changes lighting, textures, colours and camera parameters every time an image is generated.

 What technique is this?

 The answer is:

 **Domain randomisation.**

 And why?

 Because we aren't trying to create one precise transformation from simulation to reality.

 We're deliberately creating a broad training distribution.

 The target environment should then become less unusual to the model.

---

 # Part 20 — Feature-Space Alignment

 Now let's consider the second major type of domain adaptation:

 **feature-space alignment**.

 Here we don't directly change the raw input.

 Instead, we ask:

 > Can we make source and target examples produce similar internal representations?

 Suppose:

 $$
\text{Source image}
\rightarrow
\text{feature extractor}
\rightarrow
\text{features}
$$

 and:

 $$
\text{Target image}
\rightarrow
\text{feature extractor}
\rightarrow
\text{features}.
$$

 We want the resulting feature distributions to be more similar.

 Why?

 Because the classification head is then less likely to encounter a representation that looks completely unfamiliar.

 So the fundamental idea is:

 > **Make the representation domain-invariant while preserving information useful for the task.**

---

 # Part 21 — Domain Confusion

 One approach is called **domain confusion**.

 Here's the basic process.

 We take a batch from the source domain.

 We take a batch from the target domain.

 We pass both through the network.

 We calculate the ordinary task or classification loss:

 $$
L_{\text{cls}}.
$$

 Then we calculate another loss that measures the difference between source and target feature distributions:

 $$
L_{\text{dom}}.
$$

 We combine them:

 $$
\boxed{
L=L_{\text{cls}}+L_{\text{dom}}
}
$$

 The objective is to maintain good task performance while reducing the discrepancy between source and target representations.

---

 # Part 22 — Domain Confusion Question

 Let's check this.

 ### Question

 What does $L_{\text{cls}}$ represent?

 It represents the **task or classification loss**.

 What does $L_{\text{dom}}$ represent?

 It measures the discrepancy between the domain representations.

 And what is the combined objective presented in the lecture?

 $$
L=L_{\text{cls}}+L_{\text{dom}}.
$$

 The goal is to obtain a representation where source and target are more aligned.

---

 # Part 23 — DANN

 Now we have a related but particularly important technique:

 **DANN**.

 DANN stands for:

 > **Domain-Adversarial Neural Network.**

 The basic idea is slightly different from simply defining a direct feature-distance measure.

 We introduce a **domain classifier**.

 This domain classifier tries to determine:

 > Did this representation come from the source domain or the target domain?

 But the feature extractor is trained to make that distinction difficult.

 So we have two competing objectives.

 The task representation should be:

 - useful for the actual prediction task,
- but less useful for identifying the domain.

 In other words:

 > **Learn task-relevant features while removing domain-specific information.**

 That produces a more **domain-invariant representation**.

---

 # Part 24 — DANN Question

 Imagine that I train a network so that a domain classifier has difficulty determining whether a feature came from the source or target domain.

 What technique am I describing?

 **DANN.**

 Why?

 Because DANN uses adversarial training to encourage domain-invariant features.

 And what does the domain classifier want?

 It wants to identify the domain.

 What does the feature extractor want?

 It wants to make domain identification difficult while still solving the task.

 That adversarial relationship is the key idea.

---

 # Part 25 — Comparing the Methods

 Let's pause and put everything together.

 If I change the **raw data**, I'm working in:

 **input space.**

 Examples include:

 - CORAL,
- domain translation,
- certain sim2real transformations.

 If I change or align the **internal representations**, I'm working in:

 **feature space.**

 Examples include:

 - domain confusion,
- DANN.

 And domain randomisation is slightly different again.

 Rather than transforming the source to exactly match the target, I make the source distribution sufficiently broad and diverse.

 So the conceptual map is:

 $$
\text{Domain Adaptation}
$$

 splits into:

 $$
\text{Input-space alignment}
$$

 and:

 $$
\text{Feature-space alignment}.
$$

 Input-space methods change the data.

 Feature-space methods change the representation.

---

 # Part 26 — Pre-Trained Models

 Now let's connect all of this to what we actually do in practice.

 In many real projects, you don't personally perform the expensive pre-training.

 Someone has already trained a model.

 They release the model architecture and learned weights.

 We can download those weights and use them as a starting point.

 The practical workflow is:

 1. Choose an architecture.
2. Choose a pre-trained checkpoint.
3. Download the weights.
4. Modify the necessary layers.
5. Fine-tune using the target dataset.

 For example, we might take a pre-trained vision model, replace its output layer, and train it on our target problem.

 The important conceptual point is:

 > **Pre-training provides a very good initialisation.**

 Instead of:

 $$
\text{Random weights}
\rightarrow
\text{training from scratch},
$$

 we have:

 $$
\boxed{\text{Pre-trained weights}}
\rightarrow
\text{fine-tuning}.
$$

 That can save enormous amounts of time, data and computation.

---

 # Part 27 — Foundation Models

 And this idea becomes even more powerful with **foundation models**.

 Foundation models are very large models trained on enormous datasets.

 Their training is extremely expensive.

 But the important idea is that the cost of that training can be reused.

 Instead of training one enormous model for one task, we train a general-purpose model and then adapt it to many downstream applications.

 Conceptually:

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

 This is transfer learning at a much larger scale.

 The expensive pre-training is effectively amortised over many downstream applications.

---

 # Part 28 — Multimodal Transfer

 We can also combine multiple pre-trained components.

 For example, we might have:

 $$
\text{Vision backbone}
+
\text{Language backbone}
\rightarrow
\text{Multimodal model}.
$$

 The vision component provides knowledge about visual information.

 The language component provides knowledge about language.

 Rather than learning everything from scratch jointly, we can reuse specialised representations.

 Again, the same principle appears:

 > **Reuse useful knowledge rather than relearning everything.**

---

 # Part 29 — Integrated MCQ Discussion

 Now I want to revisit the entire lecture through questions.

 I would encourage you to answer each question before I give you the explanation.

---

 ## Question 1 — Central Idea

 What is the central idea of transfer learning?

 A. Train a larger model from scratch.

 B. Use knowledge learned from one task or domain to help solve another.

 C. Remove all pre-trained parameters.

 D. Use only unlabelled data.

 The answer is:

 **B.**

 Transfer learning is about reusing knowledge from a source to improve learning on a target.

---

 ## Question 2 — Why Transfer Learning?

 Which situation most strongly motivates transfer learning?

 A. Unlimited target data and compute.

 B. Identical source and target problems.

 C. Limited target data or compute where previously learned knowledge can be reused.

 D. A model with no parameters.

 The answer is:

 **C.**

 The key motivations are data efficiency, compute efficiency and faster training.

---

 ## Question 3 — Domain

 What does a domain primarily describe?

 A. Neural-network architecture.

 B. Data characteristics and distribution.

 C. Loss function only.

 D. Number of parameters.

 The answer is:

 **B.**

 Domain means the statistical characteristics of the data.

---

 ## Question 4 — Task

 What does a task describe?

 A. Where the data was collected.

 B. The problem the model is solving.

 C. Hardware.

 D. Number of samples.

 The answer is:

 **B.**

 Classification, segmentation and object detection are examples of tasks.

---

 ## Question 5 — Formal Notation

 What does $D_S$ represent?

 **Source domain.**

 What does $T_T$ represent?

 **Target task.**

 And what does transfer learning try to improve?

 The target predictive function:

 $$
f_T(\cdot).
$$

---

 ## Question 6 — Inductive Transfer

 Suppose:

 $$
D_S=D_T
$$

 and:

 $$
T_S\neq T_T.
$$

 What type of transfer is this?

 **Inductive transfer.**

 Remember:

 > Same domain, different task.

---

 ## Question 7 — Transductive Transfer

 Suppose:

 $$
D_S\neq D_T
$$

 and:

 $$
T_S\approx T_T.
$$

 What type of transfer is this?

 **Transductive transfer.**

 Remember:

 > Different domain, same or similar task.

---

 ## Question 8 — Fine-Tuning

 What happens during fine-tuning?

 A. Start from random weights.

 B. Start from a pre-trained model and continue training on target data.

 C. Delete the feature extractor.

 D. Remove target data.

 The answer is:

 **B.**

---

 ## Question 9 — Frozen Layers

 Why might we freeze early layers?

 Because early layers often contain relatively generic features such as edges, textures and simple shapes.

 But is there one universally correct number of layers to freeze?

 **No.**

 It depends on the architecture, source-target similarity and target data.

---

 ## Question 10 — Domain Shift

 If:

 $$
P_S(X)\neq P_T(X),
$$

 what does this indicate?

 **Domain shift.**

 The source and target input distributions differ.

---

 ## Question 11 — Domain Adaptation

 What is the main purpose of domain adaptation?

 To:

 > **Explicitly compensate for differences between source and target domains.**

---

 ## Question 12 — Input or Feature Space?

 If I modify the raw source images so they look more like the target images, am I doing input-space or feature-space alignment?

 **Input-space alignment.**

 If I instead make the internal representations of source and target more similar?

 **Feature-space alignment.**

 This distinction is one of the most important in the lecture.

---

 ## Question 13 — CORAL

 What does CORAL do?

 It performs **Correlation Alignment**.

 It aligns statistical properties, particularly first- and second-order statistics.

 So:

 > CORAL → statistical alignment.

---

 ## Question 14 — Domain Translation

 What does domain translation learn?

 A mapping:

 $$
g:X_S\rightarrow X_T.
$$

 In words:

 > Transform source-domain samples into target-domain-like samples.

---

 ## Question 15 — Sim2Real

 What does sim2real mean?

 **Simulation-to-reality transfer.**

 Why is it difficult?

 Because:

 $$
P_{\text{simulation}}(X)
\neq
P_{\text{real}}(X).
$$

 The simulated and real environments have different distributions.

---

 ## Question 16 — Domain Randomisation

 A simulator randomly varies lighting, colours, textures and camera parameters.

 What technique is this?

 **Domain randomisation.**

 And what is its philosophy?

 Not:

 > "Make source identical to target."

 But:

 > **"Make source sufficiently diverse that target is covered by the training distribution."**

---

 ## Question 17 — Feature-Space Alignment

 Where does feature-space alignment operate?

 **Inside the learned representation space.**

 The goal is to make source and target representations more similar.

---

 ## Question 18 — Domain Confusion

 What two losses appear in the lecture's domain-confusion objective?

 $$
L_{\text{cls}}
$$

 and:

 $$
L_{\text{dom}}.
$$

 The combined objective is:

 $$
\boxed{
L=L_{\text{cls}}+L_{\text{dom}}
}
$$

 The first preserves task performance.

 The second encourages source-target feature alignment.

---

 ## Question 19 — DANN

 What does DANN stand for?

 **Domain-Adversarial Neural Network.**

 What makes it different?

 It uses a domain classifier in an adversarial framework.

 The desired representation is:

 > useful for the task but difficult to identify by domain.

---

 ## Question 20 — Compare DANN and CORAL

 Suppose someone says:

 > "I explicitly align first- and second-order statistics."

 Which method comes to mind?

 **CORAL.**

 Suppose someone says:

 > "I train a domain classifier adversarially so that the learned features hide domain identity."

 Which method?

 **DANN.**

---

 ## Question 21 — Domain Translation vs Randomisation

 What is the difference?

 Domain translation attempts to map:

 $$
\text{Source}\rightarrow\text{Target}.
$$

 Domain randomisation instead broadens:

 $$
\text{Source distribution}.
$$

 So:

 > Translation tries to move the source toward the target.

 > Randomisation tries to make the source broad enough to contain the target.

---

 ## Question 22 — Pre-Trained Models

 Why are pre-trained models useful?

 Because they provide learned representations that can be reused.

 Instead of random initialisation, we begin with a strong learned representation.

 That is why we often describe pre-training as:

 > **A very good initialisation.**

---

 ## Question 23 — Foundation Models

 Why are foundation models useful?

 Because very expensive large-scale pre-training creates representations that can be reused across many downstream tasks.

 The cost is high up front.

 But the resulting knowledge can be reused many times.

---

 # Part 30 — Challenging Scenario Questions

 Now let's move to questions where I want you to reason rather than simply recall definitions.

---

 ## Scenario 1

 You have only 500 labelled target images.

 You have access to a model trained on millions of related images.

 What should you consider?

 The obvious answer is:

 **Transfer learning and fine-tuning.**

 Why?

 Because the pre-trained representation can reduce how much the target model has to learn from scratch.

---

 ## Scenario 2

 A model trained on photographs is used on medical scans.

 The task remains classification.

 What has changed?

 The **domain**.

 Therefore this is primarily a domain-transfer/domain-adaptation situation.

---

 ## Scenario 3

 A model trained for vehicle classification is adapted for road-image segmentation.

 What changed?

 The **task**.

 Therefore this is an example of **inductive transfer**.

---

 ## Scenario 4

 A robot learns entirely in simulation and then performs poorly in reality.

 What is the central problem?

 **Sim2real domain shift.**

---

 ## Scenario 5

 Instead of trying to make simulation look exactly like reality, you generate millions of simulated examples with random lighting, textures, colours and camera properties.

 What are you doing?

 **Domain randomisation.**

---

 ## Scenario 6

 You transform source images so their statistical properties resemble those of the target.

 What are you doing?

 **Input-space alignment.**

 If the question specifically mentions aligning first- and second-order statistics?

 **CORAL.**

---

 ## Scenario 7

 You train a network so that the domain classifier cannot reliably tell whether an internal representation came from the source or target.

 What are you doing?

 **DANN.**

 The goal is domain-invariant representation learning.

---

 # Part 31 — Higher-Level Questions

 Now let's consider a few deeper questions.

 ### Why can transfer learning work when the source and target datasets are different?

 Because some useful structures are shared.

 For example, in computer vision, edges, textures and shapes can be useful across many different datasets.

 The model doesn't need to relearn the concept of an edge every time we change datasets.

---

 ### When might transfer learning be less necessary?

 Suppose I have:

 - enormous amounts of labelled target data,
- a target distribution that closely matches my training data,
- and sufficient computational resources.

 Then training from scratch may be perfectly reasonable.

 Transfer learning is most valuable when we have a reason to reuse existing knowledge.

---

 ### Why might freezing too many layers hurt?

 Because the representation might be too rigid.

 The target problem may require adaptation that the frozen layers cannot provide.

---

 ### Why might freezing too few layers hurt when target data is scarce?

 Because we may modify too much of the useful pre-trained representation and overfit to a small target dataset.

---

 # Part 32 — The Four Equations to Remember

 Before we finish, there are four mathematical relationships I particularly want you to remember.

 First:

 $$
\boxed{
D_S=D_T,\quad T_S\neq T_T
\Rightarrow
\text{Inductive transfer}
}
$$

 Second:

 $$
\boxed{
D_S\neq D_T,\quad T_S\approx T_T
\Rightarrow
\text{Transductive transfer}
}
$$

 Third:

 $$
\boxed{
P_S(X)\neq P_T(X)
\Rightarrow
\text{Domain shift}
}
$$

 And fourth:

 $$
\boxed{
L=L_{\mathrm{cls}}+L_{\mathrm{dom}}
\Rightarrow
\text{Domain-confusion objective}
}
$$

 If you understand these four relationships, you already have a strong conceptual foundation for this lecture.

---

 # Part 33 — Final Conceptual Summary

 Let's finish by bringing the entire lecture together.

 We started with a problem:

 > Training a model from scratch can require a lot of data, computation and time.

 So we introduced:

 > **Transfer learning.**

 The idea is to reuse knowledge from a source problem to help with a target problem.

 We then distinguished:

 **Domain** — the characteristics and distribution of the data.

 **Task** — the problem the model is solving.

 Then we distinguished two transfer scenarios:

 **Inductive transfer:**

 $$
\text{same domain}+\text{different task}.
$$

 **Transductive transfer:**

 $$
\text{different domain}+\text{same/similar task}.
$$

 We then looked at **model-centric transfer**.

 The main example is:

 **Fine-tuning.**

 Start with a pre-trained model and adapt it to the target.

 We can freeze some layers because early layers often contain generic features.

 Then we introduced:

 **Domain shift.**

 That occurs when:

 $$
P_S(X)\neq P_T(X).
$$

 To address domain shift, we introduced:

 **Domain adaptation.**

 There are two broad strategies.

 First:

 **Input-space alignment.**

 Change the data.

 Examples include:

 - CORAL,
- domain translation,
- sim2real transformations.

 And a related strategy:

 **Domain randomisation.**

 Instead of making source exactly like target, make source sufficiently diverse that target is covered.

 Second:

 **Feature-space alignment.**

 Change or align the internal representations.

 Examples include:

 - domain confusion,
- DANN.

 And finally, we looked at the practical evolution of this idea:

 **Pre-trained models** give us strong initialisations.

 And at an even larger scale:

 **Foundation models** provide reusable representations across many downstream tasks.

---

 # Part 34 — The One Mental Model I Want You to Leave With

 If you remember only one question from today's lecture, remember this:

 > **"What useful information have I already learned that I can reuse?"**

 There are two main answers.

 The first is:

 > **Reuse knowledge in the model.**

 That gives us:

 $$
\text{Pre-trained model}
\rightarrow
\text{Fine-tuning}.
$$

 The second is:

 > **Deal explicitly with the difference between source and target domains.**

 That gives us:

 $$
\text{Domain adaptation}.
$$

 And domain adaptation can happen in two places:

 $$
\boxed{\text{Input space}}
$$

 or:

 $$
\boxed{\text{Feature space}}.
$$

 So the final distinction I want you to remember is:

 > **Fine-tuning transfers knowledge through the model.**

 > **Domain adaptation explicitly addresses differences between source and target domains.**

 And within domain adaptation:

 > **Input-space alignment changes the data.**

 > **Feature-space alignment changes or aligns the representations.**

 If you can explain those distinctions in your own words, identify them in a practical scenario, and recognise the equations and methods we've discussed today, then you have understood the central theory of transfer learning.
