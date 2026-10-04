Absolutely. Below is a **question-driven teaching script** designed specifically for an **audio-only lecture**. The teacher continually asks questions, gives students a few seconds to think, then provides the answer and builds toward the next idea.

 The narrative follows the conceptual progression of the notes: **why efficiency matters → numerical precision → quantisation → pruning → why compression works → parameter-efficient fine-tuning → adapters → separable convolutions → low-rank factorisation → LoRA → QLoRA**.

 # Model Compression and Parameter-Efficient Learning

 ## A Question-Focused Teaching Script

 ### Opening: Why do we need model compression?

 Let me start with a question.

 **Question:** Suppose I give you a neural network that achieves excellent accuracy. Why might we still say that the model has a problem?

 Think about what happens when the model becomes very large.

 It may contain millions, billions, or even hundreds of billions of parameters.

 So the first problem is obvious:

 **Can we afford to store the model?**

 And the second question is:

 **Can we afford to compute with it?**

 A model may be accurate but still be too slow, too expensive, or too energy-hungry to deploy.

 Imagine that you want to put a neural network on a mobile phone, a small robot, a medical device, or an edge computer.

 The model might work perfectly in the laboratory.

 But if it requires enormous memory and computation, it may be practically useless.

 So this lecture is really about one fundamental question:

 **How can we make a neural network smaller, cheaper, and faster without destroying its performance?**

 That is the central problem of **model compression and parameter-efficient learning**.

---

 # 1\. What does "performance" actually mean?

 Before we compress anything, let us ask:

 **What are we trying to optimise?**

 Is performance simply accuracy?

 Not necessarily.

 There are at least two different notions of performance.

 First:

 **How well does the model perform its task?**

 For example, classification accuracy, language-model loss, or generation quality.

 But there is another question:

 **How efficiently does the model achieve that performance?**

 We therefore care about things such as:

 - memory usage,
- number of parameters,
- computational cost,
- inference latency,
- training cost,
- and energy consumption.

 This gives us an important principle:

 **A model with slightly lower accuracy but dramatically lower cost may be more useful than a larger model with slightly higher accuracy.**

 So model compression asks us to find a good trade-off between:

 **quality and efficiency.**

---

 # 2\. Why are neural networks so large?

 Now let me ask:

 **Where does the size of a neural network actually come from?**

 The answer is:

 **parameters.**

 Weights and biases are stored as numbers.

 So if a network has one billion parameters, we have to store one billion numbers.

 But here is the interesting question:

 **Do we really need all those numbers in their original form?**

 And this question leads us to our first compression technique:

 **quantisation.**

---

 # 3\. Quantisation: Do weights really need to be high precision?

 Imagine that a neural network stores every weight using a 32-bit floating-point number.

 That is called **float32**.

 Now ask:

 **Does every weight really need 32 bits?**

 Suppose a weight is:

 0.18374291.

 Do we necessarily need all that numerical precision?

 Perhaps not.

 Instead, we could represent it using a smaller number of bits.

 For example, we might map many possible floating-point values onto a much smaller set of integer values.

 This is the basic idea of **quantisation**.

---

 ## 3.1 What exactly is quantisation?

 Here is the question:

 **What does quantisation actually do to a weight?**

 Instead of allowing a weight to take any value on a continuous floating-point scale, we map it onto a discrete set of values.

 For example, instead of representing a number using float32, we might represent it using **int8**.

 That means we have dramatically fewer possible values.

 So conceptually:

 **high-precision value → nearby low-precision value**

 We lose some precision.

 But we gain something much more valuable:

 **a much smaller representation.**

---

 ## 3.2 Why does this save memory?

 Let us ask a simple numerical question.

 If one weight uses 32 bits, how much memory do we need for one billion weights?

 One billion multiplied by 32 bits.

 Now suppose each weight uses only 8 bits.

 What happens?

 We have reduced the storage required for the weights by roughly a factor of four.

 So:

 **32-bit → 8-bit**

 means approximately:

 **4 times less storage for the raw weights.**

 And if we go from 32 bits to 4 bits?

 That is potentially an eight-fold reduction.

 So the first major lesson is:

 **Quantisation compresses the numerical representation of parameters.**

---

 # 4\. But if we throw away precision, why doesn't the model break?

 This is the important question.

 If quantisation changes the weights, then:

 **Why doesn't accuracy collapse?**

 Because neural networks are often surprisingly tolerant to small changes in their parameters.

 A weight of 0.1837 becoming approximately 0.18 may have almost no meaningful effect.

 But there is a catch.

 **What happens if the values are not evenly distributed?**

 Suppose most weights are close to zero, but a few weights are extremely large.

 If we use a simple linear mapping, those extreme values can consume a lot of our available quantisation range.

 This introduces the problem of **outliers**.

 Therefore, quantisation is not simply:

 "Round everything."

 Good quantisation schemes try to represent the important parts of the weight distribution accurately.

---

 # 5\. What exactly do we gain from quantisation?

 Let us ask three questions.

 **Question 1: Does quantisation reduce memory?**

 Yes.

 That is usually its biggest benefit.

 **Question 2: Does it always make the model faster?**

 No.

 And this is important.

 A smaller numerical representation does not automatically mean faster computation.

 The hardware must actually support efficient low-precision operations.

 So:

 **smaller model ≠ automatically faster model.**

 **Question 3: Can it reduce energy consumption?**

 Yes, particularly when low-precision operations reduce memory movement and computation.

 This is especially important for devices such as phones and embedded systems.

 So quantisation can provide:

 **less memory + potentially less computation + potentially less energy.**

---

 # 6\. What about extremely aggressive quantisation?

 Now let us push the idea.

 If 32 bits works, and 8 bits can work, then:

 **Could we use only 4 bits?**

 Surprisingly, yes.

 Modern methods can quantise very large language models to extremely low precision while retaining much of their original performance.

 But now the problem becomes harder.

 At four bits, every quantisation error matters more.

 So researchers use techniques such as:

 - per-group quantisation,
- outlier handling,
- bias correction,
- careful bit packing,
- and specialised hardware instructions.

 The deeper lesson is:

 **Compression becomes increasingly sophisticated as we push the representation to lower precision.**

---

 # 7\. When should we quantise?

 Suppose we have already trained our model.

 Do we have to retrain the entire model just because we want a smaller representation?

 Not necessarily.

 This gives us:

 **Post-Training Quantisation, or PTQ.**

---

 ## 7.1 Post-Training Quantisation

 Here is the question:

 **What happens in PTQ?**

 We first train the model normally.

 Then we take the trained model and quantise it.

 The weights are mapped from high precision to lower precision.

 Simple.

 But what about the **activations**?

 That is more difficult.

 Why?

 Because weights are fixed once the model has been trained, but activations depend on the input.

 We therefore need to estimate the range of possible activations.

 This is called **calibration**.

 We can use a small calibration dataset to observe the activation ranges.

---

 # 8\. What if PTQ damages accuracy?

 Now we encounter another important question:

 **What if quantising the trained model causes too much accuracy loss?**

 Could the model learn to compensate for the quantisation?

 Yes.

 And that gives us:

 **Quantisation-Aware Training, or QAT.**

---

 ## 8.1 What is QAT?

 The basic idea is clever.

 During training, we simulate the quantisation process.

 The model is still trained using high-precision parameters, but we insert **fake quantisation** operations into the forward pass.

 In other words:

 **quantise → dequantise → continue computation.**

 The model therefore experiences the noise caused by quantisation while it is learning.

 So the model can adapt its parameters to that noise.

 This gives us the central difference:

 **PTQ: train first, quantise afterwards.**

 **QAT: train while accounting for quantisation.**

 QAT is usually more expensive, but it can recover accuracy when simple PTQ is insufficient.

---

 # 9\. Quantisation versus pruning

 Now let us ask a completely different question.

 Suppose instead of storing every weight with fewer bits, we ask:

 **Do we need every weight at all?**

 Maybe some weights are essentially useless.

 If so, why store them?

 This leads us to:

 **pruning.**

---

 # 10\. What is pruning?

 Imagine a neural network containing one million connections.

 Suppose we discover that 500,000 of them contribute very little.

 What could we do?

 **Remove them.**

 That is pruning.

 A simple implementation is to set those weights to zero.

 The model then contains many zero-valued connections.

 Conceptually:

 **large dense network → remove unnecessary connections → sparse network.**

---

 # 11\. Why would pruning work?

 This is a very important question.

 **Why can we remove so many parameters without destroying the model?**

 Because neural networks are often highly redundant.

 A trained network may have many parameters doing similar or unnecessary things.

 So perhaps the network has more capacity than it actually needs.

 This is the idea behind **overparameterisation**.

 The network is deliberately much larger than the minimum required solution.

 That redundancy gives us room to remove parameters.

---

 # 12\. What kind of parameters should we remove?

 The simplest question is:

 **Which weights look least important?**

 One obvious answer is:

 **weights with small magnitude.**

 This gives us **magnitude pruning**.

 If a weight is very close to zero, we might assume it has little influence.

 For example:

 - 0.0001 → candidate for pruning
- 0.0002 → candidate for pruning
- 2.4 → probably more important

 We can either choose a threshold or decide on a target sparsity.

 For example:

 **"Remove the smallest 20% of weights."**

---

 # 13\. Is magnitude the only measure of importance?

 No.

 Consider two neurons.

 Suppose they produce almost identical activations for every input.

 Then ask:

 **Do we really need both?**

 Perhaps not.

 One may be redundant.

 This motivates **similarity pruning**.

 We compare neurons or feature maps and identify components performing almost the same function.

 Then we can remove one of them.

 The key idea is:

 **Don't just ask whether a parameter is small. Ask whether its function is redundant.**

---

 # 14\. Can we remove entire neurons?

 Yes.

 Instead of removing individual weights, we can remove a whole neuron.

 We might identify a neuron whose weights or activations have a very low L2 norm.

 Such a neuron may behave almost like a "dead neuron."

 So we can remove the node entirely.

 This is more structured than simply deleting individual connections.

---

 # 15\. What is structured pruning?

 Suppose we have a convolutional neural network.

 Instead of deleting random individual weights, what if we remove:

 **an entire filter?**

 Or:

 **an entire channel?**

 That is **structured pruning**.

 Why is this attractive?

 Because modern hardware likes regular, dense computational structures.

 Randomly deleting individual weights produces an irregular sparse matrix.

 Theoretically, that saves parameters.

 But the hardware may not exploit those zeros efficiently.

 Structured pruning is different.

 If we remove an entire channel, we create a genuinely smaller dense operation.

 So structured pruning can save:

 **both memory and actual computation time.**

 This leads to a very important engineering lesson:

 **Theoretical compression is not necessarily practical speed-up.**

 Hardware matters.

---

 # 16\. Should we prune once or repeatedly?

 Suppose we immediately remove 90% of the weights.

 What might happen?

 The model could suffer a large performance drop.

 So perhaps we should do something gentler.

 We could:

 1. prune a little,
2. fine-tune,
3. prune a little more,
4. fine-tune again,
5. continue.

 This is **iterative pruning**.

 Why might this work better?

 Because after each pruning step, the remaining parameters have an opportunity to compensate.

 So instead of asking the model to survive a huge change all at once, we let it adapt gradually.

---

 # 17\. Why is pruning theoretically surprising?

 Here is one of the deepest questions in this lecture:

 **If a neural network contains billions of parameters, why can we sometimes remove most of them and barely hurt performance?**

 One explanation comes from the **manifold hypothesis**.

 We already know the manifold hypothesis from data.

 High-dimensional data often appears to occupy a much lower-dimensional structure.

 The surprising extension is:

 **Perhaps trained neural networks also occupy a low-dimensional region within their enormous parameter space.**

 In other words:

 The model may have billions of possible parameter values, but the useful solutions may occupy a much smaller effective space.

 This helps explain why so much redundancy can exist.

---

 # 18\. The Lottery Ticket Hypothesis

 Now let me ask a more provocative question:

 **Why not simply train a small network from scratch?**

 If a large network can eventually be pruned into a small network, perhaps we could just start with the small network.

 But researchers found evidence suggesting something more interesting.

 This is the **Lottery Ticket Hypothesis**.

 The basic idea is:

 **A randomly initialised large network contains a smaller subnetwork that, with the right initialisation, can train successfully on its own.**

 Think about a lottery.

 You buy many tickets.

 Most are useless.

 But somewhere among them is a winning ticket.

 Similarly, a large neural network may contain a small "winning" subnetwork.

 The large network gives us a better chance of discovering it.

 Pruning identifies the useful structure.

---

 # 19\. From compression to fine-tuning

 So far, we have mostly asked:

 **How can we make an already-trained model smaller?**

 But now consider another problem.

 Suppose we have a huge pre-trained model.

 We want to adapt it to a new task.

 Do we really need to update every parameter?

 For a billion-parameter model, that could be enormously expensive.

 This gives us:

 **Parameter-Efficient Fine-Tuning, or PEFT.**

 The question becomes:

 **Can we adapt a huge model by changing only a tiny number of parameters?**

---

 # 20\. Layer freezing

 The simplest solution is:

 **freeze most layers.**

 If the model already knows useful general features, why change them?

 We update only a small part of the model.

 This reduces the training cost.

 In particular, frozen layers do not require the same optimiser states and gradient updates.

 So layer freezing primarily saves **training memory**.

 But ask:

 **Is freezing whole layers necessarily the best way to restrict the number of trainable parameters?**

 Not necessarily.

 Maybe the new task needs small changes distributed throughout the network.

 So we want something more flexible.

---

 # 21\. Can every layer contribute without training every weight?

 This is the next key question.

 Imagine that every layer contains a billion parameters.

 Instead of updating the whole layer, perhaps we could add a small trainable component.

 That leads us to:

 **adapter modules.**

---

 # 22\. What are adapter modules?

 Adapters are small neural networks inserted between the original layers.

 The original model remains frozen.

 Only the adapters are trained.

 The adapter typically has a **bottleneck**.

 For example:

 **large dimension → small dimension → large dimension.**

 Why use a bottleneck?

 Because it dramatically reduces the number of trainable parameters.

 And the adapter is usually added through a residual connection.

 So conceptually:

 **original representation \+ small learned correction.**

 This is a powerful idea.

 We are not trying to relearn the entire model.

 We are learning:

 **How should this existing model be adjusted for the new task?**

---

 # 23\. Why does the adapter approach make sense?

 Ask yourself:

 **If a pre-trained model already knows most of what it needs, how much new information should we really have to teach it?**

 Probably not very much.

 The new task may require only a small adjustment.

 Therefore:

 **the adaptation itself may be low-dimensional even though the model is enormous.**

 That is one of the central ideas running throughout this entire lecture.

---

 # 24\. Prefix tuning

 For language models, there is another approach:

 **prefix tuning.**

 Instead of modifying the main model's parameters, we learn a specialised context or prefix representation.

 Then we prepend this learned representation to the input.

 Why is this useful?

 Because we can keep the main language model frozen and swap different prefixes for different tasks.

 So one large model can have:

 **one prefix for summarisation, another for classification, another for another task, and so on.**

 This introduces a very useful idea:

 **one shared base model + many tiny task-specific modules.**

---

 # 25\. Can we compress the computation itself?

 Now let us change perspective.

 Instead of asking:

 **Can we store fewer parameters?**

 ask:

 **Can we represent the same operation using fewer parameters?**

 Consider a 3×3 convolution.

 How many numbers does it contain?

 Nine.

 Now ask:

 **Can nine parameters sometimes be represented using fewer parameters?**

 Yes.

 This leads us to **separable convolutions**.

---

 # 26\. Spatially separable convolution

 A 3×3 kernel can sometimes be represented as the combination of:

 **a 3×1 convolution**

 followed by:

 **a 1×3 convolution.**

 How many parameters?

 Three plus three.

 So:

 **9 parameters → 6 parameters.**

 That is a one-third reduction.

 And under the appropriate conditions, the resulting operation can be mathematically equivalent.

 The deeper idea is not really about convolution.

 It is:

 **A complicated operation may have a simpler factorised representation.**

---

 # 27\. Does every matrix have this nice factorisation?

 No.

 This is important.

 Not every matrix can be represented exactly as a product of very small matrices.

 But perhaps we can find an **approximate** factorisation.

 And that takes us to matrix factorisation and **low-rank representations**.

---

 # 28\. What does "low rank" mean?

 Suppose we have a large matrix.

 Mathematically, we can decompose it using **SVD**:

 **A = UΣVᵀ**

 Now ask:

 **Do we really need every component of this decomposition?**

 If some singular values are extremely small, perhaps they contribute very little.

 So we can discard them.

 This produces a **truncated SVD**.

 Instead of representing the full matrix, we represent an approximation using only its most important components.

 That is a form of compression by **dimension reduction**.

---

 # 29\. So why not simply compress all neural-network weights using SVD?

 This is where things become interesting.

 The manifold hypothesis suggests that useful solutions may have low intrinsic dimensionality.

 So we might expect trained weight matrices to be low-rank.

 But in practice:

 **the weights themselves are often not sufficiently low-rank.**

 If we aggressively truncate them, performance can degrade quickly.

 So we have a problem.

 The idea is theoretically attractive.

 But direct low-rank compression of the trained weights does not always work well.

 And this leads us to one of the most important ideas in the lecture.

---

 # 30\. If the weights are not low-rank, what is?

 Here is the crucial question:

 **Maybe the weights themselves are not low-rank. But what about the changes we make to them during fine-tuning?**

 This is the central insight behind:

 **Low-Rank Adaptation — LoRA.**

 The model's original weights may be complicated and full-rank.

 But perhaps the **adaptation required for a new task is much simpler.**

 That is:

 **W itself may be high-rank.**

 But:

 **ΔW may be approximately low-rank.**

 This distinction is fundamental.

---

 # 31\. What normally happens during fine-tuning?

 Suppose our original weight matrix is:

 **W**

 During normal fine-tuning, we calculate a gradient update:

 **ΔW**

 and update the weights:

 **W' = W + ΔW**

 The problem is that ΔW has the same enormous dimensions as W.

 So we have to train and store a huge update.

 LoRA asks:

 **Do we really need to represent ΔW as a full matrix?**

 Maybe not.

---

 # 32\. The LoRA idea

 Instead of learning the full matrix ΔW, LoRA learns two smaller matrices:

 **A**

 and

 **B**

 such that:

 **BA ≈ ΔW**

 This is the heart of LoRA.

 Suppose W is:

 **d × d**

 Instead of learning d² parameters, choose a small rank:

 **r \<\< d**

 Then:

 **A is r × d**

 and:

 **B is d × r**

 The number of trainable parameters becomes:

 **rd + dr = 2dr**

 instead of:

 **d².**

 If r is tiny compared with d, the savings are enormous.

---

 # 33\. Why does LoRA work?

 Now ask the most important conceptual question:

 **Why should a tiny low-rank update be enough to adapt a huge model?**

 Because the model is already highly capable.

 We are not training the model from scratch.

 We are not teaching it everything again.

 We are only asking:

 **What small change should we make to an already useful representation?**

 The hypothesis is that this task-specific change lies in a much lower-dimensional space.

 So LoRA exploits:

 **low-dimensional adaptation rather than low-dimensional weights.**

 That is the key distinction.

---

 # 34\. Why initialise LoRA carefully?

 Suppose we add:

 **BA**

 to the original weights.

 What should happen at the beginning of training?

 We want the model to initially behave exactly like the original pre-trained model.

 Therefore, we initialise the LoRA matrices so that:

 **BA = 0.**

 Then initially:

 **W' = W.**

 The LoRA component starts as a zero residual.

 During training, it learns the required adjustment.

 So LoRA is essentially learning:

 **a low-rank residual correction to the frozen model.**

---

 # 35\. Is LoRA slower at inference?

 Here is a very useful question.

 If we add extra LoRA matrices, have we made inference slower?

 Potentially, yes, if we keep the adapter as a separate computation.

 But there is a clever solution.

 We can **merge the LoRA weights into the original model**.

 Since:

 **W' = W + BA**

 we can simply construct the merged weight matrix.

 After merging, the model looks like the original architecture.

 So:

 **training:** base model + LoRA

 **deployment:** merged model

 This means LoRA can have essentially **no additional inference architecture cost** after merging.

---

 # 36\. Why is modular LoRA useful?

 Now imagine that we have one enormous language model.

 We want versions for:

 - English,
- French,
- German,
- medical language,
- legal language,
- programming,
- or some particular style.

 Do we need to store a complete copy of the model for every task?

 No.

 We can store:

 **one large base model**

 plus:

 **many tiny LoRA modules.**

 At inference time, we select the appropriate module.

 This is **modular fine-tuning**.

 The same idea is also widely useful in generative models, where LoRA modules can represent particular styles or characters.

---

 # 37\. How small can LoRA really be?

 Now let us ask:

 **How large does the rank r actually need to be?**

 You might expect it to have to be reasonably large.

 But empirical results show that surprisingly small ranks can work well.

 Sometimes:

 **r = 8**

 is enough.

 And in some settings, even:

 **r = 1**

 can produce useful results.

 That is remarkable.

 It tells us that the task-specific adaptation can be extremely low-dimensional.

---

 # 38\. LoRA versus ordinary fine-tuning

 Let's compare them.

 ### Ordinary fine-tuning

 We update:

 **W**

 Every parameter may change.

 Therefore:

 **many trainable parameters \+ large optimiser state \+ high memory usage.**

 ### LoRA

 We freeze:

 **W**

 and learn:

 **A and B**

 Therefore:

 **very few trainable parameters + much smaller optimiser state + much lower training memory.**

 But there is a deeper conceptual difference.

 Ordinary fine-tuning says:

 **"Change the whole model."**

 LoRA says:

 **"Keep the model and learn a small task-specific correction."**

---

 # 39\. Can we combine compression methods?

 Now we arrive at an important final question.

 **Why choose only one compression technique?**

 Could we combine them?

 Absolutely.

 For example:

 **pruning + quantisation**

 can produce both sparsity and low-precision weights.

 And:

 **LoRA + quantisation**

 can reduce the memory requirements of both the base model and the trainable adaptation.

 This combination is known as:

 **QLoRA — Quantised LoRA.**

 The important general principle is:

 **Different compression techniques attack different sources of cost.**

---

 # 40\. The big picture

 Let's stop and connect everything.

 Suppose we start with a huge neural network.

 What problems might we have?

 **Question:** Too much memory?

 Use:

 **quantisation.**

---

 **Question:** Too many unnecessary connections?

 Use:

 **pruning.**

---

 **Question:** Too many trainable parameters during fine-tuning?

 Use:

 **layer freezing, adapters, prefix tuning, or LoRA.**

---

 **Question:** Can a computation itself be represented more compactly?

 Use:

 **factorisation or separable operations.**

---

 **Question:** Can we make the adaptation low-dimensional even if the original model isn't low-rank?

 Use:

 **LoRA.**

---

 # 41\. The most important conceptual distinction

 Let me ask you one final conceptual question.

 **What is the difference between compressing a model and making fine-tuning parameter-efficient?**

 Model compression generally asks:

 **"How can I make the model cheaper to store or execute?"**

 Examples:

 - quantisation,
- pruning,
- weight factorisation.

 Parameter-efficient fine-tuning asks:

 **"How can I adapt a large model without changing or training most of it?"**

 Examples:

 - layer freezing,
- adapters,
- prefix tuning,
- LoRA.

 These goals are related, but they are not identical.

---

 # 42\. The deeper idea behind the whole lecture

 Now ask yourself:

 **Why do all these techniques work at all?**

 Why can we throw away precision?

 Why can we remove weights?

 Why can we freeze layers?

 Why can we train tiny adapters?

 Why can a tiny LoRA update adapt a giant model?

 The recurring answer is:

 **Neural networks contain a great deal of redundancy.**

 The model may have an enormous number of parameters, but the actual information needed for a particular task may occupy a much smaller effective space.

 This appears in several forms.

 For quantisation:

 **We don't need exact numerical precision everywhere.**

 For pruning:

 **We don't need every connection.**

 For structured pruning:

 **We don't need every neuron or channel.**

 For adapters:

 **We don't need to modify every parameter.**

 For LoRA:

 **We don't need a full-dimensional update.**

 For factorisation:

 **Some computations can be represented using fewer degrees of freedom.**

 So all of these techniques are exploiting some form of **redundancy or lower effective dimensionality**.

---

 # 43\. A final thought experiment

 Imagine that you have a model with one trillion parameters.

 And someone tells you:

 "We can adapt this model to a new task by changing only a few million parameters."

 Would you believe them?

 After this lecture, you should ask:

 **Why might that be possible?**

 Because the model already contains enormous amounts of knowledge.

 The new task may not require a completely new model.

 It may require only a small change in how the existing knowledge is used.

 That is the fundamental intuition behind parameter-efficient fine-tuning.

---

 # 44\. Final question: Which technique should we choose?

 Suppose you are given a large neural network and asked:

 **"Make this model cheaper."**

 What should you do?

 There is no universal answer.

 Instead, ask:

 ### Question 1:

 **Is memory the main problem?**

 Consider:

 **quantisation.**

 ### Question 2:

 **Are many parameters unnecessary?**

 Consider:

 **pruning.**

 ### Question 3:

 **Do you need actual computation speed-up?**

 Consider:

 **structured pruning and hardware-supported quantisation.**

 ### Question 4:

 **Do you need to fine-tune a huge pre-trained model?**

 Consider:

 **parameter-efficient fine-tuning.**

 ### Question 5:

 **Do you want very few trainable parameters while keeping the base model unchanged?**

 Consider:

 **LoRA.**

 ### Question 6:

 **Do you also want the base model quantised?**

 Consider:

 **QLoRA.**

---

 # 45\. Final summary — the questions you should be able to answer

 Let me finish by giving you the questions I would expect you to be able to answer after this lecture.

 **What is model compression trying to achieve?**

 Reduce memory, computation, latency, and energy while preserving as much task performance as possible.

 **What is quantisation?**

 Representing parameters or activations using fewer numerical bits.

 **Why does quantisation help?**

 Because fewer bits mean less memory and potentially cheaper computation.

 **What is PTQ?**

 Quantising a model after it has already been trained.

 **What is QAT?**

 Training a model while simulating the effects of quantisation so that it can learn to compensate.

 **What is pruning?**

 Removing unnecessary parameters, connections, neurons, or channels.

 **What is magnitude pruning?**

 Removing parameters with small magnitude.

 **What is structured pruning?**

 Removing whole structured components such as neurons, filters, or channels.

 **Why can pruning work?**

 Because neural networks are often highly redundant and overparameterised.

 **What is the Lottery Ticket Hypothesis?**

 The idea that a large randomly initialised network may contain a smaller subnetwork that can train effectively when properly identified and initialised.

 **What is parameter-efficient fine-tuning?**

 Adapting a large pre-trained model while training only a small fraction of its parameters.

 **What are adapters?**

 Small trainable bottleneck networks inserted into a frozen model to learn task-specific residual corrections.

 **What is prefix tuning?**

 Learning task-specific context representations while keeping the main language model frozen.

 **What is low-rank factorisation?**

 Representing a large matrix approximately using a product of smaller matrices.

 **Why doesn't direct low-rank compression of weights always work?**

 Because neural-network weight matrices are not necessarily sufficiently low-rank.

 **What is the key insight behind LoRA?**

 The weights themselves may be high-rank, but the **task-specific update to those weights may be low-rank**.

 **How does LoRA work?**

 Instead of learning the full update ΔW, it learns two small matrices A and B such that:

 **BA ≈ ΔW.**

 **Why is LoRA parameter-efficient?**

 Because if the rank r is much smaller than the matrix dimension d, the number of trainable parameters drops dramatically.

 **Why can LoRA have no inference overhead after training?**

 Because the low-rank update can be merged into the original weights.

 **What is QLoRA?**

 A combination of quantisation and LoRA, allowing highly memory-efficient fine-tuning of large models.

---

 # Closing question

 So here is the question I want you to leave with:

 **If modern neural networks are enormous, why don't we simply make them smaller from the beginning?**

 Because size is not necessarily the enemy.

 A large model can provide redundancy, capacity, and a rich representation.

 The real trick is:

 **Use the large model to find or represent what is useful, and then exploit the redundancy to make storage, computation, or adaptation much cheaper.**

 That is the central story of model compression and parameter-efficient learning:

 **Keep the performance. Remove the unnecessary cost.**
