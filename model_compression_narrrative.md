 # Lecture Script — Model Compression & Parameter-Efficient Learning

 ## 1\. Quantisation: Why Reduce Numerical Precision?

 “Let’s begin with one of the simplest and most powerful ideas in model compression: **quantisation**.

 The basic idea is very straightforward.

 When we train a neural network, we normally represent its parameters using relatively high numerical precision. For example, we might use 32-bit floating-point numbers, commonly called FP32.

 But do we really need all 32 bits to represent every parameter?

 Often, the answer is no.

 A neural network may contain millions or billions of parameters, and if every parameter takes 32 bits, the memory requirement becomes enormous.

 So quantisation asks a simple question:

 **Can we represent these numbers using fewer bits while keeping approximately the same model behaviour?**

 For example, instead of storing a parameter using 32 bits, we might use 16 bits, 8 bits, or in some modern applications even 4 bits.

 The immediate benefit is memory.

 If we go from 32-bit parameters to 8-bit parameters, each parameter requires only one quarter as many bits.

 So, ignoring some additional overhead, a model that required 4 gigabytes for its parameters in FP32 might require roughly 1 gigabyte using 8-bit representations.

 That's a very substantial saving.”

---

 ## 2\. What Does Quantisation Save?

 “There are actually three important benefits we should think about.

 First, **memory**.

 This is the most obvious one. Fewer bits per parameter means that the model occupies less storage and requires less memory when loaded.

 Second, potentially **time**.

 But I want to emphasise the word _potentially_.

 Using fewer bits does not automatically make every computation faster.

 The speed improvement depends heavily on the hardware.

 If the computation is limited by how quickly data can be moved through memory, then smaller representations can help because less data needs to be transferred.

 And if the hardware has specialised instructions for low-precision arithmetic, then the actual arithmetic can also become faster.

 Third, quantisation can reduce **energy consumption**.

 This is particularly important for devices such as smartphones, embedded systems, and microcontrollers.

 Moving data around and performing large numbers of operations costs energy. If we can represent and manipulate the same model using substantially less data, we may be able to reduce that energy cost.

 So the big picture is:

 **quantisation can reduce memory, sometimes reduce computation time, and often reduce energy consumption.**”

---

 ## 3\. Performance per Parameter versus Performance per Kilobyte

 “There's an important subtlety here.

 A quantised model may not have exactly the same performance **per parameter** as a full-precision model.

 That's not surprising.

 If I give a model fewer bits with which to represent each parameter, I've restricted the precision with which it can represent information.

 But that's not necessarily the right way to evaluate the model.

 Instead, we can ask:

 **How much useful performance do I get per kilobyte of memory?**

 And this is where quantisation becomes particularly attractive.

 Even if an 8-bit model is slightly worse than a 32-bit model when comparing individual parameters, the 8-bit model can fit vastly more parameters into the same amount of memory.

 So we should distinguish between:

 **performance per parameter**

 and

 **performance per unit of memory.**

 For practical deployment, particularly on memory-constrained hardware, the second quantity can be much more important.”

---

 # 4\. Extreme Quantisation

 “Now let's push the idea further.

 Do we really need 16 or 8 bits?

 Modern research has shown that large language models can sometimes be represented using extremely low precision, including **4-bit representations**, while maintaining surprisingly good performance.

 This is quite remarkable.

 Remember that a conventional floating-point representation may use 32 bits.

 We're potentially going from 32 bits down to only 4 bits.

 That's an eight-fold reduction in the number of bits per parameter.

 However, we shouldn't interpret this as simply taking every 32-bit number and rounding it independently to one of 16 possible values.

 The successful approaches use more sophisticated techniques.

 For example, we can use **per-group quantisation**.

 Instead of using exactly the same quantisation parameters for an entire model or tensor, we divide the parameters into groups and quantise each group appropriately.

 We can also deal specially with **outliers**.

 Some parameters may have unusually large values compared with the rest, and treating those values in exactly the same way can introduce substantial error.

 There are also techniques such as **bias correction** and specialised **bit-packing**.

 And finally, the hardware matters enormously.

 If the hardware provides efficient instructions for four-bit or eight-bit computation, then we can actually realise the theoretical memory and computational benefits.

 If the hardware doesn't support the relevant operations efficiently, simply reducing the number of bits may not give us the speed-up we expect.”

---

 # 5\. Why 4 Bits Is Interesting

 “There is another interesting observation from research on quantisation.

 For a fixed total memory budget, extremely low precision can actually be surprisingly effective.

 Suppose I give you a fixed number of bits with which to store a model.

 You have two choices.

 You could store relatively few parameters at high precision.

 Or you could store many more parameters at lower precision.

 There appears to be a useful trade-off here.

 Some research has found that **around four bits per parameter can be particularly effective** for this kind of scaling.

 The intuition is that reducing precision allows us to fit a substantially larger model into the same memory budget.

 So quantisation isn't merely about taking an existing model and making it smaller.

 It changes the trade-off between:

 **number of parameters**

 and

 **precision per parameter.**

 That's an important conceptual point.”

---

 # 6\. Post-Training Quantisation

 “Now let's look at how quantisation is actually performed.

 The simplest approach is called **Post-Training Quantisation**, or **PTQ**.

 The name tells us exactly what happens.

 We first train our neural network normally, using high precision.

 Then, after training has finished, we quantise it.

 So the pipeline is:

 **train first, quantise afterwards.**

 For the weights, this is relatively straightforward.

 Suppose the trained model contains weights ranging from negative values through zero to positive values.

 We define a smaller set of representable values and map the original weights onto those values.

 In other words, we're replacing the original continuous or high-precision values with a discrete set of lower-precision values.”

---

 # 7\. Why Activations Are More Difficult

 “Quantising the weights is relatively easy because we know the weights before deployment.

 But what about the **activations**?

 Activations depend on the input.

 The same neural network can produce very different activation values for different inputs.

 So when we quantise activations, we need some idea of the range of values that we're likely to encounter.

 For example, we might determine that an activation generally lies between some minimum and maximum value.

 We can then map that range onto our available quantised values.

 But how do we know the range?

 One approach is to use information collected during training.

 Another is to use a small **calibration dataset** after training.

 We pass representative examples through the model and observe the activation distributions.

 We then choose suitable quantisation ranges.

 So PTQ is relatively simple, but there is a potential problem:

 **quantisation introduces error.**

 And sometimes that error causes a noticeable decrease in task accuracy.”

---

 # 8\. Quantisation-Aware Training

 “So what do we do if ordinary post-training quantisation isn't good enough?

 We can use **Quantisation-Aware Training**, or **QAT**.

 The fundamental idea is:

 Instead of training the model as though it will eventually be full precision and only introducing quantisation at the very end, we allow the model to experience the effects of quantisation during training.

 This gives the model an opportunity to adapt to the errors introduced by quantisation.

 Imagine that I'm training a model and every time it produces an activation, I simulate what would happen if that activation were quantised.

 The model sees the resulting approximation error.

 During training, the model can then adjust its parameters so that it becomes more robust to those errors.”

---

 # 9\. Fake Quantisation

 “There's an important implementation detail here.

 During QAT, we often don't literally perform all the training computations in low precision.

 Instead, we insert what are called **fake-quantisation operations**.

 Conceptually, we do something like this:

 Take a high-precision value.

 Quantise it.

 Then de-quantise it back into a high-precision representation.

 The resulting value is slightly different from the original.

 That difference simulates the noise that will exist after deployment.

 So the forward pass effectively becomes:

 **full precision → quantisation → de-quantisation → continue computation.**

 The model therefore learns in the presence of quantisation noise.

 This tends to produce better final accuracy than simply quantising a model after it has already been trained.

 But there is an obvious cost.

 QAT requires additional training.

 So PTQ is cheaper and simpler, while QAT is more computationally expensive but can recover accuracy when PTQ causes too much degradation.”

---

 # 10\. Quantisation in Practice

 “Let's summarise the practical distinction.

 **Post-training quantisation** is simple.

 Train the model normally, then quantise it.

 It's cheap and convenient, but it can sometimes reduce accuracy.

 **Quantisation-aware training** is more sophisticated.

 The model learns while experiencing simulated quantisation.

 This generally gives us more opportunity to preserve accuracy, but training becomes slower and more expensive.

 And there's another practical issue that is easy to overlook:

 **hardware support matters.**

 A particular numerical format is only useful if our hardware can exploit it efficiently.

 For example, support for formats such as bfloat16 varies across hardware generations and architectures.

 So when deploying a quantised model, we should not only ask:

 'How many bits am I using?'

 We should also ask:

 'Does my target hardware actually accelerate this representation?'”

---

 # 11\. From Quantisation to Pruning

 “Now let's move to another major model-compression technique:

 **pruning.**

 The motivation comes from a familiar idea in neural networks.

 Large neural networks often contain substantial redundancy.

 For example, imagine a neural network layer with many different computational pathways.

 The model has enough capacity to learn many different operations, but perhaps it doesn't actually need all of them.

 So we can ask:

 **Can we train a large model first, identify the unnecessary parts, and then remove them?**

 That's the basic idea behind pruning.”

---

 # 12\. What Is Pruning?

 “Pruning means removing parameters, connections, neurons, or other components from a trained network.

 One simple way to think about it is:

 We take a trained network and identify weights that aren't particularly important.

 We then set those weights to zero or remove them from the representation.

 The network becomes **sparse**.

 For example, imagine a matrix of weights.

 Originally, almost every entry might contain a non-zero number.

 After pruning, perhaps half of those entries are zero.

 We can then represent the matrix using a sparse representation rather than storing every zero explicitly.

 That's where the memory saving comes from.”

---

 # 13\. Why Is Sparsity Useful?

 “Suppose we prune only one or two weights.

 That's not very useful.

 If we're going to use a special sparse representation, the overhead of representing which values are present might actually cancel out the savings.

 But if a very large proportion of the weights are zero, then sparsity becomes valuable.

 In some cases, models can tolerate 50 percent, 90 percent, or even higher levels of sparsity with surprisingly little performance degradation.

 And modern hardware can sometimes exploit this sparsity computationally.

 Instead of performing a multiplication by zero, the hardware can simply skip that computation.

 So pruning can potentially provide two benefits:

 **less memory**

 and

 **less computation.**

 However, there's an important qualification:

 The computational benefit depends strongly on the type of sparsity and on hardware support.”

---

 # 14\. Magnitude Pruning

 “One of the simplest pruning methods is called **magnitude pruning**.

 The principle is extremely intuitive.

 If a weight is very close to zero, perhaps it isn't doing much.

 So we remove the weights with the smallest absolute values.

 For example, suppose the weights are:

 minus 0.9, 0.5, 0.02, minus 0.01, 0.7.

 The weights 0.02 and minus 0.01 have much smaller magnitude than the others.

 Magnitude pruning would consider those weights good candidates for removal.

 There are two common ways of defining the pruning threshold.

 We could say:

 'Remove every weight whose magnitude is below some threshold.'

 Or we could specify a target sparsity.

 For example:

 'Remove the smallest 20 percent of the weights.'

 The second approach guarantees a particular compression level.”

---

 # 15\. Similarity Pruning

 “Magnitude isn't the only way to decide what's redundant.

 Another idea is **similarity pruning**.

 Instead of looking at individual weights, we look at neurons or feature maps.

 Suppose two neurons produce almost exactly the same activation patterns for the same inputs.

 What does that tell us?

 It suggests that the two neurons may be performing very similar functions.

 If they're effectively duplicating each other, perhaps we don't need both.

 So we can calculate some measure of similarity between their activations.

 If two neurons are highly similar, we may choose to remove one.

 This is a different philosophy from magnitude pruning.

 Magnitude pruning asks:

 **Which individual parameters are small?**

 Similarity pruning asks:

 **Which computational components are redundant?**”

---

 # 16\. Node Pruning

 “We can go one step further and prune entire neurons.

 This is called **node pruning**.

 One simple criterion is the L2 norm of a neuron's weights or activations.

 Recall that the L2 norm measures the overall magnitude of a vector.

 If a neuron's weights have an extremely small norm, or its activations are almost always close to zero, then that neuron may be contributing very little.

 We can therefore remove it.

 An intuitive interpretation is that these neurons are effectively **dead**.

 This connects to the well-known 'dying ReLU' phenomenon.

 If a ReLU neuron consistently receives values that result in zero output, it can become effectively inactive.”

---

 # 17\. Structured Pruning

 “There's an important distinction between removing individual weights and removing entire structures.

 Suppose I remove random individual weights throughout a large weight matrix.

 The resulting matrix is irregularly sparse.

 That is called **unstructured pruning**.

 Alternatively, I might remove an entire neuron, channel, or convolutional filter.

 That's **structured pruning**.

 Why does this matter?

 Because hardware is generally very good at performing dense operations on regular blocks.

 Irregular sparsity can be difficult to exploit efficiently.

 But if I remove an entire channel, the resulting network is simply smaller.

 The remaining computation can still use ordinary dense matrix or convolution operations.

 So structured pruning often gives a more direct reduction in actual computation time.

 This leads to an important practical distinction:

 **unstructured pruning can produce high sparsity, but structured pruning is often easier for hardware to exploit.**”

---

 # 18. Iterative Pruning

 “Pruning doesn't necessarily have to happen once.

 We can perform it **iteratively**.

 Imagine that we start with a trained model.

 We remove a small percentage of parameters.

 Then we fine-tune the model so that it can compensate for the damage.

 Then we prune some more.

 Then we fine-tune again.

 And we repeat this process.

 This is called **iterative pruning**.

 Why might this work better?

 Because the network has an opportunity to adapt after each pruning step.

 If we remove 90 percent of the parameters in one operation, we've made a massive change.

 But if we gradually move from, say, 0 percent sparsity to 20 percent, then 40 percent, then 60 percent, and so on, the model can progressively adjust.

 The compensation is learned gradually.”

---

 # 19\. Pruning and Quantisation Together

 “Now here's an important observation.

 These techniques aren't mutually exclusive.

 We don't have to choose between pruning and quantisation.

 We can combine them.

 For example:

 First, prune the model so that many parameters disappear.

 Then quantise the parameters that remain.

 The resulting model benefits from both forms of compression.

 This is a recurring theme in model efficiency:

 **different compression techniques can often be combined.**”

---

 # 20\. Why Does Pruning Work So Well?

 “Now we arrive at a deeper theoretical question.

 If neural networks need all of these parameters to perform their computations, how can we remove so many of them without destroying performance?

 Why is there so much redundancy?

 One useful way of thinking about this comes from the **manifold hypothesis**.

 In machine learning, the manifold hypothesis says that although data may technically exist in an extremely high-dimensional space, real-world data often occupies a much lower-dimensional structure within that space.

 For example, an image might contain hundreds of thousands of pixel values.

 Mathematically, that gives us a huge-dimensional space.

 But natural images don't randomly occupy every possible combination of pixel values.

 They occupy a much more constrained region.

 Something similar appears to happen with neural-network parameters.

 Although the model may have millions or billions of parameters, the solutions that perform useful functions may occupy a much smaller effective region of parameter space.

 So the model may have many parameters without requiring every parameter to represent an independent degree of freedom.

 That helps explain why compression is possible.”

---

 # 21\. The Lottery Ticket Hypothesis

 “This brings us to one of the most famous ideas associated with neural-network pruning:

 the **Lottery Ticket Hypothesis**.

 The question is:
