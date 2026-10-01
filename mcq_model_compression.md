# Advanced Machine Learning — Comprehensive MCQ Bank

 ## Lecture 4: Model Compression & Parameter-Efficient Learning

---

 # Part I — Efficiency and Resource-Constrained ML

 ### 1\. What is the central motivation for model compression and parameter-efficient learning in this lecture?

 A. Training neural networks without labelled data\
 B. Making models more efficient under computational and hardware constraints\
 C. Eliminating the need for transfer learning\
 D. Increasing the number of model parameters

 **Answer: B**

 **Explanation:** The lecture emphasizes that practical ML systems are often constrained not by data but by **compute and hardware**, particularly when models must run on mobile devices, integrated devices, robotics, or edge hardware.

---

 ### 2\. Which of the following is **not** one of the efficiency dimensions explicitly discussed?

 A. Parameter count\
 B. FLOPs\
 C. Latency/throughput\
 D. Number of training examples

 **Answer: D**

 **Explanation:** The lecture identifies parameter count, FLOPs, latency/throughput, and memory footprint as major efficiency measures. The number of training examples is a data consideration, not an efficiency dimension of the model itself.

---

 ### 3\. What does FLOPs measure?

 A. Number of trainable parameters\
 B. Amount of computation performed per model pass\
 C. Memory required to store the model\
 D. Time required to load the model from disk

 **Answer: B**

 **Explanation:** FLOPs measure the amount of floating-point computation required during a model pass.

---

 ### 4\. Why can reducing the number of **trainable** parameters be particularly useful during training?

 A. It always reduces inference latency\
 B. It reduces training memory requirements and can improve convergence/stability\
 C. It increases the number of activations\
 D. It necessarily increases model accuracy

 **Answer: B**

 **Explanation:** Fewer trainable parameters mean less gradient and optimizer-state storage. The lecture also notes that restricting trainable parameters can improve convergence and stability, effectively saving training time.

---

 ### 5\. According to the lecture's rule of thumb, how does the computational cost of a backward pass compare with a forward pass?

 A. Approximately 0.5×\
 B. Approximately 1×\
 C. Approximately 2×\
 D. Approximately 10×

 **Answer: C**

 **Explanation:** The lecture states that the backward pass usually costs about **2×** the forward pass in time and FLOPs. Therefore, a training batch costs approximately **3× inference**: one forward pass + one backward pass.

---

 ### 6\. If a forward pass costs 100 units of computation, approximately how much does one training pass cost according to the lecture's rule of thumb?

 A. 100 units\
 B. 150 units\
 C. 200 units\
 D. 300 units

 **Answer: D**

 **Explanation:** Forward ≈ 100 units and backward ≈ 200 units, giving approximately **300 units total**.

---

 ### 7\. Why does inference generally require less memory than training?

 A. Inference uses fewer parameters\
 B. Inference does not need to store gradients and training-related optimizer state\
 C. Inference uses no activations\
 D. Inference always uses integer arithmetic

 **Answer: B**

 **Explanation:** Training requires storage associated with gradients and optimizer state in addition to parameters and activations. Inference does not require these training-specific quantities.

---

 ### 8\. Which statement best summarizes the ultimate resource-saving goals discussed in the lecture?

 A. Maximize FLOPs and memory\
 B. Minimize data and maximize parameters\
 C. Save memory and time, potentially also reducing cost\
 D. Eliminate all computation

 **Answer: C**

 **Explanation:** The various efficiency objectives ultimately reduce to saving **memory and time**, which can also translate into monetary savings at scale.

---

 # Part II — Reduced Precision and Numerical Representation

 ### 9\. What is the basic idea behind reduced-precision training?

 A. Reduce the number of layers\
 B. Reduce the number of training examples\
 C. Use fewer bits to represent parameters and activations\
 D. Remove all small weights

 **Answer: C**

 **Explanation:** Reduced-precision training changes the numerical representation, for example from FP32 to FP16 or other lower-bit formats.

---

 ### 10\. Why can neural networks often tolerate reduced numerical precision?

 A. Neural networks never perform numerical operations\
 B. The exact numerical precision of every individual value is often unnecessary for the learned computation\
 C. Lower precision always improves accuracy\
 D. Floating-point values are irrelevant during training

 **Answer: B**

 **Explanation:** The lecture emphasizes that neural networks are surprisingly robust to numerical approximation. FP32 is often more precision than is necessary for the computation being learned.

---

 ### 11\. What are two direct benefits of reduced numerical precision?

 A. More layers and more parameters\
 B. Lower memory usage and potentially lower computation time\
 C. Higher precision and larger dynamic range\
 D. More training data and larger batches only

 **Answer: B**

 **Explanation:** Lower-bit representations require less memory and can reduce computation time, for both training and inference.

---

 ### 12\. Why might FP16 be problematic even though it provides lower storage requirements?

 A. It has no exponent\
 B. It can have insufficient dynamic range\
 C. It cannot represent any negative numbers\
 D. It requires more bits than FP32

 **Answer: B**

 **Explanation:** The lecture stresses that the problem with FP16 is not simply loss of precision; its **dynamic range** can be insufficient. Sometimes representing very small or very large values is more important than representing values extremely precisely.

---

 ### 13\. What is BF16 designed to change relative to FP16?

 A. It removes floating-point representation entirely\
 B. It shifts bits toward the exponent, increasing dynamic range\
 C. It increases the number of mantissa bits\
 D. It uses 32 bits instead of 16

 **Answer: B**

 **Explanation:** Brain Floating Point 16 (BF16) allocates more of its limited bits to the exponent and fewer to the mantissa, giving greater dynamic range at the expense of precision.

---

 ### 14\. Why is BF16 use partly hardware-dependent?

 A. It is a software-only compression method\
 B. Not all hardware supports BF16 computation equally\
 C. BF16 requires a GPU with exactly 175 billion parameters\
 D. BF16 can only be used during inference

 **Answer: B**

 **Explanation:** The lecture explicitly notes that the benefits of particular numerical formats depend on hardware support.

---

 ### 15\. What is mixed-precision training?

 A. Training multiple models simultaneously\
 B. Using different numerical formats for different parts or operations of a model\
 C. Randomly changing precision every epoch\
 D. Using both supervised and unsupervised data

 **Answer: B**

 **Explanation:** Mixed precision uses lower precision where it is sufficiently robust and higher precision where numerical sensitivity makes it necessary.

---

 ### 16\. Which component is described as relatively more sensitive to reduced precision?

 A. Feature activations\
 B. Parameter matrices\
 C. Optimizer states\
 D. Forward-pass multiplications

 **Answer: C**

 **Explanation:** The lecture identifies optimizer states, loss computation, batch normalization, and accumulation operations as more sensitive than many ordinary forward-pass operations.

---

 # Part III — Quantisation

 ### 17\. What is quantisation?

 A. Increasing floating-point precision\
 B. Mapping continuous/high-precision values onto a discrete set of representable values\
 C. Removing all model biases\
 D. Increasing the number of parameters

 **Answer: B**

 **Explanation:** Quantisation maps floating-point values to a discrete representation, such as INT8.

---

 ### 18\. What is a major benefit of quantising a model?

 A. It always increases accuracy\
 B. It reduces memory requirements\
 C. It always eliminates training\
 D. It increases parameter precision

 **Answer: B**

 **Explanation:** Quantisation can substantially reduce memory usage. Depending on hardware and whether the workload is memory-bound, it can also reduce time and energy consumption.

---

 ### 19\. Why can quantisation sometimes reduce computation time?

 A. Quantisation removes all operations\
 B. Lower-precision operations can be accelerated by suitable hardware\
 C. Quantised models contain no activations\
 D. Quantisation increases the number of FLOPs

 **Answer: B**

 **Explanation:** Lower-bit arithmetic can be computationally efficient when the hardware provides suitable acceleration. The lecture qualifies this: time savings are especially relevant when memory-bound and supported by hardware.

---

 ### 20\. What is an important distinction between performance per parameter and performance per kilobyte after quantisation?

 A. Quantisation necessarily improves both\
 B. Quantised models can have lower performance per parameter but much higher performance per kilobyte\
 C. Quantisation reduces performance per kilobyte\
 D. They are always identical

 **Answer: B**

 **Explanation:** A quantised parameter is less expressive numerically, so performance per individual parameter may decrease. But many more parameters can fit into the same memory budget, making **performance per kilobyte** much better.

---

 ### 21\. What is one reason extreme 4-bit quantisation can work surprisingly well?

 A. Neural networks never depend on numerical values\
 B. Special quantisation techniques can preserve useful information despite very low precision\
 C. Four-bit values are more precise than FP32\
 D. It eliminates the need for model training

 **Answer: B**

 **Explanation:** The lecture mentions techniques such as **per-group quantisation, outlier handling, bias correction, and bit-packing**, combined with specialized hardware.

---

 ### 22\. Which issue can particularly hurt nonlinear quantisation?

 A. Too many training examples\
 B. Poor representation of outlier values\
 C. Excessive number of layers\
 D. Lack of convolutional layers

 **Answer: B**

 **Explanation:** A nonlinear mapping can capture dynamic range more effectively, but extreme **outlier values** may still be poorly represented, causing performance degradation.

---

 ### 23\. What is the main difference between PTQ and QAT?

 A. PTQ happens before training, QAT after training\
 B. PTQ quantises a trained model; QAT incorporates quantisation effects during training\
 C. PTQ changes architecture; QAT only changes labels\
 D. There is no difference

 **Answer: B**

 **Explanation:** **Post-training quantisation (PTQ)** is performed after training. **Quantisation-aware training (QAT)** exposes the model to quantisation effects during training so it can learn to compensate.

---

 ### 24\. Why is activation quantisation generally more complicated than weight quantisation in PTQ?

 A. Activations never have numerical values\
 B. The possible activation range needs to be calibrated\
 C. Weights change during inference\
 D. Activations have no distributions

 **Answer: B**

 **Explanation:** Weight distributions are available from the trained model. Activation ranges depend on inputs and intermediate computations, so they need calibration, potentially using training data or a small calibration dataset.

---

 ### 25\. What does QAT typically do during training?

 A. Permanently replace all floating-point values with integers from the first step\
 B. Insert fake-quantisation operations that simulate quantisation effects\
 C. Remove all gradients\
 D. Freeze every layer

 **Answer: B**

 **Explanation:** QAT can train in full precision while simulating quantisation and dequantisation in the forward computation. This exposes the network to quantisation noise during training.

---

 ### 26\. Why can QAT outperform PTQ?

 A. The model learns to compensate for quantisation error\
 B. QAT always has fewer parameters\
 C. QAT eliminates activations\
 D. QAT uses more labelled classes

 **Answer: A**

 **Explanation:** QAT allows the model to adapt its parameters to the distortions introduced by quantisation, often producing better final task accuracy than straightforward PTQ.

---

 ### 27\. Which statement about QAT is correct?

 A. It is generally cheaper and faster than PTQ\
 B. It is slower and more costly because it requires additional training\
 C. It requires no model training\
 D. It can only be applied to convolutional networks

 **Answer: B**

 **Explanation:** PTQ is relatively straightforward after training, whereas QAT requires additional training iterations and therefore incurs extra computational cost.

---

 # Part IV — Pruning

 ### 28\. What is the fundamental idea behind pruning?

 A. Increase every weight's magnitude\
 B. Delete or zero out parameters judged unnecessary\
 C. Convert all parameters to FP32\
 D. Add additional layers

 **Answer: B**

 **Explanation:** Pruning removes unnecessary connections or structures, often by setting their weights to zero.

---

 ### 29\. Why is sparse storage important for pruning?

 A. Zero weights automatically disappear from every dense matrix\
 B. Sparse representations can avoid storing explicit zeros\
 C. Sparse storage increases the number of parameters\
 D. Sparse storage is only useful for full-precision models

 **Answer: B**

 **Explanation:** Simply setting weights to zero does not automatically save memory in a dense representation. Sparse formats can exploit the zeros and store only nonzero values and their locations.

---

 ### 30\. Why might pruning 50–90% of weights be useful?

 A. Such sparsity can produce substantial memory and computation savings when supported by suitable representations/hardware\
 B. It guarantees zero accuracy loss\
 C. It increases the number of nonzero weights\
 D. It makes the model mathematically exact

 **Answer: A**

 **Explanation:** High sparsity can substantially reduce resource requirements, although actual benefits depend on sparse storage and hardware support.

---

 ### 31\. What does magnitude pruning remove?

 A. Weights with the largest absolute values\
 B. Weights with values closest to zero\
 C. Entire datasets\
 D. All biases

 **Answer: B**

 **Explanation:** Magnitude pruning assumes weights with small absolute magnitude are less important and removes them.

---

 ### 32\. Magnitude pruning can be implemented using:

 A. Only a fixed number of layers\
 B. Either a magnitude threshold or a target sparsity\
 C. Only gradient descent\
 D. Only random selection

 **Answer: B**

 **Explanation:** One can specify a numerical threshold or specify a desired proportion of weights to prune, such as the smallest 20%.

---

 ### 33\. What is the intuition behind similarity pruning?

 A. Remove neurons whose activations are very similar to other neurons\
 B. Remove neurons with large gradients only\
 C. Remove all neurons with positive activations\
 D. Remove neurons randomly

 **Answer: A**

 **Explanation:** If two neurons produce highly similar activations for the same inputs, they may be redundant. One can therefore remove one without losing much unique functionality.

---

 ### 34\. Node pruning can use which criterion according to the lecture?

 A. Low L2 norm of weights or activations\
 B. High number of training examples\
 C. Maximum layer depth\
 D. Alphabetical ordering of neurons

 **Answer: A**

 **Explanation:** Neurons with very low L2 norm may behave like "dead" neurons whose activations remain near zero.

---

 ### 35\. What does structured pruning remove?

 A. Individual arbitrary weights only\
 B. Whole structural units such as channels, filters, or nodes\
 C. Training examples\
 D. Numerical precision

 **Answer: B**

 **Explanation:** Structured pruning removes meaningful blocks of the network, such as convolutional channels or filters.

---

 ### 36\. Why can structured pruning be more useful for actual speedups than unstructured pruning?

 A. GPUs are generally better optimized for dense, structured blocks than arbitrary sparse patterns\
 B. Structured pruning always produces higher accuracy\
 C. Structured pruning does not remove parameters\
 D. Unstructured pruning cannot reduce memory

 **Answer: A**

 **Explanation:** Hardware often handles regular dense operations efficiently. Removing entire channels or filters can directly reduce the dimensions of those operations, making the resulting speed and memory savings more practical.

---

 ### 37\. Which statement best describes iterative pruning?

 A. All pruning occurs before training begins\
 B. Increasing amounts of the network are pruned progressively during training\
 C. Parameters are never fine-tuned after pruning\
 D. Only one neuron can ever be removed

 **Answer: B**

 **Explanation:** Iterative pruning progressively removes parameters and allows the remaining network to adapt between pruning steps.

---

 ### 38\. Why can iterative pruning preserve performance better than one-shot pruning?

 A. The model has opportunities to compensate gradually for removed parameters\
 B. It increases the number of parameters after every step\
 C. It prevents any weights from becoming zero\
 D. It eliminates the need for optimization

 **Answer: A**

 **Explanation:** Gradual pruning allows the remaining parameters to adjust to the changing network structure.

---

 ### 39\. What is the relationship between pruning and quantisation?

 A. They are mutually exclusive\
 B. They can be combined to obtain the benefits of both\
 C. Quantisation reverses pruning\
 D. Pruning is a form of quantisation

 **Answer: B**

 **Explanation:** The lecture explicitly presents pruning and quantisation as complementary compression methods.

---

 # Part V — Compression Theory and the Lottery Ticket Hypothesis

 ### 40\. What does the manifold hypothesis suggest in this context?

 A. Every parameter configuration is equally important\
 B. Useful high-dimensional structures may effectively occupy a lower-dimensional space\
 C. Neural networks require all possible parameters\
 D. Compression is impossible

 **Answer: B**

 **Explanation:** The lecture extends the intuition of the manifold hypothesis to trained neural networks: although the parameter space is huge, the solutions actually used may occupy a much lower-dimensional region.

---

 ### 41\. Why is pruning surprising from the perspective of model capacity?

 A. It can remove many parameters while causing surprisingly little performance degradation\
 B. It always increases the number of parameters\
 C. It only removes unused training examples\
 D. It necessarily makes the model larger

 **Answer: A**

 **Explanation:** The lecture asks why models can tolerate removing enormous numbers of parameters with relatively small accuracy losses. Low-dimensional structure is one conceptual explanation.

---

 ### 42\. What does the Lottery Ticket Hypothesis propose?

 A. Every large network must be trained from scratch multiple times\
 B. A randomly initialized dense network may contain a sparse subnetwork that can train effectively in isolation\
 C. Pruning always destroys generalisation\
 D. Smaller networks always outperform larger ones

 **Answer: B**

 **Explanation:** The hypothesis proposes that a randomly initialized dense network contains a "winning ticket": a subnetwork whose initialization allows it to achieve comparable test accuracy when trained appropriately.

---

 ### 43\. According to the Lottery Ticket Hypothesis as presented, why can starting with a larger network help?

 A. It guarantees better hardware utilization\
 B. It provides more opportunities for a suitable winning subnetwork to exist\
 C. It reduces the number of initial parameters\
 D. It eliminates the need for initialization

 **Answer: B**

 **Explanation:** A larger dense network gives more possible subnetworks, increasing the chance that it contains a particularly effective sparse "winning ticket."

---

 ### 44\. What is distinctive about the iterative Lottery Ticket procedure described in the lecture?

 A. Surviving weights are reset to their original initialization and retrained\
 B. Surviving weights are permanently frozen\
 C. All weights are randomly reinitialized after every batch\
 D. No pruning occurs

 **Answer: A**

 **Explanation:** This is a key distinction. The procedure does not merely prune and continue training; the surviving weights are reset to their **original initial values** and trained again.

---

 ### 45\. Which claim was highlighted about winning-ticket subnetworks?

 A. They necessarily require more parameters than the original model\
 B. They can sometimes train faster and generalise better than the original dense network\
 C. They cannot generalise\
 D. They only work with quantised networks

 **Answer: B**

 **Explanation:** The lecture presents experimental results in which sparse subnetworks trained from their original initialization could generalise well, sometimes using only a fraction of the original parameters.

---

 # Part VI — Parameter-Efficient Fine-Tuning

 ### 46\. What is the central idea of parameter-efficient fine-tuning (PEFT)?

 A. Train every parameter more aggressively\
 B. Adapt a pretrained model while updating only a small subset or adding a small number of trainable parameters\
 C. Train without a pretrained model\
 D. Remove all model parameters

 **Answer: B**

 **Explanation:** PEFT exploits the idea that the adaptation needed for a new task/domain can have much lower dimensionality than the full parameter space.

---

 ### 47\. What is the main benefit of layer freezing?

 A. It reduces inference computation\
 B. It reduces training memory by avoiding gradients and optimizer states for frozen layers\
 C. It always improves accuracy\
 D. It increases the total number of trainable parameters

 **Answer: B**

 **Explanation:** Frozen layers do not need trainable optimizer states or gradient storage, reducing training memory requirements.

---

 ### 48\. Why can freezing early layers sometimes improve training efficiency?

 A. Early layers are often frozen, reducing the amount of backward computation through trainable parameters\
 B. Early layers contain no parameters\
 C. Early layers are always removed\
 D. Freezing early layers increases the model's FLOPs

 **Answer: A**

 **Explanation:** Freezing early layers can reduce backward-pass computation and associated memory, although the exact benefit depends on the architecture and training setup.

---

 ### 49\. What happens to inference when layers are frozen?

 A. Inference necessarily becomes twice as fast\
 B. Inference is unchanged because frozen parameters still participate in the forward computation\
 C. The frozen layers disappear\
 D. The model becomes quantised

 **Answer: B**

 **Explanation:** Freezing is primarily a **training-time** optimization. The frozen layers are still part of the model and therefore still execute during inference.

---

 ### 50\. What is the conceptual limitation of layer freezing as a PEFT method?

 A. It restricts updates according to entire layers, which may not match the true low-dimensional adaptation directions\
 B. It cannot reduce training memory\
 C. It changes every parameter\
 D. It only works with convolutional models

 **Answer: A**

 **Explanation:** The lecture emphasizes that the useful update directions may cross layer boundaries. Restricting all updates to selected layers may therefore be unnecessarily restrictive.

---

 ### 51\. What is an adapter module?

 A. A module inserted between layers whose parameters are trained while the original model remains frozen\
 B. A method for deleting neurons\
 C. A numerical representation\
 D. A replacement for the dataset

 **Answer: A**

 **Explanation:** Adapters add small trainable modules between existing layers while keeping the original pretrained parameters frozen.

---

 ### 52\. Why does an adapter typically use a bottleneck architecture?

 A. To increase the number of parameters\
 B. To represent the adaptation with relatively few parameters\
 C. To prevent any nonlinear transformations\
 D. To eliminate residual connections

 **Answer: B**

 **Explanation:** The bottleneck compresses the intermediate representation, allowing the adapter to introduce only a small number of trainable parameters.

---

 ### 53\. What is the role of the skip connection in an adapter?

 A. It removes the pretrained model output\
 B. It allows the adapter to learn a residual offset to the frozen model's behavior\
 C. It freezes the adapter\
 D. It converts the model to INT8

 **Answer: B**

 **Explanation:** The adapter learns an additive residual correction while the original network remains intact.

---

 ### 54\. What is a major disadvantage of adapter modules compared with LoRA?

 A. Adapters require training the entire model\
 B. Adapters introduce additional layers and therefore can introduce inference latency\
 C. Adapters cannot be trained\
 D. Adapters always require more parameters than full fine-tuning

 **Answer: B**

 **Explanation:** Although adapters can dramatically reduce trainable parameters, they modify the computation graph by adding modules, which can add inference overhead.

---

 ### 55\. What is prefix tuning?

 A. Removing the beginning of every sequence\
 B. Learning a specialized context/prefix that is prepended to the input while the main model remains frozen\
 C. Pruning the first layer\
 D. Quantising only input tokens

 **Answer: B**

 **Explanation:** Prefix tuning learns task-specific context vectors that can be prepended to inputs. A key advantage highlighted in the lecture is that different prefixes can be swapped for different tasks.

---

 # Part VII — Separable Convolutions and Low-Rank Factorisation

 ### 56\. Why can spatially separable convolution reduce the number of parameters in a convolutional kernel?

 A. It represents a kernel as a composition of smaller kernels\
 B. It removes the activation function\
 C. It increases kernel dimensionality\
 D. It uses more independent parameters

 **Answer: A**

 **Explanation:** Instead of learning all entries of a kernel directly, the kernel can sometimes be represented as a product/composition of smaller structures.

---

 ### 57\. According to the lecture's 3×3 example, how many parameters can be used in a separable representation instead of 9?

 A. 3\
 B. 6\
 C. 9\
 D. 18

 **Answer: B**

 **Explanation:** The lecture describes representing a 3×3 kernel using a 3×1 and a 1×3 kernel, requiring **3 + 3 = 6 parameters** instead of 9.

---

 ### 58\. What is the approximate parameter reduction in the 3×3 → 3×1 + 1×3 example?

 A. 10%\
 B. 20%\
 C. 33%\
 D. 66%

 **Answer: C**

 **Explanation:** The number of parameters drops from 9 to 6. The reduction is $3/9 = 33.3\%$.

---

 ### 59\. Does every matrix admit an exact strict separable decomposition?

 A. Yes\
 B. No\
 C. Only matrices with integer entries\
 D. Only neural-network matrices

 **Answer: B**

 **Explanation:** The lecture explicitly notes that not every matrix can be expressed using such a strict composition. Approximate matrix factorisation, however, can often provide useful compression.

---

 ### 60\. What is the key conceptual distinction between low-rank weight factorisation and LoRA?

 A. Both directly assume the pretrained weights themselves are low-rank\
 B. Weight factorisation compresses the existing weights, while LoRA assumes the **fine-tuning update** can be low-rank\
 C. LoRA removes all pretrained weights\
 D. Weight factorisation only works during inference

 **Answer: B**

 **Explanation:** This is one of the most important ideas in the lecture. A trained model's **weights themselves may not be structurally low-rank**, even if the model is redundant. LoRA instead exploits the observation that the **update needed during fine-tuning** may have low intrinsic dimension.

---

 # Part VIII — SVD and Low-Rank Weight Factorisation

 ### 61\. What is the standard SVD decomposition of a matrix $A$?

 A. $A=UV$\
 B. $A=U\Sigma V^T$\
 C. $A=U+V+\Sigma$\
 D. $A=\Sigma^{-1}UV$

 **Answer: B**

 **Explanation:** Singular Value Decomposition expresses a matrix as

 $$
A = U\Sigma V^T.
$$

---

 ### 62\. What does truncated SVD do?

 A. Keeps only the largest singular components\
 B. Keeps only the smallest singular components\
 C. Randomly removes singular vectors\
 D. Converts the matrix to binary

 **Answer: A**

 **Explanation:** If a matrix is approximately low-rank, small singular components can be discarded while retaining the dominant structure.

---

 ### 63\. Why might truncated SVD be expected to compress a neural network?

 A. It can represent a matrix approximately using fewer dimensions\
 B. It increases matrix rank\
 C. It adds parameters\
 D. It eliminates activations

 **Answer: A**

 **Explanation:** A low-rank approximation replaces a large matrix with a product of smaller matrices, reducing the number of stored parameters.

---

 ### 64\. What empirical problem with direct SVD compression of DNN weights is highlighted in the lecture?

 A. Neural-network weights are always exactly rank one\
 B. Model parameters can be highly sensitive to SVD truncation and degrade quickly\
 C. SVD cannot be computed for matrices\
 D. SVD increases precision

 **Answer: B**

 **Explanation:** The lecture makes an important distinction: **features may be low-rank, but weights are not necessarily structurally low-rank** in the way required for aggressive SVD compression.

---

 # Part IX — LoRA

 ### 65\. What is the central insight behind LoRA?

 A. The pretrained weights themselves must be low-rank\
 B. Fine-tuning updates to a pretrained model can often be represented approximately with a low-rank matrix\
 C. All model weights should be deleted\
 D. Quantisation and fine-tuning are incompatible

 **Answer: B**

 **Explanation:** LoRA does not claim that the pretrained weight matrix $W$ is low-rank. Instead, it assumes that the update $\Delta W$ required for adaptation can be represented using a low-rank factorization.

---

 ### 66\. In ordinary fine-tuning, what is the shape of $\Delta W$ relative to $W$?

 A. It has half the dimensions\
 B. It has the same shape\
 C. It is always scalar\
 D. It has twice the dimensions

 **Answer: B**

 **Explanation:** Ordinary gradient updates produce a matrix $\Delta W$ with the same dimensions as $W$.

---

 ### 67\. In LoRA, how is the update approximated?

 A. $\Delta W \approx A+B$\
 B. $\Delta W \approx BA$\
 C. $\Delta W \approx A-B$\
 D. $\Delta W \approx A^TB^T$

 **Answer: B**

 **Explanation:** LoRA represents the update using two smaller matrices:

 $$
\Delta W \approx BA.
$$

---

 ### 68\. Suppose $W$ is a $d\times d$ matrix and LoRA uses rank $r$. What are the dimensions of $A$ and $B$ according to the lecture?

 A. $A:d\times r,\ B:r\times d$\
 B. $A:r\times d,\ B:d\times r$\
 C. $A:d\times d,\ B:r\times r$\
 D. $A:r\times r,\ B:d\times d$

 **Answer: B**

 **Explanation:**

 $$
A\in\mathbb{R}^{r\times d},
\qquad
B\in\mathbb{R}^{d\times r}.
$$

 Therefore,

 $$
BA\in\mathbb{R}^{d\times d},
$$

 matching the shape of $W$.

---

 ### 69\. Why does LoRA use $r\ll d$?

 A. To make the low-rank update contain far fewer trainable parameters than the original matrix\
 B. To increase the rank of the update\
 C. To increase the size of the pretrained model\
 D. To eliminate matrix multiplication

 **Answer: A**

 **Explanation:** The original $d\times d$ matrix has $d^2$ parameters, while the LoRA factors contain

 $$
rd+dr=2rd
$$

 parameters. When $r\ll d$, this is dramatically smaller than $d^2$.

---

 ### 70\. What happens to the original pretrained weights in LoRA during fine-tuning?

 A. They are randomly reinitialized\
 B. They are frozen\
 C. They are quantised to zero\
 D. They are deleted

 **Answer: B**

 **Explanation:** LoRA freezes the pretrained weights and trains only the low-rank adaptation matrices.

---

 ### 71\. Why is LoRA's initialization often arranged so that $BA=0$?

 A. So the adapted model initially behaves like the original pretrained model\
 B. So the model initially produces random outputs\
 C. So all gradients become zero permanently\
 D. So the original weights disappear

 **Answer: A**

 **Explanation:** If $BA=0$, the adapter initially introduces no change:

 $$
W_{\text{effective}}=W+BA=W.
$$

 Thus fine-tuning starts from the original pretrained behavior.

---

 ### 72\. Which statement correctly describes how LoRA relates to matrix decomposition?

 A. LoRA first computes the SVD of the gradient matrix\
 B. LoRA directly factorizes the pretrained weights using SVD\
 C. LoRA does not perform an actual matrix decomposition; it learns a low-rank update directly\
 D. LoRA requires every weight matrix to be rank one

 **Answer: C**

 **Explanation:** This distinction is crucial. LoRA **assumes** the useful update can be represented in low rank and learns the factors directly. It does not first perform an SVD decomposition of the existing matrix or gradient.

---

 ### 73\. What happens during the LoRA forward pass?

 A. Only $BA$ is used\
 B. Only the original weights are used\
 C. The original-weight computation and low-rank adaptation are combined\
 D. Both matrices are discarded

 **Answer: C**

 **Explanation:** Conceptually,

 $$
W_{\text{effective}}=W+BA,
$$

 possibly with an additional scaling factor in practical LoRA formulations.

---

 ### 74\. What is a major deployment advantage of LoRA?

 A. It necessarily requires extra inference layers\
 B. The low-rank update can be merged into the original weights\
 C. It requires storing all optimizer states during inference\
 D. It doubles the model's inference architecture

 **Answer: B**

 **Explanation:** Because $W+BA$ can be precomputed, the adapter can be merged into the original weights. The resulting inference architecture can therefore be identical to the base architecture.

---

 ### 75\. Why is LoRA especially useful for modular fine-tuning?

 A. Each task requires storing a complete copy of the pretrained model\
 B. Different low-rank updates can be stored and swapped while sharing the same base model\
 C. LoRA prevents task specialization\
 D. LoRA requires a new architecture for every task

 **Answer: B**

 **Explanation:** A single large pretrained model can be shared, while small task-specific LoRA matrices are stored separately and activated as needed.

---

 ### 76\. Which scenario best illustrates modular LoRA?

 A. Store one complete model for every language\
 B. Store one base model plus small language-specific LoRA updates\
 C. Delete the base model after every task\
 D. Train each task entirely from scratch

 **Answer: B**

 **Explanation:** The lecture specifically gives the example of a full pretrained model plus small low-rank updates for different languages/tasks.

---

 ### 77\. According to the lecture's GPT-3 example, LoRA can reduce the number of trainable parameters by approximately:

 A. 2×\
 B. 10×\
 C. 100×\
 D. 10,000×

 **Answer: D**

 **Explanation:** The lecture cites the original LoRA work's GPT-3 example as using roughly **10,000× fewer trainable weights**, corresponding to about 0.01% of the full parameter count.

---

 ### 78\. Why can the LoRA rank $r$ be surprisingly small?

 A. The useful fine-tuning update may have very low intrinsic dimensionality\
 B. The pretrained model contains no information\
 C. All language tasks are identical\
 D. A low rank always means perfect reconstruction

 **Answer: A**

 **Explanation:** The lecture connects LoRA's effectiveness to the empirical observation that fine-tuning movement through parameter space may have far fewer effective dimensions than the full parameter space.

---

 ### 79\. A $12{,}288\times12{,}288$ weight matrix uses LoRA with $r=8$. What are the dimensions of its two LoRA matrices?

 A. $12{,}288\times8$ and $8\times12{,}288$\
 B. $12{,}288\times12{,}288$ and $8\times8$\
 C. $8\times8$ and $12{,}288\times12{,}288$\
 D. $12{,}288\times4$ and $4\times12{,}288$

 **Answer: A**

 **Explanation:** For $d=12{,}288$ and $r=8$:

 $$
A\in\mathbb{R}^{8\times12{,}288},
\qquad
B\in\mathbb{R}^{12{,}288\times8}.
$$

 The lecture gives these dimensions in the GPT-3 rank-selection example.

---

 # Part X — Integrated and Application Questions

 These questions combine multiple concepts and are closer to the kind of questions that test **actual mastery**.

 ### 80\. You have a trained model whose inference memory is the main bottleneck. Which technique most directly targets this issue?

 A. Quantisation\
 B. Layer freezing\
 C. Increasing rank in LoRA\
 D. Increasing batch size

 **Answer: A**

 **Explanation:** Quantisation directly reduces the number of bits needed to represent model parameters and therefore can substantially reduce inference memory. Layer freezing primarily affects training.

---

 ### 81\. You have a pretrained model and want to adapt it to many tasks while keeping one shared base model. You also want **no additional inference architecture after deployment**. Which technique is especially appropriate?

 A. Adapter modules without merging\
 B. Layer freezing alone\
 C. LoRA with weight merging\
 D. Training a separate model for every task

 **Answer: C**

 **Explanation:** LoRA allows small task-specific updates to be learned while the base model remains frozen. Those updates can subsequently be merged into the base weights, avoiding additional inference latency.

---

 ### 82\. You quantise a model after training and find that accuracy drops substantially. What is the lecture's natural next step?

 A. Increase parameter count randomly\
 B. Consider quantisation-aware training\
 C. Remove all activations\
 D. Use layer freezing

 **Answer: B**

 **Explanation:** PTQ is straightforward but can sometimes fail to preserve the original accuracy. QAT allows the model to learn to compensate for quantisation effects.

---

 ### 83\. You prune 90% of a model's weights but observe almost no inference speedup. What is the most plausible explanation?

 A. Pruning never reduces computation\
 B. The hardware/runtime may not exploit the resulting unstructured sparsity efficiently\
 C. The model has become larger\
 D. Quantisation has reversed the pruning

 **Answer: B**

 **Explanation:** Sparsity only translates into practical speedups when the storage format and hardware can exploit it. Structured pruning is often more hardware-friendly because dense GPU blocks are efficiently optimized.

---

 ### 84\. You freeze the first 80% of the layers during fine-tuning. What is the most likely direct benefit?

 A. Lower training memory\
 B. Lower inference memory automatically\
 C. Higher numerical precision\
 D. More total parameters

 **Answer: A**

 **Explanation:** Frozen layers do not need trainable gradients and optimizer states. However, they still participate in inference, so freezing alone does not necessarily reduce inference memory or latency.

---

 ### 85\. You want the model to adapt using parameters distributed across **every layer**, but you do not want to train every original weight. Which approach most directly addresses this?

 A. Freeze the first 90% of layers\
 B. Train only the classification head\
 C. Use adapters or LoRA across layers\
 D. Delete all intermediate layers

 **Answer: C**

 **Explanation:** Layer freezing restricts adaptation to selected layers. PEFT approaches such as adapters and LoRA can introduce small trainable updates throughout the network.

---

 ### 86\. Which pair consists entirely of **compression** techniques rather than primarily PEFT techniques?

 A. Quantisation and pruning\
 B. LoRA and adapters\
 C. Prefix tuning and layer freezing\
 D. LoRA and prefix tuning

 **Answer: A**

 **Explanation:** Quantisation and pruning reduce the representation/size of a model. LoRA, adapters, prefix tuning, and freezing primarily address efficient **adaptation/fine-tuning**.

---

 ### 87\. Which pair can be combined to achieve compounded efficiency gains?

 A. Pruning and quantisation\
 B. Layer freezing and increasing model size\
 C. Prefix tuning and increasing precision\
 D. SVD and adding random parameters

 **Answer: A**

 **Explanation:** The lecture explicitly presents pruning and quantisation as compatible techniques. It also presents LoRA + quantisation through QLoRA.

---

 ### 88\. Which statement best captures the relationship between **model compression** and **parameter-efficient fine-tuning**?

 A. They are exactly the same technique\
 B. Compression primarily reduces the representation/compute/storage of a model, while PEFT primarily reduces the number of parameters that must be updated during adaptation\
 C. PEFT always reduces inference model size\
 D. Compression only applies during training

 **Answer: B**

 **Explanation:** This distinction is fundamental to the lecture. Quantisation and pruning target model representation and computation. PEFT methods target the cost of adapting a pretrained model.

---

 ### 89\. A researcher argues: "Because DNNs have low intrinsic dimensionality, their trained weight matrices should always be easy to compress using truncated SVD." What does the lecture suggest is wrong with this argument?

 A. DNNs have no intrinsic dimensionality\
 B. Low intrinsic dimensionality of the solution/update space does not imply that the actual weight matrices are structurally low-rank\
 C. SVD increases rank\
 D. Weight matrices cannot be factorised

 **Answer: B**

 **Explanation:** This is one of the lecture's most important conceptual subtleties. The model may have a low-dimensional **solution/update space** without the actual weight matrices being low-rank enough for aggressive SVD truncation.

---

 ### 90\. Why does LoRA exploit this distinction particularly well?

 A. It compresses the existing weight matrix directly\
 B. It assumes the fine-tuning update is low-rank rather than assuming the pretrained weights themselves are low-rank\
 C. It removes the pretrained model\
 D. It requires full-rank fine-tuning

 **Answer: B**

 **Explanation:** LoRA effectively says: the full pretrained model can remain high-dimensional, but the **change needed to adapt it** may occupy a much smaller subspace.

---

 # Part XI — High-Value "Distinguish Between" Questions

 ### 91\. Which statement correctly distinguishes PTQ from QAT?

 A. PTQ requires retraining; QAT does not\
 B. PTQ is performed after training; QAT simulates quantisation effects during training\
 C. PTQ changes architecture; QAT deletes weights\
 D. PTQ uses adapters; QAT uses LoRA

 **Answer: B**

---

 ### 92\. Which statement correctly distinguishes magnitude pruning from similarity pruning?

 A. Magnitude pruning examines weight size; similarity pruning examines redundancy between units/activations\
 B. Both use exactly the same criterion\
 C. Magnitude pruning removes whole layers; similarity pruning removes individual bits\
 D. Similarity pruning quantises weights

 **Answer: A**

 **Explanation:** Magnitude pruning asks whether a weight is small. Similarity pruning asks whether a neuron/feature map is redundant with another.

---

 ### 93\. Which statement correctly distinguishes node pruning from unstructured weight pruning?

 A. Node pruning removes entire neurons/structural units rather than arbitrary individual connections\
 B. Node pruning only changes precision\
 C. Node pruning cannot save memory\
 D. Node pruning is identical to QAT

 **Answer: A**

---

 ### 94\. Which statement correctly distinguishes adapters from LoRA?

 A. Adapters add trainable modules to the architecture; LoRA adds low-rank weight updates that can be merged into the base weights\
 B. Adapters always train all original weights; LoRA deletes them\
 C. Adapters are numerical formats; LoRA is pruning\
 D. They are mathematically identical in every respect

 **Answer: A**

 **Explanation:** Both are PEFT approaches, but their implementation and inference implications differ. Adapters introduce additional modules, whereas LoRA's low-rank update can be merged into the original weights.

---

 ### 95\. Which statement correctly distinguishes layer freezing from LoRA?

 A. Layer freezing freezes selected layers and trains the rest; LoRA freezes the original model while learning low-rank updates\
 B. Both train every parameter\
 C. Layer freezing is a quantisation method\
 D. LoRA removes layers

 **Answer: A**

---

 ### 96\. Which statement correctly distinguishes ordinary fine-tuning from LoRA?

 A. Ordinary fine-tuning learns a full-rank parameter update; LoRA constrains the learned update to a low-rank parameterization\
 B. Ordinary fine-tuning cannot use gradients\
 C. LoRA changes every pretrained parameter directly\
 D. Ordinary fine-tuning always uses fewer trainable parameters

 **Answer: A**

---

 # Part XII — Calculation and Reasoning Questions

 ### 97\. A $d\times d$ matrix has $d^2$ parameters. A LoRA update uses rank $r$, with $A\in\mathbb{R}^{r\times d}$ and $B\in\mathbb{R}^{d\times r}$. How many trainable parameters does LoRA introduce?

 A. $d^2+r^2$\
 B. $2dr$\
 C. $d+r$\
 D. $dr^2$

 **Answer: B**

 **Explanation:**

 $$
A: rd,\qquad B: dr.
$$

 Therefore:

 $$
N_{\text{LoRA}}=rd+dr=2dr.
$$

 When $r\ll d$, this is much smaller than $d^2$.

---

 ### 98\. Suppose $d=1000$ and $r=10$. How many parameters are in the original matrix and in the two LoRA matrices?

 A. Original: 10,000; LoRA: 20,000\
 B. Original: 1,000,000; LoRA: 20,000\
 C. Original: 1,000,000; LoRA: 10,000\
 D. Original: 100,000; LoRA: 2,000

 **Answer: B**

 **Explanation:**

 Original:

 $$
1000^2=1,000,000.
$$

 LoRA:

 $$
2(1000)(10)=20,000.
$$

 Thus the trainable representation is dramatically smaller.

---

 ### 99\. A model has 100 million FP32 parameters. Ignoring overhead and assuming one parameter uses 32 bits, approximately how much parameter storage is needed?

 A. 40 MB\
 B. 100 MB\
 C. 400 MB\
 D. 3.2 GB

 **Answer: C**

 **Explanation:**

 FP32 = 4 bytes per parameter.

 $$
100\,000\,000\times4
=400\,000\,000\text{ bytes}
$$

 which is approximately **400 MB** using decimal units.

---

 ### 100\. If the same 100 million parameters are represented using 8 bits per parameter instead of 32 bits, approximately how much storage is needed?

 A. 25 MB\
 B. 100 MB\
 C. 200 MB\
 D. 400 MB

 **Answer: B**

 **Explanation:**

 INT8 uses 1 byte per parameter:

 $$
100\,000\,000\times1
=100\,000\,000\text{ bytes}
$$

 ≈ **100 MB**.

 This illustrates the fundamental memory advantage of quantisation.

---

 # Part XIII — Tricky Exam-Style Questions

 ### 101\. A student says: "If I freeze 90% of the model's parameters, inference will become approximately 90% faster." What is wrong?

 A. Frozen parameters are not used during inference\
 B. Freezing primarily affects training; frozen parameters still participate in the forward pass\
 C. Freezing increases inference computation\
 D. Freezing automatically quantises the model

 **Answer: B**

 **Explanation:** This is a classic conceptual trap. **Frozen ≠ removed**. The parameters remain in the network and are used during inference.

---

 ### 102\. A student says: "If a weight is zero, the model automatically gets a speedup." What is the best response?

 A. Yes, always\
 B. No; speedup depends on whether sparse storage and hardware/software can exploit the zero\
 C. Zero weights increase computation\
 D. Zero weights only affect accuracy

 **Answer: B**

 **Explanation:** A zero stored in a conventional dense matrix is still part of the computation. Practical savings require appropriate sparse representations and/or hardware support.

---

 ### 103\. Which situation is most likely to make quantisation particularly useful?

 A. A model is constrained by memory bandwidth and runs on hardware with efficient low-bit arithmetic\
 B. A model has unlimited memory and no hardware support\
 C. A model has no numerical parameters\
 D. The only goal is increasing parameter precision

 **Answer: A**

 **Explanation:** Quantisation is especially attractive when memory and low-bit computation are hardware-efficient.

---

 ### 104\. Why isn't "fewer parameters" alone sufficient to define model efficiency?

 A. Parameter count is unrelated to ML\
 B. FLOPs, latency, memory, and hardware characteristics also matter\
 C. Parameter count cannot be measured\
 D. More parameters always means a faster model

 **Answer: B**

 **Explanation:** A model can have fewer parameters but still require substantial computation or have poor latency depending on its architecture and hardware.

---

 ### 105\. Which statement best summarizes the lecture's view of compression?

 A. There is one universally optimal compression technique\
 B. Different techniques optimize different resources and involve different trade-offs\
 C. Compression always reduces accuracy to zero\
 D. Compression only matters for inference

 **Answer: B**

 **Explanation:** The lecture repeatedly emphasizes that efficiency gains have different advantages and trade-offs. The correct technique depends on whether the bottleneck is memory, computation, latency, training resources, etc.

---

 ### 106\. Suppose a model's weights are highly redundant but not low-rank in the SVD sense. Which approach from the lecture is particularly motivated?

 A. Direct truncated-SVD compression of the weights\
 B. LoRA for fine-tuning\
 C. Increasing the rank of the weight matrix\
 D. Removing all activation functions

 **Answer: B**

 **Explanation:** The lecture's central insight is that **redundancy does not necessarily mean structural low rank of the weights**. LoRA instead exploits low-dimensionality in the **adaptation/update**.

---

 ### 107\. Why can LoRA be described as a generalization of parameter-efficient fine-tuning?

 A. It adapts the model using a restricted low-dimensional update rather than updating all original weights\
 B. It always modifies every parameter\
 C. It removes the need for a pretrained model\
 D. It only applies to convolutional layers

 **Answer: A**

 **Explanation:** LoRA is a particularly powerful PEFT method because it allows adaptation across many layers while requiring only a small number of trainable parameters.

---

 ### 108\. Which combination best matches the primary resource affected by each method?

 A. Quantisation → memory; layer freezing → training memory; LoRA → trainable parameter count\
 B. Quantisation → training data; layer freezing → inference architecture; LoRA → dataset size\
 C. Pruning → labels; adapters → precision; LoRA → batch size\
 D. QAT → model depth; pruning → optimizer type; freezing → numerical range

 **Answer: A**

 **Explanation:** This is an important synthesis:

 - **Quantisation:** reduces bits/storage and potentially computation.
- **Layer freezing:** reduces training memory and trainable computation.
- **LoRA:** dramatically reduces trainable parameter count.

---

 # Part XIV — Final Mastery Questions

 ### 109\. Which technique is most directly described as "learn the model first, then cut away what it doesn't need"?

 A. Pruning\
 B. QAT\
 C. Prefix tuning\
 D. LoRA

 **Answer: A**

 **Explanation:** This is essentially the motivation given for pruning: train a sufficiently expressive model, identify unnecessary parameters/structures, and remove them.

---

 ### 110\. Which technique is most directly described as "train while simulating the errors introduced by quantisation"?

 A. PTQ\
 B. QAT\
 C. Magnitude pruning\
 D. Layer freezing

 **Answer: B**

 **Explanation:** QAT inserts fake-quantisation operations during training so the model can adapt to quantisation noise.

---

 ### 111\. Which technique is most directly described as "freeze the pretrained weights and learn two small matrices whose product represents the update"?

 A. Adapter tuning\
 B. Prefix tuning\
 C. LoRA\
 D. Magnitude pruning

 **Answer: C**

 **Explanation:** This is the defining mechanism of LoRA:

 $$
\Delta W\approx BA.
$$

---

 ### 112\. Which statement captures the deepest conceptual lesson connecting manifold hypothesis, intrinsic dimensionality, and LoRA?

 A. Large parameter spaces necessarily require equally large updates\
 B. Although neural networks may have enormous parameter spaces, useful adaptations can occupy a much smaller effective dimensionality\
 C. All neural networks are inherently low-rank matrices\
 D. Parameter count has no relationship to model efficiency

 **Answer: B**

 **Explanation:** This is arguably the central theoretical intuition behind the PEFT portion of the lecture. The **ambient parameter space can be huge**, while the directions needed to adapt a pretrained model can be surprisingly few.

---

 ### 113\. Why can LoRA provide parameter-efficient fine-tuning without changing the deployed architecture?

 A. It removes all layers\
 B. Its low-rank update can be merged into the original weight matrix\
 C. It never participates in the forward pass\
 D. It replaces the model with a lookup table

 **Answer: B**

 **Explanation:** After training,

 $$
W' = W + BA
$$

 can be computed ahead of deployment. The low-rank factors do not need to remain separate during inference.

---

 ### 114\. Which technique would be the most natural choice if the primary objective is to make a trained model occupy substantially less storage **without retraining it**?

 A. PTQ\
 B. QAT\
 C. Layer freezing\
 D. Prefix tuning

 **Answer: A**

 **Explanation:** PTQ can quantise an already-trained model without requiring additional training. QAT, in contrast, explicitly involves training with quantisation effects.

---

 ### 115\. Which technique would be the most natural choice if you need a single pretrained model to support many different tasks, with small task-specific components that can be swapped?

 A. Train a separate full model for every task\
 B. LoRA or adapters\
 C. Increase FP32 precision\
 D. Dense retraining from scratch

 **Answer: B**

 **Explanation:** Both adapters and LoRA support modular task-specific adaptations. LoRA has the additional advantage that the updates can be merged for deployment.

---

 ### 116\. Which of the following represents the lecture's overall progression most accurately?

 A. Quantisation → pruning → PEFT → intrinsic dimensionality → LoRA\
 B. LoRA → data augmentation → classification → quantisation\
 C. Pruning → larger models → more parameters → full fine-tuning\
 D. PEFT → removing all pretrained knowledge → random initialization

 **Answer: A**

 **Explanation:** The lecture moves from general efficiency constraints through **numerical compression**, **quantisation**, **pruning**, theoretical ideas about **intrinsic dimensionality**, and finally **parameter-efficient fine-tuning**, culminating in LoRA.

---

 # Compact Mastery Table

 | Concept | Core idea | Main benefit | Key limitation/trade-off |
| --- | --- | --- | --- |
| Reduced precision | Fewer bits for numerical representation | Memory + potentially speed | Precision/dynamic-range issues |
| BF16 | More exponent bits, fewer mantissa bits | Better dynamic range than FP16 | Hardware dependent |
| Mixed precision | Different precisions for different operations | Efficiency while retaining numerical stability | Requires careful implementation |
| Quantisation | Map continuous values to discrete representations | Major memory reduction | Quantisation error |
| PTQ | Quantise after training | Simple, cheap | May lose accuracy |
| QAT | Simulate quantisation during training | Better accuracy after quantisation | Additional training cost |
| Magnitude pruning | Remove small-magnitude weights | Sparsity/compression | Unstructured sparsity may not accelerate hardware |
| Similarity pruning | Remove redundant neurons/features | Removes redundant computation | Requires similarity analysis |
| Node pruning | Remove low-norm/dead neurons | Structural reduction | Architecture changes |
| Structured pruning | Remove channels/filters/etc. | Better practical speedups | More constrained than arbitrary pruning |
| Iterative pruning | Prune progressively during training | Allows gradual compensation | More complex training |
| Lottery Ticket | Find trainable sparse subnetworks | Sparse networks can train effectively | Identification can be expensive |
| Layer freezing | Freeze selected layers | Lower training memory | Restricts update locations |
| Adapters | Small bottleneck modules | Very few trainable parameters | Added inference layers/latency |
| Prefix tuning | Learn task-specific context vectors | Tiny task-specific representation | Mainly suited to sequence/language settings |
| Low-rank factorisation | Approximate weights with smaller matrices | Compression | Actual weights may not be sufficiently low-rank |
| LoRA | Low-rank parameter **updates** | Huge reduction in trainable parameters | Requires choosing rank |
| LoRA merging | Add $BA$ into $W$ | No extra inference architecture | Merging loses the separate modular representation |
| QLoRA | LoRA + quantisation | Compounded memory efficiency | More complex numerical/training setup |

---

 # The 15 Facts I Would Memorize First

 If you're preparing for an exam, these are the **highest-yield relationships** from the lecture:

 1. **Efficiency ≠ parameter count alone.** Consider parameters, FLOPs, latency/throughput, and memory.
2. **Backward ≈ 2× forward**, so training is approximately **3× inference** in computation under the lecture's rule of thumb.
3. **Reduced precision saves memory and potentially time/energy.**
4. **FP16's problem can be dynamic range**, not merely precision.
5. **BF16 trades mantissa precision for exponent/dynamic range.**
6. **PTQ = quantise after training.**
7. **QAT = train while simulating quantisation**, allowing the model to compensate for quantisation error.
8. **Pruning = remove unnecessary parameters/structures.**
9. **Magnitude pruning = small weights; similarity pruning = redundant neurons/features.**
10. **Structured pruning is generally more hardware-friendly** than arbitrary unstructured sparsity.
11. **Lottery Ticket Hypothesis:** a dense random network may contain a sparse, trainable "winning ticket"; the surviving weights can be reset to their original initialization.
12. **Layer freezing reduces training cost, not inference cost.**
13. **Adapters add small trainable bottleneck modules; LoRA adds low-rank weight updates.**
14. **The actual weights need not be low-rank for LoRA to work.** The important assumption is that the **fine-tuning update $\Delta W$** can be low-rank.
15. **LoRA:**\

    $$
    \boxed{\Delta W\approx BA}
    $$
    \
     with\

    $$
    A\in\mathbb{R}^{r\times d},\qquad
       B\in\mathbb{R}^{d\times r},\qquad r\ll d.
    $$
    \
     The base weights remain frozen, and $BA$ can ultimately be **merged into the base weights**, giving no additional inference architecture.

 ### One mental model for the whole lecture

 Think of the lecture as answering **three different questions**:

 > **"How can I make the model itself smaller/faster?"**\
>  → **Quantisation + pruning + factorisation**

 > **"How can I train/adapt a huge model without updating everything?"**\
>  → **Freezing + adapters + prefix tuning + LoRA**

 > **"Why can this work despite huge neural networks?"**\
>  → **Redundancy + manifold hypothesis + intrinsic dimensionality + low-dimensional updates**

 And the key conceptual endpoint is:

 $$
\boxed{\text{Huge model} \;\neq\; \text{huge adaptation}}
$$

 A model can have an enormous parameter space while the **useful change required for a new task occupies a tiny subspace**. That is the central intuition connecting the lecture's PEFT material to LoRA.
