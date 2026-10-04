# Model Compression 

 - **Conceptual questions** — “What is quantisation and why does it reduce memory?”
- **Application/comparison questions** — “A model loses accuracy after PTQ. What would you do and why?”
- **Exam-grade reasoning questions** — questions where students must explain trade-offs, interpret a scenario, compare methods, or derive a result.

 For these lecture notes, the exam questions should particularly test whether students understand **why a method works, what it saves, what it costs, and when to choose it**.

 ### Example exam-grade questions

 **Q1. Quantisation**

 A neural network is trained using FP32 weights. After deployment, its weights are converted to INT8.

 **a)** Explain what has happened to the representation of the weights.\
 **b)** Why does this reduce memory usage?\
 **c)** Why might model accuracy decrease?\
 **d)** Why does INT8 not necessarily make inference faster on every device?

 **Answer:**\
 a) The continuous range of FP32 values is mapped onto a discrete set of INT8 values.\
 b) FP32 uses 32 bits per parameter whereas INT8 uses 8 bits, giving roughly a **4× reduction in weight storage**.\
 c) Quantisation introduces approximation/error because many floating-point values map to the same discrete value. Outliers can make this worse.\
 d) Speedups depend on hardware support for efficient low-precision arithmetic. If the hardware does not accelerate INT8 operations, the memory reduction may not translate into a large runtime improvement.

---

 **Q2. PTQ vs QAT**

 A model has 95% accuracy before compression but only 88% after post-training quantisation.

 **What technique from the lecture would you try next, and why?**

 **Answer:**\
 Use **quantisation-aware training (QAT)**. During training, fake quantisation operations simulate the quantisation error. The model can therefore learn to compensate for the errors that will occur after deployment. QAT is more expensive than PTQ but can recover accuracy.

---

 **Q3. Pruning**

 A model contains 10 million parameters. You discover that 90% of its weights can be set to zero with only a small reduction in accuracy.

 **Does this automatically mean that inference will be 10× faster? Explain.**

 **Answer:**\
 No. If the zeros are stored and processed using a dense representation, little computational benefit may occur. To obtain substantial benefits, the model needs an appropriate **sparse representation and hardware/software support for sparse operations**. Structured pruning can be particularly useful because modern hardware handles dense blocks efficiently.

---

 **Q4. Magnitude pruning vs similarity pruning**

 Explain the difference between magnitude pruning and similarity pruning.

 **Answer:**

 - **Magnitude pruning:** remove weights with small absolute values because they are assumed to have little influence.
- **Similarity pruning:** identify neurons/features that produce very similar activations and remove redundant ones because they are performing similar functions.

 The first looks at the **size of individual weights**; the second looks at **redundancy between learned features**.

---

 **Q5. Structured vs unstructured pruning**

 Why can structured pruning sometimes provide greater practical speedups than unstructured pruning, even when both remove the same percentage of parameters?

 **Answer:**\
 Unstructured pruning produces scattered zeros throughout weight matrices. These require specialised sparse representations and operations to exploit effectively.

 Structured pruning removes whole neurons, channels, filters, etc. This produces smaller dense matrices/tensors that conventional GPU hardware can process efficiently. Therefore, the reduction in parameters is more directly translated into reduced computation and memory.

---

 **Q6. Lottery Ticket Hypothesis**

 The Lottery Ticket Hypothesis suggests that a large randomly initialised network contains a smaller subnetwork that can achieve comparable performance when trained appropriately.

 **Why might starting with a large network be advantageous compared with simply training a small network from scratch?**

 **Answer:**\
 The large network provides many possible subnetworks. Some of these may have particularly favourable initialisations — the “winning tickets.” Pruning attempts to identify these useful subnetworks. not need gradients or optimiser states, reducing training memory and potentially training computation. During trainable modules are inserted into the network, often using a bottleneck architecture. Only of parameters, but we can remove or's questions start simple enough for students to follow by audio, but gradually become A small network trained from scratch may not contain the same advantageous initialisation.

---

 **Q7. Layer freezing**

 A student says:

 > “If I freeze 90% of the layers, inference will require 90% less computation.”

 Is this correct?

 **Answer:**\
 No. Layer freezing primarily reduces the **cost of training**, not inference. Frozen parameters do not need gradients or optimiser states, reducing training memory and potentially training computation. During inference, the frozen layers still perform their normal forward computations.

 This is a very good exam distinction.

---

 **Q8. Adapter modules**

 Why are adapter modules considered parameter-efficient?

 **Answer:**\
 The original model parameters remain frozen. Small trainable modules are inserted into the network, often using a bottleneck architecture. Only these relatively small modules need to be trained and stored for a new task.

 The trade-off is that adapters modify the architecture and can introduce some inference latency.

---

 **Q9. LoRA**

 Suppose a pretrained layer has a weight matrix

 $$
W\in\mathbb{R}^{d\times d}.
$$

 Full fine-tuning learns a complete update

 $$
\Delta W\in\mathbb{R}^{d\times d}.
$$

 LoRA instead represents the update as

 $$
\Delta W \approx BA
$$

 where

 $$
A\in\mathbb{R}^{r\times d},\qquad
B\in\mathbb{R}^{d\times r},
$$

 with $r\ll d$.

 **Why does this reduce the number of trainable parameters?**

 **Answer:**

 Full fine-tuning requires

 $$
d^2
$$

 parameters for the update.

 LoRA requires

 $$
rd+dr=2rd.
$$

 Therefore, the ratio is

 $$
\frac{2rd}{d^2}=\frac{2r}{d}.
$$

 When $r\ll d$, this is much smaller than 1.

 For example, if $d=10,000$ and $r=10$:

 $$
d^2=100,000,000
$$

 whereas

 $$
2rd=200,000.
$$

 So LoRA trains only **0.2%** as many parameters for that matrix.

---

 **Q10. Why LoRA rather than low-rank factorisation of the weights?**

 The lecture makes an important distinction:

 > The weights themselves do not necessarily have low rank, but their **fine-tuning updates** can have low intrinsic dimension.

 Explain why this makes LoRA more successful than simply applying SVD compression to the pretrained weights.

 **Answer:**\
 Direct SVD compression assumes that the pretrained weight matrix itself can be accurately approximated by a low-rank matrix. In practice, transformer weights can be sensitive to this approximation and performance can degrade quickly.

 LoRA makes a different assumption: the **change required to adapt the pretrained model to a new task** lies in a low-dimensional subspace. Therefore, the original weights remain intact while only the low-rank adaptation is learned.

 This is one of the **most important conceptual questions in the lecture**.

---

 ### Higher-level synthesis questions

 These are the questions I would particularly expect in a demanding exam because they require students to connect multiple parts of the lecture.

 **Q11. You have a huge pretrained LLM and need to adapt it to 20 different tasks. You have limited GPU memory. Design a strategy using techniques from the lecture. Justify each component.**

 **Answer:**\
 A sensible solution would be:

 1. **Quantise** the base model to reduce its memory footprint.
2. Keep the base model **frozen**.
3. Use **LoRA** to learn a small adaptation for each task.
4. Store a separate pair of low-rank matrices $A,B$ for each task.
5. At inference, either dynamically attach the appropriate LoRA module or merge it into the base weights when only one task is required.

 This combines the advantages of quantisation and parameter-efficient fine-tuning. **QLoRA** is the natural extension discussed in the lecture.

---

 **Q12. Compare the following techniques: quantisation, pruning, adapters and LoRA.**

 A strong answer should discuss:

 | Method | What changes? | Main saving | Training required? | Inference effect |
| --- | --- | --- | --- | --- |
| Quantisation | Numerical precision | Memory/computation | PTQ: no; QAT: yes | Can reduce memory/time |
| Pruning | Number of active parameters | Parameters/memory/computation | Often fine-tuning | Can reduce inference cost |
| Adapters | Adds small trainable modules | Training parameters | Yes | Some extra latency |
| LoRA | Low-rank weight updates | Training parameters | Yes | Can be merged → essentially no extra inference cost |

---

 **Q13. Explain the following apparent contradiction:**

 > “A model has billions of parameters, but we can remove or avoid training most of them without losing much performance.”

 **Answer:**\
 The lecture's underlying explanation is that the effective/intrinsic dimensionality of the problem can be much smaller than the nominal parameter dimensionality.

 The model may be **overparameterised**. Many parameters can be redundant, and the changes required for a new task may occupy only a small subspace.

 This motivates several techniques:

 $$
\boxed{
\text{Pruning}
\rightarrow
\text{remove redundancy}
}
$$

 $$
\boxed{
\text{Quantisation}
\rightarrow
\text{represent parameters more efficiently}
}
$$

 $$
\boxed{
\text{LoRA}
\rightarrow
\text{represent adaptation in a low-dimensional space}
}
$$

---

 ## The key exam distinction

 I would therefore **not replace the original question-focused teaching script**. I would extend it.

 The teaching questions should lead students from:

 **“What is it?” → “Why does it work?” → “What does it save?” → “What are its limitations?” → “When would I use it?”**

 Then the exam questions should add:

 **“Given a situation, which method should you choose and why?”**

 **“Compare two methods under specific constraints.”**

 **“Calculate the parameter/memory reduction.”**

 **“Explain an apparent contradiction.”**

 **“Predict what happens when a method is applied.”**

 **“Combine several methods into a compression/PEFT strategy.”**

 That progression is much closer to how I would structure an **exam-ready lecture script**: the teacher's questions start simple enough for students to follow by audio, but gradually become analytical enough that students practise the reasoning expected in an exam.
