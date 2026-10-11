# LoRA: Simple MCQs with Answers and Explanations

These 12 multiple-choice questions explain what LoRA is, what problem it solves, how it works, and when it is or is not appropriate. They progress from basic concepts to practical applications.

## Question 1 — What is LoRA?

What does LoRA stand for?

- A. Local Regression Algorithm
- B. Low-Rank Adaptation
- C. Long-Range Architecture
- D. Learned Output Representation

Correct answer: B. Low-Rank Adaptation

Explanation: LoRA is a parameter-efficient fine-tuning method. It allows a pretrained neural network to adapt to a new task by learning a relatively small number of additional parameters rather than updating all the original weights.

## Question 2 — Is LoRA an architecture or an algorithm?

Which description best characterizes LoRA?

- A. It is a completely new neural network architecture that replaces Transformers.
- B. It is a parameter-efficient fine-tuning technique that modifies how a pretrained model is adapted.
- C. It is an optimization algorithm that replaces gradient descent.
- D. It is a dataset preprocessing method.

Correct answer: B

Explanation: LoRA is primarily a parameter-efficient fine-tuning (PEFT) method. It changes which parameters are learned during adaptation and how the weight updates are represented.

- Architecture: The structure of a model, such as a Transformer, CNN, or LSTM.
- Optimization algorithm: A method for updating parameters, such as SGD or Adam.
- LoRA: A method for parameterizing trainable updates to an existing model.

LoRA is often used with Transformers, but it can also be applied to suitable linear layers in other neural network architectures.

## Question 3 — What problem does LoRA solve?

Suppose a pretrained language model contains 7 billion parameters. You want to adapt it to a specialized task.

Why might full fine-tuning be expensive?

- A. It requires rebuilding the model's architecture from scratch.
- B. It requires updating all the model's parameters, which can require substantial memory and computation.
- C. It cannot use gradient descent.
- D. It always reduces the model's accuracy.

Correct answer: B

Explanation: Full fine-tuning typically computes gradients and updates the pretrained model's trainable weights. For a very large model, storing gradients and optimizer states can be expensive.

LoRA reduces the number of trainable parameters by freezing the original weights and learning smaller low-rank updates.

Importantly, LoRA does not eliminate all the costs of training or inference. The pretrained model must still be loaded, and forward and backward computations are still required.

## Question 4 — How does LoRA work?

Suppose the original weight matrix is $W$. LoRA represents the adapted weight as

$$
W'=W+BA.
$$

What do $A$ and $B$ represent?

- A. Two copies of the original pretrained matrix.
- B. Two small trainable matrices whose product represents the weight update.
- C. Two datasets used during training.
- D. Two activation functions.

Correct answer: B

Explanation: In LoRA, the original matrix $W$ is frozen. The trainable matrices are

$$
A\in\mathbb{R}^{r\times d},
\qquad
B\in\mathbb{R}^{d\times r}.
$$

Their product has shape

$$
BA\in\mathbb{R}^{d\times d}.
$$

Instead of learning all $d^2$ entries of an unrestricted update matrix, LoRA learns only $2rd$ entries.

Deeply Learning 4: LoRA (Low Rank Adaptation) - Math, Intuition and Implementation from Scratch | Deeply Learning

Conceptual illustration: the original weights are frozen while the smaller matrices learn the task-specific update.

## Question 5 — Why is LoRA parameter-efficient?

A weight matrix has dimensions $4096\times4096$. LoRA uses rank $r=8$.

How many trainable parameters does LoRA introduce for this matrix, ignoring biases and other layers?

- A. 16,777,216
- B. 32,768
- C. 65,536
- D. 8,192

Correct answer: C. 65,536

Explanation:

Full matrix:

$$
N_{\text{full}}=4096^2=16,777,216.
$$

LoRA factors:

$$
N_{\text{LoRA}}=rd+dr=2rd.
$$

Therefore,

$$
N_{\text{LoRA}}=2(8)(4096)
=\boxed{65,536}.
$$

That is approximately 0.39% of the original matrix's parameter count. The original matrix is still present, but its weights are not updated.

## Question 6 — Does LoRA train the original model weights?

During LoRA fine-tuning, what normally happens to the original pretrained matrix $W$?

- A. Every entry of $W$ is updated normally.
- B. $W$ is frozen, while the LoRA matrices are trained.
- C. $W$ is deleted and replaced by $BA$.
- D. $W$ is randomly initialized before every training step.

Correct answer: B

Explanation: The usual LoRA setup freezes the original pretrained weights and learns the low-rank update.

The effective weight is

$$
W'=W+BA.
$$

The pretrained knowledge remains in $W$, while the trainable update $BA$ allows the model to adapt.

Some training setups also unfreeze selected components, such as a classification head or biases. Such configurations are LoRA-based fine-tuning, but they train more than just the LoRA matrices.

## Question 7 — When can LoRA be used?

Which is the most appropriate example of using LoRA?

- A. Adapting a pretrained language model to answer questions about a company's internal documents.
- B. Sorting a list of integers without using a pretrained model.
- C. Calculating the average of five numbers.
- D. Compressing an image file without training a neural network.

Correct answer: A

Explanation: LoRA is especially useful when a pretrained model already has relevant general knowledge but needs to adapt to a specialized task, domain, style, or dataset.

Examples include:

- Adapting an LLM to customer support or technical documentation.
- Fine-tuning a pretrained vision model for a particular image classification task.
- Adapting a text-to-image diffusion model to a specific artistic style or subject.
- Specializing a pretrained speech model for a particular domain.

LoRA is not a general replacement for ordinary programming, data processing, or training a model from scratch.

## Question 8 — When is LoRA not the right choice?

You need to train a neural network for a completely new problem. There is no suitable pretrained model, and you have a large labeled dataset and sufficient computing resources.

Which statement is most accurate?

- A. LoRA must always be used because it is more modern.
- B. LoRA cannot ever be used in this situation.
- C. Training from scratch or using another training strategy may be more appropriate because LoRA is designed primarily to adapt existing pretrained weights.
- D. LoRA automatically creates the required pretrained knowledge.

Correct answer: C

Explanation: LoRA is most useful when there is a suitable pretrained model to adapt. If no suitable pretrained model exists, training from scratch or pretraining a model may be necessary.

LoRA can sometimes be applied during broader training workflows, but it does not magically replace the need to learn knowledge that is absent from the starting model.

## Question 9 — Does LoRA reduce the model's inference size automatically?

After fine-tuning a model using LoRA, what is generally true about its size?

- A. The entire pretrained model disappears, leaving only the LoRA matrices.
- B. The original model remains necessary, and the LoRA adapter adds relatively little storage.
- C. The model always becomes smaller than the pretrained model.
- D. The number of parameters in the pretrained model becomes zero.

Correct answer: B

Explanation: A LoRA adapter is usually much smaller than the base model, making it convenient to store multiple task-specific adaptations.

For example, one pretrained model can be paired with separate adapters for customer support, legal documents, and technical writing.

The base model is still required. However, the LoRA update can often be merged into the original weights for inference:

$$
W'=W+BA.
$$

After merging, a compatible implementation can run the adapted model without separately applying the adapter at every layer. Merging does not remove the underlying base model's parameter count.

## Question 10 — Can LoRA make a model learn anything?

A pretrained model is adapted using LoRA. Which statement is most accurate?

- A. LoRA guarantees that the model will learn any new task perfectly.
- B. LoRA always improves performance compared with full fine-tuning.
- C. LoRA provides a restricted trainable update, and its success depends on the task, the pretrained model, the data, and the chosen rank.
- D. LoRA eliminates the need for training data.

Correct answer: C

Explanation: LoRA restricts the update to a low-rank form. This can work very well, but a rank that is too small or a poorly chosen adaptation target can limit performance.

LoRA does not guarantee better accuracy, prevent all overfitting, or eliminate the need for suitable training data.

## Question 11 — Which LoRA rank should you choose?

For a square matrix of dimension $d$, the number of trainable LoRA parameters is $2rd$.

What generally happens when you increase the rank $r$?

- A. The number of trainable parameters decreases.
- B. The number of trainable parameters increases, and the update can represent a broader range of changes.
- C. The pretrained matrix is automatically converted into a rank-$r$ matrix.
- D. The number of trainable parameters stays constant.

Correct answer: B

Explanation: The LoRA parameter count is

$$
N_{\text{LoRA}}=2rd.
$$

It increases linearly with $r$. A higher rank gives the update more representational capacity, but it also increases the number of trainable parameters and training costs.

A higher rank is not automatically better. The appropriate rank depends on the task, the model, and the available resources.

## Question 12 — LoRA versus full fine-tuning

Which statement best summarizes the trade-off between LoRA and full fine-tuning?

- A. LoRA always achieves the same accuracy while using no memory.
- B. Full fine-tuning never works with pretrained models.
- C. LoRA usually reduces the trainable parameter count and optimizer-state requirements, while full fine-tuning offers more freedom to update the model.
- D. LoRA changes only the dataset, not the model.

Correct answer: C

Explanation: Full fine-tuning can update all model weights, giving it greater flexibility. LoRA limits the trainable update to a low-rank parameterization, which can make adaptation considerably cheaper.

The best choice depends on the task, compute budget, model, and required performance.

## Final summary: When should you use LoRA?

| Situation | Is LoRA a good candidate? | Reason |
| --- | --- | --- |
| Adapt a pretrained LLM to a specialized domain | Yes | Reuses pretrained knowledge with fewer trainable parameters |
| Create multiple task-specific versions of one large model | Yes | Adapters can be stored separately |
| Fine-tune a pretrained image-generation model | Often | Low-rank updates can adapt style or subject |
| Train a model entirely from scratch | Usually not the first choice | LoRA is primarily intended for adaptation of pretrained weights |
| Solve a simple deterministic calculation | No | Ordinary code is generally more appropriate |
| Update every pretrained weight for maximum flexibility | Not necessarily | Full fine-tuning may be preferable |
| Work with very limited GPU memory | Often useful | Fewer trainable parameters and optimizer states, although the base model still needs to fit or be handled using memory-saving techniques |

The key concept to remember: LoRA is a parameter-efficient way to adapt an existing model. It freezes most or all of the pretrained weights and learns a small, low-rank update instead of learning a full weight update. Its primary advantage is reducing the cost of fine-tuning, not eliminating the need for a pretrained model, data, or computation.
