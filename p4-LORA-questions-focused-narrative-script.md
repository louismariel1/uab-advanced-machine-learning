# Advanced Machine Learning – Practical 4: LoRA and Efficient Learning

## Question-and-Answer Focused Narrative Script

Narrator's script for audio delivery

Teaching approach: This script is designed for students who can hear the narrator but cannot see the screen. Therefore, all explanations are self-contained. Questions are spoken clearly, pauses are provided for students to think, and references to notebook sections, code, plots, and experimental outputs are explained verbally.

The script follows the practical notebook in sequence, guiding students through the theoretical concepts, implementation of LoRA, fine-tuning experiments, performance comparisons, experimental analysis, and assessment requirements.

# Part 1: Introduction to LoRA and Parameter-Efficient Learning

### Question 1: What is fine-tuning, and why do we need it?

Welcome to Practical 4 of Advanced Machine Learning.

In this practical, we are going to explore an important technique in modern deep learning called Low-Rank Adaptation, commonly known as LoRA.

Let's begin with a question.

Imagine that we have already trained a neural network on a large dataset.

The network has learned to recognise useful patterns and features from images.

Now, suppose we want to use this network for a slightly different task, involving images from a new dataset or a different visual domain.

Should we train an entirely new neural network from scratch?

Pause for a moment and think about your answer.

Usually, the answer is no.

We can often reuse the knowledge learned by the original model and adapt it to the new task.

This process is called fine-tuning.

Fine-tuning starts with a pretrained model and continues training it using data from a new target task or domain.

For example, imagine that a neural network was originally trained to recognise objects in everyday photographs.

We might want to adapt it to recognise a different collection of objects, perhaps containing images with different backgrounds, lighting conditions, or visual characteristics.

Instead of learning everything again, we reuse the pretrained model's existing knowledge.

This can save training time and data.

However, conventional fine-tuning has an important disadvantage.

It may require updating a very large number of model parameters.

This brings us to the main problem investigated in this practical.

### Question 2: What is parameter-efficient fine-tuning?

Suppose our pretrained neural network contains one million parameters.

During conventional fine-tuning, we might update a large proportion of those parameters using the new training dataset.

What if we could adapt the model by updating only a small fraction of its parameters?

Would that be useful?

Absolutely.

This is the motivation behind parameter-efficient fine-tuning, often abbreviated as PEFT.

Parameter-efficient fine-tuning describes techniques that adapt pretrained models while updating substantially fewer parameters than conventional fine-tuning.

There are several advantages.

First, fewer parameters need to be updated during training.

Second, the optimiser needs to maintain less training state.

Third, the additional parameters required for adaptation can be relatively small.

Fourth, it becomes easier to maintain multiple task-specific adaptations of the same pretrained model.

One important parameter-efficient fine-tuning technique is LoRA.

### Question 3: What is LoRA?

LoRA stands for Low-Rank Adaptation.

It is a technique that allows us to adapt a pretrained neural network without directly updating all its original weight matrices.

Instead, LoRA introduces additional trainable matrices alongside selected pretrained layers.

The original weights remain frozen.

Only the newly introduced low-rank matrices are trained.

Let's understand the intuition.

Imagine that a pretrained layer contains a large weight matrix.

Conventional fine-tuning changes this entire matrix.

LoRA instead learns a relatively small update that can be added to the original matrix.

The update is represented using two smaller matrices.

This allows us to modify the behaviour of the original layer without training its entire weight matrix.

The central idea is simple:

Reuse the pretrained knowledge, freeze the original weights, and learn a much smaller adaptation.

Keep this principle in mind throughout the practical.

# Part 2: Understanding the Practical Objective and Dataset

### Question 4: What are we trying to achieve in this practical?

The main objective is to compare conventional dense fine-tuning with LoRA-based fine-tuning.

We begin with a pretrained convolutional neural network.

We then adapt it to a target dataset using two different approaches.

The first approach is dense fine-tuning.

In this approach, the selected fully connected layers are updated directly.

The second approach is LoRA fine-tuning.

Here, the original fully connected weights are frozen, and low-rank adapter matrices are introduced.

We then compare the two approaches.

In particular, we investigate:

- Validation accuracy.
- Number of trainable parameters.
- Model state dictionary size.
- Optimiser state size.
- Training computation time.
- The effect of changing the LoRA rank.

Why are we investigating all these measurements?

Because an efficient learning technique should not be judged exclusively by accuracy.

We also need to understand the computational resources required to achieve that accuracy.

Our objective is to investigate the balance between predictive performance and computational efficiency.

### Question 5: What is the difference between the source domain and the target domain?

An important concept in this practical is domain adaptation.

Let's imagine that our original model was trained using images from one dataset.

This is called the source domain.

We then want to adapt the model to images from a different dataset.

This is called the target domain.

Why might the two domains be different?

They may contain different image categories, different backgrounds, different lighting conditions, or different visual characteristics.

Even when both datasets contain images, the underlying data distributions may differ.

A model that performs well on the source domain may not necessarily perform equally well on the target domain.

This is why fine-tuning is useful.

We reuse the knowledge learned from the source domain and adapt the model to the target domain.

### Question 6: Why do we need a dense fine-tuning baseline?

Before evaluating LoRA, we need a reference point.

This is called the baseline model.

In this practical, the baseline is a model trained using conventional dense fine-tuning.

Why is this important?

Suppose LoRA achieves eighty per cent validation accuracy.

Is that good?

We cannot answer confidently without knowing how well the dense fine-tuning model performs under comparable conditions.

If dense fine-tuning achieves eighty-two per cent, LoRA is slightly less accurate.

If dense fine-tuning achieves seventy-five per cent, LoRA has actually improved performance.

Therefore, the baseline provides a reference for evaluating LoRA.

Throughout the practical, we aim to keep the training conditions as consistent as possible.

This includes the target-domain dataset, training epochs, learning rate, and regularisation settings.

This makes the comparison more meaningful.

# Part 3: Understanding the Neural Network Architecture

### Question 7: What architecture are we adapting?

The practical uses a convolutional neural network called `ConvMLP`.

The model contains convolutional layers followed by fully connected layers.

Let's understand the purpose of these components.

The convolutional layers are responsible for extracting visual features from images.

Early convolutional layers may learn relatively simple patterns, such as edges, textures, and colour transitions.

Deeper layers can learn more complex visual patterns and combinations of features.

After feature extraction, the fully connected layers process the extracted features and produce the final classification predictions.

In this practical, the fully connected layers are particularly important because they are the layers we adapt using LoRA.

The model contains four fully connected layers:

- FC1.
- FC2.
- FC3.
- FC4.

These layers are the targets for our LoRA implementation.

### Question 8: Why do we freeze some convolutional layers?

An important part of the practical is that the convolutional layers remain frozen during the adaptation experiments.

What does freezing a layer mean?

It means that its parameters are not updated during training.

The layer still participates in the forward pass, but its weights remain unchanged.

Why would we do this?

The pretrained convolutional layers already contain useful visual representations.

We want to reuse these representations rather than relearn them from the target dataset.

This reduces the number of parameters that need to be trained.

It also allows us to concentrate the adaptation on the fully connected layers.

Remember an important distinction:

A frozen layer is not removed from the network.

It still processes the input and contributes to the predictions.

It simply does not receive parameter updates.

# Part 4: Understanding Conventional Dense Fine-Tuning

### Question 9: What happens during dense fine-tuning?

Before implementing LoRA, we need to understand the conventional approach.

In dense fine-tuning, the selected fully connected layers retain their original weight matrices.

These matrices are allowed to change during training.

Suppose a fully connected layer has a weight matrix called W.

During dense fine-tuning, the training algorithm calculates gradients with respect to W.

The optimiser then uses those gradients to update the weight matrix.

Mathematically, a simplified gradient-based update can be written as:

$$
W_{\text{new}}=W_{\text{old}}-\eta\nabla_W L
$$

Here:

- W represents the original weight matrix.
- Eta represents the learning rate.
- The gradient represents how the loss changes with respect to the weights.
- L represents the training loss.

What does this equation tell us?

The original weight matrix is directly modified to reduce the training loss.

This is the conventional approach to fine-tuning.

### Question 10: What are the advantages and disadvantages of dense fine-tuning?

The main advantage is flexibility.

The model can directly modify all the selected weight parameters to adapt to the target domain.

This gives it considerable freedom to learn the changes required by the new task.

However, there are disadvantages.

First, many parameters may need to be updated.

Second, the optimiser must maintain additional information for those trainable parameters.

Third, the backward pass must calculate gradients for the trainable weights.

Fourth, maintaining separate fully fine-tuned models for multiple tasks can require substantial storage.

These limitations motivate the use of LoRA.

# Part 5: Understanding the Mathematics of LoRA

### Question 11: What is the main mathematical idea behind LoRA?

We have established that dense fine-tuning directly modifies a large weight matrix.

LoRA takes a different approach.

Instead of learning an unrestricted update to the original matrix, it represents the update as the product of two smaller matrices.

Let's denote the original pretrained weight matrix by W.

LoRA introduces two additional matrices, A and B.

The adapted weight matrix can be expressed as:

$$
W'=W+\Delta W
$$

where:

$$
\Delta W=BA
$$

Therefore:

$$
W'=W+BA
$$

This is the central mathematical idea behind LoRA.

The original matrix W remains frozen.

The new matrices A and B are trainable.

Instead of learning a complete weight update, we learn a structured low-rank update.

### Question 12: Why do we use two smaller matrices instead of one large matrix?

Let's consider a weight matrix with one thousand input dimensions and one thousand output dimensions.

A full weight matrix contains:

$$
1000\times1000=1,000,000
$$

parameters.

Now suppose we choose a LoRA rank of eight.

The first adapter matrix contains:

$$
8\times1000=8,000
$$

parameters.

The second adapter matrix also contains:

$$
1000\times8=8,000
$$

parameters.

Together, the two matrices contain:

$$
8,000+8,000=16,000
$$

parameters.

Compare this with the original one-million-parameter matrix.

Instead of training one million parameters, we train only sixteen thousand adapter parameters.

That is a substantial reduction.

The example is illustrative, but it demonstrates why low-rank adaptation can be so effective.

### Question 13: What does the rank parameter r represent?

The parameter r is called the LoRA rank.

It controls the dimensions of the two adapter matrices.

For a weight matrix with dimensions:

$$
W\in\mathbb{R}^{d_{\text{out}}\times d_{\text{in}}}
$$

the adapter matrices have dimensions:

$$
A\in\mathbb{R}^{r\times d_{\text{in}}}
$$

and:

$$
B\in\mathbb{R}^{d_{\text{out}}\times r}
$$

The total number of trainable adapter parameters is therefore:

$$
r(d_{\text{in}}+d_{\text{out}})
$$

Compare this with the original matrix, which contains:

$$
d_{\text{in}}d_{\text{out}}
$$

parameters.

What happens when we increase r?

The adapter matrices become larger.

This increases the number of trainable parameters and the flexibility of the adaptation.

What happens when we decrease r?

The adapter matrices become smaller.

This improves parameter efficiency, but it may restrict how much the model can adapt.

Therefore, rank controls an important trade-off between adaptation capacity and parameter efficiency.

### Question 14: What does low-rank mean?

The term low-rank refers to the mathematical rank of the update matrix.

Because the update is formed by multiplying matrices A and B through an intermediate dimension r, its rank cannot exceed r.

In other words:

$$
\operatorname{rank}(BA)\leq r
$$

This means that the update is constrained to a lower-dimensional structure.

Why might this work?

The pretrained model already contains useful knowledge.

We may not need to change every component of its original weight matrices to adapt to a new task.

A relatively small, structured update may be sufficient.

This is the fundamental assumption behind LoRA.

# Part 6: Understanding the LoRA Forward Pass

### Question 15: How does LoRA change the forward pass?

Now let's connect the mathematics to the actual neural network computation.

In a conventional fully connected layer, the forward pass can be written as:

$$
y=Wx+b
$$

Here:

- x is the input.
- W is the weight matrix.
- b is the bias.
- y is the output.

With LoRA, the adapted layer produces:

$$
y=Wx+BAx+b
$$

We can rearrange this as:

$$
y=Wx+B(Ax)+b
$$

This rearrangement is important.

Instead of explicitly constructing the complete updated weight matrix, we can calculate the low-rank contribution using two smaller matrix operations.

First, we calculate:

$$
h=Ax
$$

Then:

$$
\Delta y=Bh
$$

Finally, we add the adapter contribution to the original output.

Therefore:

$$
y=Wx+\Delta y+b
$$

What is the benefit?

We retain the original layer's functionality while introducing a trainable correction.

The original weight matrix remains frozen, while A and B are updated during training.

### Question 16: Why should the initial LoRA model behave similarly to the original model?

A useful implementation detail is how the adapter matrices are initialised.

A common LoRA initialisation strategy sets one adapter matrix to zero and initialises the other using a suitable random distribution.

For example, if B is initially zero:

$$
BA=0
$$

Therefore, the initial adapted weight matrix becomes:

$$
W'=W+0
$$

Which means:

$$
W'=W
$$

What does this tell us?

At the beginning of training, the LoRA layer initially behaves like the original pretrained layer.

The adapter contribution is initially zero.

As training progresses, the adapter matrices learn a useful update.

This allows LoRA to begin from the pretrained model's existing behaviour rather than introducing an arbitrary initial change.

### Question 17: What is the role of the LoraLayer class?

In the notebook, the LoRA implementation is encapsulated inside a class called `LoraLayer`.

The purpose of this class is to wrap an existing fully connected layer.

It retains the original layer and introduces the two trainable adapter matrices.

The class is responsible for several operations.

First, it stores the original layer.

Second, it freezes the original layer's weights.

Third, it creates the A and B adapter matrices using the specified rank.

Fourth, it implements the adapted forward pass.

The forward pass calculates the original layer's output and adds the adapter contribution.

Why is this class useful?

Because we can reuse the same implementation for multiple fully connected layers.

We do not need to write a separate LoRA implementation for FC1, FC2, FC3, and FC4.

We simply wrap each original layer with the same `LoraLayer` class.

# Part 7: Implementing LoRA in the Practical

### Question 18: Which layers do we replace with LoRA adapters?

We now reach the implementation section of the practical.

The notebook replaces the four fully connected layers with LoRA-wrapped versions.

These are:

- FC1.
- FC2.
- FC3.
- FC4.

The initial rank is set to eight.

The practical uses the following conceptual operation:

Take the original FC1 layer and wrap it with a LoRA adapter of rank eight.

Repeat this operation for FC2, FC3, and FC4.

The result is a model in which all four fully connected layers contain trainable LoRA adapters.

What happens to the original weight matrices?

They remain in the model, but their gradients are disabled.

### Question 19: How do we verify that the original weights are frozen?

The notebook includes an assertion to check that the original weight matrix of FC1 does not require gradients.

In PyTorch, this property is called:

`requires_grad`

If it is set to false, the parameter will not receive gradient updates.

Why is this verification important?

Because the main objective of LoRA is to avoid updating the original full-rank weights.

If these weights accidentally remain trainable, our implementation is no longer performing the intended parameter-efficient adaptation.

The practical also checks the trainable parameters by inspecting the model's named parameters.

What should we expect to see?

We should see:

- The A and B matrices of FC1.
- The A and B matrices of FC2.
- The A and B matrices of FC3.
- The A and B matrices of FC4.
- The classifier's weight and bias, where applicable.

We should not see the original frozen weight matrices among the trainable parameters.

This is an important sanity check.

### Question 20: How do we measure the number of trainable parameters?

The notebook uses a function called `num_parameters`.

This function calculates two quantities.

The first is the total number of parameters in the model.

The second is the number of parameters that are currently trainable.

Why do we need both?

The total number of parameters tells us how large the model is.

The trainable parameter count tells us how many parameters are being updated during fine-tuning.

These are not necessarily the same.

For LoRA, the original pretrained weights are still present.

However, they are frozen.

Therefore, the total model parameter count can remain similar to the original model, while the trainable parameter count becomes much smaller.

The practical also calculates the proportion of trainable parameters.

This is:

$$
\text{Trainable fraction}
=
\frac{\text{Trainable parameters}}
{\text{Total parameters}}
$$

The smaller this value, the smaller the proportion of the model being updated.

### Question 21: Why do we compare LoRA with dense fine-tuning?

We now have two models.

The dense fine-tuning model updates the selected original fully connected weights.

The LoRA model freezes those original weights and updates the adapter matrices instead.

Both models are adapted to the same target domain.

We now compare their validation accuracy.

Why?

Because parameter efficiency is only useful if the adapted model still performs well.

A LoRA model that uses very few parameters but performs extremely poorly would not be a particularly useful solution.

The ideal outcome is for LoRA to approach the validation accuracy of dense fine-tuning while using substantially fewer trainable parameters.

The practical trains the LoRA model using the same general training settings as the baseline.

After training, it compares the best validation accuracy of both models.

The difference is calculated as:

$$
\text{Accuracy difference}
=
\text{LoRA accuracy}
-
\text{Dense accuracy}
$$

A positive value means LoRA performed better.

A negative value means LoRA performed worse.

This completes the initial LoRA implementation and comparison.

# Part 8: Experiment 1 – Investigating Different LoRA Ranks

### Question 22: Why do we experiment with different LoRA ranks?

So far, we have used a rank of eight.

But is rank eight necessarily the best choice?

Not necessarily.

Remember that the rank controls the dimensions of the adapter matrices.

A smaller rank provides fewer trainable parameters.

A larger rank provides more adaptation capacity.

We therefore need to investigate how rank affects performance.

The practical tests five different values:

- Rank two.
- Rank four.
- Rank eight.
- Rank sixteen.
- Rank thirty-two.

Why do we use several values?

Because we want to understand how the amount of trainable adapter capacity affects target-domain accuracy.

A low rank may be sufficient for a relatively simple adaptation.

A more challenging target domain may require a higher rank.

However, increasing the rank also increases the number of trainable parameters.

The best rank is therefore not necessarily the largest one.

### Question 23: Why must we create a new model for each rank?

An important experimental design decision is that every rank experiment starts with a fresh model.

Why?

Imagine that we train a rank-two model and then increase its rank to eight.

If we continue training the same model, the rank-eight experiment may inherit information from the previous experiment.

This would make the comparison less controlled.

Instead, the practical creates a new `ConvMLP` model for each rank.

It then loads the same pretrained source-domain weights.

The convolutional layers are frozen again.

The fully connected layers are replaced with LoRA adapters using the selected rank.

This ensures that each experiment starts from the same pretrained foundation.

### Question 24: Why do we keep the other hyperparameters unchanged?

When comparing different ranks, we want rank to be the main variable.

Therefore, the practical keeps the following settings consistent:

- The source-domain pretrained weights.
- The target-domain training dataset.
- The target-domain validation dataset.
- The number of training epochs.
- The learning rate.
- The weight decay.
- The frozen convolutional layers.

Why is this important?

Suppose rank sixteen achieves higher accuracy than rank eight.

If we also changed the learning rate and number of epochs, we would not know whether the improvement was caused by rank or by the other changes.

Keeping the settings consistent makes the experiment more reliable.

This is an example of a controlled experiment.

### Question 25: How does the rank-sweep training loop work?

The notebook uses a loop to train the five LoRA configurations.

For each rank, the following steps are performed.

First, the random seed is reset to zero to improve consistency.

Second, a fresh model is created.

Third, the pretrained weights are loaded.

Fourth, the convolutional layers are frozen.

Fifth, the four fully connected layers are replaced with LoRA adapters using the current rank.

Sixth, the notebook checks that the original fully connected weights are frozen.

Seventh, it counts the total and trainable parameters.

Finally, the model is trained and evaluated.

The best validation accuracy is recorded.

The model and its training metrics are also stored so that we can analyse the results later.

This process is repeated for ranks two, four, eight, sixteen, and thirty-two.

### Question 26: What should we look for in the rank-sweep results?

Once the experiments are complete, the notebook creates a summary table.

The table includes:

- LoRA rank.
- Total model parameters.
- Trainable parameters.
- Trainable parameter fraction.
- Best validation accuracy.

What patterns should we look for?

First, the number of trainable adapter parameters should generally increase as rank increases.

Second, validation accuracy may also increase as rank increases.

However, this is not guaranteed.

A higher rank gives the model more adaptation capacity, but it does not guarantee better generalisation.

It is possible that a smaller rank performs just as well as a larger rank.

It is also possible that a higher rank improves accuracy but provides a smaller efficiency advantage.

### Question 27: How do we identify the best LoRA rank?

The practical identifies the LoRA configuration with the highest validation accuracy.

It then reports:

- The best rank.
- Its validation accuracy.
- Its number of trainable parameters.
- Its difference from the dense baseline.

Suppose rank sixteen produces the highest validation accuracy.

That would make rank sixteen the best-performing configuration among the tested ranks.

However, we should not automatically assume that it is the most efficient configuration.

For example, rank eight might achieve almost identical accuracy while using fewer trainable parameters.

Therefore, we should consider both accuracy and parameter count when choosing a configuration.

The best practical choice depends on the balance between these two objectives.

# Part 9: Interpreting the Rank-Sweep Plots

### Question 28: What does the validation accuracy versus rank plot tell us?

The notebook produces a plot showing LoRA rank on the horizontal axis and validation accuracy on the vertical axis.

Each point represents one of the five LoRA experiments.

A horizontal dashed line represents the dense fine-tuning baseline.

Why is this plot useful?

It allows us to see whether increasing the rank improves target-domain performance.

For example, if accuracy increases from rank two to rank eight but then remains relatively stable, this suggests that rank eight may already provide sufficient adaptation capacity.

If accuracy continues to improve at higher ranks, a larger adapter may be beneficial.

The most important comparison is with the dense baseline.

If a LoRA point is above the baseline line, that configuration achieved higher validation accuracy than dense fine-tuning.

If it is below the line, it achieved lower accuracy.

Remember that a higher validation accuracy does not automatically mean a more efficient model.

We must also consider how many parameters were required.

### Question 29: What does the accuracy versus trainable parameters plot tell us?

The second plot compares validation accuracy with the number of trainable parameters.

The horizontal axis represents trainable parameters.

The vertical axis represents validation accuracy.

The LoRA configurations are shown as individual points, and the dense baseline is shown separately.

What is the purpose of this plot?

It helps us visualise the relationship between model performance and parameter efficiency.

Imagine two LoRA configurations.

The first uses ten thousand trainable parameters and achieves eighty per cent accuracy.

The second uses twenty thousand trainable parameters and achieves eighty-one per cent accuracy.

Is the second configuration necessarily preferable?

Not always.

We need to decide whether the one-percentage-point improvement justifies doubling the number of trainable parameters.

This is the trade-off we are investigating.

The ideal configuration achieves high validation accuracy with relatively few trainable parameters.

### Question 30: Can LoRA outperform dense fine-tuning?

Yes, it is possible.

Although LoRA uses fewer trainable parameters, its performance can sometimes match or exceed dense fine-tuning.

Why might this happen?

The low-rank constraint can act as a form of regularisation.

Instead of allowing unrestricted changes to the original weights, LoRA restricts the adaptation to a lower-dimensional update.

This restriction may reduce the risk of overfitting, particularly when the target dataset is relatively small.

However, this is not guaranteed.

A higher-capacity dense model may still perform better, especially when the target task requires substantial changes to the pretrained representations.

The actual outcome depends on the dataset, architecture, rank, and training conditions.

Therefore, we must use experimental results rather than assume that LoRA will always outperform dense fine-tuning.

# Part 10: Experiment 2 – Comparing Model State Dictionary Size

### Question 31: What is a model state dictionary?

We have now investigated validation accuracy and trainable parameters.

The next question is whether LoRA also reduces the amount of storage required for the model.

To answer this, the practical compares the model state dictionaries.

In PyTorch, a model's `state_dict` contains the tensors representing its parameters and registered buffers.

It is commonly used when saving and loading model checkpoints.

For example, it includes learned weights and biases.

Why do we compare state dictionaries?

Because the model's stored parameters contribute to the amount of disk space required to save the model.

### Question 32: Does LoRA always reduce the complete model state dictionary size?

This is a particularly important question.

You might expect LoRA to produce a much smaller model because it trains fewer parameters.

However, this is not necessarily true.

Why?

Because LoRA does not remove the original pretrained weights.

It freezes them.

The original weights are still required during the forward pass, so they remain part of the model.

LoRA then adds the adapter matrices A and B.

Therefore, the complete LoRA model contains:

- The original pretrained weights.
- The LoRA adapter matrices.
- The other model parameters, including the classifier where applicable.

This means that the complete LoRA state dictionary can actually be slightly larger than the dense model state dictionary.

This is an important distinction.

Fewer trainable parameters do not necessarily mean a smaller complete model.

### Question 33: How do we calculate the model state dictionary size?

The practical defines a function that counts the number of tensor elements in a state dictionary.

It loops through the tensors and adds up their number of elements.

For example, a matrix containing one thousand rows and one thousand columns contains one million scalar values.

If each value uses thirty-two-bit floating-point precision, each value occupies four bytes.

Therefore, the approximate storage requirement is:

$$
\text{Storage}
=
\text{Number of values}\times4
$$

This is how the practical estimates the model storage in bytes.

It then converts the result into more readable units, such as kilobytes or megabytes.

The notebook performs this calculation for both the dense model and the LoRA model.

The results allow us to determine whether the complete LoRA model is smaller or larger.

### Question 34: Why might the LoRA state dictionary be larger?

Let's revisit the main principle.

The original pretrained weights remain stored.

The LoRA adapters are additional parameters.

Therefore, LoRA introduces extra storage rather than replacing the original model weights.

For example, imagine that the original model requires one hundred megabytes.

If the adapters require an additional two megabytes, the complete LoRA model would require approximately one hundred and two megabytes.

This is an illustrative example.

The key point is that the complete model still needs the original pretrained weights.

However, there is another way of using LoRA that can reduce the storage needed for task-specific adaptations.

Instead of saving a complete copy of the model for every task, we can retain one shared pretrained model and save only the adapter matrices for each task.

This is an important practical advantage of LoRA.

We will return to this idea in the optional extension.

# Part 11: Experiment 3 – Comparing Optimiser State Size

### Question 35: What is optimiser state, and why does it matter?

We have seen that LoRA does not necessarily reduce the complete model state dictionary.

However, it can significantly reduce the optimiser state.

To understand why, we first need to understand what an optimiser stores.

The practical uses AdamW.

AdamW is an optimisation algorithm that maintains additional information for trainable parameters.

In particular, it maintains estimates of the first and second moments of the gradients.

These are commonly called:

- First moment estimates.
- Second moment estimates.

The first moment is related to the average gradient.

The second moment is related to the average squared gradient.

These values help AdamW determine appropriate parameter updates.

What is important here?

The optimiser needs to maintain these additional tensors for the parameters it is optimising.

If a parameter is frozen and excluded from the optimiser, it does not need AdamW moment estimates.

### Question 36: Why does LoRA require less optimiser state?

Let's compare the two approaches.

During dense fine-tuning, many original fully connected weights are trainable.

AdamW maintains optimiser state for these parameters.

During LoRA fine-tuning, the original weights are frozen.

Only the adapter matrices and other trainable parameters are included in the optimiser.

Therefore, AdamW maintains state for a much smaller collection of parameters.

This is why LoRA can substantially reduce optimiser memory requirements.

For a simplified Adam-style optimiser using two full-sized moment tensors, the optimiser may need approximately twice as many additional scalar values as there are trainable parameters, excluding smaller auxiliary state.

This explains why reducing trainable parameters can have a significant effect on optimiser storage.

### Question 37: Why do we initialise the optimiser state before measuring it?

There is an important detail in the notebook.

AdamW creates its optimiser state lazily.

This means that the moment tensors are not necessarily created when we first construct the optimiser.

They are usually created after the first optimisation step.

Therefore, if we measure the optimiser state immediately after creating the optimiser, we might obtain a misleading result.

The practical solves this by performing one measurement training step.

It calculates predictions, computes the loss, performs backpropagation, and executes one optimiser step.

This initialises the optimiser state.

The notebook then counts the tensor values stored in the optimiser.

Importantly, this is a measurement operation rather than an additional full training experiment.

### Question 38: What should we expect from the optimiser comparison?

We expect the LoRA optimiser state to be substantially smaller than the dense optimiser state.

Why?

Because LoRA has far fewer trainable parameters.

The original frozen weights do not require optimiser moment estimates.

This demonstrates one of the strongest advantages of LoRA.

Even when the complete model storage remains similar, the additional memory required to optimise the model can be substantially reduced.

This is especially important when training large neural networks or adapting models to multiple tasks.

# Part 12: Experiment 4 – Comparing Training Computation Time

### Question 39: Does using fewer trainable parameters automatically make LoRA faster?

This is another important question.

We have established that LoRA reduces the number of trainable parameters and the optimiser state.

Does this automatically mean that LoRA trains faster?

Not necessarily.

To understand why, we need to consider the complete forward and backward computations.

During LoRA training, the original pretrained layer still participates in the forward pass.

The original matrix multiplication is still required.

In addition, the model must calculate the LoRA adapter contribution.

Therefore, LoRA introduces additional computations.

However, it avoids calculating gradients and performing optimiser updates for the frozen original weights.

This can reduce the backward-pass and optimiser workload.

The final effect on training time depends on which computational costs dominate.

### Question 40: What exactly does the practical measure?

The notebook creates fresh copies of the dense and LoRA models.

It also creates separate AdamW optimisers containing only trainable parameters.

The practical then measures the time required to process fifty training batches.

For each batch, the model performs:

First, a forward pass.

Second, loss calculation.

Third, backpropagation.

Fourth, an optimiser update.

The total time is recorded.

The average time per batch is calculated by dividing the total time by the number of batches.

Why do we use the same number of batches?

To make the timing comparison consistent.

We want to compare the computational cost of processing equivalent training workloads.

### Question 41: Why do we synchronise the GPU before measuring time?

The notebook includes CUDA synchronisation when a GPU is available.

Why is this necessary?

GPU operations are often executed asynchronously.

This means that Python may continue running before the GPU has completed all the requested operations.

If we measure time without synchronisation, we might measure how long it takes to submit the GPU operations rather than how long the computations actually take.

Synchronisation ensures that the GPU has completed its queued operations before the timing measurement is finalised.

This makes the measurement more meaningful.

### Question 42: What should we expect from the training-time comparison?

There is no guaranteed outcome.

LoRA may be faster, slower, or approximately equal in wall-clock training time.

Why?

Because LoRA has two competing computational effects.

On the positive side, fewer parameters need gradients and optimiser updates.

On the negative side, the adapter computations introduce additional operations.

Furthermore, the original frozen network still needs to perform its forward computations.

The result depends on several factors:

- Model architecture.
- LoRA rank.
- Batch size.
- GPU or CPU hardware.
- Memory bandwidth.
- The relative cost of forward and backward operations.

Therefore, we should interpret the measured results rather than assume that fewer parameters always produce faster training.

### Question 43: How do we interpret the training-time reduction?

The practical calculates the difference between the average batch times.

The percentage reduction is:

$$
\text{Time reduction}
=
1-
\frac{t_{\text{LoRA}}}
{t_{\text{Dense}}}
$$

If the result is positive, LoRA required less time per batch.

If the result is negative, LoRA required more time per batch.

The practical also calculates an approximate speed-up:

$$
\text{Speed-up}
=
\frac{t_{\text{Dense}}}
{t_{\text{LoRA}}}
$$

For example, if dense fine-tuning takes zero point one seconds per batch and LoRA takes zero point zero eight seconds, the speed-up is:

$$
\frac{0.1}{0.08}=1.25
$$

This means LoRA is approximately one point two five times as fast for that measurement.

These are illustrative values, not guaranteed experimental results.

The important lesson is that parameter efficiency and computational speed are related, but they are not identical.

# Part 13: Understanding the Overall Experimental Results

### Question 44: How do we bring all the experimental results together?

We have now investigated several aspects of LoRA.

Let's review the four main areas.

First, accuracy.

We compared the validation accuracy of dense fine-tuning and different LoRA ranks.

Second, parameter efficiency.

We measured how many trainable parameters were required by each approach.

Third, storage and memory.

We compared the complete model state dictionary and the optimiser state.

Fourth, computation time.

We measured the average time required to process training batches.

Why is it important to consider all four?

Because a model may perform well in one area but poorly in another.

For example, dense fine-tuning may produce the highest accuracy but require substantially more trainable parameters.

A LoRA configuration may achieve slightly lower accuracy but require much less optimiser state.

Another LoRA configuration might achieve similar accuracy with even fewer parameters.

We therefore need to evaluate the complete trade-off.

### Question 45: What is the difference between total parameters and trainable parameters?

Let's make sure this distinction is clear.

Total parameters represent all the model parameters that are stored in the network.

Trainable parameters represent the subset that can be updated during training.

For dense fine-tuning, the selected fully connected weights are trainable.

For LoRA, those original weights are frozen, and the adapter matrices are trainable.

The original weights still contribute to the total parameter count.

This is why the total parameter count and trainable parameter count tell us different things.

When discussing LoRA, we should be precise.

We are reducing the number of trainable parameters, not necessarily reducing the number of parameters stored in the complete model.

### Question 46: What is the main difference between model storage and optimiser storage?

Model storage represents the parameters and buffers required to save and restore the model.

Optimiser storage represents additional information required by the optimisation algorithm during training.

LoRA may have a similar or slightly larger complete model state dictionary because the original pretrained weights remain stored alongside the adapters.

However, its optimiser state can be substantially smaller because only trainable parameters require optimiser state.

This is one of the most important findings to understand from the practical.

### Question 47: What does the final experimental report tell us?

The final notebook section consolidates the results into a single report.

It presents:

- Dense fine-tuning accuracy.
- The rank-sweep results.
- The best LoRA rank.
- Trainable parameter counts.
- Model state dictionary sizes.
- Optimiser state sizes.
- Training-time measurements.

It then produces an overall interpretation.

The notebook's final conclusion is intended to explain whether LoRA provides a useful alternative to dense fine-tuning for the particular experiment.

However, remember that the actual numerical results must come from running your own notebook.

We should never invent or assume the final validation accuracy, storage reduction, or training-time improvement.

Instead, we use the values produced by the experiments.

# Part 14: Optional Extension – Merging LoRA Weights

### Question 48: Can we remove the additional LoRA computations after training?

We now move to the optional extension exercise.

This exercise explores an important advantage of LoRA.

Recall that during training, the adapted weight matrix is:

$$
W'=W+BA
$$

The original weight matrix is frozen, while A and B are trained.

Once training is complete, do we always need to calculate the original layer output and the adapter output separately?

No.

We can merge the learned adapter update into the original weight matrix.

We calculate:

$$
W_{\text{merged}}=W+BA
$$

The merged matrix now contains the adapted weights.

The original LoRA branch is no longer needed for inference.

### Question 49: What are the advantages of merging LoRA weights?

The main advantage is that we can return to the original layer architecture.

Instead of performing two sets of operations, we use one merged weight matrix.

The forward pass becomes:

$$
y=W_{\text{merged}}x+b
$$

This can remove the additional adapter computations during inference.

It also makes deployment simpler because the resulting model can use the ordinary fully connected layer structure.

An important point is that merging should preserve the adapted model's predictions, subject to normal floating-point numerical differences.

This provides a useful way to separate training efficiency from inference efficiency.

During training, we use adapters.

After training, we can merge them into the original weights.

### Question 50: Can we save only the LoRA adapter matrices?

Yes.

This is another optional extension.

Instead of saving the complete adapted model for every target task, we can save only the A and B matrices.

The original pretrained model can be stored once.

When we want to restore a particular adaptation, we load the original pretrained model and then load the corresponding adapter matrices.

Why is this useful?

Imagine that we have one pretrained model and want to adapt it to five different tasks.

With conventional fine-tuning, we might need to store five complete fine-tuned model checkpoints.

With LoRA, we can store one shared pretrained model and five much smaller adapter checkpoints.

This can substantially reduce the storage required to maintain multiple task-specific adaptations.

### Question 51: Can LoRA reduce GPU memory usage?

Yes, it can reduce certain components of GPU memory usage.

During training, memory is required for several things:

- Model parameters.
- Gradients.
- Optimiser state.
- Intermediate activations.
- Other computational buffers.

LoRA reduces the memory required for gradients and optimiser state associated with trainable weights.

However, the frozen pretrained model still occupies memory.

Its forward-pass activations may also need to be retained where required for gradient computation through earlier trainable operations.

Therefore, LoRA does not eliminate all memory requirements.

The optional exercise asks us to compare the GPU memory allocated to the dense and LoRA models.

The practical also includes a memory-checking function.

This reports allocated memory, reserved memory, and maximum reserved memory.

Why must we be careful when interpreting these measurements?

Because GPU memory is shared by all the objects currently loaded in the session.

To obtain a fair comparison, we should restart the notebook kernel and measure the models separately under consistent conditions.

This helps avoid including unrelated models and tensors in the measurements.

# Part 15: Assessment and Reflection

### Question 52: What do you need to submit for this practical?

We have now completed the main experimental investigation.

Let's review the submission requirements.

First, you need to explain how you implemented LoRA.

Describe how the original fully connected layers were wrapped with LoRA adapters.

Explain the purpose of matrices A and B.

Explain how the forward pass combines the original layer output with the low-rank adapter contribution.

Second, show a comparison between dense fine-tuning and LoRA.

Your report should include training and validation plots where available.

The plots should make it clear how the two approaches performed during training.

Third, report the results of the rank experiments.

You should compare ranks two, four, eight, sixteen, and thirty-two.

Identify the best-performing rank and discuss whether any configuration improved on the dense baseline.

Fourth, compare the model state dictionary sizes.

Explain why the LoRA model may have a slightly larger complete state dictionary.

Fifth, compare the optimiser state dictionary sizes.

Explain why LoRA requires less optimiser state.

Sixth, report the forward and backward computation times.

Discuss whether LoRA was faster or slower in your experiment and explain why.

Finally, describe any implementation problems you encountered and how you resolved them.

### Question 53: What problems might you encounter when implementing LoRA?

There are several possible implementation challenges.

The first is incorrect matrix dimensions.

Remember that the dimensions of A and B must be compatible with the input and output dimensions of the original fully connected layer.

If these dimensions are incorrect, the forward pass will produce a matrix multiplication error.

The second problem is failing to freeze the original weights.

If the original weight matrices remain trainable, the model will update more parameters than intended.

The third problem is incorrect initialisation.

If the adapter contribution is not initialised appropriately, the model's initial predictions may differ unnecessarily from those of the pretrained model.

The fourth problem is incorrect parameter counting.

It is important to distinguish between total parameters and trainable parameters.

The fifth problem is inconsistent experimental conditions.

If we change multiple hyperparameters when comparing ranks, it becomes difficult to interpret the results.

Finally, plotting and measurement functions can introduce their own implementation issues.

For example, the rank-sweep training wrapper in the notebook avoids a plotting problem in the original training function by using a simpler training loop.

The important lesson is that successful experiments require both a correct implementation and careful experimental design.

### Question 54: What are the main insights we should gain from this practical?

Let's summarise the most important ideas.

First, fine-tuning allows us to adapt a pretrained model to a new task or domain.

Second, conventional dense fine-tuning can require updating many parameters.

Third, LoRA introduces two smaller trainable matrices to represent a low-rank update to the original weight matrix.

Fourth, the original pretrained weights remain frozen.

Fifth, the LoRA rank controls the number of adapter parameters and the flexibility of the adaptation.

Sixth, LoRA can substantially reduce the number of trainable parameters.

Seventh, fewer trainable parameters can lead to much smaller optimiser state.

Eighth, the complete model state dictionary may not be smaller because the original pretrained weights remain stored.

Ninth, LoRA does not automatically guarantee faster training because the frozen layers still participate in the forward pass and the adapter computations introduce additional operations.

Finally, LoRA adapters can be merged into the original weights after training, allowing the model to return to an ordinary forward-pass architecture.

The overall objective is to achieve effective model adaptation while reducing the resources required for training.

# Part 16: Final Knowledge Check

Before we finish, let's test your understanding with a series of questions.

Take a moment to think about each question before listening to the answer.

### Question 1: What is the main purpose of fine-tuning?

Pause.

The answer is to adapt a pretrained model to a new task or target domain by continuing training using new data.

### Question 2: What does PEFT stand for?

Pause.

The answer is Parameter-Efficient Fine-Tuning.

It describes techniques that adapt pretrained models while updating substantially fewer parameters than conventional fine-tuning.

### Question 3: What does LoRA stand for?

Pause.

The answer is Low-Rank Adaptation.

### Question 4: What is the main idea behind LoRA?

Pause.

LoRA freezes the original pretrained weight matrices and introduces two smaller trainable matrices, A and B, to represent a low-rank weight update.

### Question 5: How is the adapted weight matrix represented mathematically?

Pause.

The answer is:

$$
W'=W+BA
$$

Here, W is the original frozen weight matrix, while A and B are the trainable adapter matrices.

### Question 6: What does the rank parameter r control?

Pause.

The answer is the intermediate dimension of the adapter matrices.

A higher rank generally introduces more trainable parameters and provides greater adaptation capacity.

A lower rank reduces the number of trainable parameters but may restrict the flexibility of the adaptation.

### Question 7: What happens to the original pretrained weights during LoRA training?

Pause.

They remain frozen.

They still participate in the forward pass, but they are not updated through gradient-based optimisation.

### Question 8: How many fully connected layers are replaced with LoRA adapters in this practical?

Pause.

The answer is four.

They are FC1, FC2, FC3, and FC4.

### Question 9: What is the purpose of the LoRA forward pass?

Pause.

It combines the original layer output with the additional low-rank adapter contribution.

Mathematically, the adapted layer can be expressed as:

$$
y=Wx+BAx+b
$$

### Question 10: Why is one adapter matrix commonly initialised to zero?

Pause.

To ensure that the initial adapter contribution is zero.

This means the LoRA layer initially behaves like the original pretrained layer.

### Question 11: What is the difference between total parameters and trainable parameters?

Pause.

Total parameters include all the model parameters, including frozen weights.

Trainable parameters include only those parameters that can be updated during training.

### Question 12: Which LoRA ranks are tested in this practical?

Pause.

The answer is two, four, eight, sixteen, and thirty-two.

### Question 13: Why do we create a fresh model for each rank experiment?

Pause.

To ensure that each experiment starts from the same pretrained weights and does not inherit training updates from another rank experiment.

This makes the comparison more controlled.

### Question 14: Why do we keep the other training hyperparameters unchanged during the rank sweep?

Pause.

To make LoRA rank the main experimental variable.

This allows us to investigate the effect of rank without introducing unnecessary differences in the training conditions.

### Question 15: Does increasing the LoRA rank always improve validation accuracy?

Pause.

No.

Increasing the rank provides more adaptation capacity, but it does not guarantee better generalisation.

The best rank must be identified experimentally.

### Question 16: Why might LoRA outperform dense fine-tuning?

Pause.

The low-rank constraint can act as a form of regularisation.

It may help prevent overfitting when the target dataset is relatively small.

However, this is not guaranteed.

### Question 17: Does LoRA necessarily reduce the complete model state dictionary size?

Pause.

No.

The original pretrained weights remain stored, and the LoRA adapter matrices are added to the model.

Therefore, the complete LoRA state dictionary can be slightly larger than that of the dense model.

### Question 18: Why does LoRA reduce optimiser state size?

Pause.

Because the optimiser maintains state for trainable parameters.

The original frozen weights do not require AdamW moment estimates.

Therefore, the optimiser only needs to maintain state for the smaller collection of trainable parameters.

### Question 19: Why must we perform an optimiser step before measuring its state size?

Pause.

Because AdamW creates its moment tensors lazily.

The optimiser state may not exist until the first optimisation step has been performed.

### Question 20: Does using fewer trainable parameters automatically make LoRA faster?

Pause.

No.

Although LoRA reduces gradient and optimiser computations, the original frozen weights still participate in the forward pass.

The adapter also introduces additional computations.

Therefore, actual training speed depends on the model architecture, hardware, batch size, and LoRA rank.

### Question 21: Why do we synchronise CUDA when measuring GPU computation time?

Pause.

Because GPU operations are often asynchronous.

Synchronisation ensures that the GPU has completed the queued computations before the timing measurement is finalised.

### Question 22: What is the main advantage of merging LoRA weights?

Pause.

It allows us to combine the trained adapter update with the original weight matrix.

The merged weight becomes:

$$
W_{\text{merged}}=W+BA
$$

The model can then perform inference using the ordinary layer architecture without a separate adapter computation.

### Question 23: Why might we save only the LoRA adapter matrices?

Pause.

Because we can store one shared pretrained model and save much smaller task-specific adapter files.

This is particularly useful when adapting the same pretrained model to multiple tasks.

### Question 24: What aspects of GPU memory can LoRA reduce?

Pause.

LoRA can reduce memory required for gradients and optimiser state associated with trainable parameters.

However, the original frozen model weights still occupy memory, and other computation-related memory requirements remain.

### Question 25: What is the main objective of this practical?

Pause.

The answer is to investigate whether LoRA can achieve competitive target-domain performance while updating substantially fewer parameters than conventional dense fine-tuning.

We also investigate the effects on optimiser storage, model storage, and computational time.

# Conclusion

Congratulations\! You have reached the end of Practical 4 on LoRA and Efficient Learning.

Let's finish by remembering the central idea.

When we adapt a pretrained neural network to a new task, we do not necessarily need to update all its original parameters.

LoRA provides an alternative.

It freezes the original pretrained weights and learns a low-rank update using two smaller matrices.

This can dramatically reduce the number of trainable parameters and the amount of optimiser state required during training.

However, we have also learned that parameter efficiency is not the same as reducing the complete model size or automatically increasing training speed.

The original weights are still stored and used during the forward pass.

The real advantages of LoRA must therefore be evaluated through several measurements, including accuracy, trainable parameters, optimiser state, storage requirements, and computation time.

In this practical, you have explored the mathematical foundations of low-rank adaptation, implemented LoRA layers, compared different ranks, investigated model and optimiser storage, and measured training efficiency.

You have also explored how trained adapters can be merged into the original model and saved separately for different tasks.

As a final reflection, consider this question:

Can we adapt powerful pretrained neural networks to new tasks by learning only a small number of additional parameters, rather than updating the entire model?

That is the central question behind LoRA and parameter-efficient fine-tuning.

Thank you for listening, and good luck with your experiments and assessment\!

End of Practical 4 – LoRA and Efficient Learning.
