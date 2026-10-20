# Knowledge Distillation on CIFAR-10: Training a Compact Student Model

## 1. Introduction and objectives

The aim of this experiment was to investigate whether knowledge distillation can improve the performance of a small neural network by transferring knowledge from a larger, more accurate teacher model.

Knowledge distillation is a model compression and training technique in which a student network learns from two sources: the ground-truth labels of the training data and the output predictions of a pretrained teacher network. Instead of learning only which class an image belongs to, the student also learns from the teacher's probability distribution across all possible classes.

The experiment compared three models:

- Independent student: a small network trained conventionally using labelled images.
- Teacher: a larger network trained on the same dataset, used to provide additional learning signals.
- Distilled student: a new instance of the small network trained using both the ground-truth labels and the teacher's predictions.

The primary objective was to improve the independent student's validation accuracy by at least five percentage points while retaining the smaller architecture. The results show that this objective was achieved.

## 2. Experimental setup and preconfigured notebook

The experiment was conducted in Google Colab using Python and PyTorch. The provided notebook supplied the basic infrastructure for the experiment, including dataset preparation, neural network definitions, training and evaluation functions, metric tracking, and plotting utilities.

The main libraries used were:

- PyTorch: model construction, automatic differentiation, loss functions, optimisation, and training.
- Torchvision: downloading and loading CIFAR-10 and applying image transformations.
- NumPy and pandas: numerical operations and smoothing training metrics.
- Matplotlib: visualisation of training curves, image samples, and output distributions.
- torchinfo: inspection of network architectures and parameter counts.
- ipdb: an optional debugging utility.

The notebook was extended to implement three components required for knowledge distillation: temperature-scaled softmax, a KL-divergence distillation loss, and a teacher-student training loop.

The experiment retained the notebook's predefined training configuration to ensure a fair comparison between the independent and distilled students.

## 3. Dataset

The experiment used the CIFAR-10 image classification dataset. CIFAR-10 contains 60,000 colour images, each measuring $32 \times 32$ pixels, divided into ten classes.

The classes are:

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

To reduce computational requirements and investigate distillation in a limited-data setting, the experiment used a reduced subset of the original dataset.

The data was divided into three portions:

| Dataset portion | Percentage of original dataset | Number of images |
| --- | --- | --- |
| Training | 30% | 18,000 |
| Validation | 20% | 12,000 |
| Discarded | 50% | 30,000 |
| Total | 100% | 60,000 |

The training and validation subsets were created using PyTorch's `random_split`, with a fixed random seed to improve reproducibility.

Images were converted to tensors and normalised using a mean of 0.5 and a standard deviation of 0.2 for each RGB channel. The data loaders used a batch size of 256.

The same training and validation data was used for all three models. This ensured that the performance differences were not caused by differences in the dataset.

## 4. Model architectures

### 4.1. Residual neural network

The notebook defined a custom residual network, `BasicResnet`, constructed from residual blocks. Each block contains convolutional layers, batch normalisation, ReLU activations, and a shortcut connection. Projection convolutions are used when the number of channels or spatial dimensions changes.

Residual connections help information and gradients propagate through deeper networks. Adaptive average pooling reduces the spatial dimensions before a final linear classification layer produces ten output logits.

The same model class was used for the teacher and student, but with different backbone configurations.

### 4.2. Independent student

The independent student used the following convolutional channel configuration:

`[32, 64, 128]`

This model had 299,082 trainable parameters. It was trained from scratch using the standard cross-entropy loss, which compares the model's predicted class distribution with the ground-truth labels.

The independent student established the baseline performance that the distilled student needed to exceed.

### 4.3. Teacher

The teacher used a deeper and wider architecture:

`[64, 128, 128, 256, 256]`

Spatial dropout with probability 0.2 was applied within its residual blocks to help control overfitting.

The teacher contained 2,666,122 trainable parameters, approximately 8.9 times as many as the student. Its greater capacity allowed it to learn a more informative representation of the training data.

The teacher was trained independently before distillation. Its weights were saved to a checkpoint file, `teacher.ckpt`, and reloaded before the student training stage.

### 4.4. Distilled student

The distilled student used the same architecture and parameter count as the independent student. A fresh model was initialised rather than continuing training from the independent student's weights.

The only substantive difference was its training objective: it received both the ground-truth labels and the teacher's output distributions.

This design allowed the experiment to isolate the effect of distillation more directly.

## 5. Experimental methodology

The experiment consisted of the following stages.

### 5.1. Training the independent student

The independent student was trained using cross-entropy loss and the AdamW optimiser.

The main hyperparameters were:

| Hyperparameter | Value |
| --- | --- |
| Epochs | 20 |
| Batch size | 256 |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Exponential learning-rate decay | 0.95 |

Validation accuracy and loss were recorded after each epoch. The best validation accuracy was 66.80%.

This result provided the baseline for assessing whether knowledge distillation offered a measurable improvement.

### 5.2. Training the teacher

The teacher was trained using the same dataset and number of epochs, with a weight decay of 0.1 and spatial dropout of 0.2.

Its best validation accuracy was 78.71%, approximately 11.91 percentage points higher than that of the independent student.

The teacher therefore provided a stronger source of predictions for guiding the student.

### 5.3. Inspecting teacher predictions

Before implementing distillation, the notebook visualised the teacher's predictions for example images.

A neural network produces logits, which are unnormalised class scores. Applying softmax converts these logits into a probability distribution over the ten classes.

The teacher's distribution typically assigned a high probability to its predicted class while retaining smaller probabilities for other classes. These secondary probabilities can contain useful information about similarities between classes.

For example, a teacher might consider a horse more similar to a deer than to a truck. A conventional hard label does not express this relationship, but the teacher's output distribution can.

### 5.4. Temperature-scaled softmax

The notebook implemented temperature-scaled softmax to control how smooth the teacher's probability distribution becomes.

For logits $z_i$ and temperature $T$, the probability assigned to class $i$ is

$$
p_i =
\frac{\exp(z_i/T)}
{\sum_j \exp(z_j/T)}.
$$

When $T=1$, this is the ordinary softmax function. Increasing the temperature produces a softer distribution, making the relative probabilities of less likely classes more informative.

The experiment used a temperature of 20. This allowed the student to learn from more than just the teacher's most confident class prediction.

### 5.5. Implementing the distillation loss

The next step was to compare the softened output distributions of the student and teacher using Kullback–Leibler (KL) divergence.

The implemented loss was

$$
L_{\mathrm{distill}}
=
T^2 D_{\mathrm{KL}}(p_t\|p_s),
$$

where $p_t$ is the teacher's temperature-scaled distribution and $p_s$ is the student's temperature-scaled distribution.

The KL-divergence implementation used PyTorch's `F.kl_div` with log-probabilities for the student and probabilities for the teacher, together with `reduction='batchmean'`.

Multiplying by $T^2$ compensates for the reduction in gradient magnitude associated with temperature scaling.

An initial sanity check confirmed that the loss returned a scalar value and was positive for the example predictions.

### 5.6. Training the distilled student

The final training loop combined the conventional supervised loss with the distillation loss:

$$
L_{\mathrm{total}}
=
(1-\alpha)L_{\mathrm{hard}}
+
\alpha L_{\mathrm{distill}}.
$$

Here, $L_{\mathrm{hard}}$ is the cross-entropy loss against the ground-truth labels, and $\alpha$ controls the relative importance of the teacher's guidance.

The experiment used:

| Hyperparameter | Value |
| --- | --- |
| Temperature $T$ | 20 |
| Distillation weight $\alpha$ | 0.5 |
| Epochs | 20 |
| Batch size | 256 |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Exponential learning-rate decay | 0.95 |

The teacher was placed in evaluation mode and its parameters were frozen. Its predictions were computed under `torch.no_grad()`, preventing unnecessary gradient calculations. Only the student's parameters were updated.

The student was placed in training mode during training, while validation was performed using the existing evaluation function.

Training and validation metrics were recorded, and the resulting curves were compared with those of the independent student.

## 6. Results

The experiment produced the following best validation accuracies:

| Model | Parameters | Best validation accuracy |
| --- | --- | --- |
| Independent student | 299,082 | 66.80% |
| Distilled student | 299,082 | 72.94% |
| Teacher | 2,666,122 | 78.71% |

The distilled student improved on the independent student by

$$
72.94\%-66.80\% = 6.14
$$

percentage points.

This exceeded the target improvement of five percentage points. The distilled student also recovered approximately 51.6% of the performance gap between the independent student and the teacher.

The results demonstrate that the teacher's predictions provided useful additional supervision. Importantly, this improvement was achieved without increasing the student's model size.

### Figure 1. Comparison of best validation accuracy

Insert your validation accuracy comparison graph here.

Suggested caption: Best validation accuracy of the independent student, distilled student, and teacher on the reduced CIFAR-10 dataset. The distilled student achieved 72.94%, compared with 66.80% for the independent student and 78.71% for the teacher.

### Figure 2. Training and validation curves

Insert your training curves here.

Suggested caption: Training and validation loss and accuracy for the distilled student compared with the independent student baseline. The curves illustrate the learning process and the effect of teacher supervision over the training epochs.

When discussing Figure 2, describe the actual curves you observe, including convergence, any divergence between training and validation performance, and whether validation accuracy stabilises toward the end of training.

## 7. Issues encountered and how they were resolved

A few implementation issues arose during the experiment.

First, the initial distillation training cell contained placeholder ellipses (`...`), which caused a Python `SyntaxError`. This was resolved by completing the function signature and implementing the missing training loop, optimiser, validation, and metric logging.

Second, the distillation loss required careful handling of the probability distributions. PyTorch's KL-divergence function expects log-probabilities for its first argument and probabilities for its second. The implementation used the student's log-probabilities and the teacher's probabilities, together with the batch-mean reduction and temperature-squared correction.

Third, the teacher and student needed different training modes. The teacher had to remain in evaluation mode so that dropout and batch-normalisation behaviour remained stable, while the student needed training mode during optimisation. Freezing the teacher's parameters and using `torch.no_grad()` prevented unnecessary gradient computation and ensured that only the student was updated.

Finally, the experiment required a fair comparison. The distilled student was reinitialised from scratch and trained using the same architecture and main optimisation hyperparameters as the independent student. This reduced the risk of attributing improvements to a larger model or a more favourable training configuration.

## 8. Discussion and insights

The results illustrate an important property of knowledge distillation: a model can benefit from the information contained in a stronger model's output distribution, even when both models are trained on the same labelled examples.

The independent student learned directly from the ground-truth classes. The distilled student received that same supervision, together with information about the teacher's relative confidence across classes. The improvement from 66.80% to 72.94% suggests that this additional signal helped the smaller network learn a more effective decision boundary.

Temperature was particularly important conceptually. At a low temperature, the teacher's distribution can be highly concentrated on one class, making the distillation signal more similar to a hard label. A higher temperature reveals more of the relative probabilities assigned to other classes. However, the best temperature and loss weighting depend on the dataset, architecture, and training configuration; the experiment did not establish that the chosen values are optimal.

The teacher remained more accurate than the student, which is expected given its greater capacity. Nevertheless, the distilled student recovered approximately half of the original performance gap while using about 11.2% of the teacher's parameter count.

A limitation is that the experiment used a single training configuration and a fixed random seed rather than repeated runs. Therefore, the measured improvement is evidence of success in this run, but it does not establish the average improvement or statistical significance across multiple runs.

## 9. Conclusion

This experiment successfully implemented output-based knowledge distillation using a custom residual neural network and the CIFAR-10 dataset.

The independent student achieved 66.80% best validation accuracy, while the teacher achieved 78.71%. After incorporating temperature-scaled KL-divergence loss into the student training procedure, the distilled student reached 72.94%.

The improvement of 6.14 percentage points exceeded the required five-point target. Furthermore, the distilled student retained the original student architecture, with 299,082 parameters compared with the teacher's 2,666,122.

Overall, the experiment demonstrates that knowledge distillation can improve the predictive performance of a compact model by transferring information from a larger teacher, without increasing the student's parameter count. It also highlights the importance of correct loss implementation, temperature scaling, model training modes, and controlled comparisons when evaluating distillation.
