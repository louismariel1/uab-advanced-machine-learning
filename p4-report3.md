# Part 3 — Training infrastructure and CIFAR-10 pretraining

This section introduces the functions used to measure model performance and reports the first actual training result: 70.8% best validation accuracy on CIFAR-10.

This is an important milestone. Before testing full fine-tuning and LoRA, the notebook needs a model that has learned useful visual features from the source dataset.

## 1. TrainingMetrics: how performance is monitored

The `TrainingMetrics` class records four quantities during training:

| Metric | What it measures | Desired direction |
| --- | --- | --- |
| Training loss | Prediction error on training batches | Decrease |
| Validation loss | Prediction error on validation batches | Decrease, ideally |
| Training accuracy | Fraction of training examples classified correctly | Increase |
| Validation accuracy | Fraction of validation examples classified correctly | Increase |

Training metrics are recorded for every batch, while validation metrics are recorded once per epoch.

The class also calculates `best_val_acc`, which returns the highest validation accuracy recorded throughout training.

### How should we interpret the plots?

- Training loss decreases and validation loss decreases: the model is learning patterns that generalise to unseen validation examples.
- Training accuracy rises while validation accuracy stagnates: the model may be overfitting.
- Both accuracies remain low: the model may be undertrained or the optimisation settings may be unsuitable.
- Validation loss rises while training loss falls: this can be a warning sign of overfitting.

The plotting function smooths the batch-level training curves using an exponentially weighted moving average. Validation curves remain epoch-level measurements.

One important distinction is that the final training accuracy and the best validation accuracy need not occur at the same epoch.

## 2. Prediction accuracy and evaluation

The notebook defines the accuracy calculation as:

python
```
def top1_acc(pred, y):
    return (pred.argmax(axis=1) == y).float().mean().item()
```

The model outputs a vector of class scores, called logits, for each image. `argmax` selects the class with the highest score. Accuracy is the proportion of predictions that match the ground-truth labels.

For example, 70.8% accuracy means approximately 708 correct predictions per 1,000 examples, if the same accuracy holds over a representative set of that size.

The model is trained using cross-entropy loss:

$$
\mathcal{L}_{CE}
=-\frac{1}{N}\sum_{i=1}^{N}\log p(y_i\mid x_i)
$$

This loss penalises incorrect predictions, particularly when the model assigns low probability to the correct class.

The optimiser is AdamW, configured with a learning rate and weight decay. AdamW adapts parameter updates using gradient information, while weight decay helps discourage excessively large weights.

During validation, the notebook uses `model.eval()` and `torch.no_grad()`. These ensure evaluation is performed in inference mode without calculating gradients, reducing unnecessary computational work.

Implementation caveat: validation loss and accuracy are calculated for each batch and then averaged equally. Because the final batch can be smaller than the others, this is not exactly the same as calculating metrics over every validation example with equal weighting. The difference should usually be small here.

## 3. The CIFAR-10 pretraining experiment

The next cell performs the first actual model-training experiment.

Key settings:

| Setting | Value |
| --- | --- |
| Source dataset | CIFAR-10 |
| Training examples | 40,000 |
| Validation examples | 10,000 |
| Number of epochs | 20 |
| Batch size | 256 |
| Learning rate | 0.002 |
| Weight decay | 0.001 |
| Optimiser | AdamW |
| Loss function | Cross-entropy |
| Model parameters | 1,334,026 |

The code creates a fresh `ConvMLP` model and initialises its random weights using `torch.manual_seed(1)` before training.

The training loop then repeats the following process for each batch:

1. Calculate predictions with a forward pass.
2. Compute cross-entropy loss.
3. Calculate gradients through backpropagation.
4. Update the model parameters with AdamW.
5. Record training loss and accuracy.
6. At the end of each epoch, evaluate the model on the validation dataset.

The notebook uses the following configuration:

python
```
num_pt_epochs = 20
lr = 2e-3
l2_reg = 1e-3
```

The `train_model` function creates the optimiser from all model parameters, which are initially trainable.

## 4. Interpreting the result: 70.8% validation accuracy

Best CIFAR-10 validation accuracy

# 70.8%

Measured result reported by the notebook after 20 training epochs.

### What does this tell us?

The model has learned features that enable it to distinguish between the ten CIFAR-10 classes substantially better than random guessing.

For a balanced ten-class classification task, uniform random guessing has an expected accuracy of 10%. The observed 70.8% is considerably higher, providing evidence that the training process has learned meaningful predictive structure.

However, this is not evidence that the model is optimal. The architecture is deliberately simple, and the result depends on the training configuration, data preprocessing and optimisation.

It is also important to distinguish between these two quantities:

- Best validation accuracy: 70.8%, explicitly reported by the notebook.
- Final-epoch validation accuracy: not provided in the pasted output.

We should therefore report 70.8% as the best observed validation accuracy, not automatically as the accuracy at epoch 20.

### Why is pretraining necessary for the LoRA experiment?

The model has learned visual representations from 40,000 CIFAR-10 training images. These representations may transfer to the target vehicle classification task, even though the target labels differ from the source labels.

The next experiment should reuse this trained model as a starting point for CIFAR-100 vehicle classification. The target task has only 2,000 training examples, so reusing the learned representation may be more effective than learning all the visual features from scratch.

The key question will then be whether LoRA can adapt this representation with far fewer trainable parameters than full fine-tuning.

## 5. An important methodological point

The pretraining cell reports the best validation accuracy, but the output does not include the training curves or per-epoch metrics. Consequently, we cannot yet determine:

- Whether training and validation accuracy converged smoothly.
- Whether the model overfit during the 20 epochs.
- At which epoch the 70.8% best validation accuracy was achieved.
- How much training time or GPU computation was required.

We should avoid drawing conclusions about these points until the relevant outputs are available.

There is also a minor reproducibility consideration: the notebook seeds PyTorch before model initialisation and uses shuffled data loaders. The experiment should be reasonably reproducible in the same environment, although complete reproducibility can depend on CUDA operations and other execution details.

## 6. Interim findings for the final report

We can now record the following results:

| Experiment | Result | Interpretation |
| --- | --- | --- |
| CIFAR-10 pretraining | 70.8% best validation accuracy | The model learns useful source-domain representations. |
| Model size | 1,334,026 parameters | Establishes the parameter-count baseline. |
| Training duration | 20 epochs | Provides the source-training configuration. |

Interim conclusion: the notebook has successfully established a pretrained model for the subsequent transfer-learning experiments. The 70.8% validation accuracy indicates useful, though imperfect, source-domain classification performance. Whether these learned features transfer well to the five vehicle classes remains to be tested.

Please paste the next cell and its output. The next important stage is likely target-domain full fine-tuning, which will give us the performance baseline against which the LoRA results should be compared.
