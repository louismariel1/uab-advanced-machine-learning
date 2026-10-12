# Part 4 — Saving the pretrained model and establishing the fine-tuning baseline

This section moves from source-domain pretraining to the target-domain experiment. We now have the first result for the actual five-class vehicle task: 65.0% best validation accuracy using dense fine-tuning.

This is the baseline against which we will evaluate LoRA. An important detail is that the notebook freezes the convolutional layers in both the baseline and the intended LoRA comparison, so the baseline is not conventional full fine-tuning of every parameter.

## 1. Verifying and saving the pretrained model

The notebook first obtains a batch of validation images from CIFAR-10 and displays the model's predictions:

python
```
source_x, source_y = next(iter(val_loader_source))

inspect_batch(
    source_x, source_y,
    predictions=pt_model(source_x),
    class_names=source_data.classes
)
```

The `inspect_batch()` function displays images alongside their true labels and predicted labels. Correct predictions receive a green check mark, while incorrect predictions receive a red cross.

This is a qualitative sanity check: it allows us to inspect individual predictions and see which classes the model appears to confuse. It complements the 70.8% aggregate validation accuracy, but does not replace it.

The model is then saved:

python
```
torch.save(pt_model.state_dict(), 'pretrained_model.ckpt')
```

Saving the `state_dict` preserves the model's learned parameters so they can be loaded into another instance of the same architecture. It does not save the architecture definition itself, the optimiser state or the training history.

The output confirms that the checkpoint was saved successfully. However, because the actual image grid was not included in the pasted output, we cannot draw conclusions about specific misclassifications.

## 2. The dense fine-tuning baseline

The next experiment adapts the pretrained model to the five CIFAR-100 vehicle classes.

The notebook deliberately freezes the two convolutional layers and fine-tunes the MLP and the new classifier. This is called dense fine-tuning with layer freezing (DFT).

### What changes between pretraining and fine-tuning?

- Source task: classify images into 10 CIFAR-10 classes.
- Target task: classify images into five vehicle classes.
- Convolutional layers: their pretrained weights are retained and frozen.
- MLP layers: pretrained weights are loaded and updated during fine-tuning.
- Classification head: replaced with a newly initialised five-class output layer.

The target classifier must be reinitialised because its output dimension changes from 10 to 5. The meanings of the target labels also differ from the source labels, so the original classification head cannot simply be reused.

### Which parameters are trainable?

The baseline model has the following parameter counts:

Dense fine-tuning baseline

# 1,332,741

Total parameters in the five-class model

Trainable

## 1,183,237

Frozen

## 149,504

88.8% trainable; 11.2% frozen.

The trainable parameters are:

- `fc1.weight` and `fc1.bias`
- `fc2.weight` and `fc2.bias`
- `fc3.weight` and `fc3.bias`
- `fc4.weight` and `fc4.bias`
- `classifier.weight` and `classifier.bias`

The frozen parameters belong to `conv1` and `conv2`.

The total parameter count is slightly smaller than the original 1,334,026 because the new classifier has five outputs instead of ten.

### Why freeze the convolutional layers?

The convolutional layers learn general visual features such as edges, textures and local patterns. The practical assumes that these features can be reused for vehicle classification, allowing the MLP to specialise the representation to the new task.

Freezing these layers also reduces the number of trainable parameters and avoids updating their weights during backpropagation. It does not remove their computational cost from the forward pass.

This design creates a fairer comparison for the later LoRA experiment, which will focus on adapting the fully connected layers.

## 3. Loading the pretrained weights

The notebook loads the saved checkpoint into the target model:

python
```
pt_state_dict = torch.load(pt_save_name)

del pt_state_dict['classifier.weight']
del pt_state_dict['classifier.bias']

baseline_ft_model.load_state_dict(
    pt_state_dict, strict=False
)
```

Deleting the classifier weights and bias prevents a shape mismatch between the original ten-class classifier and the new five-class classifier.

The other pretrained parameters are loaded into the target model. Consequently, the MLP starts from learned CIFAR-10 weights rather than random initialisation, while the new classification head starts randomly initialised.

This is the core transfer-learning mechanism: reuse learned representations and adapt them to a new task.

## 4. Fine-tuning configuration and results

The baseline is trained using the following settings:

| Setting | Value |
| --- | --- |
| Target training images | 2,000 |
| Target validation images | 500 |
| Training epochs | 30 |
| Learning rate | 0.0001 |
| Weight decay | 0.01 |
| Optimiser | AdamW |
| Loss | Cross-entropy |
| Trainable parameters | 1,183,237 |

The learning rate is lower than the pretraining rate of 0.002. This is a common transfer-learning strategy: pretrained weights already contain useful information, so smaller updates can help preserve that information while adapting to the target domain.

The reported result is:

Best target validation accuracy — dense fine-tuning

# 65.0%

Best observed validation accuracy during the 30-epoch run.

### Interpretation

The model achieves 65.0% validation accuracy on the five vehicle classes. This indicates that the pretrained representation can be adapted to the target task, although there is still considerable room for improvement.

For context, uniform random guessing across five balanced classes would have an expected accuracy of 20%. The baseline substantially exceeds that reference level.

However, we cannot conclude that transfer learning is better than training from scratch without a separate from-scratch experiment on the same target split.

We also cannot yet determine which vehicle classes are hardest to classify or whether the model overfits, because the per-class predictions and training curves have not been supplied.

## 5\. Comparison with the pretraining result

| Experiment | Best validation accuracy | Dataset |
| --- | --- | --- |
| Source pretraining | 70.8% | CIFAR-10, 10 classes |
| Dense fine-tuning | 65.0% | CIFAR-100 vehicles, 5 classes |

The source accuracy is 5.8 percentage points higher, but these figures should not be interpreted as a direct performance decline caused by fine-tuning. They come from different datasets and classification tasks, with different class definitions and levels of difficulty.

The correct role of the 65.0% result is to serve as the target-domain reference for comparing LoRA against dense fine-tuning under comparable conditions.

## 6\. Interim conclusion

The notebook has established a useful transfer-learning baseline:

- Pretrained convolutional and MLP weights are reused.
- The convolutional layers are frozen.
- The MLP and new five-class classifier are fine-tuned.
- The resulting model reaches 65.0% best validation accuracy while updating 1,183,237 parameters.

This gives us two quantities to compare against LoRA: target validation accuracy and the number of trainable parameters.

One nuance for the final report is that this baseline already freezes part of the network. Therefore, when we evaluate LoRA, we should describe the comparison as LoRA versus dense fine-tuning with frozen convolutional layers, rather than LoRA versus unrestricted full-model fine-tuning.

Please paste the next cell and output. The next stage should reveal how LoRA is implemented and how its parameter efficiency and classification performance compare with this baseline.
