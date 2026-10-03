# The key pieces are:

 1. **Pre-training:** `BasicCNN(num_classes=10)` is trained on CIFAR-10.
2. **Checkpointing:** the pretrained weights are saved.
3. **Classifier replacement:** the final `10 → 5` classifier is replaced for the CIFAR-100 target classes.
4. **Fine-tuning:** the whole model is trained on the target dataset with a smaller learning rate.
5. **Evaluation:** baseline and transfer-learning validation accuracies are compared.

 A few small changes would make the solution more robust and easier to explain in your report.

 ### Recommended fine-tuning code

 Your current code is valid, but I would load the checkpoint with `map_location=device` and explicitly use the best validation accuracy:

```
# Track fine-tuning metrics
ft_metrics = TrainingMetrics()

# Create a model with the same architecture used during pre-training
ft_model = BasicCNN(num_classes=source_data.num_classes).to(device)

# Load pretrained CIFAR-10 weights
ft_model.load_state_dict(
    torch.load('pretrained_model.ckpt', map_location=device)
)

# Replace the CIFAR-10 classifier with a CIFAR-100-target classifier
ft_model.classifier = nn.Linear(
    128,
    target_data.num_classes
).to(device)

# Fine-tuning hyperparameters
num_ft_epochs = 20
lr_ft = 1e-4
l2_reg_ft = 1e-3

loss_func_ft = nn.CrossEntropyLoss()

opt_ft = torch.optim.Adam(
    ft_model.parameters(),
    lr=lr_ft,
    weight_decay=l2_reg_ft
)

for e in range(num_ft_epochs):

    ft_model.train()

    for x, y in train_loader_target:

        opt_ft.zero_grad()

        pred = ft_model(x)
        batch_loss = loss_func_ft(pred, y)

        batch_loss.backward()
        opt_ft.step()

        ft_metrics.log_train(
            loss=float(batch_loss.item()),
            acc=float(get_batch_acc(pred, y))
        )

    # Validation
    ft_model.eval()

    val_losses = []
    val_accs = []

    with torch.no_grad():
        for vx, vy in val_loader_target:
            vpred = ft_model(vx)

            val_loss = loss_func_ft(vpred, vy)
            val_acc = get_batch_acc(vpred, vy)

            val_losses.append(val_loss.item())
            val_accs.append(val_acc)

    ft_metrics.log_val(
        loss=float(np.mean(val_losses)),
        acc=float(np.mean(val_accs))
    )

    if not DYNAMIC_PLOT:
        train_loss, val_loss = ft_metrics.epoch_loss()
        train_acc, val_acc = ft_metrics.epoch_acc()

        print(
            f'epoch:{e:>3} | '
            f'loss: {train_loss:.2f} | {val_loss:.2f}   '
            f'acc: {train_acc:.1%} | {val_acc:.1%}'
        )
```

 Then your final comparison is exactly what you need:

```
print(f'Best validation accuracy (baseline): '
      f'{np.max(baseline_metrics.val_acc):.1%}')

print(f'Best validation accuracy (transfer learning): '
      f'{np.max(ft_metrics.val_acc):.1%}')

transfer_boost = (
    np.max(ft_metrics.val_acc) -
    np.max(baseline_metrics.val_acc)
)

print(f'Transfer-learning improvement: {transfer_boost:.1%}')
```

 ### One important detail

 You **should not** simply load the CIFAR-10 model and leave its classifier unchanged. CIFAR-10 has 10 output classes, whereas your target has 5:

```
CIFAR-10:
...
128 → 10
       ↑
   discard this

Target:
...
128 → 5
       ↑
   new classifier
```

 The convolutional layers and intermediate fully connected layers retain the features learned from CIFAR-10, while the final classifier is freshly initialized for:

```
["bicycle", "bus", "motorcycle", "pickup_truck", "train"]
```

 This is the central transfer-learning idea in the exercise.

 ## What to write in the report

 Once you've actually run the notebook and have your two accuracies, you can use something like this, replacing the placeholders with your results:

 Transfer Learning Practical Report

# Transfer Learning on CIFAR-100

 ## Method

 I first trained the baseline `BasicCNN` model directly on the target dataset, consisting of five CIFAR-100 vehicle classes: bicycle, bus, motorcycle, pickup truck, and train. The target dataset was split into 80% training data and 20% validation data.

 For transfer learning, I first pretrained the same CNN architecture on the full CIFAR-10 training set. CIFAR-10 provides substantially more labelled training examples, allowing the convolutional layers to learn general visual features such as edges, textures, shapes, and object patterns.

 After pre-training, I saved the model weights and created a new model with the pretrained weights. Since CIFAR-10 has ten output classes while the target problem has five classes, I replaced the final classifier with a new linear layer containing five outputs. I then fine-tuned the resulting model on the target training set using a smaller learning rate than was used during pre-training.

 ## Results

 The baseline model achieved a maximum validation accuracy of **\[BASELINE\]%**.

 The transfer-learning model achieved a maximum validation accuracy of **\[TRANSFER\]%**, giving an improvement of approximately **\[IMPROVEMENT\] percentage points**.

 The training curves show that the transfer-learning model generally reached a higher validation accuracy than the baseline model. The pretrained model also started fine-tuning with useful representations already learned from CIFAR-10, rather than having to learn all visual features from the relatively small target dataset.

 ## Discussion

 The results demonstrate that transfer learning can improve performance when the target dataset is small. Although CIFAR-10 and the selected CIFAR-100 classes contain different categories, they share low-level visual characteristics. Features learned from the larger source dataset therefore provide a useful starting point for the target task.

 The main implementation issue was that the source and target datasets have different numbers of classes. I solved this by replacing the pretrained model's final ten-class classifier with a newly initialized five-class classifier before fine-tuning. I also used a lower learning rate during fine-tuning so that the pretrained features were not changed too aggressively.

 Overall, the transfer-learning approach produced better validation performance than training the same architecture from scratch.

 For the submission, include **two plots**: the baseline training/validation curves and the fine-tuning curves with the baseline comparison. Your existing `TrainingMetrics.plot()` machinery already produces these.

 One caveat: **don't invent the `[BASELINE]`, `[TRANSFER]`, or `[IMPROVEMENT]` numbers**—fill them in from the actual output of your run. The expected 5–10 percentage-point improvement in the exercise is only a guideline, not a guaranteed result.
