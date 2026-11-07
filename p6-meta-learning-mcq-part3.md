# Advanced Machine Learning — Practical 6

## Part 3: Reptile — Learning a Useful Initialisation

Step 1 — MCQ with answers and explanationsThis MCQ set follows the same workflow as Parts 1 and 2. It focuses on Q10–Q14 of the Practical 6 question-focused summary.

Topics covered:

1. Understanding Reptile and its purpose.
2. The Reptile meta-learning update.
3. Understanding the meta-training loop.
4. Task-specific classification heads.
5. Evaluating Reptile and identifying implementation issues.

There are 20 questions, progressing from fundamental concepts to mathematical calculations, PyTorch code interpretation and exam-style scenarios.

## Section A — Understanding Reptile

### Question 1. What is Reptile?

A. A supervised learning algorithm that trains a fixed classifier on one dataset.

B. A gradient-based meta-learning algorithm that learns an initialisation from which new tasks can be learned quickly.

C. A clustering algorithm that groups images without labels.

D. A distance-based classifier that uses class prototypes.

Correct answer: BExplanation:

Reptile is a gradient-based meta-learning algorithm.

Its objective is to learn a useful initialisation of the model parameters so that the model can adapt quickly to new tasks using only a small number of labelled examples.

Instead of simply learning one fixed classification problem, Reptile learns across multiple tasks.

### Question 2. What is the main objective of Reptile?

A. Maximise the number of training images.

B. Learn a parameter initialisation that can adapt quickly to different tasks.

C. Eliminate the need for gradient descent.

D. Create a separate backbone for every image class.

Correct answer: BExplanation:

Reptile aims to find initial parameters that are useful across many tasks.

The model should require relatively little fine-tuning when presented with a new few-shot classification task.

This is the central idea of learning to learn.

### Question 3. How does Reptile differ from ordinary supervised learning?

A. Reptile trains only on unlabelled images.

B. Reptile repeatedly samples tasks and learns an initialisation that supports adaptation across tasks.

C. Reptile never uses backpropagation.

D. Reptile requires a fixed classification head for every possible class.

Correct answer: BExplanation:

Ordinary supervised learning generally optimises a model on a fixed collection of training classes.

Reptile repeatedly samples different tasks, fine-tunes the model on each task and uses the adapted parameters to improve its initialisation.

The aim is to generalise to new tasks rather than simply classify the original training classes.

### Question 4. What does the term meta-learning mean?

A. Learning how to learn across different tasks.

B. Learning how to increase image resolution.

C. Learning without any model parameters.

D. Learning how to remove classification labels.

Correct answer: AExplanation:

Meta-learning is often described as learning how to learn.

The model is exposed to many tasks during meta-training so that it can become better at adapting to new tasks.

For Reptile, this means learning an initialisation that can be fine-tuned efficiently.

### Question 5. What happens when Reptile receives a new few-shot task during meta-training?

A. It immediately deletes its current model.

B. It fine-tunes the model on the task and then uses the adapted parameters to update the meta-model.

C. It calculates class prototypes without using the backbone.

D. It trains a completely independent model and discards all results.

Correct answer: BExplanation:

Reptile follows two important stages:

1. Inner-loop adaptation: Fine-tune the current model on a sampled task.
2. Meta-update: Move the original model parameters towards the task-adapted parameters.

Repeating this process across tasks helps produce an initialisation suitable for rapid adaptation.

## Section B — Understanding the Reptile mathematical update

### Question 6. Which equation represents the Reptile meta-update?

A. $\theta \leftarrow \theta-\epsilon(\theta_i'-\theta)$

B. $\theta \leftarrow \theta+\epsilon(\theta_i'-\theta)$

C. $\theta \leftarrow \theta_i'+\epsilon$

D. $\theta \leftarrow \theta\times\theta_i'$

Correct answer: BExplanation:

The Reptile update is:

$$
\boxed{
\theta\leftarrow
\theta+\epsilon(\theta_i'-\theta)
}
$$

Where:

- $\theta$: Original meta-model parameters.
- $\theta_i'$: Parameters after fine-tuning on task $i$.
- $\epsilon$: Meta-step size.

The update moves the original parameters towards the task-adapted parameters.

### Question 7. In the Reptile equation, what does $\theta$ represent?

A. The query labels.

B. The current meta-model parameters before the meta-update.

C. The number of training episodes.

D. The classification accuracy.

Correct answer: BExplanation:

$\theta$ represents the current initialisation of the meta-model.

It is the parameter vector that Reptile is trying to improve across different tasks.

### Question 8. In the Reptile equation, what does $\theta_i'$ represent?

A. The model parameters after fine-tuning on task $i$.

B. The original model parameters before training.

C. The query-set accuracy.

D. The class labels of the support set.

Correct answer: AExplanation:

$\theta_i'$ represents the task-adapted parameters.

These are obtained after performing inner-loop gradient updates on the sampled task, usually using its support set.

The meta-model then moves towards these adapted parameters.

### Question 9. What is the purpose of the meta-step size $\epsilon$?

A. It determines the number of classes in a task.

B. It controls how far the meta-model moves towards the task-adapted parameters.

C. It determines the number of support images.

D. It controls the image resolution.

Correct answer: BExplanation:

The meta-step size determines the proportion of the task vector used in the meta-update.

- Small $\epsilon$: Small movement towards the adapted parameters.
- Large $\epsilon$: Larger movement.
- $\epsilon=1$: Moves completely to the task-adapted parameters.

### Question 10. Suppose the initial parameter is 2, the task-adapted parameter is 6, and the meta-step size is 0.5. What is the updated parameter?

A. 2

B. 3

C. 4

D. 6

Correct answer: CExplanation:

Use the Reptile update:

$$
\theta_{\text{new}}
=\theta+\epsilon(\theta_i'-\theta)
$$

Substitute the values:

$$
=2+0.5(6-2)
$$

$$
=2+0.5(4)
$$

$$
\boxed{\theta_{\text{new}}=4}
$$

The meta-model moves halfway from 2 towards 6.

### Question 11. What happens if the Reptile meta-step size is equal to 1?

A. The meta-model does not change.

B. The meta-model moves completely to the task-adapted parameters.

C. The meta-model moves halfway towards the task-adapted parameters.

D. The model parameters become zero.

Correct answer: BExplanation:

When $\epsilon=1$:

$$
\theta_{\text{new}}
=\theta+1(\theta_i'-\theta)
$$

Therefore:

$$
\boxed{\theta_{\text{new}}=\theta_i'}
$$

The updated meta-model becomes equal to the task-adapted model.

### Question 12. Suppose $\theta=10$, $\theta_i'=4$, and $\epsilon=0.25$. What is the updated parameter?

A. 4

B. 5.5

C. 8.5

D. 10

Correct answer: CExplanation:

Apply the Reptile equation:

$$
\theta_{\text{new}}
=10+0.25(4-10)
$$

$$
=10+0.25(-6)
$$

$$
=10-1.5
$$

$$
\boxed{\theta_{\text{new}}=8.5}
$$

The model moves 25% of the distance towards the adapted parameter.

## Section C — Understanding the Reptile training loop

### Question 13. What are the main operations in one Reptile meta-training iteration?

A. Sample a task, fine-tune the model, perform a meta-update and record performance.

B. Generate images, delete the model and calculate image resolution.

C. Calculate prototypes, freeze all parameters and stop training.

D. Train a fixed classifier without sampling tasks.

Correct answer: AExplanation:

The Reptile training loop performs four important operations:

1. Sample a task from the meta-training classes.
2. Fine-tune the model using the task's support set.
3. Move the meta-model towards the adapted parameters.
4. Record query performance and periodically evaluate the model.

### Question 14. What is the purpose of `finetune_backbone_on_task()` in Reptile?

A. To calculate the average embedding of every class.

B. To adapt the backbone to a sampled task using gradient-based fine-tuning.

C. To permanently remove the backbone.

D. To calculate only the meta-test accuracy.

Correct answer: BExplanation:

The function performs the inner-loop adaptation.

It fine-tunes the backbone on the support examples of the sampled task and returns task metrics and a task-specific classification head.

The adapted backbone parameters are subsequently used in the Reptile meta-update.

### Question 15. What is a task vector in Reptile?

A. The difference between the task-adapted parameters and the original parameters.

B. The number of images in the query set.

C. The classification accuracy of the model.

D. The feature vector of a single image.

Correct answer: AExplanation:

The task vector is:

$$
\boxed{
\text{Task vector}=\theta_i'-\theta
}
$$

It represents the change in model parameters caused by fine-tuning on a particular task.

Reptile uses this difference to move the meta-model towards a better initialisation.

### Question 16. Consider the following code:

python

```python
param_task_vector = current_param - prev_param

new_param = (
    prev_param
    + param_task_vector * meta_step_size
)
```

What does `param_task_vector` represent?

A. The query-set loss.

B. The difference between the adapted parameter and the original parameter.

C. The learning rate of SGD.

D. The number of classes in the task.

Correct answer: BExplanation:

In this implementation:

- `prev_param` represents the original meta-model parameter.
- `current_param` represents the task-adapted parameter.
- `param_task_vector` is their difference.

The new parameter is calculated by moving a fraction of this difference towards the adapted parameter.

### Question 17. Why does the Reptile implementation create a new classification head for each task?

A. Every task contains exactly the same classes.

B. Different tasks can contain different classes, so they need task-specific classifiers.

C. The backbone cannot extract features.

D. The classification head is responsible for updating the meta-step size.

Correct answer: BExplanation:

Each sampled task can contain a different combination of classes.

Therefore, the original classifier for the meta-training categories cannot automatically be reused.

A new task-specific head is created and fine-tuned alongside the backbone during inner-loop adaptation.

Importantly, the Reptile meta-update is applied to the backbone parameters, not the temporary task-specific classification head.

## Section D — Evaluating Reptile

### Question 18. What is the most important metric for evaluating Reptile on unseen few-shot tasks?

A. Baseline training loss.

B. Average query accuracy across meta-test tasks.

C. Number of backbone parameters.

D. Number of training epochs only.

Correct answer: BExplanation:

The key evaluation metric is average query accuracy:

$$
\overline{A}_{\text{query}}
=
\frac{1}{N}
\sum_{i=1}^{N} A_i
$$

Where:

- $N$: Number of meta-test tasks.
- $A_i$: Query accuracy for task $i$.

This measures how successfully Reptile generalises to previously unseen classes.

### Question 19. Why might Reptile fail to outperform the supervised baseline?

A. Reptile never uses model parameters.

B. Its performance depends on the task distribution, backbone and inner-loop hyperparameters.

C. Reptile cannot use labelled examples.

D. Reptile always requires a prototype classifier.

Correct answer: BExplanation:

Reptile does not automatically outperform supervised learning.

Its performance depends on factors such as:

- Quality of the backbone.
- Number of inner-loop fine-tuning steps.
- Inner-loop learning rate.
- Meta-step size.
- Distribution of sampled tasks.

A strong supervised baseline can sometimes outperform Reptile.

The important lesson is to compare actual meta-test performance rather than assume that meta-learning must be better.

### Question 20. The notebook contains the following final evaluation code:

python

```python
final_meta_results = meta_evaluate(meta_model, ...)
```

However, the Reptile model is called `reptile_backbone`. What is the likely problem?

A. The model has too many classes.

B. The code may reference an undefined or incorrect variable and should evaluate the intended Reptile model.

C. The learning rate is automatically set to zero.

D. The query set has no labels.

Correct answer: BExplanation:

The notebook's Reptile model is named `reptile_backbone`.

If `meta_model` has not been defined elsewhere, this produces a variable-name error.

The evaluation should use the intended model:

python

```python
final_meta_results = meta_evaluate(
    reptile_backbone,
    model_name="Final meta-trained Reptile model",
    baseline_avg=baseline_metatest_query_acc,
    full_state=False,
    freeze_backbone=True
)
```

This assumes the best Reptile checkpoint has already been loaded into `reptile_backbone`.

## Final revision sheet — Part 3

These are the essential concepts to remember for an oral or written assessment.

| Concept | Essential takeaway |
| --- | --- |
| Reptile | Gradient-based meta-learning algorithm |
| Main objective | Learn an initialisation that adapts quickly |
| Inner loop | Fine-tune the model on a sampled task |
| Meta-update | Move initial parameters towards adapted parameters |
| $\theta$ | Original meta-model parameters |
| $\theta_i'$ | Task-adapted parameters |
| $\epsilon$ | Meta-step size |
| Task vector | Difference between adapted and original parameters |
| Task-specific head | New classifier for each sampled task |
| Meta-test accuracy | Measures generalisation to unseen tasks |

### Five particularly important exam questions

1. What is Reptile, and why is it a meta-learning algorithm?
2. Explain the difference between inner-loop adaptation and the outer meta-update.
3. Explain the mathematical Reptile update and the role of the meta-step size.
4. Why does Reptile create a new classification head for every task?
5. Why is average meta-test query accuracy more important than training accuracy?

## Key formula to memorise

$$
\boxed{
\theta\leftarrow
\theta+\epsilon(\theta_i'-\theta)
}
$$

Remember the intuition:

Fine-tune on a task, then move the original model towards the adapted model. Repeat across many tasks to learn a useful initialisation.

Part 3 — Step 1 completeNext, we can convert these exact 20 questions into the interactive practice exam (Step 2), with answer selection, question navigation, automatic scoring and explanations revealed after submission.
