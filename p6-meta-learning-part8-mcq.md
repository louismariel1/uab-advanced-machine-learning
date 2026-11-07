# Advanced Machine Learning — Practical 6

## Part 8: Reporting Your Results

MCQ Practice Questions with Answers and Explanations

Part 8 focuses on how to present, analyse, and discuss the experimental results obtained from the three few-shot learning approaches:

1. Baseline supervised learning.
2. Reptile meta-learning.
3. Prototypical Networks (ProtoNet).

The questions test your understanding of training plots, performance comparisons, average meta-test query accuracy, percentage-point improvements, relative percentage improvements, and scientific reporting.

25 MCQsAnswers includedExplanations included

## Section A — Understanding experimental plots

### Question 1. What is the main purpose of including training and validation plots in your practical report?

A. To make the report visually attractive.

B. To demonstrate how the model learns and whether its performance improves during training.

C. To replace the final meta-test evaluation.

D. To show only the model architecture.

Correct answer: BExplanation: Training and validation plots help demonstrate the learning process. They show how loss and accuracy change over time and can reveal problems such as overfitting or unstable optimisation.

### Question 2. Which plots should you include for the baseline supervised model?

A. Only the final meta-test accuracy.

B. Training and validation loss and accuracy during supervised pre-training.

C. Only the Reptile query accuracy.

D. Only the ProtoNet prototype distances.

Correct answer: BExplanation: The baseline should be evaluated throughout supervised training. Training and validation loss and accuracy help demonstrate how effectively the model learns its original meta-training classes.

### Question 3. What should the Reptile training plot ideally show?

A. Only the number of model parameters.

B. Meta-training query performance and meta-test evaluation performance across episodes.

C. Only the baseline training loss.

D. Only the class prototype coordinates.

Correct answer: BExplanation: Reptile uses episodic meta-training. Its plots should show how query performance changes during meta-training and how well the model performs when evaluated on meta-test tasks.

### Question 4. What should the ProtoNet training plot ideally show?

A. Only support-set accuracy.

B. Meta-training query performance and meta-test evaluation performance across episodes.

C. Only the backbone's embedding dimension.

D. Only the number of training classes.

Correct answer: BExplanation: ProtoNet learns through episodic training using support and query sets. Its plots should demonstrate training progress and generalisation to meta-test tasks.

### Question 5. Why is it useful to include a final comparison plot?

A. To compare the average meta-test query accuracy achieved by the three approaches.

B. To compare the number of lines of Python code.

C. To show which method has the largest backbone.

D. To eliminate the need for a discussion section.

Correct answer: AExplanation: A final comparison plot makes it easier to identify which approach performs best on unseen few-shot tasks. It provides a clear visual comparison of the baseline, Reptile, and ProtoNet.

## Section B — Understanding evaluation metrics

### Question 6. What is the main metric used to compare the three methods in this practical?

A. Training loss.

B. Number of training episodes.

C. Average meta-test query accuracy.

D. Number of model parameters.

Correct answer: CExplanation: Average meta-test query accuracy measures how well each method classifies additional examples from previously unseen classes. It is the key metric for evaluating few-shot generalisation.

### Question 7. Why is baseline training accuracy not sufficient for evaluating few-shot learning?

A. Because the baseline cannot classify images.

B. Because training accuracy measures performance on the original training classes, not necessarily adaptation to unseen classes.

C. Because accuracy cannot be expressed as a percentage.

D. Because the baseline has no classification head.

Correct answer: BExplanation: A model can achieve excellent training accuracy on known classes but still perform poorly on unfamiliar classes. Few-shot learning specifically aims to generalise to new classes using limited labelled examples.

### Question 8. How is average query accuracy calculated across multiple meta-test tasks?

A. By adding all task accuracies and dividing by the number of tasks.

B. By selecting only the highest task accuracy.

C. By multiplying all task accuracies.

D. By dividing the number of classes by the number of tasks.

Correct answer: AExplanation: The average query accuracy is:

$$
\overline{A}_{query}
=
\frac{1}{N}\sum_{i=1}^{N}A_i
$$

where $A_i$ represents the query accuracy for task $i$, and $N$ is the number of evaluated tasks.

### Question 9. Suppose a model achieves query accuracies of 60%, 70%, 80%, 50%, and 90% across five meta-test tasks. What is its average query accuracy?

A. 60%.

B. 65%.

C. 70%.

D. 75%.

Correct answer: CExplanation:

$$
\frac{60+70+80+50+90}{5}
=
\frac{350}{5}
=70\%
$$

The average query accuracy is 70%.

### Question 10. What does a high average meta-test query accuracy indicate?

A. The model has a large number of parameters.

B. The model generalises effectively to unseen few-shot classification tasks.

C. The model has memorised every training image.

D. The model does not need a support set.

Correct answer: BExplanation: High average query accuracy indicates that the model can classify additional examples from unfamiliar classes using the few-shot task setup.

## Section C — Comparing the three approaches

### Question 11. Which three methods are compared in the practical?

A. CNN, RNN, and Transformer.

B. Baseline supervised learning, Reptile, and ProtoNet.

C. K-means, PCA, and SVM.

D. ResNet, VGG, and AlexNet.

Correct answer: BExplanation: The practical investigates three approaches to few-shot image classification: an ordinary supervised baseline, the Reptile meta-learning algorithm, and Prototypical Networks.

### Question 12. What is the main difference between the baseline and Reptile?

A. The baseline uses ordinary supervised training, whereas Reptile learns an initialisation that can adapt to different tasks.

B. The baseline uses prototypes, whereas Reptile uses no backbone.

C. The baseline is always more accurate than Reptile.

D. Reptile does not use gradient-based optimisation.

Correct answer: AExplanation: The baseline learns visual features through conventional supervised training. Reptile uses episodic fine-tuning and meta-updates to learn initial parameters that are useful across multiple tasks.

### Question 13. What is the main difference between Reptile and ProtoNet?

A. Both methods use exactly the same adaptation mechanism.

B. Reptile learns an adaptable parameter initialisation, whereas ProtoNet learns an embedding space for prototype-based classification.

C. ProtoNet uses no support examples.

D. Reptile classifies queries using Euclidean distances to prototypes.

Correct answer: BExplanation: Reptile adapts the model through gradient-based fine-tuning. ProtoNet calculates class prototypes from support embeddings and classifies query examples using distances.

### Question 14. If ProtoNet achieves 65%, Reptile achieves 58%, and the baseline achieves 52% average meta-test query accuracy, which method performs best?

A. Baseline.

B. Reptile.

C. ProtoNet.

D. All perform equally.

Correct answer: CExplanation: ProtoNet achieves the highest average query accuracy at 65%.

The ranking is:

1. ProtoNet: 65%.
2. Reptile: 58%.
3. Baseline: 52%.

This suggests ProtoNet generalises best under these illustrative experimental conditions.

### Question 15. What should you conclude if the baseline outperforms both Reptile and ProtoNet?

A. The experiment must be incorrect.

B. The baseline should be reported as the best-performing method under the tested experimental conditions.

C. The meta-learning results should be removed.

D. ProtoNet must be more accurate theoretically.

Correct answer: BExplanation: Experimental results must be reported honestly. Meta-learning does not guarantee better performance than supervised pre-training. The outcome depends on factors such as the dataset, backbone, task distribution, and hyperparameters.

## Section D — Percentage points and relative improvement

### Question 16. The baseline achieves 55% accuracy and ProtoNet achieves 64%. What is the improvement in percentage points?

A. 9 percentage points.

B. 16.4 percentage points.

C. 64 percentage points.

D. 119 percentage points.

Correct answer: AExplanation:

$$
64\%-55\%=9
$$

The improvement is 9 percentage points.

Percentage points measure the absolute difference between two percentages.

### Question 17. The baseline achieves 55% accuracy and ProtoNet achieves 64%. What is the relative percentage improvement?

A. 9%.

B. 12.5%.

C. Approximately 16.4%.

D. 64%.

Correct answer: CExplanation:

Relative improvement is calculated as:

$$
\frac{64-55}{55}\times100
$$

$$
=\frac{9}{55}\times100
\approx16.4\%
$$

ProtoNet improves on the baseline by approximately 16.4% relative to the baseline's accuracy.

### Question 18. What is the difference between percentage points and relative percentage improvement?

A. They always represent the same quantity.

B. Percentage points measure the absolute difference, whereas relative improvement measures the difference relative to the original value.

C. Percentage points are used only for loss values.

D. Relative improvement is calculated by adding the two accuracies.

Correct answer: BExplanation: Consider an improvement from 50% to 60%.

- Percentage-point improvement: $60-50=10$ percentage points.
- Relative improvement: $\frac{60-50}{50}\times100=20\%$.

These are two different ways of describing the same performance change.

### Question 19. The baseline achieves 40% accuracy and Reptile achieves 50%. What is Reptile's relative percentage improvement?

A. 10%.

B. 20%.

C. 25%.

D. 50%.

Correct answer: CExplanation:

$$
\frac{50-40}{40}\times100
=
25\%
$$

Reptile achieves a 10-percentage-point improvement, equivalent to a 25% relative improvement over the baseline.

### Question 20. Why is it important to distinguish percentage-point improvement from relative percentage improvement in your report?

A. Because they communicate different quantities and prevent misleading interpretations.

B. Because percentage points are only used in Reptile.

C. Because relative improvement is always larger than the original accuracy.

D. Because accuracy cannot be compared between models.

Correct answer: AExplanation: Reporting the wrong quantity can exaggerate or misrepresent the improvement achieved by a model. Clearly distinguishing the two makes the experimental discussion more accurate.

## Section E — Writing the discussion and interpreting results

### Question 21. What should the discussion section of your report include?

A. Only the final accuracy of ProtoNet.

B. A summary of the methods, comparison of measured results, implementation challenges, and lessons learned.

C. Only screenshots of the notebook.

D. Only the Python source code.

Correct answer: BExplanation: A good discussion explains what was investigated, what the results show, how the approaches compare, what problems occurred, and what was learned from the practical.

### Question 22. Why should you describe implementation challenges in your discussion?

A. To demonstrate that you understand the practical work and can explain how problems were addressed.

B. To make the model achieve higher accuracy.

C. To avoid reporting experimental results.

D. To prove that every implementation was error-free.

Correct answer: AExplanation: Discussing genuine implementation challenges demonstrates practical understanding. Examples include tensor shape errors, incorrect label mappings, learning-rate problems, and checkpoint issues.

### Question 23. What should you do if ProtoNet does not outperform the baseline?

A. Change the reported accuracy.

B. Claim that ProtoNet performed best regardless of the results.

C. Report the actual results and discuss possible explanations.

D. Exclude the baseline from the comparison.

Correct answer: CExplanation: Scientific reporting requires honest interpretation. Possible explanations include the quality of the embedding space, initialisation, learning rate, regularisation, and number of meta-training episodes.

### Question 24. Which statement is the most appropriate conclusion for a practical report?

A. "ProtoNet is always better than every supervised learning model."

B. "Reptile must outperform the baseline because it is a meta-learning algorithm."

C. "Under the experimental conditions tested, ProtoNet achieved the highest average meta-test query accuracy, suggesting better generalisation in this task setup."

D. "The results do not matter because all methods use neural networks."

Correct answer: CExplanation: This conclusion is evidence-based and appropriately limited to the experimental conditions. It avoids making unsupported claims about the universal superiority of a method.

### Question 25. What is the most important overall lesson from Part 8?

A. The most important result is the model's training accuracy.

B. A model with more parameters always performs better.

C. Experimental results should be compared using average meta-test query accuracy, supported by plots and an honest discussion.

D. Meta-learning always outperforms supervised learning.

Correct answer: CExplanation: Part 8 focuses on presenting and interpreting experimental evidence. The objective is to compare how well the three approaches generalise to unseen few-shot tasks and explain the results accurately.

# Final revision summary

| Topic | What you must remember |
| --- | --- |
| Baseline plots | Training and validation loss and accuracy |
| Reptile plots | Meta-training and evaluation performance across episodes |
| ProtoNet plots | Meta-training and evaluation performance across episodes |
| Main evaluation metric | Average meta-test query accuracy |
| Performance comparison | Baseline vs Reptile vs ProtoNet |
| Percentage points | Absolute difference between accuracies |
| Relative improvement | Difference divided by baseline accuracy × 100 |
| Discussion | Methods, results, challenges and interpretation |
| Scientific reporting | Report actual results honestly |
| Final conclusion | Identify which method generalises best under the tested conditions |

## Five essential oral-exam questions

Make sure you can answer these without looking at your notes:

1. Why is average meta-test query accuracy more important than training accuracy?
2. What plots should you include for the baseline, Reptile, and ProtoNet?
3. What is the difference between percentage points and relative percentage improvement?
4. What should you discuss if ProtoNet performs worse than the baseline?
5. Why should conclusions be based on actual experimental results rather than theoretical expectations?

Key takeaway: Part 8 is about turning your implementation and experimental results into a clear, evidence-based scientific report. You must not only know how the models work, but also demonstrate which approach generalises best and explain why your results might have occurred.

When you're ready, we can move on to Part 9 — Advanced Understanding and Exam Preparation, using the same 25-question MCQ format.
