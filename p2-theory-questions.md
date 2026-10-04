 # Curriculum Learning & Semi-Supervised Learning — Question-Focused Exam-Ready Notes

 > **How to use these notes:** Cover the answers and try to answer each question yourself first.\
>  Questions marked **⭐** are especially important for an exam.

---

 # Part I — Curriculum Learning

 ## 1\. What is curriculum learning?

 **Answer:**\
 Curriculum learning is a training strategy where examples are presented to a model in a meaningful order, typically starting with **easier examples** and gradually introducing **harder examples**.

 The key idea is:

 > Instead of training on all examples with equal difficulty from the beginning, control the order in which the model encounters them.

 The analogy is human education: learn simple concepts before advanced ones.

---

 ## 2\. ⭐ Why might curriculum learning improve neural-network training?

 **Answer:**

 A curriculum can:

 - make early optimisation easier;
- provide more useful gradients during the initial stages of training;
- help the model learn simple patterns before complex ones;
- reduce the influence of noisy or ambiguous examples early on;
- potentially improve convergence speed;
- sometimes improve final generalisation.

 The important intuition is that **the training dynamics depend not only on what data the model sees, but also on when it sees it**.

---

 ## 3\. What is the difference between example difficulty and label quality?

 **Answer:**

 They are related but distinct.

 - **Example difficulty:** how difficult an input is for the model to learn correctly.
- **Label quality:** how trustworthy/correct the associated label is.

 For example:

 | Example | Difficulty | Label quality |
| --- | --- | --- |
| Clear image of a cat | Easy | High |
| Blurry image of a cat | Hard | High |
| Ambiguous image | Hard | High/uncertain |
| Clearly identifiable image with wrong label | Easy | Low |

Curriculum learning can exploit either **difficulty**, **label quality**, or both.

---

 ## 4\. ⭐ What does a curriculum define?

 **Answer:**

 A curriculum defines a schedule for the training data.

 Conceptually:

 $$
\text{easy examples}
\rightarrow
\text{medium examples}
\rightarrow
\text{hard examples}
$$

 The schedule can be defined using a measure of difficulty.

---

 ## 5\. How can we determine whether an example is "easy"?

 **Answer:**

 There is no universal definition of difficulty.

 Possible measures include:

 - model loss;
- prediction confidence;
- number of previous errors;
- human-provided difficulty;
- label quality;
- input complexity;
- distance from a prototype;
- uncertainty;
- teacher-model confidence.

 Therefore, curriculum learning is not simply "sort the dataset by difficulty"; we first need a **difficulty criterion**.

---

 ## 6\. ⭐ What is self-paced learning?

 **Answer:**

 Self-paced learning is closely related to curriculum learning, but instead of relying entirely on a predefined ordering, the **model itself determines which examples it is currently capable of learning from**.

 The model may initially select easy examples and gradually include harder ones.

 A simplified conceptual objective is:

 $$
\min_{\theta}
\sum_i v_i L_i(\theta)
+
\lambda R(v)
$$

 where:

 - $L\_i(\\theta)$ is the loss for example $i$;
- $v\_i$ determines whether/how strongly example $i$ is selected;
- $\\lambda$ controls how aggressively examples are included.

---

 ## 7\. What is the difference between curriculum learning and self-paced learning?

 **Answer:**

 | Curriculum learning | Self-paced learning |
| --- | --- |
| Training order is often externally specified | Model determines selection |
| Can use known difficulty | Usually based on current model loss/confidence |
| Teacher-designed schedule is common | Adaptive schedule |
| "Learn easy → hard" | "Learn what I can currently handle" |

Both exploit the principle that **training on appropriately chosen examples can improve optimisation**.

---

 ## 8\. ⭐ What is the main danger of an overly aggressive curriculum?

 **Answer:**

 If we train too long only on easy examples, the model may:

 - fail to learn difficult regions of the data distribution;
- overfit the easy subset;
- develop biased representations;
- never receive sufficient gradient information from hard examples.

 Therefore, a good curriculum must eventually expose the model to the **full relevant data distribution**.

---

 # Part II — Semi-Supervised Learning

 ## 9\. ⭐ What is semi-supervised learning?

 **Answer:**

 Semi-supervised learning uses a combination of:

 - a relatively small labelled dataset;
- a larger unlabelled dataset.

 We can write:

 $$
\mathcal{D}_L = \{(x_i,y_i)\}
$$

 for labelled data and

 $$
\mathcal{D}_U = \{x_j\}
$$

 for unlabelled data.

 The objective is to use both datasets to improve the learned model.

---

 ## 10\. Why is semi-supervised learning useful?

 **Answer:**

 In many real-world problems:

 > Obtaining labels is expensive, but obtaining raw data is cheap.

 For example, collecting millions of images may be easy, while manually labelling every image may be expensive.

 Semi-supervised learning attempts to exploit the information contained in the large unlabelled dataset.

---

 ## 11\. ⭐ What is the basic semi-supervised learning objective?

 **Answer:**

 The total objective typically contains:

 $$
\mathcal{L} = \mathcal{L}_{\text{sup}} + \lambda\mathcal{L}_{\text{unsup}}
$$

 where:

 - $\\mathcal{L}\_{\\text{sup}}$ is computed using labelled examples;
- $\\mathcal{L}\_{\\text{unsup}}$ exploits unlabelled examples;
- $\\lambda$ controls the importance of the unsupervised objective.

 For classification:

 $$
\mathcal{L}_{\text{sup}} = \operatorname{CE}(y_i,f_\theta(x_i))
$$

---

 ## 12\. What is the key assumption behind semi-supervised learning?

 **Answer:**

 Semi-supervised learning assumes that the unlabelled data contains useful information about the underlying task.

 Common assumptions include:

 - **smoothness assumption:** nearby examples should have similar predictions;
- **cluster assumption:** examples in the same cluster tend to have the same label;
- **manifold assumption:** data lies on a lower-dimensional structure;
- **low-density separation:** decision boundaries should avoid dense regions of the data.

---

 # Part III — Self-Training

 ## 13\. ⭐ What is self-training?

 **Answer:**

 Self-training is a simple semi-supervised technique where the model:

 1. trains on labelled data;
2. predicts labels for unlabelled examples;
3. treats some predictions as **pseudo-labels**;
4. trains on those pseudo-labelled examples;
5. repeats.

---

 ## 14\. What is a pseudo-label?

 **Answer:**

 A pseudo-label is a label generated by the model rather than provided by a human.

 For an unlabelled example $x\_j$:

 $$
\hat{y}_j = \arg\max_y f_\theta(x_j)_y
$$

 The model's most likely class becomes the pseudo-label.

---

 ## 15\. ⭐ What is the loss for basic self-training?

 **Answer:**

 A simplified objective is:

 $$
\mathcal{L} = \operatorname{CE}(y_i,f_\theta(x_i)) + \operatorname{CE}(\hat{y}_j,f_\theta(x_j))
$$

 The first term is the supervised loss and the second is the pseudo-labelled unsupervised loss.

---

 ## 16\. Why is naive self-training dangerous?

 **Answer:**

 Because the model can make incorrect predictions and then **train on its own mistakes**.

 This creates a feedback loop:

 $$
\text{wrong prediction}
\rightarrow
\text{wrong pseudo-label}
\rightarrow
\text{training on wrong label}
\rightarrow
\text{stronger wrong prediction}
$$

 This is called **confirmation bias**.

---

 ## 17\. ⭐ What is mode collapse in the context of naive self-training?

 **Answer:**

 Mode collapse occurs when the model increasingly predicts a limited set of classes or behaviours, potentially ignoring other modes/classes in the data.

 Because the model generates its own training labels, errors can reinforce themselves.

 This is one reason why semi-supervised learning requires a **stability mechanism**.

---

 # Part IV — Co-Training

 ## 18\. ⭐ What is co-training?

 **Answer:**

 Co-training uses **two different models** that learn from each other.

 Each model:

 1. learns from labelled data;
2. predicts labels for unlabelled data;
3. provides confident predictions to the other model;
4. the other model uses those predictions as additional supervision.

---

 ## 19\. Why can co-training be more stable than naive self-training?

 **Answer:**

 The two models ideally make **different errors** because they use different views/features of the data.

 Therefore:

 > One model can provide useful information to the other without simply reinforcing exactly the same mistakes.

 The key requirement is some form of **independence or diversity** between the models/views.

---

 ## 20\. ⭐ What is the central problem that co-training and consistency regularisation are trying to solve?

 **Answer:**

 Both address the instability of naive self-training.

 The central problem is:

 $$
\text{model's prediction}
\rightarrow
\text{training target}
$$

 If the prediction is wrong, the model can reinforce its own error.

 Co-training addresses this using **different models/views**.

 Consistency regularisation addresses it using **different perturbations of the same input**.

---

 # Part V — Consistency Regularisation

 ## 21\. ⭐ What is consistency regularisation?

 **Answer:**

 Consistency regularisation forces a model to produce similar predictions for different perturbed versions of the same input.

 If $x$ and $x'$ are two versions of the same example:

 $$
f_\theta(x)
\approx
f_\theta(x')
$$

 The assumption is:

 > Small or semantically irrelevant changes to an input should not change its predicted class.

---

 ## 22\. What is the general consistency objective?

 **Answer:**

 A generic formulation is:

 $$
\mathcal{L}_{\text{cons}} = d\left(f_\theta(x),f_\theta(x')\right)
$$

 where $d$ is a distance between predictions.

 For example:

 $$
\mathcal{L}_{\text{cons}} = \operatorname{MSE}\left(f_\theta(x),f_\theta(x')\right)
$$

 or a cross-entropy/KL-divergence-based objective.

---

 ## 23\. ⭐ Why does consistency regularisation work particularly well with unlabelled data?

 **Answer:**

 An unlabelled example does not provide a ground-truth $y$, but it **does provide multiple views of the same underlying example**.

 Therefore we can still impose:

 $$
f_\theta(x)
\approx
f_\theta(\operatorname{augment}(x))
$$

 without knowing the true class.

---

 # Part VI — $\\Pi$-Model

 ## 24\. ⭐ What is the $\\Pi$-model?

 **Answer:**

 The $\\Pi$-model is an early implementation of consistency regularisation.

 The model applies stochastic perturbations/noise and encourages the predictions to agree.

 Conceptually:

 $$
x+\eta
\quad\text{and}\quad
x+\eta'
$$

 are two perturbed versions of the same input.

 The model minimises a consistency loss between their predictions.

---

 ## 25\. What is the main idea of the $\\Pi$-model in one sentence?

 **Answer:**

 > **The model should give the same answer when the same input is subjected to different stochastic perturbations.**

---

 # Part VII — UDA

 ## 26\. ⭐ What is UDA?

 **Answer:**

 UDA stands for **Unsupervised Data Augmentation**.

 It applies consistency training by comparing:

 - an original/unmodified example;
- a strongly augmented version of the same example.

---

 ## 27\. How does UDA treat labelled and unlabelled examples differently?

 **Answer:**

 For labelled examples, we can directly use the ground-truth label:

 $$
\mathcal{L}_{\text{sup}} = \operatorname{CE}(y,f_\theta(x))
$$

 For unlabelled examples, we impose consistency:

 $$
\mathcal{L}_{\text{unsup}} = d\left(f_\theta(x),f_\theta(\operatorname{augment}(x))\right)
$$

---

 ## 28\. ⭐ Why does UDA only apply the consistency loss when the original prediction is sufficiently confident?

 **Answer:**

 Because the original prediction is being used as a target.

 If:

 $$
f_\theta(x)
$$

 is uncertain or incorrect, forcing the augmented example to match it can reinforce a bad prediction.

 Therefore, UDA uses **confidence thresholding**.

 Conceptually:

 $$
\text{if confidence}(f_\theta(x)) > \tau:
\quad
\text{apply consistency loss}
$$

 Otherwise, ignore the unlabelled example for that update.

---

 # Part VIII — Stability in Semi-Supervised Learning

 ## 29\. ⭐ Why is stability so important in semi-supervised learning?

 **Answer:**

 Because the model is partially training against targets that are themselves generated by models.

 This creates a potential feedback loop:

 $$
\text{prediction}
\rightarrow
\text{pseudo-target}
\rightarrow
\text{training}
\rightarrow
\text{new prediction}
$$

 Without stabilisation, small errors can be amplified.

---

 ## 30\. What are the main stability mechanisms discussed in the lecture?

 **Answer:**

 Three major approaches are:

 1. **Co-training**
   - Use different models/views.
2. **Consistency regularisation**
   - Require predictions to be stable under perturbations.
3. **Temporal ensembling / Mean Teacher**
   - Construct more stable targets using information from previous model states.

---

 # Part IX — Temporal Ensembling

 ## 31\. ⭐ What problem does temporal ensembling solve?

 **Answer:**

 Consistency regularisation has a problem:

 > The target prediction and the current prediction may both be unreliable, especially early in training.

 Temporal ensembling makes the target more stable by averaging predictions from previous training steps.

---

 ## 32\. What is the temporal ensembling update?

 **Answer:**

 The lecture gives an exponential moving average (EMA):

 $$
\tilde{z}_i^{\,t} = \alpha \tilde{z}_i^{\,t-1} + (1-\alpha)z_i^t
$$

 where:

 - $z\_i^t$ is the current prediction;
- $\\tilde{z}\_i^{,t}$ is the smoothed prediction;
- $\\alpha$ controls how strongly we retain the previous estimate.

 The lecture gives approximately:

 $$
\alpha \approx 0.6
$$

---

 ## 33\. ⭐ Why is this called an exponential moving average?

 **Answer:**

 Repeatedly expanding the recurrence gives:

 $$
\tilde{z}^{\,t} = (1-\alpha)z^t + \alpha(1-\alpha)z^{t-1} + \alpha^2(1-\alpha)z^{t-2}+\cdots
$$

 Therefore, older predictions receive exponentially decreasing weights.

---

 ## 34\. Why does temporal averaging stabilise predictions?

 **Answer:**

 Individual predictions can be noisy.

 Averaging across time smooths this noise:

 $$
\text{noisy predictions}
\rightarrow
\text{EMA}
\rightarrow
\text{stable target}
$$

 The temporal lag also creates a form of resistance against sudden noisy changes.

---

 ## 35\. ⭐ What is the connection between EMA and Adam?

 **Answer:**

 Adam also uses exponential moving averages to smooth quantities such as gradients and squared gradients.

 The general EMA mechanism is:

 $$
m_t = \beta m_{t-1} + (1-\beta)x_t
$$

 The reason is similar: **smooth noisy quantities using information from previous steps**.

---

 # Part X — Limitations of Temporal Ensembling

 ## 36\. ⭐ What are the two main problems with temporal ensembling?

 **Answer:**

 ### 1\. Poor scaling with dataset size

 We need to store a moving-average prediction $\\tilde{z}\_i$ for **every training example**.

 For a huge dataset, this becomes expensive.

 ### 2\. Updates only once per epoch

 The stored prediction for an example is updated only when that example is encountered.

 Therefore, the target can lag behind the rapidly changing model.

---

 # Part XI — Mean Teacher

 ## 37\. ⭐ What is the Mean Teacher algorithm?

 **Answer:**

 Mean Teacher replaces the stored EMA prediction for every individual example with an **EMA of the model parameters**.

 Instead of:

 $$
\tilde{z}_i = \operatorname{EMA}(\text{past predictions})
$$

 we maintain:

 $$
\theta' = \operatorname{EMA}(\theta)
$$

 where:

 - $\\theta$ = student parameters;
- $\\theta'$ = teacher parameters.

---

 ## 38\. ⭐ Why is the second model called the "teacher"?

 **Answer:**

 The teacher generates the targets that the student is trained to match.

 Conceptually:

 $$
\boxed{
\text{teacher}
\rightarrow
\text{stable target}
\rightarrow
\text{student}
}
$$

 The student is updated using gradient descent, while the teacher is updated using an EMA of the student.

---

 ## 39\. What is the Mean Teacher parameter update?

 **Answer:**

 The teacher parameters are updated using:

 $$
\theta'
\leftarrow
\alpha\theta'
+
(1-\alpha)\theta
$$

 Thus, the teacher is a smoothed version of the student's historical parameters.

---

 ## 40\. ⭐ What is the Mean Teacher consistency loss?

 **Answer:**

 The lecture gives the objective:

 $$
\mathcal{L} = \operatorname{CE}\left(y_i,f_\theta(x_i)\right) + \operatorname{MSE}\left(f_\theta(x_j+\eta),f_{\theta'}(x_j+\eta')\right)
$$

 where:

 - $f\_\\theta$ is the student;
- $f\_{\\theta'}$ is the teacher;
- $\\eta,\\eta'$ are perturbations/noise.

 The first term is supervised learning.

 The second term is consistency training.

---

 ## 41\. ⭐ Why is Mean Teacher more scalable than temporal ensembling?

 **Answer:**

 Temporal ensembling stores an EMA prediction for every example:

 $$
\{\tilde{z}_1,\tilde{z}_2,\ldots,\tilde{z}_N\}
$$

 Mean Teacher instead stores one additional model:

 $$
\theta'
$$

 Therefore, it does not need a separate prediction vector for every training example.

---

 ## 42\. Why does Mean Teacher provide more up-to-date targets?

 **Answer:**

 The teacher parameters are updated continuously:

 $$
\theta'
\leftarrow
\alpha\theta'
+
(1-\alpha)\theta
$$

 Thus the teacher can generate a new target every batch/example.

 Temporal ensembling can only update the stored prediction when the corresponding example is encountered.

---

 # Part XII — Comparing the Main Semi-Supervised Methods

 ## 43\. ⭐ How does online self-training work?

 **Answer:**

 For an unlabelled example $x\_j$:

 $$
\hat{y}_j = \arg\max f_\theta(x_j)
$$

 Then:

 $$
\mathcal{L} = \operatorname{CE}(y_i,f_\theta(x_i)) + \operatorname{CE}(\hat{y}_j,f_\theta(x_j))
$$

 The model uses its own prediction as a pseudo-label.

---

 ## 44\. ⭐ How does consistency regularisation differ from self-training?

 **Answer:**

 Self-training says:

 > "My predicted class is the target."

 Consistency regularisation says:

 > "My prediction should remain stable under perturbations."

 Self-training:

 $$
\hat{y} = \arg\max f_\theta(x)
$$

 Consistency:

 $$
f_\theta(x)
\approx
f_\theta(x+\eta)
$$

 The second approach does not necessarily require converting the prediction into a hard class label.

---

 ## 45\. ⭐ How does temporal ensembling differ from ordinary consistency regularisation?

 **Answer:**

 Ordinary consistency regularisation compares the current prediction with another prediction:

 $$
f_\theta(x+\eta)
\approx
f_\theta(x+\eta')
$$

 Temporal ensembling uses a **historically averaged target**:

 $$
f_\theta(x+\eta)
\approx
\tilde{z}
$$

 where $\\tilde{z}$ is an EMA of previous predictions.

 Thus temporal ensembling makes the target more stable.

---

 ## 46\. ⭐ How does Mean Teacher improve upon temporal ensembling?

 **Answer:**

 It moves the EMA from the **predictions** to the **model parameters**.

 ### Temporal ensembling

 $$
\tilde{z}_i = \operatorname{EMA}(\text{predictions for example }i)
$$

 ### Mean Teacher

 $$
\theta' = \operatorname{EMA}(\theta)
$$

 The teacher can therefore generate fresh predictions at every update.

---

 ## 47\. Can you summarise the evolution of the methods?

 **Answer:**

 A useful conceptual progression is:

 $$
\boxed{
\text{Self-training}
\rightarrow
\text{Consistency}
\rightarrow
\text{Temporal Ensembling}
\rightarrow
\text{Mean Teacher}
}
$$

 ### Self-training

 Use the model's own prediction as a label.

 ### Consistency

 Force predictions to agree under perturbations.

 ### Temporal ensembling

 Make the target more stable by averaging predictions over time.

 ### Mean Teacher

 Make the target model itself an EMA of previous student models.

---

 # Part XIII — The Timeline Formula

 ## 48\. ⭐ What is the general supervised \+ unsupervised structure shared by these methods?

 **Answer:**

 The overall objective is generally:

 $$
\mathcal{L} = \mathcal{L}_{\text{sup}} + \lambda\mathcal{L}_{\text{unsup}}
$$

 with:

 $$
\mathcal{L}_{\text{sup}} = \operatorname{CE} (y_i,f_\theta(x_i))
$$

 The difference between methods lies mainly in how $\\mathcal{L}\_{\\text{unsup}}$ is constructed.

---

 ## 49\. What is the unsupervised loss for online self-training?

 **Answer:**

 Using a pseudo-label:

 $$
\mathcal{L}_{\text{unsup}} = \operatorname{CE}\left(\hat{y}_j,f_\theta(x_j)\right)
$$

 where:

 $$
\hat{y}_j = \arg\max f_\theta(x_j)
$$

---

 ## 50\. What is the unsupervised loss for consistency regularisation/UDA?

 **Answer:**

 A conceptual form is:

 $$
\mathcal{L}_{\text{unsup}} = \operatorname{CE}\left(f_\theta(x_j),f_\theta(x_j+\eta)\right)
$$

 or another suitable distance between the two predictions.

 The important idea is:

 $$
\boxed{
\text{same input + different perturbation}
\Rightarrow
\text{same prediction}
}
$$

---

 ## 51\. What is the unsupervised loss for temporal ensembling?

 **Answer:**

 The lecture gives:

 $$
\mathcal{L}_{\text{unsup}} = \operatorname{MSE}\left(f_\theta(x_j+\eta),\tilde{z}_j\right)
$$

 where $\\tilde{z}\_j$ is the EMA target based on previous predictions.

---

 ## 52\. What is the unsupervised loss for Mean Teacher?

 **Answer:**

 The lecture gives:

 $$
\mathcal{L}_{\text{unsup}} = \operatorname{MSE}\left(f_\theta(x_j+\eta),f_{\theta'}(x_j+\eta')\right)
$$

 where:

 $$
\theta' = \operatorname{EMA}(\theta)
$$

---

 # Part XIV — Exam Comparison Questions

 ## 53\. ⭐ Compare self-training, UDA, temporal ensembling and Mean Teacher.

 **Answer:**

 | Method | Target | Main stabilisation idea |
| --- | --- | --- |
| Self-training | Model's own pseudo-label | Confidence filtering can help |
| UDA | Prediction on original/weakly perturbed input | Consistency under augmentation |
| Temporal ensembling | EMA of past predictions | Smooth target over time |
| Mean Teacher | EMA teacher model prediction | Smooth model parameters |

---

 ## 54\. ⭐ Which method stores predictions for each training example?

 **Answer:**

 **Temporal ensembling.**

 It maintains:

 $$
\tilde{z}_i
$$

 for each example $i$.

 Mean Teacher avoids this by maintaining an EMA model.

---

 ## 55\. ⭐ Which method maintains a separate teacher model?

 **Answer:**

 **Mean Teacher.**

 The teacher parameters are:

 $$
\theta' = \operatorname{EMA}(\theta)
$$

---

 ## 56\. Which method uses two different models that teach one another?

 **Answer:**

 **Co-training.**

 The key idea is that the models should ideally make different errors because they have different views or representations.

---

 ## 57\. Which method is most directly based on data augmentation?

 **Answer:**

 **UDA — Unsupervised Data Augmentation.**

 It explicitly compares predictions for an original/weakly transformed example and a strongly augmented version.

---

 # Part XV — Active Learning

 ## 58\. ⭐ What is active learning?

 **Answer:**

 Active learning strategically chooses which unlabelled examples should be labelled by a human.

 The loop is:

 $$
\text{unlabelled data}
\rightarrow
\text{select informative examples}
\rightarrow
\text{human labels them}
\rightarrow
\text{retrain}
$$

---

 ## 59\. ⭐ How is active learning different from semi-supervised learning?

 **Answer:**

 ### Semi-supervised learning

 Uses unlabelled data **without necessarily obtaining new labels**.

 $$
\text{labelled + unlabelled}
\rightarrow
\text{model}
$$

 ### Active learning

 Selects examples and **requests labels** for the most informative ones.

 $$
\text{unlabelled}
\rightarrow
\text{select}
\rightarrow
\text{human annotation}
\rightarrow
\text{new labelled data}
$$

---

 ## 60\. How is active learning related to curriculum learning?

 **Answer:**

 Both strategically control which examples are used during training.

 - Curriculum learning chooses examples based on **difficulty/quality/order**.
- Active learning chooses examples based on **informativeness/value of obtaining a label**.

---

 # Part XVI — High-Value "Explain Why" Questions

 ## 61\. ⭐ Why can confidence thresholding improve pseudo-labelling?

 **Answer:**

 A high-confidence prediction is more likely to be correct.

 If:

 $$
\max_y f_\theta(x) > \tau
$$

 we accept the pseudo-label.

 Otherwise, we discard it.

 This reduces the probability of feeding incorrect pseudo-labels back into training.

---

 ## 62\. ⭐ Why is confidence thresholding not a perfect solution?

 **Answer:**

 Because neural networks can be **confident and wrong**.

 Therefore:

 $$
\text{high confidence}
\not\Rightarrow
\text{guaranteed correctness}
$$

 It reduces noisy labels but does not eliminate confirmation bias.

---

 ## 63\. ⭐ Why does consistency regularisation rely on augmentation being meaningful?

 **Answer:**

 The augmentation must preserve the semantic label.

 If:

 $$
y(x) \neq y(\operatorname{augment}(x))
$$

 then forcing:

 $$
f(x)
\approx
f(\operatorname{augment}(x))
$$

 would be harmful.

 Therefore, augmentations should change irrelevant properties while preserving the underlying class.

---

 ## 64\. Why can consistency regularisation fail early in training?

 **Answer:**

 Early in training, the model's predictions may be unreliable.

 If both predictions are bad:

 $$
f_\theta(x)
\approx
f_\theta(x+\eta)
$$

 the model may simply learn to be **consistently wrong**.

 This motivates stabilisation techniques such as temporal ensembling and Mean Teacher.

---

 ## 65\. ⭐ Why does Mean Teacher provide a more stable target than the student?

 **Answer:**

 The teacher is an EMA of the student:

 $$
\theta' = \alpha\theta'_{\text{old}} + (1-\alpha)\theta
$$

 Therefore, sudden changes in the student are smoothed out.

 The teacher changes more slowly and therefore provides a more stable target.

---

 # Part XVII — Conceptual Exam Traps

 ## 66\. ⭐ Is semi-supervised learning the same as self-supervised learning?

 **Answer:**

 No.

 ### Semi-supervised learning

 Uses some labelled examples and some unlabelled examples:

 $$
\boxed{
\text{labelled + unlabelled}
}
$$

 ### Self-supervised learning

 Constructs a learning signal from the data itself, typically without human labels.

 The lecture notes that consistency regularisation can be generalised into more powerful self-supervised approaches.

---

 ## 67\. Is consistency regularisation the same as pseudo-labelling?

 **Answer:**

 No.

 Pseudo-labelling typically creates a hard target:

 $$
\hat{y} = \arg\max f(x)
$$

 Consistency regularisation instead compares predictions:

 $$
f(x)
\approx
f(x')
$$

 It can therefore retain more information from the model's probability distribution.

---

 ## 68\. Is Mean Teacher the same as training two independent models?

 **Answer:**

 No.

 In Mean Teacher:

 - the student is trained using gradient descent;
- the teacher is an EMA of the student.

 They are therefore **not independently trained models**.

 $$
\theta'\leftarrow\alpha\theta' + (1-\alpha)\theta
$$

---

 ## 69\. ⭐ Does the teacher receive gradient updates?

 **Answer:**

 No, conceptually the teacher is updated through the EMA mechanism rather than ordinary backpropagation.

 The student receives the gradient from the supervised and consistency losses.

---

 ## 70\. Why is the teacher called a "moving target"?

 **Answer:**

 Because its predictions change as the teacher model parameters change.

 However, because the teacher is an EMA of previous students, it moves more slowly and smoothly than the current student.

---

 # Part XVIII — Short Exam Questions

 ## 71\. Define curriculum learning in one sentence.

 **Answer:**\
 Curriculum learning trains a model using examples in a strategically chosen order, often from easy to difficult.

---

 ## 72\. Define semi-supervised learning in one sentence.

 **Answer:**\
 Semi-supervised learning combines labelled and unlabelled data to improve model learning.

---

 ## 73\. Define pseudo-labelling.

 **Answer:**\
 Pseudo-labelling treats a model's prediction on an unlabelled example as a temporary training label.

---

 ## 74\. Define consistency regularisation.

 **Answer:**\
 Consistency regularisation encourages a model to produce similar predictions for different perturbations of the same input.

---

 ## 75\. Define temporal ensembling.

 **Answer:**\
 Temporal ensembling creates a stable target by maintaining an exponential moving average of predictions over previous training iterations.

---

 ## 76\. Define Mean Teacher.

 **Answer:**\
 Mean Teacher maintains a teacher model whose parameters are an exponential moving average of the student's parameters.

---

 ## 77\. Define active learning.

 **Answer:**\
 Active learning selects informative unlabelled examples for human annotation.

---

 # Part XIX — Formula Sheet

 ## 78\. ⭐ What formulas should you know?

 ### General semi-supervised objective

 $$
\boxed{\mathcal{L} = \mathcal{L}_{\text{sup}} + \lambda\mathcal{L}_{\text{unsup}}}
$$

 ### Supervised classification loss

 $$
\boxed{\mathcal{L}_{\text{sup}} = \operatorname{CE}(y_i,f_\theta(x_i))}
$$

 ### Pseudo-label

 $$
\boxed{\hat{y}_j = \arg\max_y f_\theta(x_j)_y}
$$

 ### Self-training

 $$
\boxed{\mathcal{L}_{\text{unsup}} = \operatorname{CE}(\hat{y}_j,f_\theta(x_j))}
$$

 ### Consistency regularisation

 $$
\boxed{\mathcal{L}_{\text{cons}} = d\left(f_\theta(x),f_\theta(x')\right)}
$$

 ### UDA

 $$
\boxed{
f_\theta(x)
\approx
f_\theta(\operatorname{augment}(x))
}
$$

 ### Temporal ensembling

 $$
\boxed{\tilde{z}_i^{\,t} = \alpha\tilde{z}_i^{\,t-1} + (1-\alpha)z_i^t}
$$

 ### Temporal ensembling loss

 $$
\boxed{\mathcal{L}_{\text{unsup}} = \operatorname{MSE}\left(f_\theta(x_j+\eta),\tilde{z}_j\right)}
$$

 ### Mean Teacher parameter update

 $$
\boxed{\theta' = \alpha\theta'_{\text{old}} + (1-\alpha)\theta}
$$

 ### Mean Teacher loss

 $$
\boxed{\mathcal{L}_{\text{unsup}} = \operatorname{MSE} \left(f_\theta(x_j+\eta), f_{\theta'}(x_j+\eta') \right)}
$$

---

 # Part XX — The "Explain the Whole Lecture" Question

 ## 79\. ⭐ If asked to explain the evolution of semi-supervised learning methods, what should you say?

 **Answer:**

 Start with the fundamental problem:

 > We have limited labelled data but lots of unlabelled data.

 A simple solution is **self-training**: let the model generate pseudo-labels for unlabelled examples.

 The problem is **confirmation bias**: incorrect predictions become training targets and can reinforce themselves.

 One solution is **co-training**, where different models/views teach each other and hopefully make independent errors.

 Another solution is **consistency regularisation**, which does not require a human label. Instead, it requires the model to make consistent predictions for different perturbations of the same input.

 The $\\Pi$-model is an early implementation of this idea, while **UDA** uses strong data augmentation and confidence filtering.

 However, consistency targets can themselves be unreliable early in training. **Temporal ensembling** improves stability by using an EMA of historical predictions.

 Temporal ensembling has two major limitations: storing a prediction for every example and updating those predictions relatively infrequently.

 **Mean Teacher** solves these problems by maintaining an EMA of the model parameters instead:

 $$
\theta' = \operatorname{EMA}(\theta)
$$

 The teacher provides stable targets while the student is trained using both supervised and consistency losses.

 The overall evolution is therefore:

 $$
\boxed{
\text{Self-training}
\rightarrow
\text{Consistency}
\rightarrow
\text{Temporal Ensembling}
\rightarrow
\text{Mean Teacher}
}
$$

---

 # Part XXI — 30-Second Memory Map

 ## 80\. ⭐ Can you remember the entire topic using one diagram?

 **Answer:**

```
                    SEMI-SUPERVISED LEARNING
                              |
               +--------------+--------------+
               |                             |
          Labelled data                 Unlabelled data
               |                             |
               +--------------+--------------+
                              |
                     How do we exploit it?
                              |
          +-------------------+-------------------+
          |                   |                   |
     Self-training        Consistency          Co-training
          |              regularisation             |
     Pseudo-labels            |              Two models/views
          |             +-----+-----+
   Confirmation bias     |           |
                       UDA       Pi-model
                                  |
                           unstable targets
                                  |
                         Temporal Ensembling
                                  |
                        EMA predictions
                                  |
                         scaling / lag issues
                                  |
                            Mean Teacher
                                  |
                           EMA parameters
                                  |
                     +------------+------------+
                     |                         |
                  Student                  Teacher
                     |                         |
               gradient update             EMA update
                     |                         |
                     +------ consistency -----+
```

---

 # Part XXII — Final Exam Checklist

 Before the exam, make sure you can answer these **without looking at the notes**:

 - [ ] What is curriculum learning?
- [ ] Why can easy-to-hard training help optimisation?
- [ ] What is self-paced learning?
- [ ] What is semi-supervised learning?
- [ ] What assumptions make semi-supervised learning possible?
- [ ] What is self-training?
- [ ] What is a pseudo-label?
- [ ] Why does naive self-training suffer from confirmation bias?
- [ ] What is mode collapse?
- [ ] What is co-training?
- [ ] Why can two models be more stable than one?
- [ ] What is consistency regularisation?
- [ ] Why should predictions be invariant to perturbations?
- [ ] What is the $\\Pi$-model?
- [ ] What is UDA?
- [ ] Why does UDA use confidence thresholding?
- [ ] Why can consistency regularisation be unstable early in training?
- [ ] What is temporal ensembling?
- [ ] What is an EMA?
- [ ] Derive/explain the EMA equation.
- [ ] Why does temporal ensembling stabilise predictions?
- [ ] Why does temporal ensembling scale poorly?
- [ ] What is Mean Teacher?
- [ ] How is the teacher updated?
- [ ] Why is the teacher more stable than the student?
- [ ] Why does Mean Teacher scale better than temporal ensembling?
- [ ] What is the Mean Teacher loss?
- [ ] Compare self-training, UDA, temporal ensembling and Mean Teacher.
- [ ] What is active learning?
- [ ] How does active learning differ from semi-supervised learning?
- [ ] Explain the historical progression from self-training to Mean Teacher.

---

 # One-Minute Final Summary

 The central problem is:

 $$
\boxed{
\text{few labels} + \text{many unlabelled examples}
}
$$

 The simplest solution is **self-training**:

 $$
\text{prediction}
\rightarrow
\text{pseudo-label}
$$

 but this suffers from **confirmation bias**.

 **Consistency regularisation** instead says:

 $$
\boxed{
f(x) \approx f(\operatorname{augment}(x))
}
$$

 The problem is that the target may still be unreliable.

 **Temporal ensembling** stabilises the target:

 $$
\boxed{\tilde z_t = \alpha\tilde z_{t-1} + (1-\alpha)z_t}
$$

 but storing predictions for every example does not scale well.

 **Mean Teacher** moves the EMA from predictions to model parameters:

 $$
\boxed{\theta' = \alpha\theta' + (1-\alpha)\theta}
$$

 and trains the student to match the teacher:

 $$
\boxed{\mathcal{L} = \mathcal{L}_{\text{sup}} + \lambda\mathcal{L}_{\text{consistency}}}
$$

 The single most important conceptual progression is:

 $$
\boxed{\text{Self-training}\rightarrow\text{Consistency}\rightarrow\text{Temporal Ensembling}\rightarrow\text{Mean Teacher}}
$$

 while **curriculum learning** provides the broader idea of strategically controlling which examples the model learns from and when.

 This is structured so you can save it directly as something like `curriculum-semi-supervised-exam-questions.md` and have the equations render correctly on GitHub.
