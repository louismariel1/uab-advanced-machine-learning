# Ensemble Learning — Answers and Explanations

## Answer key

| Question | Correct answer | Main concept                     |
| -------- | -------------- | -------------------------------- |
| 1        | B              | Combining predictions            |
| 2        | D              | Diversity and error correlation  |
| 3        | C              | Correlated errors                |
| 4        | A              | Variance reduction               |
| 5        | B              | Ensembles for linear regression  |
| 6        | D              | Ensembles for decision trees     |
| 7        | B              | Bagging, boosting, and averaging |

## Detailed explanations

### Question 1 — Why does averaging improve accuracy?

Correct answer: B. The models can make different errors, allowing their predictions to complement one another and reduce aggregate error.

Each network learns a slightly different representation because of random initialization, stochastic optimization, or other sources of variation. When their predictions are combined, some individual errors may cancel out.

For example, three classifiers might predict the correct class with probabilities:

- Model 1: 0.55
- Model 2: 0.80
- Model 3: 0.75

Their average probability is \\(0.70\\), potentially giving a more reliable prediction than an individual model.

The improvement is not guaranteed: it depends on the models' errors and the aggregation method.

### Question 2 — Why is diversity important?

Correct answer: D. Diverse models can make different errors, allowing one model's correct prediction to compensate for another's mistake.

If all models make exactly the same predictions, averaging them does not change the result.

An effective ensemble therefore benefits from two properties:

- Individual competence: the models should make reasonably good predictions.
- Diversity: their errors should not be too strongly correlated.

The ideal ensemble combines strong individual models whose mistakes differ.

### Question 3 — What if all models make the same mistake?

Correct answer: C. The ensemble will likely remain wrong because the models agree on the same incorrect answer.

Averaging is not a mechanism that automatically discovers the truth. If all five networks confidently predict the same incorrect class, their averaged probabilities will probably preserve that error.

This illustrates why the number of models alone is insufficient. Adding more highly correlated models often yields diminishing returns.

### Question 4 — How Does Averaging Reduce Variance?

**Correct answer: A.**

For \(M\) predictors with equal error variance \(\sigma^2\) and pairwise error correlation \(\rho\), the variance of their average is

\[
\operatorname{Var}(\bar f)
= \sigma^2\left(\rho+\frac{1-\rho}{M}\right).
\]

This equation assumes equal variances and equal pairwise error correlations.

- If errors are independent, \(\rho = 0\), and the variance decreases to \(\sigma^2/M\).
- If errors are perfectly correlated, \(\rho = 1\), and the variance remains \(\sigma^2\).
- If errors are positively correlated but not perfectly correlated, averaging can still reduce variance, although less dramatically.

This is one of the central theoretical explanations for ensemble learning.

### Question 5 — Can linear regression be ensembled?

Correct answer: B. Averaging predictions from multiple fitted regressors can reduce variance when their errors differ.

For example, train several linear regression models on different bootstrap samples of the training data, then average their predictions:

\\[ \hat y\_{\text{ensemble}}(x) = \frac{1}{M}\sum\_{m=1}^{M}\hat y_m(x). \\]

This is a valid regression ensemble.

However, if every model is trained on identical data with an identical deterministic procedure and produces exactly the same fitted coefficients, averaging provides no improvement. An ensemble also does not necessarily improve bias or accuracy.

### Question 6 — Can decision trees be ensembled?

Correct answer: D. Random forests combine trees using voting or averaging, while boosting combines trees sequentially.

[A Guide to Random Forest in Machine Learning | EJable](https://images.openai.com/static-rsc-4/fl-v2RxUl45Rxdto_QvjS_KzjbO0o0HAsfuXurw1AFBQ8mKm2dl8RbHfqTF-kOFRIqFrLR4QRdPXm-zM3sER60h4gX0aq0gDN2SWWzZBGafRBSksfCJvmUSSJgNHcx8Dc99KcNBmRCRL11mAvSsGkYfkuBcgUxdZiH75Pswbf20?purpose=inline)

Random forest — bagging

Multiple decision trees are trained using randomized data samples and feature selection. Their predictions are combined to reduce variance and improve robustness.

[Gradient Boosting. Gradient Boosting is a powerful machine… | by Mausam | Mar, 2026 | Medium](https://images.openai.com/static-rsc-4/5DPv2SS_Dn7ASNfRMgfEbgPQn4EEeal4l6m8F8zoERWnw09W0A8W6xQuovqfpei1OrZlnRDXqkUstvjGXkTtG5gi76kwn68of3EvBMmhyx-KaEgwLVSX-qR8-UytTFmMwrCXX1FTGQD-WXJQAYOyUNVOSfxa0mJW5ua5WRMB2mo?purpose=inline)

Boosting

Trees are built sequentially, with later learners focusing on residuals, errors, or difficult examples according to the boosting algorithm.

For classification, tree predictions are commonly combined through majority voting or probability averaging. For regression, predictions are commonly averaged or combined through weighted sums.

### Question 7 — Bagging vs. boosting vs. averaging

Correct answer: B.

| Method           | How it works                                             | Main purpose                                                            |
| ---------------- | -------------------------------------------------------- | ----------------------------------------------------------------------- |
| Bagging          | Trains models on bootstrap samples, usually in parallel  | Often reduces variance                                                  |
| Boosting         | Trains learners sequentially, adapting to earlier errors | Builds a strong predictor from successive learners                      |
| Simple averaging | Averages model predictions                               | Combines existing predictions; does not specify how models were trained |

A key distinction is that averaging is an aggregation rule, while bagging and boosting are ensemble-training strategies.

## Final takeaway

The central theoretical idea behind ensemble learning is that combining multiple competent but imperfect predictors can reduce prediction variance and improve robustness when their errors are sufficiently diverse.

The benefit is largest when individual models are accurate and their errors are not too strongly correlated. Ensemble learning applies to neural networks, linear regression, decision trees, and many other machine learning models.
