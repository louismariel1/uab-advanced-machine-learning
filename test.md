### Q7. LoRA Parameter Count

**Answer: B —****2dr****.**

For a square matrix:

$$
W \in \mathbb{R}^{d \times d}
$$

LoRA uses:

$$
A \in \mathbb{R}^{r \times d}
$$

and:

$$
B \in \mathbb{R}^{d \times r}
$$

Therefore:

$$
N_A = rd
$$

$$
N_B = dr
$$

so:

$$
\boxed{N_{\text{LoRA}} = 2dr}
$$

:::

For **GitHub Markdown**, the key is to use `\(...\)` for inline math and `\[...\]` for display math, rather than escaping every backslash as `\\`.
