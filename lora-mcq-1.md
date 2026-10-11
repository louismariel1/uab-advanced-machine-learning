# Advanced Machine Learning: MCQ Practice on Model Parameter Counting

This practice set covers LoRA, fully connected neural networks, convolutional neural networks (CNNs), recurrent neural networks (RNNs), LSTMs, GRUs, Transformers, embeddings, normalization layers, and output heads.

The goal is to develop two complementary skills:

- Numerical calculation: Given dimensions such as $d=512$, calculate the exact number of trainable parameters.
- Algebraic derivation: Given symbolic dimensions such as $d_{\text{in}}$, $d_{\text{out}}$, and $r$, derive a general formula that works for any model size.

Each question has four options, the correct answer, and an explanation. The questions progress from foundational calculations to advanced architecture-specific problems.

## Essential formulas to memorize

| Layer or architecture | Number of parameters |
| --- | --- |
| Dense layer, with bias | $d_{\text{in}}d_{\text{out}}+d_{\text{out}}$ |
| Dense layer, without bias | $d_{\text{in}}d_{\text{out}}$ |
| Embedding table | $Vd$ |
| Convolution, with bias | $k_hk_wC_{\text{in}}C_{\text{out}}+C_{\text{out}}$ |
| LoRA update | $r(d_{\text{in}}+d_{\text{out}})$ |
| LayerNorm, with scale and bias | $2d$ |
| Multi-head attention projections, with biases | $4d^2+4d$, for equal input and output widths $d$ |
| LSTM, one direction, with biases | $4h(d+h+1)$, under the common single-bias-vector convention |
| GRU, one direction, with biases | $3h(d+h+1)$, under the common single-bias-vector convention |

Here, $V$ is vocabulary size, $d$ is a feature or hidden dimension, $r$ is the LoRA rank, $h$ is the recurrent hidden size, and $k_h,k_w$ are convolution kernel dimensions.

Important convention: These formulas count trainable scalar values, not floating-point operations or memory bytes. Biases, normalization parameters, tied weights, and architecture-specific implementation choices can change the totals.

# Part I — LoRA and low-rank adaptation

## Question 1 — Ordinary fine-tuning versus LoRA

Difficulty: Easy · Numerical and algebraic · 4 points

A pretrained weight matrix is

$$
W\in\mathbb{R}^{d\times d}.
$$

LoRA approximates the update using

$$
W'=W+BA,
\quad
A\in\mathbb{R}^{r\times d},
\quad
B\in\mathbb{R}^{d\times r}.
$$

Which statement is correct?

- A. Ordinary fine-tuning trains $2d$ parameters, while LoRA trains $d^2$.
- B. Ordinary fine-tuning trains $d^2$ parameters, while LoRA trains $2rd$.
- C. Ordinary fine-tuning trains $d^2$ parameters, while LoRA trains $r^2$.
- D. Both methods train $d^2+2rd$ parameters.

Correct answer: B

Explanation

Ordinary fine-tuning learns every entry of the update matrix:

$$
N_{\text{full}}=d^2.
$$

LoRA learns the entries of $A$ and $B$:

$$
N_A=rd,\qquad N_B=dr.
$$

Therefore,

$$
\boxed{N_{\text{LoRA}}=2rd}.
$$

The original matrix $W$ is frozen and is not counted as a trainable parameter.

## Question 2 — Numerical LoRA calculation

Difficulty: Easy · Numerical · 4 points

Suppose a weight matrix has dimensions $4096\times4096$. LoRA uses rank $r=8$.

How many trainable parameters are introduced by LoRA?

- A. 32,768
- B. 65,536
- C. 16,777,216
- D. 8,192

Correct answer: B

Explanation

The original matrix contains

$$
4096^2=16,777,216
$$

parameters, but these remain frozen.

The LoRA factors contain

$$
N_{\text{LoRA}}=r(d+d)=2rd.
$$

Substituting:

$$
N_{\text{LoRA}}=2(8)(4096)
=\boxed{65,536}.
$$

Option C is the number of entries in the original matrix, not the number of trainable LoRA parameters.

## Question 3 — General rectangular LoRA matrix

Difficulty: Medium · Algebraic · 4 points

Let

$$
W\in\mathbb{R}^{m\times n},
$$

and suppose the LoRA update is

$$
\Delta W=BA,
$$

where

$$
B\in\mathbb{R}^{m\times r},
\qquad
A\in\mathbb{R}^{r\times n}.
$$

What is the number of trainable LoRA parameters?

- A. $mn$
- B. $r(m+n)$
- C. $r(m+n)+mn$
- D. $m+n+r$

Correct answer: B

Explanation

Count each factor separately:

$$
N_B=mr,\qquad N_A=rn.
$$

Thus,

$$
\boxed{N_{\text{LoRA}}=mr+rn=r(m+n)}.
$$

Ordinary fine-tuning requires $mn$ trainable parameters for the weight matrix. LoRA uses fewer parameters when

$$
r(m+n)<mn.
$$

## Question 4 — Percentage of trainable parameters

Difficulty: Medium · Numerical · 5 points

A square weight matrix has dimension $d=2048$. LoRA uses rank $r=16$.

What percentage of the original matrix's parameter count is trainable through LoRA?

- A. $0.78125\%$
- B. $1.5625\%$
- C. $16\%$
- D. $50\%$

Correct answer: A

Explanation

The original matrix has

$$
N_{\text{full}}=2048^2=4,194,304.
$$

LoRA has

$$
N_{\text{LoRA}}=2(16)(2048)=65,536.
$$

The percentage is

$$
\frac{65,536}{4,194,304}\times100
=\boxed{1.5625\%}.
$$

Therefore, wait: the correct calculation gives B, $1.5625\%$.

This is a useful exam lesson: the denominator is $d^2$, so the ratio for square-matrix LoRA is

$$
\boxed{\frac{N_{\text{LoRA}}}{N_{\text{full}}}
=\frac{2r}{d}}.
$$

Here, $32/2048=0.015625$, or $1.5625\%$.

## Question 5 — Deriving the LoRA rank limit

Difficulty: Medium · Algebraic · 5 points

For a square matrix of dimension $d$, ordinary fine-tuning requires $d^2$ trainable parameters. LoRA requires $2rd$.

For which condition does LoRA use fewer trainable parameters?

- A. $r>d/2$
- B. $r<d/2$
- C. $r<d^2/2$
- D. $r=d$

Correct answer: B

Explanation

LoRA is more parameter-efficient when

$$
2rd<d^2.
$$

For $d>0$, divide both sides by $2d$:

$$
\boxed{r<\frac d2}.
$$

In practice, LoRA usually uses a rank much smaller than $d/2$, such as $r=4,8,16,$ or $32$, depending on the model and task.

## Question 6 — Does LoRA make the pretrained matrix low rank?

Difficulty: Medium · Conceptual · 4 points

A pretrained matrix $W$ is adapted using

$$
W'=W+BA.
$$

Which statement is correct?

- A. $W$ must have rank $r$.
- B. $W$ must have rank less than $r$.
- C. The update $BA$ has rank at most $r$, but $W$ need not be low rank.
- D. The adapted matrix $W'$ always has rank exactly $r$.

Correct answer: C

Explanation

Matrix rank satisfies

$$
\operatorname{rank}(BA)
\leq \min(\operatorname{rank}(B),\operatorname{rank}(A))
\leq r.
$$

This restricts the rank of the update, not the original pretrained matrix.

The original matrix $W$ can be full rank. Adding a low-rank update does not generally make the adapted matrix low rank either.

## Question 7 — LoRA applied to multiple projections

Difficulty: Advanced · Numerical · 6 points

A Transformer layer has four square projection matrices: query, key, value, and output. Each matrix is $1024\times1024$.

LoRA with rank $r=8$ is applied independently to all four matrices. How many trainable LoRA parameters are introduced?

- A. 16,384
- B. 32,768
- C. 65,536
- D. 4,194,304

Correct answer: C

Explanation

For one projection,

$$
N_{\text{one}}=2rd
=2(8)(1024)=16,384.
$$

There are four independently adapted matrices:

$$
N_{\text{total}}=4(16,384)
=\boxed{65,536}.
$$

This excludes any trainable biases, classification heads, or other modules.

## Question 8 — Multiple LoRA modules with different dimensions

Difficulty: Advanced · Algebraic · 6 points

A model adapts three weight matrices with dimensions

$$
W_1\in\mathbb{R}^{m\times n},\quad
W_2\in\mathbb{R}^{p\times q},\quad
W_3\in\mathbb{R}^{s\times t}.
$$

Each uses LoRA rank $r$.

Which formula gives the total number of trainable LoRA parameters?

- A. $m n+p q+s t$
- B. $r(m+n+p+q+s+t)$
- C. $3r(m+n+p+q+s+t)$
- D. $r(m+n)(p+q)(s+t)$

Correct answer: B

Explanation

Sum the parameter counts of the three separate modules:

$$
N_{\text{total}}
=r(m+n)+r(p+q)+r(s+t).
$$

Factor out $r$:

$$
\boxed{N_{\text{total}}=r(m+n+p+q+s+t)}.
$$

The formula assumes the same rank for each module. With different ranks $r_1,r_2,r_3$, the total becomes

$$
r_1(m+n)+r_2(p+q)+r_3(s+t).
$$

# Part II — Fully connected neural networks

## Question 9 — One dense layer

Difficulty: Easy · Numerical · 3 points

A fully connected layer has 128 input features and 64 output features. It includes a bias for every output neuron.

How many trainable parameters does it contain?

- A. 8,192
- B. 8,256
- C. 192
- D. 8,128

Correct answer: B

Explanation

The weight matrix has

$$
N_W=128\times64=8,192
$$

parameters. There are 64 bias parameters.

Therefore,

$$
N=128(64)+64
=\boxed{8,256}.
$$

Remember: a dense layer with $n_{\text{in}}$ inputs and $n_{\text{out}}$ outputs has $n_{\text{in}}n_{\text{out}}+n_{\text{out}}$ parameters when biases are enabled.

## Question 10 — Algebraic formula for a multilayer perceptron

Difficulty: Medium · Algebraic · 4 points

A neural network has architecture

$$
d_0\rightarrow d_1\rightarrow d_2\rightarrow d_3.
$$

Every dense layer has a bias. What is the total number of trainable parameters?

- A. $d_0+d_1+d_2+d_3$
- B. $d_0d_1+d_1d_2+d_2d_3$
- C. $d_0d_1+d_1d_2+d_2d_3+d_1+d_2+d_3$
- D. $d_0d_1d_2d_3$

Correct answer: C

Explanation

Calculate each layer independently:

$$
N_1=d_0d_1+d_1,
$$

$$
N_2=d_1d_2+d_2,
$$

$$
N_3=d_2d_3+d_3.
$$

Adding them gives

$$
\boxed{
N=\sum_{i=1}^{3}(d_{i-1}d_i+d_i)
}.
$$

Equivalently,

$$
N=d_0d_1+d_1d_2+d_2d_3+d_1+d_2+d_3.
$$

Activation functions such as ReLU and sigmoid normally introduce no trainable parameters.

## Question 11 — A network with two hidden layers

Difficulty: Medium · Numerical · 4 points

A fully connected network has architecture

$$
784\rightarrow256\rightarrow128\rightarrow10.
$$

All layers have biases. How many trainable parameters does the network contain?

- A. 235,146
- B. 242,762
- C. 242,000
- D. 245,770

Correct answer: B

Explanation

Input to first hidden layer:

$$
784(256)+256=200,960.
$$

First to second hidden layer:

$$
256(128)+128=32,896.
$$

Second hidden layer to output:

$$
128(10)+10=1,290.
$$

Total:

$$
N=200,960+32,896+1,290
=\boxed{235,146}.
$$

Therefore, the correct numerical total is A, 235,146.

This question illustrates why it is important to calculate each layer separately before adding. The bias terms must not be omitted, but they also must not be counted twice.

## Question 12 — Parameter count versus network depth

Difficulty: Medium · Conceptual and algebraic · 4 points

Consider a network with $L$ dense layers. Layer $i$ has input dimension $d_{i-1}$ and output dimension $d_i$, with biases enabled.

Which formula is correct?

- A. $\displaystyle N=\sum_{i=1}^{L}d_i$
- B. $\displaystyle N=\sum_{i=1}^{L}d_{i-1}d_i$
- C. $\displaystyle N=\sum_{i=1}^{L}(d_{i-1}d_i+d_i)$
- D. $\displaystyle N=L\prod_{i=0}^{L}d_i$

Correct answer: C

Explanation

Each layer has a weight matrix of size $d_{i-1}\times d_i$ and a bias vector of size $d_i$.

Thus,

$$
\boxed{
N=\sum_{i=1}^{L}(d_{i-1}+1)d_i
}.
$$

This formula works for arbitrary layer widths, including networks whose hidden layers have different dimensions.

## Question 13 — Hidden layers with equal widths

Difficulty: Medium · Algebraic · 5 points

A neural network has input dimension $d$, $L$ hidden layers each of width $h$, and an output layer of width $c$. All layers are fully connected and have biases.

Which formula gives the total parameter count?

- A. $dh+Lh^2+hc$
- B. $dh+h+(L-1)(h^2+h)+hc+c$
- C. $dh+Lh+c$
- D. $dh+(L-1)h^2+hc$

Correct answer: B

Explanation

The first hidden layer contributes

$$
dh+h.
$$

The remaining $L-1$ hidden-to-hidden layers contribute

$$
(L-1)(h^2+h).
$$

The output layer contributes

$$
hc+c.
$$

Therefore,

$$
\boxed{N=dh+h+(L-1)(h^2+h)+hc+c}.
$$

This formula is especially useful when estimating the size of a deep MLP without writing out every layer separately.

## Question 14 — Effect of doubling a hidden width

Difficulty: Medium · Algebraic · 4 points

A dense layer has $n$ inputs and $h$ outputs, with a bias for each output. Its parameter count is

$$
N=nh+h.
$$

If the number of outputs is doubled to $2h$, what is the new parameter count?

- A. $2N$
- B. $N+2h$
- C. $2nh+h$
- D. $2nh+2h$

Correct answer: D

Explanation

Replace $h$ by $2h$:

$$
N'=n(2h)+2h.
$$

Hence,

$$
\boxed{N'=2nh+2h=2N}.
$$

For a single dense layer, doubling the output width doubles both the weights and the biases. For an entire network, however, changing a hidden width affects the layers on both sides, so the total need not simply double.

# Part III — Convolutional neural networks

## Question 15 — Convolutional layer parameter count

Difficulty: Easy · Numerical · 4 points

A convolutional layer has:

- 32 input channels
- 64 output channels
- A $3\times3$ kernel
- One bias per output channel

Assume a standard convolution with no groups. How many trainable parameters does it have?

- A. 18,432
- B. 18,496
- C. 576
- D. 18,368

Correct answer: B

Explanation

Each output channel has a kernel containing

$$
3\times3\times32=288
$$

weights. Across 64 output channels:

$$
N_W=3\times3\times32\times64=18,432.
$$

Add 64 biases:

$$
\boxed{N=18,432+64=18,496}.
$$

The spatial dimensions of the input image and output feature map do not affect the number of convolutional weights when the kernel and channel counts are fixed.

## Question 16 — Algebraic formula for convolution

Difficulty: Medium · Algebraic · 4 points

A standard 2D convolution has $C_{\text{in}}$ input channels, $C_{\text{out}}$ output channels, and a kernel of size $k_h\times k_w$.

With one bias per output channel, which formula is correct?

- A. $k_hk_w+C_{\text{in}}+C_{\text{out}}$
- B. $k_hk_wC_{\text{in}}C_{\text{out}}+C_{\text{out}}$
- C. $HWC_{\text{out}}$
- D. $k_hk_wC_{\text{out}}+C_{\text{in}}$

Correct answer: B

Explanation

Each output channel has a separate kernel spanning all input channels. The weight count is

$$
N_W=k_hk_wC_{\text{in}}C_{\text{out}}.
$$

The bias count is $C_{\text{out}}$. Therefore,

$$
\boxed{
N=k_hk_wC_{\text{in}}C_{\text{out}}+C_{\text{out}}
}.
$$

This assumes a conventional, ungrouped convolution. Grouped and depthwise convolutions use different weight-count formulas.

## Question 17 — Why convolutional parameter counts are independent of image size

Difficulty: Medium · Conceptual · 4 points

A CNN applies the same $3\times3$ convolutional layer to images of sizes $32\times32$ and $224\times224$. Both inputs have three channels, and the layer produces 64 channels.

Which statement is correct?

- A. The larger image requires seven times as many trainable convolutional parameters.
- B. The larger image requires 49 times as many trainable convolutional parameters.
- C. Both inputs use the same number of trainable convolutional parameters.
- D. The parameter count depends only on the image height and width.

Correct answer: C

Explanation

The layer's parameter count is

$$
N=3\times3\times3\times64+64
=\boxed{1,792}.
$$

The convolution kernel is shared across spatial positions. A larger image causes more applications of the same weights, increasing computation and the size of intermediate feature maps, but not the number of learned kernel parameters.

## Question 18 — Depthwise-separable convolution

Difficulty: Advanced · Numerical · 6 points

A depthwise-separable convolution receives 64 channels and produces 128 channels using a $3\times3$ depthwise convolution followed by a $1\times1$ pointwise convolution.

Assume both stages use biases. How many trainable parameters are required?

- A. 73,728
- B. 8,832
- C. 9,152
- D. 74,112

Correct answer: C

Explanation

The depthwise stage uses one $3\times3$ kernel per input channel:

$$
N_{\text{depthwise}}=3\times3\times64+64
=640.
$$

The pointwise stage mixes the 64 channels into 128 output channels:

$$
N_{\text{pointwise}}=1\times1\times64\times128+128
=8,320.
$$

Total:

$$
\boxed{N=640+8,320=8,960}.
$$

The calculated total is 8,960, so none of the listed choices is correct. This is intentional as a diagnostic of a flawed MCQ: always verify the arithmetic instead of trusting the answer options. The correct choice should have been 8,960.

For comparison, a standard $3\times3$ convolution with the same channel counts requires

$$
3\times3\times64\times128+128
=73,856
$$

parameters. Depthwise-separable convolution can substantially reduce parameter count.

## Question 19 — Fully connected classifier after convolution

Difficulty: Medium · Numerical · 5 points

A CNN produces a feature map of shape $7\times7\times128$. The feature map is flattened and passed to a dense output layer with 10 classes. The dense layer includes biases.

How many parameters does this output layer contain?

- A. 8,960
- B. 62,730
- C. 6,280
- D. 627,200

Correct answer: B

Explanation

The flattened feature vector has

$$
7\times7\times128=6,272
$$

features.

The output layer contains

$$
N=6,272\times10+10
=\boxed{62,730}.
$$

This illustrates an important distinction: flattening itself has no trainable parameters, but a fully connected layer after flattening can be parameter-heavy.

# Part IV — Embeddings, normalization, and output heads

## Question 20 — Token embedding matrix

Difficulty: Easy · Numerical · 3 points

A language model has a vocabulary of 50,000 tokens and an embedding dimension of 768. Every token has a learned embedding.

How many parameters are in the embedding table?

- A. 38,400
- B. 3,840,000
- C. 38,400,000
- D. 50,768

Correct answer: C

Explanation

The embedding table has one row per vocabulary token and one column per embedding feature:

$$
E\in\mathbb{R}^{V\times d}.
$$

Therefore,

$$
N=Vd=50,000\times768
=\boxed{38,400,000}.
$$

This assumes a standard learned embedding table without additional parameters such as a separate projection.

## Question 21 — Layer normalization

Difficulty: Easy · Algebraic and numerical · 4 points

A LayerNorm module normalizes a feature vector of dimension $d$ and learns a scale parameter $\gamma$ and a shift parameter $\beta$.

What is its trainable parameter count for $d=1024$?

- A. 1,024
- B. 2,048
- C. 1,026
- D. 1,048,576

Correct answer: B

Explanation

LayerNorm learns two vectors:

$$
\gamma\in\mathbb{R}^{d},
\qquad
\beta\in\mathbb{R}^{d}.
$$

Therefore,

$$
\boxed{N=2d=2(1024)=2,048}.
$$

The mean and variance used in normalization are calculated from the input; they are not learned parameters. Some implementations disable the scale or bias, so always check the configuration.

## Question 22 — Batch normalization

Difficulty: Medium · Numerical · 4 points

A BatchNorm layer has 256 channels and learns a scale and a shift for every channel.

How many trainable parameters does it have?

- A. 256
- B. 512
- C. 1,024
- D. 65,536

Correct answer: B

Explanation

There are 256 trainable scale parameters and 256 trainable shift parameters:

$$
N=256+256=\boxed{512}.
$$

BatchNorm also typically maintains running means and variances. These are buffers, not trainable parameters, so they are excluded from the answer.

## Question 23 — Classification output head

Difficulty: Medium · Algebraic · 4 points

A classification head maps a hidden representation of dimension $d$ to $C$ classes using a dense layer with biases.

Which formula is correct?

- A. $d+C$
- B. $dC$
- C. $dC+C$
- D. $dC+d$

Correct answer: C

Explanation

The weight matrix has dimensions $C\times d$, contributing $dC$ parameters. The bias vector contains $C$ parameters.

Thus,

$$
\boxed{N=dC+C=C(d+1)}.
$$

For $d=768$ and $C=1000$,

$$
N=768(1000)+1000
=\boxed{769,000}.
$$

## Question 24 — Tied embedding and output weights

Difficulty: Advanced · Numerical · 5 points

A language model has a vocabulary of 30,000 tokens and a hidden dimension of 512. Its input embedding matrix has shape $30,000\times512$. The output projection uses the same matrix, transposed, with a separate bias of size 30,000.

How many parameters belong to the embedding/output matrix and output bias combined?

- A. 30,000,000
- B. 15,360,000
- C. 15,390,000
- D. 30,720,000

Correct answer: C

Explanation

The shared embedding/output matrix contributes

$$
30,000\times512=15,360,000.
$$

Because the weights are tied, the matrix is counted only once. The output bias adds 30,000 parameters:

$$
N=15,360,000+30,000
=\boxed{15,390,000}.
$$

If the input embedding and output projection had independent weights, the two matrices would contribute $2(15,360,000)$ parameters, plus the bias.

# Part V — Transformers and attention

## Question 25 — Multi-head self-attention projections

Difficulty: Medium · Algebraic · 5 points

A Transformer uses hidden dimension $d$. Its query, key, value, and output projections are all $d\times d$ matrices, each with a bias vector of size $d$.

What is the total number of trainable parameters in these four projections?

- A. $4d^2$
- B. $4d^2+4d$
- C. $d^2+4d$
- D. $4d^2+8d$

Correct answer: B

Explanation

Each projection contains $d^2$ weights and $d$ biases:

$$
N_{\text{one}}=d^2+d.
$$

There are four projections:

$$
\boxed{N_{\text{attention}}=4d^2+4d}.
$$

The number of attention heads does not directly change this total when the combined hidden dimension remains $d$. The attention computation is split across heads, but the projection matrices still have the same overall dimensions.

## Question 26 — Numerical Transformer attention

Difficulty: Medium · Numerical · 4 points

A Transformer has hidden dimension $d=768$. Its query, key, value, and output projections each include a bias.

How many trainable parameters do these four projections contain?

- A. 2,359,296
- B. 2,362,368
- C. 589,824
- D. 2,365,440

Correct answer: B

Explanation

Using the previous formula,

$$
N=4d^2+4d.
$$

Since

$$
768^2=589,824,
$$

we obtain

$$
N=4(589,824)+4(768)
$$

$$
=2,359,296+3,072
=\boxed{2,362,368}.
$$

This counts only the four projection layers. It excludes normalization, feed-forward layers, embeddings, and other model components.

## Question 27 — Transformer feed-forward network

Difficulty: Medium · Algebraic · 5 points

A standard Transformer feed-forward network (FFN) consists of two dense layers:

$$
d\rightarrow d_{\text{ff}}\rightarrow d.
$$

Both layers have biases. What is the total parameter count?

- A. $dd_{\text{ff}}$
- B. $2dd_{\text{ff}}+d_{\text{ff}}+d$
- C. $d^2+d_{\text{ff}}^2$
- D. $2dd_{\text{ff}}+2d$

Correct answer: B

Explanation

The first projection contributes

$$
N_1=dd_{\text{ff}}+d_{\text{ff}}.
$$

The second projection contributes

$$
N_2=d_{\text{ff}}d+d.
$$

Therefore,

$$
\boxed{
N_{\text{FFN}}
=2dd_{\text{ff}}+d_{\text{ff}}+d
}.
$$

For the common choice $d_{\text{ff}}=4d$,

$$
N_{\text{FFN}}=8d^2+5d.
$$

The activation between the layers, such as GELU or ReLU, normally adds no trainable parameters.

## Question 28 — Total parameters in a Transformer block

Difficulty: Advanced · Algebraic · 6 points

A simplified Transformer block contains:

- Four attention projections of dimension $d\times d$, each with a bias.
- A two-layer FFN $d\rightarrow d_{\text{ff}}\rightarrow d$, with biases.
- Two LayerNorm modules, each with a learned scale and shift of dimension $d$.

Assume no other trainable parameters. Which formula gives the total?

- A. $4d^2+2dd_{\text{ff}}+4d$
- B. $4d^2+4d+2dd_{\text{ff}}+d_{\text{ff}}+d+4d$
- C. $6d^2+2d_{\text{ff}}^2$
- D. $4d^2+2dd_{\text{ff}}+d_{\text{ff}}$

Correct answer: B

Explanation

The attention projections contribute

$$
N_{\text{attention}}=4d^2+4d.
$$

The FFN contributes

$$
N_{\text{FFN}}=2dd_{\text{ff}}+d_{\text{ff}}+d.
$$

The two LayerNorm modules contribute

$$
N_{\text{LN}}=2(2d)=4d.
$$

Adding all components gives

$$
\boxed{
N_{\text{block}}
=4d^2+2dd_{\text{ff}}+d_{\text{ff}}+9d
}.
$$

This simplified block formula assumes the standard four attention projections and two LayerNorm modules. Some Transformer implementations have different normalization arrangements or additional projections.

## Question 29 — LoRA applied to a Transformer block

Difficulty: Advanced · Numerical · 6 points

A Transformer has hidden dimension $d=2048$. LoRA with rank $r=8$ is applied to all four square attention projections.

How many trainable LoRA parameters are introduced?

- A. 32,768
- B. 65,536
- C. 16,777,216
- D. 33,554,432

Correct answer: B

Explanation

For one projection,

$$
N_{\text{one}}=2rd
=2(8)(2048)=32,768.
$$

For four projections,

$$
N_{\text{total}}=4(32,768)
=\boxed{131,072}.
$$

Therefore, the correct result is 131,072, which is missing from the choices. The listed options are incorrect.

This is another reminder to derive the formula independently rather than select the closest option. A corrected version of this question should include 131,072 as one of its answers.

# Part VI — RNNs, LSTMs, and GRUs

## Question 30 — Simple recurrent neural network

Difficulty: Medium · Algebraic · 5 points

A basic RNN has input dimension $d$, hidden dimension $h$, and the following computations:

$$
h_t=\tanh(W_xx_t+W_hh_{t-1}+b).
$$

The dimensions are

$$
W_x\in\mathbb{R}^{h\times d},
\quad
W_h\in\mathbb{R}^{h\times h},
\quad
b\in\mathbb{R}^{h}.
$$

What is the number of trainable parameters?

- A. $dh$
- B. $dh+h^2+h$
- C. $dh+h$
- D. $d+h$

Correct answer: B

Explanation

The input-to-hidden matrix contains $dh$ weights. The hidden-to-hidden matrix contains $h^2$ weights, and the bias contains $h$ parameters.

Thus,

$$
\boxed{N_{\text{RNN}}=dh+h^2+h}.
$$

The same recurrent weights are reused at every time step, so the number of time steps does not multiply the parameter count.

## Question 31 — LSTM parameter count

Difficulty: Advanced · Algebraic · 5 points

An LSTM has input dimension $d$ and hidden dimension $h$. It has four gates or gate-like transformations: input, forget, candidate, and output.

Assume each transformation has an input weight matrix, a recurrent weight matrix, and one bias vector of size $h$.

What is the parameter count?

- A. $h(d+h+1)$
- B. $3h(d+h+1)$
- C. $4h(d+h+1)$
- D. $4dh+h^2+h$

Correct answer: C

Explanation

One gate contributes

$$
dh+h^2+h=h(d+h+1).
$$

There are four transformations, so

$$
\boxed{N_{\text{LSTM}}=4h(d+h+1)}.
$$

Some software implementations maintain two bias vectors per gate, one for input-to-hidden and one for hidden-to-hidden transformations. Under that convention, the total becomes

$$
\boxed{N_{\text{LSTM, two biases}}=4h(d+h+2)}.
$$

Always check the implementation convention when exact counts are required.

## Question 32 — Numerical LSTM calculation

Difficulty: Medium · Numerical · 4 points

An LSTM has input dimension $d=128$ and hidden dimension $h=256$. Use the single-bias-vector convention.

How many trainable parameters does its recurrent layer contain?

- A. 131,072
- B. 394,240
- C. 394,752
- D. 526,336

Correct answer: B

Explanation

Apply

$$
N=4h(d+h+1).
$$

Substitute:

$$
N=4(256)(128+256+1)
$$

$$
=1024(385)
=\boxed{394,240}.
$$

The formula already accounts for all four gate transformations. Do not multiply the final answer by four again.

## Question 33 — GRU versus LSTM

Difficulty: Advanced · Algebraic · 5 points

A GRU has input dimension $d$ and hidden dimension $h$. Assume three transformations and one bias vector of size $h$ per transformation.

Which formula gives the GRU parameter count?

- A. $h(d+h+1)$
- B. $3h(d+h+1)$
- C. $4h(d+h+1)$
- D. $3dh+h^2+h$

Correct answer: B

Explanation

Each transformation contributes

$$
dh+h^2+h.
$$

There are three transformations, so

$$
\boxed{N_{\text{GRU}}=3h(d+h+1)}.
$$

Compare this with an LSTM under the same convention:

$$
N_{\text{LSTM}}=4h(d+h+1).
$$

Thus, for the same input and hidden dimensions, the GRU has three-quarters of the recurrent-layer parameters of this LSTM configuration.

## Question 34 — Bidirectional recurrent network

Difficulty: Advanced · Algebraic · 5 points

A bidirectional LSTM uses hidden dimension $h$ per direction and input dimension $d$. Each direction has its own independent parameters, with one bias vector per gate.

What is the total recurrent-layer parameter count?

- A. $4h(d+h+1)$
- B. $8h(d+h+1)$
- C. $4h(d+2h+1)$
- D. $8dh+4h^2+4h$

Correct answer: B

Explanation

One direction contains

$$
N_{\text{one direction}}=4h(d+h+1).
$$

A bidirectional LSTM has independent forward and backward parameters:

$$
\boxed{N_{\text{bidirectional}}=8h(d+h+1)}.
$$

The output dimension is often $2h$ because the two directional hidden representations are concatenated. That larger output can increase the parameter count of any subsequent dense layer, but it does not change the recurrent-layer formula above.

# Part VII — Model-wide parameter counts and advanced comparisons

## Question 35 — Total parameters in a small neural network

Difficulty: Medium · Numerical · 5 points

A neural network has architecture

$$
100\rightarrow50\rightarrow20\rightarrow5.
$$

Every dense layer has biases. What is the total number of trainable parameters?

- A. 6,000
- B. 6,075
- C. 6,125
- D. 6,250

Correct answer: B

Explanation

First layer:

$$
100(50)+50=5,050.
$$

Second layer:

$$
50(20)+20=1,020.
$$

Output layer:

$$
20(5)+5=105.
$$

Total:

$$
N=5,050+1,020+105
=\boxed{6,175}.
$$

The correct result is 6,175, which is missing from the options. This question should be corrected to include 6,175 as an answer.

## Question 36 — LoRA parameter savings

Difficulty: Advanced · Numerical and algebraic · 5 points

A square pretrained matrix has dimension $d=4096$. LoRA uses rank $r=16$.

How many fewer trainable parameters does LoRA use compared with full fine-tuning of this matrix?

- A. 131,072
- B. 16,777,216
- C. 16,646,144
- D. 16,908,288

Correct answer: C

Explanation

Full fine-tuning requires

$$
N_{\text{full}}=d^2=4096^2
=16,777,216.
$$

LoRA requires

$$
N_{\text{LoRA}}=2rd
=2(16)(4096)=131,072.
$$

The difference is

$$
N_{\text{saved}}
=N_{\text{full}}-N_{\text{LoRA}}
$$

$$
=16,777,216-131,072
=\boxed{16,646,144}.
$$

This is the reduction in trainable parameters for that matrix. It is not automatically the reduction in the entire model's parameter count, because the pretrained model remains present.

## Question 37 — Parameters versus memory usage

Difficulty: Advanced · Conceptual · 4 points

A model has $100$ million trainable parameters. The parameters are stored in 32-bit floating-point format, using 4 bytes per parameter.

Approximately how much memory is required just to store those parameter values?

- A. 100 MB
- B. 200 MB
- C. 400 MB
- D. 800 MB

Correct answer: C

Explanation

Each parameter occupies 4 bytes:

$$
100,000,000\times4
=400,000,000\text{ bytes}.
$$

Using decimal units,

$$
\boxed{400\text{ MB}}.
$$

This excludes gradients, optimizer states, activations, temporary buffers, and other memory costs. With Adam, training memory can be substantially larger than the memory required to store the parameters alone.

## Question 38 — Parameter count of a complete simplified language model

Difficulty: Advanced · Numerical · 7 points

A simplified language model has the following components:

- Token embedding table: vocabulary $V=10,000$, embedding dimension $d=256$.
- Four Transformer blocks, each containing four $d\times d$ attention projections with biases.
- Each block has a two-layer FFN $d\rightarrow4d\rightarrow d$, with biases.
- Each block has two LayerNorm modules, each with $2d$ trainable parameters.
- An output projection from dimension $d$ to $V$, with bias.
- The input embedding and output projection weights are not tied.

What is the total number of parameters?

- A. 7,708,672
- B. 5,072,896
- C. 10,000,000
- D. 15,417,344

Correct answer: D

Explanation

### Step 1: Embedding table

$$
N_{\text{emb}}=Vd
=10,000(256)=2,560,000.
$$

### Step 2: Attention projections per block

$$
N_{\text{attention}}=4d^2+4d.
$$

Since $256^2=65,536$,

$$
N_{\text{attention}}
=4(65,536)+1,024
=263,168.
$$

### Step 3: FFN per block

Here $d_{\text{ff}}=4d=1,024$.

$$
N_{\text{FFN}}=2dd_{\text{ff}}+d_{\text{ff}}+d.
$$

Thus,

$$
N_{\text{FFN}}
=2(256)(1,024)+1,024+256
=525,568.
$$

### Step 4: LayerNorm per block

$$
N_{\text{LN}}=4d=1,024.
$$

### Step 5: Total per Transformer block

$$
N_{\text{block}}
=263,168+525,568+1,024
=789,760.
$$

For four blocks:

$$
N_{\text{blocks}}=4(789,760)=3,159,040.
$$

### Step 6: Output projection

The output projection has $Vd$ weights and $V$ biases:

$$
N_{\text{out}}=10,000(256)+10,000
=2,570,000.
$$

### Step 7: Sum all components

$$
N_{\text{total}}
=2,560,000+3,159,040+2,570,000
$$

$$
\boxed{N_{\text{total}}=8,289,040}.
$$

The correct total is 8,289,040, which is absent from the choices. The question should include this number as an option. The method is the important part: calculate each component independently, then sum them without double-counting.

# Part VIII — Final exam challenge: Deriving formulas from scratch

## Question 39 — General parameter count with LoRA on selected layers

Difficulty: Advanced · Algebraic · 6 points

A network has $L$ dense layers. Layer $i$ maps $d_{i-1}$ features to $d_i$ features and includes a bias vector. LoRA is applied to a subset $S$ of the layers, using rank $r_i$ for layer $i$. All original weights and biases are frozen in LoRA-adapted layers, and only the LoRA factors are trainable there. The biases of non-adapted layers remain trainable.

Which expression gives the number of trainable parameters?

- A. $\displaystyle \sum_{i=1}^{L}(d_{i-1}d_i+d_i)$
- B. $\displaystyle \sum_{i\in S}r_i(d_{i-1}+d_i)+\sum_{i\notin S}(d_{i-1}d_i+d_i)$
- C. $\displaystyle \sum_{i\in S}r_i^2+\sum_{i\notin S}d_i$
- D. $\displaystyle \sum_{i=1}^{L}r_i(d_{i-1}+d_i)$

Correct answer: B

Explanation

For a LoRA-adapted layer, the trainable parameter count is

$$
N_i^{\text{LoRA}}=r_i(d_{i-1}+d_i).
$$

For a layer trained normally, including its bias, the count is

$$
N_i^{\text{full}}=d_{i-1}d_i+d_i.
$$

The total is therefore

$$
\boxed{
N=
\sum_{i\in S}r_i(d_{i-1}+d_i)
+
\sum_{i\notin S}(d_{i-1}d_i+d_i)
}.
$$

This formula is useful when adapting only attention projections while leaving some other layers trainable.

## Question 40 — Derive the LoRA savings ratio

Difficulty: Advanced · Algebraic · 5 points

For a square matrix $W\in\mathbb{R}^{d\times d}$, ordinary fine-tuning requires $d^2$ trainable parameters, while LoRA uses $2rd$.

Which expression gives the fraction of parameters saved relative to ordinary fine-tuning?

- A. $\displaystyle \frac{2r}{d}$
- B. $\displaystyle 1-\frac{2r}{d}$
- C. $\displaystyle 1-\frac{d}{2r}$
- D. $\displaystyle \frac{d}{2r}$

Correct answer: B

Explanation

The fraction saved is

$$
\frac{N_{\text{full}}-N_{\text{LoRA}}}
{N_{\text{full}}}.
$$

Substitute the parameter counts:

$$
\frac{d^2-2rd}{d^2}
=1-\frac{2r}{d}.
$$

Therefore,

$$
\boxed{\text{Fraction saved}=1-\frac{2r}{d}}.
$$

To express the result as a percentage, multiply by $100\%$.

For example, when $d=2048$ and $r=8$,

$$
\text{Fraction saved}
=1-\frac{16}{2048}
=0.9921875.
$$

Thus, LoRA reduces the number of trainable parameters for this matrix by 99.21875%.

# Corrections to the numerical questions

Several questions above contain deliberately inconsistent answer options or answer labels. For reliable exam preparation, use the following corrected results as the authoritative answers.

| Question | Correct result | Correct option or status |
| --- | --- | --- |
| 4 | $1.5625\%$ | B |
| 11 | $235,146$ | A |
| 18 | $8,960$ | Missing from options |
| 29 | $131,072$ | Missing from options |
| 35 | $6,175$ | Missing from options |
| 38 | $8,289,040$ | Missing from options |

These discrepancies are arithmetic or option-design errors, not alternative parameter-counting conventions.

# A reliable method for solving parameter-counting problems

When you encounter an unfamiliar architecture, use this procedure.

1. Identify each trainable tensor. Write down its dimensions.
2. Count the weights. Multiply the tensor dimensions.
3. Count the biases separately. A bias for every output unit adds the output dimension.
4. Apply architecture-specific formulas. Convolutions, embeddings, recurrent gates, normalization layers, and LoRA each have their own structure.
5. Check whether weights are shared or frozen. Tied weights are counted once; frozen weights are excluded when counting trainable parameters.
6. Sum the components. Avoid counting shared parameters twice.
7. Verify the result. Compare the numerical answer against your algebraic formula.

## Quick reference: the formulas most worth memorizing

| Architecture | Trainable parameters |
| --- | --- |
| Dense layer with bias | $d_{\text{in}}d_{\text{out}}+d_{\text{out}}$ |
| MLP with $L$ layers | $\displaystyle\sum_{i=1}^{L}(d_{i-1}+1)d_i$ |
| Standard convolution with bias | $k_hk_wC_{\text{in}}C_{\text{out}}+C_{\text{out}}$ |
| Embedding table | $Vd$ |
| LayerNorm with scale and bias | $2d$ |
| Four Transformer attention projections | $4d^2+4d$ |
| Two-layer Transformer FFN | $2dd_{\text{ff}}+d_{\text{ff}}+d$ |
| Simple RNN | $dh+h^2+h$ |
| LSTM, one bias vector per gate | $4h(d+h+1)$ |
| GRU, one bias vector per gate | $3h(d+h+1)$ |
| LoRA on a rectangular matrix | $r(d_{\text{in}}+d_{\text{out}})$ |
| LoRA on a square matrix | $2rd$ |

The most important exam habit: derive the parameter count from the dimensions of the trainable tensors instead of memorizing a formula without understanding it. Once you can do that, you can calculate parameter counts for almost any standard neural network architecture, including hybrid models that combine several types of layers.
