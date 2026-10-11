# Understanding spaces in machine learning: inputs, features, embeddings, and outputs

The key idea is that a space in machine learning is a set of possible values or representations that a model can work with. Different spaces describe different stages of the transformation from input data to a prediction.

An embedding space is closely related to a feature space, but the terms are not always interchangeable. Similarly, a CNN's feature maps and a transformer's embeddings both represent data in a learned numerical form, but they organize that information differently.

Let's build a consistent picture of how all these concepts relate.

## 1. The big picture: a neural network transforms one space into another

Imagine a neural network that receives an image and predicts which animal it contains.

### 1. Input space

Raw images represented as pixel arrays

$\mathcal{X} \subseteq \mathbb{R}^{H\times W\times C}$

Convolutions, patch embeddings, or other layers

### 2. Feature / representation space

Learned numerical descriptions of the image

Feature maps, token representations, embeddings

Further layers and a prediction head

### 3. Output space

Possible predictions, such as cat, dog, or bird

$\mathcal{Y}=\{\text{cat},\text{dog},\text{bird}\}$

Mathematically, we can describe the network as a composition of functions:

$$
f:\mathcal{X}\rightarrow\mathcal{Y}
$$

Here:

- $\mathcal{X}$ is the input space: all valid inputs the model could receive.
- Intermediate spaces contain representations produced by individual layers.
- $\mathcal{Y}$ is the output space: all possible outputs of the task.
- $f$ is the complete model that maps an input to a prediction.

For example, an image might pass through the following transformations:

$$
\underbrace{x}_{\text{image}}
\rightarrow
\underbrace{h_1}_{\text{low-level features}}
\rightarrow
\underbrace{h_2}_{\text{higher-level features}}
\rightarrow
\underbrace{z}_{\text{embedding}}
\rightarrow
\underbrace{\hat y}_{\text{prediction}}
$$

The intermediate representations are often the most interesting part: the network learns how to describe the input in a form that makes the task easier.

One important distinction: the input space, intermediate spaces, and output space need not have the same dimensionality or even the same mathematical structure.

## 2. What is a space in machine learning?

In mathematics, a space is a set equipped with some structure. In machine learning, the word is often used more informally to describe a collection of possible data points or representations.

For example, suppose each input is a vector containing three measurements:

$$
x=(\text{height},\text{weight},\text{age})
$$

If each measurement is represented by a real number, the input belongs to a three-dimensional numerical space:

$$
x\in\mathbb{R}^{3}
$$

Each example is one point in that space.

Matplotlib - 3D Scatter Plots

The same idea applies to neural networks, except their representations may have hundreds, thousands, or millions of dimensions.

For instance, a vector with 768 numbers is a point in $\mathbb{R}^{768}$. It is common to call this a 768-dimensional representation space or, depending on how it was learned and used, an embedding space.

A few related concepts are worth separating:

- Space: The set of possible representations.
- Point or vector: One particular representation in that space.
- Dimension: The number of independent coordinates used to describe a vector in a vector space.
- Representation: The actual numerical encoding of a particular input.
- Mapping: A function that transforms a point from one space into another.

A neural network layer is generally a mapping between representation spaces.

## 3. Input domain, input space, output domain, and output space

These terms are related, but there is a useful distinction between a domain of a function and the space in which its values live.

Consider a model:

$$
f:\mathcal{X}\rightarrow\mathcal{Y}
$$

| Term | Meaning | Example |
| --- | --- | --- |
| Input space $\mathcal{X}$ | The set of possible inputs | All valid images |
| Input domain | Often another name for the input space; sometimes specifically the set where the function is defined | RGB images of a particular size |
| Output space $\mathcal{Y}$ | The set of possible outputs | Class labels or real-valued predictions |
| Output domain | Often another name for the output space, although usage varies | Valid values a prediction can take |
| Actual input $x$ | One specific input | One photograph of a dog |
| Actual output $\hat y$ | One prediction | The label `dog` |

In practice, input domain and input space are frequently used interchangeably, as are output domain and output space. However, domain has a more specific mathematical meaning: it is the set on which a function is defined.

There is also a subtle distinction between the output space and the range of the model.

- The codomain $\mathcal{Y}$ is the declared set of possible outputs.
- The range or image $f(\mathcal{X})$ is the set of outputs the function actually produces.

For a three-class classifier, the output space might be:

$$
\mathcal{Y}=\mathbb{R}^{3}
$$

if the model outputs three logits, meaning three unnormalized class scores. But if it applies softmax, the output lies in the probability simplex:

$$
\Delta^2 =
\left\{p\in\mathbb{R}^{3}:
p_i\geq0,\ \sum_{i=1}^{3}p_i=1
\right\}
$$

If the model instead returns a single class label, its output space could be the discrete set:

$$
\mathcal{Y}=\{\text{cat},\text{dog},\text{bird}\}
$$

These are three different ways to define the output of a classification system.

### Why the distinction matters

A model can have a continuous numerical output space even when the final task is to choose a discrete class. Conversely, a language model's next-token output is selected from a finite vocabulary, but its internal computations involve continuous vectors.

The nature of the task and the representation used to solve it are different things.

## 4. What is a feature space?

A feature space is a space whose coordinates represent features of the examples being modeled.

A feature is a measurable or computed property of an example.

Suppose you want to predict whether a house is expensive. You might start with three features:

$$
x=(\text{area},\text{bedrooms},\text{age})
$$

The feature space is then represented by:

$$
\mathcal{F}=\mathbb{R}^{3}
$$

assuming the features are numerical.

A point in this space describes one house.

Now consider a neural network. Instead of manually specifying the features, the network can learn them from data.

For an image, the early layers may learn responses to edges and color transitions. Later layers may respond to textures, shapes, object parts, or combinations of these.

The learned representation can be written as:

$$
h=\phi(x)
$$

where:

- $x$ is the original input.
- $\phi$ is a feature-extraction function.
- $h$ is the resulting feature vector.

For example:

$$
\phi:\mathbb{R}^{224\times224\times3}
\rightarrow\mathbb{R}^{512}
$$

This particular feature extractor maps an image to a vector of 512 numbers.

The set of possible vectors $h$ is the feature space of that representation.

### Handcrafted versus learned feature spaces

PPT - Edge Detection PowerPoint Presentation, free download - ID:2412250

Handcrafted features

A human defines the measurements or extraction procedure, such as edge counts, color histograms, or texture statistics. The feature space is defined by those choices.

Welcome to Lingxi Xie's Homepage!

Learned features

A neural network learns transformations that produce useful representations. The feature coordinates may not have simple human-readable meanings.

In modern deep learning, feature spaces are usually learned and can change from layer to layer.

A crucial point: a feature space is not necessarily the original input space. A network can map an image containing hundreds of thousands of pixel values into a feature vector with only a few hundred coordinates, or it can expand a representation into a larger space.

## 5. What is an embedding space? Is it a type of feature space?

Yes, an embedding space is generally a learned representation space and can be considered a type of feature space. The term embedding emphasizes that objects are mapped into a numerical space in which useful relationships can be represented geometrically.

An embedding is typically a vector:

$$
e=\operatorname{Embed}(x)\in\mathbb{R}^{d}
$$

The embedding space is the space containing these vectors.

For example, suppose a language model maps words or tokens into a 768-dimensional space:

$$
\operatorname{Embed}(\text{cat})\in\mathbb{R}^{768}
$$

$$
\operatorname{Embed}(\text{dog})\in\mathbb{R}^{768}
$$

$$
\operatorname{Embed}(\text{airplane})\in\mathbb{R}^{768}
$$

Each token is represented by a point in the space.

Vector search with Azure AI Vision and Azure Cosmos DB for PostgreSQL | Microsoft Learn

The training process can make some geometric relationships useful. For example, representations of semantically similar items may be close under a distance metric such as cosine distance.

However, similarity is not guaranteed simply because vectors are embeddings. It depends on the training objective, the model, the particular representation being examined, and the similarity measure.

### Feature space versus embedding space

| Aspect | Feature space | Embedding space |
| --- | --- | --- |
| Main emphasis | Features describing an example | A learned vector representation of an object or example |
| Features | Can be handcrafted or learned | Usually learned |
| Typical objects | Images, houses, documents, sensor readings | Words, tokens, images, users, products, documents |
| Geometry | May be numerical, structured, or categorical | Usually vector geometry, often with a meaningful similarity measure |
| Typical use | Prediction and analysis | Similarity, retrieval, clustering, and downstream prediction |

The distinction is about emphasis, not a strict mathematical boundary. The same vector can be called a feature vector, a learned representation, or an embedding.

For example, a network might learn a 512-dimensional image embedding. That embedding is simultaneously:

- A point in a 512-dimensional vector space.
- A learned feature vector.
- An image representation.
- An embedding.

Not every intermediate activation is conventionally called an embedding, though. The term is often reserved for a representation of a particular token, object, item, or example.

### One subtle point: the embedding space is not necessarily the entire vector space

Suppose an embedding has 768 coordinates. Its ambient space is:

$$
\mathbb{R}^{768}
$$

But the model may produce only a small subset of all possible vectors in that space.

For example, if the embedding layer contains a vocabulary of 50,000 tokens, its learned token embeddings form 50,000 vectors in $\mathbb{R}^{768}$. They do not fill the entire space.

So it is useful to distinguish:

- Ambient space: $\mathbb{R}^{768}$.
- Embedding set: The particular vectors learned for the vocabulary.
- Embedding space: Commonly used to mean the ambient space together with the learned embedding geometry, or more loosely the set of embeddings.

The precise usage varies across papers and implementations.

## 6. Feature maps versus feature spaces

This is one of the most important distinctions in your question.

A feature map is typically a tensor of activations produced by a neural network layer. A feature space is the set of possible feature representations.

In CNN terminology, a feature map often means one spatial activation matrix associated with a particular channel. People also use feature maps more broadly to refer to the entire stack of channels produced by a convolutional layer.

Let's examine a convolutional layer.

CNNs : breakdown and breakthroughs from 1998 to 2019 ! | by Advait Shinde | Mar, 2026 | Medium

Suppose an input image has dimensions:

$$
x\in\mathbb{R}^{224\times224\times3}
$$

A convolutional layer produces:

$$
H\in\mathbb{R}^{112\times112\times64}
$$

This output has:

- Height: 112.
- Width: 112.
- Channels: 64.

You can think of it as 64 spatial feature maps, each of size $112\times112$.

Each channel may respond to a learned pattern, although a channel rarely corresponds perfectly to a single human-interpretable concept.

### What is the feature space here?

There are two useful levels of description.

1. The local channel feature space

At each spatial position $(i,j)$, the network has a 64-dimensional feature vector:

$$
h_{ij}=H[i,j,:]\in\mathbb{R}^{64}
$$

Each pixel location in the feature map therefore has a vector of 64 learned features.

2. The entire layer representation space

The entire output tensor belongs to a tensor space:

$$
\mathcal{H}\subseteq
\mathbb{R}^{112\times112\times64}
$$

The layer transforms an input image into a point in this larger representation space.

The important distinction is:

- A feature map is an actual activation produced for a particular input.
- A feature space describes the possible activations or representations.
- A feature vector is a single vector of feature values, such as the 64-dimensional vector at one spatial location.

The same terms can also be used in broader ways across different machine-learning fields, so context matters.

## 7. Embedding spaces in transformers versus feature maps in CNNs

This is the central architectural comparison.

Both transformers and CNNs transform inputs into learned representations. The main difference is how they organize those representations and how they allow information to interact.

## A. CNN: spatial feature maps

A convolutional neural network processes data with a spatial structure, such as an image.

The Secret to Mastering Feature Extraction in Convolutional Neural Network | by Wiem Souai | AI Process Automation | Medium

Consider a CNN receiving an RGB image.

Golden Retriever Smiles From Ear to Ear After Spending Day With His Humans - PetHelpful

Input image

$224\times224\times3$

Goodfellow Deep Learning — Chapter 9.2: Motivation for Convolutional Networks – ∇ ickma.dev

Early convolutional features

Edges, color transitions, simple textures

Difference between image processing, computer vision, and Artificial intelligence | by Pallawi | Medium

Deeper convolutional features

More complex patterns and object parts

Pooled feature vector

For example, 512 numbers summarizing the image

A convolutional layer preserves a spatial grid, although the grid can become smaller through strides or pooling.

At each location, it produces a feature vector. Across the image, those vectors form a tensor of feature maps.

For example:

$$
H\in\mathbb{R}^{56\times56\times256}
$$

This tensor contains $56\times56$ spatial positions, with 256 feature values at each position.

A CNN can eventually convert this tensor into one image-level embedding:

$$
z=\operatorname{GlobalAveragePool}(H)
\in\mathbb{R}^{256}
$$

Here, global average pooling averages each channel over all spatial locations.

The resulting 256-dimensional vector is a compact representation of the whole image. It can reasonably be called an embedding if it is used as a learned vector representation of that image.

Therefore, a CNN can produce embeddings too. Embeddings are not exclusive to transformers.

## B. Transformer: token embeddings and contextual representations

A transformer often starts with discrete tokens, such as words, subwords, or image patches.

Embeddings: The Bridge Between Tokens and Context

For a text example:

> The dog chased the ball.

The tokenizer converts the text into tokens. An embedding layer maps each token ID to a vector.

Suppose the model uses an embedding dimension of 768 and the sequence contains five tokens.

Its initial token representations have shape:

$$
E\in\mathbb{R}^{5\times768}
$$

Each row corresponds to one token, and each row contains 768 numbers.

A transformer then applies attention and feed-forward layers to produce new representations:

$$
E\rightarrow H_1\rightarrow H_2\rightarrow\cdots\rightarrow H_L
$$

The shape may remain $5\times768$ through all these layers, even though the contents of the vectors change substantially.

Why? Because self-attention allows each token's representation to incorporate information from other tokens.

For instance, the representation of `it` in a sentence might change depending on which earlier noun it refers to.

The final contextual representation of each token is often written as:

$$
h_i^{(L)}\in\mathbb{R}^{768}
$$

The collection of these vectors forms the final hidden-state tensor:

$$
H_L\in\mathbb{R}^{5\times768}
$$

This is a learned representation of the sequence. Each row is contextual: it depends on the surrounding sequence, subject to the attention mask and architecture.

A model may then:

- Use the final hidden state of a designated token for classification.
- Pool the token vectors into a single sentence representation.
- Project each token vector to vocabulary scores for next-token prediction.
- Use all token vectors for token-level tasks such as tagging.

### CNN versus transformer: the important differences

| Property | CNN | Transformer |
| --- | --- | --- |
| Typical input | Images, audio, spatial or temporal arrays | Text tokens, image patches, audio tokens, multimodal tokens |
| Common intermediate structure | Spatial feature maps | Sequence of token representations |
| Typical representation shape | $H\times W\times C$ | $T\times d$ |
| Meaning of position | Spatial location, such as a pixel region | Sequence position or patch location |
| Interaction mechanism | Convolutions combine nearby locations; deeper layers expand the receptive field | Self-attention can combine information across permitted token positions |
| Learned representations | Feature maps, feature vectors, global embeddings | Token embeddings, contextual hidden states, pooled embeddings |
| Can produce an embedding? | Yes | Yes |

The dimensions $H,W,C,T,d$ describe different axes. In particular, $C$ is the number of CNN channels, $T$ is the number of tokens, and $d$ is the representation dimension per token.

Notice the architectural distinction: a CNN generally uses its spatial structure to define local processing, while a transformer commonly represents each token as a vector and uses attention to exchange information among tokens.

Neither architecture is restricted to a single kind of space.

## 8. An important subtlety: input embeddings are not the same as contextual embeddings

The word embedding is used for several related but distinct things in transformers.

Consider a language model.

### Stage 1: Token IDs

The tokenizer might turn the sentence into:

$$
[101,\ 2054,\ 2003,\ 102]
$$

These are illustrative token IDs, not vectors.

The token IDs belong to a discrete set of integer indices. Their numerical magnitudes do not inherently encode semantic similarity: token ID 101 is not necessarily more similar to token ID 102 than to token ID 900.

### Stage 2: Token embeddings

An embedding matrix maps each ID to a learned vector:

$$
\operatorname{Embedding}:
\{1,\ldots,V\}\rightarrow\mathbb{R}^{d}
$$

Here $V$ is the vocabulary size and $d$ is the embedding dimension.

For a sequence of $T$ tokens:

$$
E\in\mathbb{R}^{T\times d}
$$

These vectors are usually called input token embeddings. Depending on the model, positional information is added or incorporated in another way.

### Stage 3: Contextual hidden states

The transformer processes these initial representations:

$$
H^{(\ell+1)}=F_\ell(H^{(\ell)})
$$

where $F_\ell$ represents a transformer block.

After several layers, each token representation contains information influenced by other tokens.

These are commonly called hidden states or contextual representations.

They may also be called contextual embeddings, especially when used for retrieval, similarity, or representation learning.

### Stage 4: Output logits

For a language model predicting the next token, the final hidden state is projected into a vocabulary-sized vector:

$$
z=W_{\text{out}}h+b
$$

where:

$$
z\in\mathbb{R}^{V}
$$

The vector $z$ contains one logit per possible vocabulary token. Softmax converts those scores into probabilities.

Notice the change in dimensions:

$$
\underbrace{\mathbb{R}^{d}}_{\text{token representation}}
\rightarrow
\underbrace{\mathbb{R}^{V}}_{\text{vocabulary logits}}
$$

The embedding dimension and the vocabulary size serve different purposes. They do not need to be equal.

Some models also share their input embedding matrix with their output projection weights, but this is an architectural choice, not a requirement.

## 9. What other kinds of spaces exist in machine learning architectures?

There is no single, universally agreed taxonomy of machine-learning spaces. Different names describe different mathematical properties, different stages in a model, or different purposes.

Here is a practical classification.

### A. Spaces describing data and representations

1. Input space

The set of possible inputs to the model.

Example: $\mathcal X$, the space of valid images, text sequences, or sensor readings.

2. Feature space

A space in which examples are described by features.

Example: a vector of 100 learned image features.

3. Embedding space

A learned vector space used to represent tokens, images, documents, users, or other objects.

Example: $\mathbb R^{768}$ for token representations.

4. Hidden-state or activation space

The possible activation values at a particular layer.

Example: the vectors produced by a transformer's eighth layer, or the tensors produced by a CNN block.

5. Output space

The possible predictions produced by the model.

Example: class labels, class probabilities, continuous values, or vocabulary logits.

### B. Spaces defined by their mathematical structure or purpose

These categories can overlap with the ones above.

| Type of space | What distinguishes it? | Example |
| --- | --- | --- |
| Vector space | Vectors can be added and multiplied by scalars while satisfying vector-space axioms | $\mathbb R^{768}$ |
| Metric space | A distance function measures separation between points | Embeddings with Euclidean distance |
| Similarity space | A similarity function measures how alike two representations are | Cosine similarity between document embeddings |
| Probability space | Provides a formal setting for random events and probabilities | A probabilistic model of generated data |
| Probability simplex | Vectors represent nonnegative probabilities summing to one | Three-class softmax output |
| Latent space | A learned or inferred representation of underlying factors | A variational autoencoder's latent variables |
| Manifold | A space with locally Euclidean structure, possibly embedded in a higher-dimensional space | A manifold model of constrained data representations |
| Parameter space | The possible values of a model's trainable parameters | All possible weight configurations |
| Hypothesis or function space | The collection of functions a learning algorithm can select from | A family of candidate classifiers |
| Attention-weight space | The possible attention distributions or weight matrices | A row of attention weights over tokens |

A few qualifications are important.

First, embedding space and latent space are not synonyms in every context. An embedding is a vector representation of an object. A latent representation is a variable or representation associated with underlying factors not directly observed. An embedding can be latent, but the terms emphasize different things.

Second, not every feature space has a meaningful distance metric. If you want to use geometric notions such as nearest neighbors, clustering, or cosine similarity, you need an appropriate distance or similarity measure.

Third, a neural network's parameter space is different from its activation space. The parameters are the weights and biases being learned; the activations are the values produced when those parameters are applied to an input.

## 10. How do these spaces fit into different network architectures?

The same basic concepts appear in different architectures, but the intermediate representation structures vary.

## CNN

Neural Network Diagram Generator — CNN Visualizer | SciFig

Typical path for image classification

Input tensor $(H,W,C)$

Feature-map tensors $(H',W',C')$

Pooled image embedding $(d)$

Class logits $(K)$

## Transformer

Cryptography-based privacy-preserving large language models: a lifecycle survey of frameworks, methods, and future directions | Artificial Intelligence Review | Springer Nature Link

Typical path for text encoding

Token IDs $(T)$

Token embeddings $(T,d)$

Contextual hidden states $(T,d)$

Pooled representation $(d)$, token logits $(T,V)$, or another task-specific output

## Autoencoder

Input $x$

Encoder: learned representation

### Latent space

Compact code $z\in\mathbb R^d$

Decoder: reconstruction $\hat x$

An autoencoder maps an input into a latent representation and then maps it back into the reconstruction space. A variational autoencoder additionally imposes a probabilistic structure on its latent variables.

These examples demonstrate that the name of a space depends partly on the role its representation plays in the architecture.

A transformer's hidden states may also be used as embeddings. A CNN's feature maps may be flattened or pooled into a latent vector. An autoencoder's latent representation may simultaneously be a feature vector.

There are no completely separate mathematical worlds for these concepts.

## 11. One final distinction: representation space versus physical tensor shape

When examining neural network architectures, you will often see shapes such as:

- `(batch_size, height, width, channels)`
- `(batch_size, sequence_length, hidden_size)`
- `(batch_size, embedding_dim)`
- `(batch_size, num_classes)`

These are tensor shapes, not necessarily different kinds of mathematical spaces.

For example, a transformer hidden-state tensor has shape:

$$
(B,T,d)
$$

where:

- $B$ is the batch size.
- $T$ is the sequence length.
- $d$ is the hidden dimension.

The batch axis organizes multiple examples. The sequence axis organizes tokens within each example. The final axis holds the vector representation of each token.

For one example, the representation has shape $(T,d)$. Each token's vector lies in $\mathbb R^d$, while the entire sequence representation belongs to a tensor space with shape $(T,d)$.

Similarly, a CNN tensor of shape $(B,H,W,C)$ contains a $C$-dimensional feature vector at each spatial location.

A tensor's shape tells you how values are organized; the term used for its space tells you what those values represent or what mathematical structure they have.

## 12. Test your understanding

Consider a CNN that transforms an image into a 512-dimensional vector and then predicts one of 10 classes.

1. What is the 512-dimensional vector most naturally called?

The input domain

A learned feature vector or image embedding

The output class space

2. What does a feature map represent in a CNN?

A spatial array of activations from a learned filter or channel

The set of possible class labels

The model's trainable weights only

3. In a transformer with hidden size 768, what does the number 768 describe?

The number of possible vocabulary tokens

The number of layers in the transformer

The dimensionality of each token's hidden representation

 Check answers

## The essential takeaways

1. Input space: where the original examples live.
2. Feature space: where examples are represented by features, whether handcrafted or learned.
3. Embedding space: a learned vector representation space, often designed or trained to capture useful relationships between objects.
4. Feature maps: spatially organized activation arrays produced by CNN layers; the complete tensor is a layer representation.
5. Transformer hidden states: vector representations of tokens that become contextual as information passes through transformer layers.
6. Output space: the set of possible predictions, such as class labels, probabilities, numerical values, or vocabulary scores.
7. Latent space: an internal representation used to capture underlying structure, particularly in generative models and autoencoders.

The unifying idea is that a neural network learns mappings between spaces of representations. CNNs typically preserve spatial feature structure in intermediate layers; transformers typically preserve a sequence of token vectors. Both can learn feature spaces, embeddings, latent representations, and task-specific outputs.
