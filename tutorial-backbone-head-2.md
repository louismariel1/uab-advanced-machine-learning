In advanced machine learning, especially deep learning and transfer learning, a neural network is often divided into two main parts: the backbone and the head.

The easiest way to understand them is to imagine a neural network that identifies whether a picture contains a cat or a dog.

## 1. The main idea

cat in the garden

Input: An image

Raw pixels

## Backbone

Learns useful features from the image

Edges → textures → shapes → object features

## Head

Uses those features to make a prediction

Cat: 95% · Dog: 5%

Output: Cat

- Backbone = feature extractor. It transforms raw input into a useful representation.
- Head = task-specific predictor. It takes that representation and produces the desired output.

The backbone learns what information matters. The head learns how to use that information for a particular task.

## 2. A small numerical example

Let's simplify the neural network so we can follow the numbers.

Suppose we want to classify an image as a cat or a dog.

### Step 1: The backbone extracts features

Imagine the backbone examines the image and produces two features:

- $x_1$: how strongly the image contains cat-like ears
- $x_2$: how strongly the image contains dog-like features

For one image, suppose it outputs:

$$
\mathbf{x} =
\begin{bmatrix}
0.9\\
0.2
\end{bmatrix}
$$

Here, 0.9 indicates a strong cat-like feature, while 0.2 indicates a weaker dog-like feature.

These numbers are illustrative; real neural networks learn more complex features that aren't necessarily so easy to interpret.

### Step 2: The head makes a prediction

The head receives these two features and calculates a score for each class.

For simplicity, suppose its learned weights are:

$$
W =
\begin{bmatrix}
2 & -1\\
-1 & 2
\end{bmatrix}
$$

The first row calculates the cat score, and the second calculates the dog score.

The head computes:

$$
\mathbf{z}=W\mathbf{x}
$$

Therefore,

$$
\mathbf{z} =
\begin{bmatrix}
2(0.9)-0.2\\
-0.9+2(0.2)
\end{bmatrix}
=
\begin{bmatrix}
1.6\\
-0.5
\end{bmatrix}
$$

The scores are:

- Cat: 1.6
- Dog: −0.5

The cat score is higher, so the network predicts cat. A real classification head often applies a softmax function to convert these scores (called logits) into probabilities.

## 3. Why separate a network into backbone and head?

The key benefit is reuse.

Suppose you have already trained a backbone on millions of images. It has learned useful visual features such as edges, textures, shapes, and object parts.

Now you want to solve a new task: classify images of apples versus oranges.

Fresh red apple isolated on white. With clipping path

Task A: Cat vs. dog

Backbone A

Cat/dog head

One orange isolated on white background

Task B: Apple vs. orange

Reused backbone

Fruit head

You may be able to reuse the same backbone and train a new head for the fruit-classification task.

This is a common form of transfer learning.

There are two typical strategies:

1. Freeze the backbone: Keep its learned weights fixed and train only the new head. This is often faster and requires less data.
2. Fine-tune the backbone: Train the head and also update some or all of the backbone weights. This can improve performance when the new task differs from the original one.

## 4. What does this look like in a real neural network?

A common example is ResNet, a convolutional neural network used for image recognition.

Image input

ResNet backbone

Convolutional layers and residual blocks

Extracts visual features

Classification head

Usually a linear layer, followed by softmax for probabilities

Predicted class

Cat, dog, car, etc.

In a typical ResNet classifier, the backbone produces a feature vector, and the final fully connected layer maps that vector to class scores. If you change the number of target classes, you often replace the final classification layer while retaining the backbone.

The same general concept appears in other areas:

| Application | Backbone | Head |
| --- | --- | --- |
| Image classification | CNN or Vision Transformer | Class prediction |
| Object detection | CNN or Vision Transformer | Object classes and bounding boxes |
| Semantic segmentation | Feature-extraction network | Pixel-level class predictions |
| Large language models | Transformer layers | Next-token prediction head |

One important nuance: the backbone/head boundary depends on the architecture. For example, an object-detection model can have multiple heads, and a language model's output head may be a relatively simple projection from hidden states to vocabulary logits.

## 5. Check your understanding

You take a pretrained image model, keep its feature-extraction layers, and replace its final layer to classify 10 new types of flowers. Which part did you replace?

A. The backbone

B B. The head

C. Both necessarily

Correct!

The head was replaced to produce predictions for the 10 flower classes. The pretrained backbone can still extract useful visual features.

The takeaway: Think of the backbone as the part that turns raw data into useful features, and the head as the part that turns those features into a task-specific answer. This separation makes neural networks easier to adapt to new problems without training everything from scratch.
