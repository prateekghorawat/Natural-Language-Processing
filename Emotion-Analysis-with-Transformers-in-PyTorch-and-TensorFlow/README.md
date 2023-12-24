# Emotion-Analysis-with-Transformers-in-PyTorch-and-TensorFlow
Emotion Analysis using state-of-the-art Transformer models implemented in both PyTorch and TensorFlow. Emotion analysis is a critical task in natural language processing, enabling the detection of sentiments and emotions in text data. In this, I learned about sentiment analysis and worked on both the framework PyTorch and TensorFlow.
**Text Classification :** can be used for a broad range of
applications, such as tagging customer feedback into categories or routing support tickets
according to their language. Another common type of text classification is sentiment analysis

---

We’ll tackle this task using a variant of **BERT** called
**DistilBERT**. The main advantage of this model is that it achieves comparable performance to
BERT, while being significantly smaller and more efficient. This enables us to train a classifier
in a few minutes, and if you want to train a larger BERT model you can simply change the
checkpoint of the pretrained model. A checkpoint corresponds to the set of weights that are
loaded into a given transformer architecture.

---


**Note :**Datasets provides a
set_format() method that allows us to change the output format of the Dataset. Note that
this does not change the underlying data format (which is an Arrow table), and you can switch
to another format later if needed

---

**NOTE :** In this case, we can see that the dataset is heavily imbalanced; the joy and sadness classes
appear frequently, whereas love and surprise are about 5–10 times rarer. There are several
ways to deal with imbalanced data, including:
Randomly oversample the minority class.
Randomly undersample the majority class.
Gather more labeled data from the underrepresented classes.

---

**Note :**From the plot we see that for each emotion, most tweets are around 15 words long and the
longest tweets are well below DistilBERT’s maximum context size. Texts that are longer than a
model’s context size need to be truncated, which can lead to a loss in performance if the
truncated text contains crucial information.

---

**Transformer models like DistilBERT cannot receive raw strings as input; instead, they assume the text has been tokenized and encoded as numerical vectors**

---

**Note :**Our model expects each character to be converted
to an integer, a process sometimes called numericalization. One simple way to do this is by
encoding each unique token (which are characters in this case) with a unique integer

---

# One-Hot Encoding and Ordinal Scales

## Introduction

- **One-Hot Encoding**: One-hot encoding is a method to represent categorical data as binary vectors.
- **Ordinal Scale**: Ordinal scale refers to a scale of measurement where values have order but lack meaningful numerical differences.

## Ordinal Scale

- In an ordinal scale, data values have a clear order or ranking, but the differences between values don't have a meaningful interpretation.
- For example, token IDs in natural language processing represent the order of tokens in a vocabulary but don't convey the relationship between tokens based on numerical differences.

## Advantages of One-Hot Encodings

- One-hot encodings provide a more interpretable representation, especially for operations involving the presence or absence of categories.
- Each token is represented by a binary vector with a single "hot" entry (1) and 0s elsewhere, indicating the presence of a specific token.
- Operations on one-hot encodings, such as addition, produce results with easily interpretable meaning.
- When you add two one-hot encodings, both tokens are "hot," indicating their co-occurrence.

## Example

- Consider two one-hot encodings:
   - "cat" -> [1, 0]
   - "dog" -> [0, 1]
- Adding these two one-hot encodings results in:
   - "cat" + "dog" -> [1, 0] + [0, 1] = [1, 1]
- In the result, both entries are "hot," indicating that both "cat" and "dog" co-occur.

## Conclusion

- One-hot encodings are valuable for preserving the categorical nature of data and performing operations involving the presence or absence of categories.
- They are particularly useful when the ordinal scale of raw IDs lacks interpretation, making data more interpretable and meaningful.

---

**Note :** Having a large vocabulary is a problem because it requires neural networks to have an
enormous number of parameters. To illustrate this, suppose we have 1 million unique words
and want to compress the 1-million-dimensional input vectors to 1-thousand-dimensional
vectors in the first layer of our neural network. This is a standard step in most NLP
architectures, and the resulting weight matrix of this first layer would contain 1 million × 1
thousand = 1 billion weights. This is already comparable to the largest GPT-2 model, which
has around 1.5 billion parameters in total!
Naturally, we want to avoid being so wasteful with our model parameters since models are
expensive to train, and larger models are more difficult to maintain. A common approach is to
limit the vocabulary and discard rare words by considering, say, the 100,000 most common
words in the corpus. Words that are not part of the vocabulary are classified as “unknown” and
mapped to a shared UNK token. This means that we lose some potentially important information
in the process of word tokenization, since the model has no information about words associated
with UNK.

---

# Subword Tokenization in NLP

## Introduction

Subword tokenization is a technique in natural language processing (NLP) that strikes a balance between character and word-level tokenization. It is designed to address the limitations of both character and word tokenization methods.

### Challenges with Character and Word Tokenization

- **Character Tokenization**: Treats each character in the text as a separate token, leading to highly granular but often impractical representations.
- **Word Tokenization**: Treats each word as a token, maintaining word-level meaning but struggling with rare words, misspellings, and complex word structures.

### The Concept of Subword Tokenization

- **Balancing Act**: Subword tokenization aims to combine the advantages of character and word tokenization.
- It divides text into subword units, which are larger than individual characters but smaller than whole words.
- Subword units are learned from a pretraining corpus, allowing adaptability to specific languages and data.

### Learning from Data

- **Distinctive Feature**: Subword tokenization is data-driven and adaptive.
- It's learned from large text corpora during a pretraining phase.
- Statistical rules and algorithms are employed to determine how to split words into subword units based on observed linguistic patterns.

### Benefits and Applications

- **Adaptability**: Subword tokenization adapts to the language and data it is applied to.
- Particularly useful in multilingual and low-resource settings.
- Commonly used in modern NLP models for tasks like text classification, machine translation, and language modeling.

## Subword Tokenization with WordPiece

### Introduction

Subword tokenization is a technique in NLP that breaks words into smaller units. One common algorithm for subword tokenization is WordPiece, used by models like BERT and DistilBERT.

#### Understanding WordPiece

- **WordPiece**: A popular subword tokenization algorithm.
- It divides words into smaller units, allowing handling of complex words and rare words.

#### Implementing WordPiece

- **Transformers Library**: Provides tools for implementing WordPiece.
- Use the `AutoTokenizer` class to load the tokenizer for pretrained models like BERT and DistilBERT.

#### Practical Application

- Subword tokenization is essential for various NLP tasks and pretrained models.
- It helps models understand complex words and adapt to different languages and data.

#### Conclusion

- WordPiece, a subword tokenization algorithm, is a crucial part of modern NLP.
- It makes NLP models like BERT and DistilBERT effective in understanding and processing text.


---

**Note :** The AutoTokenizer class belongs to a larger set of “auto” classes whose job is to
automatically retrieve the model’s configuration, pretrained weights, or vocabulary from the
name of the checkpoint.

---

**If we wish to manually load the required Tokenizer or any other thing this is the method below**

    from transformers import DistilBertTokenizer
    distilbert_tokenizer = DistilBertTokenizer.from_pretrained(model_ckpt)

---

#### **Three things to Note :**
* [CLS] and [sep] state the start and end of a sentence. 
* Everthing is Lowercase now. 
* transformer word is seperated to Transform and ##er

---

**We all get Attention Ids with Input Ids , it helps to understand and not get confused with respect to padding**

![Attention](https://github.com/prateekghorawat/Emotion-Analysis-with-Transformers-in-PyTorch-and-TensorFlow/blob/main/Images/Attention_mask.png)

---

![Distil](https://github.com/prateekghorawat/Emotion-Analysis-with-Transformers-in-PyTorch-and-TensorFlow/blob/main/Images/distil.png)

## Step 1: Tokenization and One-Hot Encoding

- **Tokenization**: Your text, like "I love ice cream," is split into individual words or subwords: ["I", "love", "ice", "cream"].
- **One-Hot Encoding**: Each word is turned into a unique binary code. For instance, "love" might be represented as [0, 1, 0, 0], where 1 means "love."

## Step 2: Token Embeddings

- **Token Embeddings**: These convert one-hot codes into more meaningful, compact vectors that capture word relationships. For example, "love" becomes a vector like [0.1, 0.9, 0.2, 0.4].

## Step 3: Encoder Blocks

- **Encoder Blocks**: Imagine an AI understanding words in context. It assigns hidden states to each word, considering the surrounding words. "Love" gets a hidden state that knows it's about positive feelings.

## Step 4: Language Modeling

- In a **language modeling task**, the AI predicts missing words. For instance, if "I love" is given, the AI might predict "chocolate" and "cookies" based on its understanding of word meanings.

## Step 5: Classification

- If the AI's task is different, like determining if a sentence is positive or negative, it replaces language modeling with a classification layer.
- It uses hidden states to make decisions, e.g., saying, "Based on 'I love,' this sentence is positive."

In simple terms, this process helps AI understand and work with words in context, filling in missing words, classifying text, and performing various language tasks. The key is starting with words, turning them into numbers, and learning relationships between them to better understand language.

---

**NOTE:**
**In simple terms, a "hidden state" is like a secret memory that a computer program, like an AI or a deep learning model, uses to understand and remember information from the past. It's a way for the computer to keep track of what it's read or processed so far.**

---

**We have two Option :**
* Feature Extraction 
* Fine Tuning 

**We Will perform both and compare the Results**

---

**Reason we don't need to do Embedding here is :**
The AutoModel class converts the token encodings to embeddings, and then feeds them through
the encoder stack to return the hidden states.

---

#### **WE WILL USE BOTH THE FRAMEWORKS AND TRY TO SOLVE THE PROBLEM IN BOTH THE FRAMEWORK**

**How to use transformers for Tensorflow:**
* For AutoModel or AutoTokenizer or any such just use TF in Start
  * TFAutoModel(model_ckpt)
  * If model was never defined in Tensorflow than just use other paramter from_pt = True , like TFAutoModel(model_ckpt , from_pt = True)
* At every place we use "pt" replace it with tf.

---

**Depending on the model configuration, the output can contain several objects,such as the hidden states, losses, or attentions, arranged in a class similar to a namedtuple in Python.** 

**The model output is an instance of BaseModelOutput, and we can
simply access its attributes by name. The current model returns only one attribute, which is the
last hidden state**

---

**Before performing the classifier action it is good to visualize our data. Since out input feature is in 768 dimension it will be very hard to visualize it and hence we are using *UMAP* to convert to 2D and then present the data.**

---

# Using UMAP for Dimensionality Reduction

In this code snippet, UMAP (Uniform Manifold Approximation and Projection) is used for dimensionality reduction, allowing the creation of a 2D representation of the data. Let's break down the steps:

1. **Data Scaling**: The input data `X_train` is first scaled using Min-Max scaling, which transforms the data to a specified range (commonly between 0 and 1). This step ensures that all features are on the same scale, which is important for dimensionality reduction.

2. **UMAP Initialization**: UMAP is then initialized and applied to the scaled data (`X_scaled`). UMAP is a technique for visualizing high-dimensional data in a lower-dimensional space while preserving data structure. In this case:
   - `n_components=2` specifies the reduction to two dimensions (X and Y).
   - `metric="cosine"` specifies the distance metric used by UMAP, with "cosine" being suitable for high-dimensional data.

3. **DataFrame Creation**: A DataFrame named `df_emb` is created to store the 2D embeddings. These embeddings are derived from the UMAP reduction and are stored in columns labeled "X" and "Y."

4. **Label Inclusion**: The DataFrame is further enhanced by adding a "label" column, which typically contains the labels or target values associated with the data points. This column is essential for visualizing the 2D embeddings.

5. **Printing**: Finally, the code prints the first few rows of the DataFrame using `df_emb.head()`, offering a preview of the 2D embeddings and their corresponding labels.

The result of this process is a 2D representation of the data that facilitates the visualization and exploration of high-dimensional data in a more accessible form.

---

# Visualizing Data with Hexbin Plots

In this code snippet, hexbin plots are used to visualize data points associated with different emotions. Here's a breakdown of the key components:

1. **Subplots Initialization**: Subplots are created with a 2x3 layout using the `plt.subplots()` function, forming a grid of subplots to display data related to various emotions. The `figsize` parameter specifies the size of the overall figure.

2. **Iterating Through Emotions**: The code iterates through different emotions, each represented by an index `i`. For each emotion:
   - The code selects and filters a subset of data from the `df_emb` DataFrame using the `.query` method. This filtering is based on the condition `f"label == {i}"`, where "label" is the column containing emotion labels, and `i` represents the current emotion index.

3. **Hexbin Plotting**: A hexbin plot is created using the `axes[i].hexbin()` function, where `i` is the current subplot index.
   - The hexbin plot visualizes the distribution of data points specific to the current emotion.
   - The color map (`cmap`) is specified based on the emotion. Each emotion is associated with a different colormap.
   - `gridsize` determines the size of hexagonal bins in the plot.
   - `linewidths=(0,)` removes the outline around the hexagons for a cleaner appearance.

4. **Title and Axes Ticks**: Each subplot is given a title corresponding to the emotion being visualized. X and Y-axis ticks are removed to focus on the distribution of data points.

5. **Plot Layout and Display**: The `plt.tight_layout()` function ensures that the subplots are neatly arranged, and `plt.show()` displays the final visualization.

The result is a grid of hexbin plots, each showing the distribution of data points for a specific emotion. This visualization provides insights into the distribution and clustering of data points related to different emotions.

---

**Note : Scikit-learn there is a DummyClassifier that can be used to build a classifier with simple heuristics such as always choosing the majority class or always drawing a random class.**

---

#### Training Configuration for Fine-Tuning

In this code snippet, the training configuration for fine-tuning a model is set using the `TrainingArguments` class from the Hugging Face Transformers library. Let's understand the key parameters:

1. **`num_train_epochs`**:
   - `num_train_epochs` specifies the number of complete passes through the training data during fine-tuning. A value of 2, for example, means the model will iterate through the training dataset twice.

2. **`weight_decay`**:
   - `weight_decay` is a regularization term used during training to prevent overfitting. It encourages model weights to stay closer to zero. A typical value, such as 0.01, is applied to control the extent of regularization.

3. **`log_level`**:
   - `log_level` determines the verbosity of logging messages during training. It can be set to "info," "error," or "warning," among other levels.
   - When set to "error," only error-level log messages are displayed, reducing the amount of information shown during training.


---




