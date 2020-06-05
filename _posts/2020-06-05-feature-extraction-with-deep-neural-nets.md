---
layout: post
title: A quick glimpse on feature extraction with deep neural networks
tags: [deep learning, machine learning, feature extraction]
comments: true
---

Nowadays it is common to think deep learning as a suitable approach to images, text, and audio.
Many breakthroughs happened since the seminal work of AlexNet [[1]](#reference1) back in 2012, which gave rise to a large amount of techniques and improvements for deep neural networks.
Fast forward to 2020, I'm constantly impressed with the state-of-the-art results deep neural networks are able to achieve.

Nevertheless, there are two aspects of deep neural networks that might be overlooked by newcomers: feature extraction and transfer learning.
They are powerful techniques that can be employed in different contexts and they enable one to leverage pre-trained deep neural networks for their applications.

In this post, I'll give a brief introduction on feature extraction, explain what it is and how it works, and show code examples on how to apply it.

## Feature Extraction
Feature extraction is an important step of any machine learning pipeline.
It refers to using different algorithms and techniques to compute representations (also called features, or feature vectors) that facilitate a downstream task.
One of the main goals of the process is to reduce the original data to a representation that is compact but meaningful.
By meaningful, we could understand that it should somewhat summarize the essential information of the original data.
For example, traditional techniques like principal component analysis (PCA) can be employed to reduce the dimensionality of the data.

Let's look at an example with image classification to understand why feature extraction is a key feature of deep learning.

### "Traditional pipeline"

![Feature Extraction](/assets/img/feature_extraction.png){: .mx-auto.d-block :}

In the figure you can see a pipeline comprised of the input image, feature extraction, and classification.
In this context, the feature extraction routine should compute characteristics of the image that are meaningful to the classification somehow.
Histograms of pixel intensities, detection of specific shapes and edges are examples.

These techniques demand expert knowledge, they're time consuming, and are domain specific (usually).
That is, you need experts in computer vision to craft algorithms that are able to extract such characteristics.
Examples of these techniques are SIFT, SURF, and HOG, only to name a few.

Before the rise of deep learning, a traditional image classification pipeline comprised preprocessing, feature extraction with one of the above-mentioned techniques, and the training of a machine learning model (e.g., support vector machine - SVM).
Pipelines based on these traditional feature descriptions combined with SVM were very successful and a common choice for different problems.
However, the challenge is the feature extraction algorithm could be too specific that it would work only for a given class of problems or too general that it would work poorly in specific problems.
That's where deep learning enters.

### Deep learning pipeline
What if we could input the raw data and the algorithm could figure out the "best" features for our problem?
Deep learning can help exactly in that sense.
Instead of having the so-called hand-crafted feature extraction process, deep neural networks such as convolutional neural networks are able to extract high-level and hierarchical features from raw data.
During the training process, the network not only learns how to classify an image, but also how to extract the best features that can facilitate such classification.

![Feature Extraction Deep Learning](/assets/img/feature_extraction_2.png){: .mx-auto.d-block :}

The figure illustrates a simple way to compare a traditional pipeline and a deep learning pipeline.
In the traditional case, the features are used to train a machine learning classifier, for example.
The performance of the trained model is highly dependent on the quality of the features provided.
On the other hand, in the deep learning pipeline the model learns both at the same time.

That's great. But there is more.
Research showed the features extracted by deep neural networks carry semantic meaning.
That is, the feature vector that represents an image of a cat can be similar to the feature vector that represents another cat.
On the contrary, the feature vector of a person is less similar than both cat feature vectors.
Therefore, the representations learned by deep neural networks could be leveraged in information retrieval applications such as visual search.

### Extracting features with a pre-trained model
We'll now see an example of how to compute features using a pre-trained model.
Deep learning frameworks such as PyTorch and Tensorflow offer pre-trained models for different domains like computer vision.
In this case, we'll be using a VGG16 model available on Tensorflow/Keras.

{% highlight python linenos %}
# Imports
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

# Creating the model
model = VGG16(weights='imagenet', include_top=False, pooling='avg')
{% endhighlight %}

With Keras it becomes straightforward to use a pre-trained convolutional neural network.
After importing the correct packages, we can instantiate our model using the `VGG16` class.
In this case, we are passing some parameters to the constructor:
* `weights='imagenet'`: to use weights pre-trained on ImageNet.
* `include_top=False`: to remove the final three fully-connected layers.
* `pooling='avg'`: applies global average pooling [[2]](#reference2) at the output of the last convolutional layer.

{% highlight python linenos %}
# Loading the image, preprocessing, and extracting features
img = image.load_img('dog.jpeg', target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Extracting the features
features = model.predict(x)
print(features.shape, features.dtype) # (1, 512) float32
{% endhighlight %}

To extract the features of our image we need to prepare it accordingly.
First, the loaded PIL image `img` is transformed into a `float32` Numpy array.
Next, we create an extra dimension in the image since the network expects a batch as input.
Finally, we preprocess the input with respect to the statistics from ImageNet dataset.

{: .box-note}
**Note:** An interesting overview of preprocessing can be found [here](https://cs231n.github.io/neural-networks-2/#datapre).

This is a very important step because whenever we use a trained model we need to apply the exact same preprocessing steps.
A common mistake is to forget this step, which can significantly affect the resulting output, be it features or classification labels.
As we did all the required steps, we can simply call the method `predict` to extract the features of the image.

That's all it takes to extract features used a pre-trained model.
I encourage you to explore this, testing different pre-trained models with different images.
You can find a notebook with feature extraction using the above example in Keras and a similar example in PyTorch [here](https://github.com/tspthomas/blog_notebooks/tree/master/2020/feature_extraction).
I also added a simple example on how you can compute similarity between two images using their respective feature vectors.

### Final thoughts

Feature extraction is a relevant aspect of deep neural networks and it is very important to learn how to use it.
As I mentioned in the beginning of this post, we can use these features in other downstream tasks.
Examples include general classification tasks, clustering, and retrieval (please refer to [[3]](#reference3) for an early study on using CNN features for different computer vision tasks).

In the case of classification, one could extract features for all the images and train a traditional classifier like Naive-Bayes or Logistic Regression on top of them.
In clustering, one could group different images according to their similarity using an algorithm like K-means.
Because of the semantic information carried by features, you can end up with a cluster of forest photos, for example.
Lastly, in retrieval applications one could build a database of features and use an image as a query.
The image query would have its features extracted with the same model used to build the database of features and a similarity measure like Euclidean distance could be employed to compare your query against the database.

That's it for today!
I hope you enjoyed to read this post, learned something new or a different perspective on the subject.
Feedback is appreciated, so in case you find any errors, have any suggestions, or general comments, please share your thoughts below!

-- Thomas Paula

---
Cite this article as:
```
@article{thomaspaula2020featureextraction,
  title   = "A quick glimpse on feature extraction with deep neural networks",
  author  = "Paula, Thomas",
  journal = "tspthomas.github.io",
  year    = "2020",
  url     = "https://tspthomas.github.io/2020-06-04-feature-extraction-with-deep-neural-nets/"
}
```
---


## References

* <a name="reference1">[1]</a> Krizhevsky et al. [ImageNet Classification with Deep Convolutional Neural](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf), 2012.
* <a name="reference2">[2]</a> Lin et al. [Network in network](https://arxiv.org/abs/1312.4400), 2013.
* <a name="reference3">[3]</a> Razavian et al. [CNN Features off-the-shelf: an Astounding Baseline for Recognition](https://arxiv.org/pdf/1403.6382.pdf), 2014.
* <a name="reference4">[4]</a> Stanford's CS231n Lecture Notes on [Neural Networks Training](https://cs231n.github.io/neural-networks-2/#datapre)
* <a name="reference5">[5]</a> [Feature Extraction Example Notebooks](https://github.com/tspthomas/blog_notebooks/tree/master/2020/feature_extraction)