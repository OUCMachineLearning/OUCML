# Connections: Log Likelihood, Cross-Entropy, KL Divergence, Logistic Regression, and Neural Networks

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-12-08-082252.png)

This article will cover the relationships between the negative log likelihood, entropy, softmax vs. sigmoid cross-entropy loss, maximum likelihood estimation, Kullback-Leibler (KL) divergence, logistic regression, and neural networks. If you are not familiar with the connections between these topics, then this article is for you!

## **Recommended Background**

- Basic understanding of neural networks. If you would like more background in this area please read [Introduction to Neural Networks](https://glassboxmedicine.com/2019/01/17/introduction-to-neural-networks/).
- Thorough understanding of the difference between multiclass and multilabel classification. If you are not familiar with this topic, please read the article [Multi-label vs. Multi-class Classification: Sigmoid vs. Softmax](https://glassboxmedicine.com/2019/05/26/classification-sigmoid-vs-softmax/). The short refresher is as follows: in multiclass classification we want to assign a single class to an input, so we apply a softmax function to the raw output of our neural network. In multilabel classification we want to assign multiple classes to an input, so we apply an element-wise sigmoid function to the raw output of our neural network.

# **Problem Setup: Multiclass Classification with a Neural Network**

First we will use a multiclass classification problem to understand the relationship between log likelihood and cross-entropy. Here’s our problem setup:



![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-12-08-82308.png)



![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-12-08-082301.png)

Cat image source: [Wikipedia](https://en.wikipedia.org/wiki/File:Cat_November_2010-1a.jpg).

# **Likelihood**

Let’s say we’ve chosen a particular neural network architecture to solve this multiclass classification problem — for example, [VGG](https://neurohive.io/en/popular-networks/vgg16/), [ResNet](https://arxiv.org/abs/1512.03385), [GoogLeNet](https://arxiv.org/abs/1409.4842), etc. Our chosen architecture represents a family of possible models, where each member of the family has different weights (different parameters) and therefore represents a different relationship between the input image *x* and some output class predictions *y*. We want to choose the member of the family that has a good set of parameters for solving our particular problem of [image] -> [airplane, train, or cat].

One way of choosing good parameters to solve our task is to choose the parameters that maximize the likelihood of the observed data:



![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-12-08-082259.png)

# **Negative Log Likelihood**

The negative log likelihood is then, literally, the negative of the log of the likelihood:

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-12-08-082304.png)

Reference for Setup, Likelihood, and Negative Log Likelihood: [“Cross entropy and log likelihood” by Andrew Webb](http://www.awebb.info/blog/cross_entropy)

# **Side Note on Maximum Likelihood Estimation (MLE)**

Why do we “minimize the negative log likelihood” instead of “maximizing the likelihood” when these are mathematically the same? It’s because we typically minimize loss functions, so we talk about the “negative log likelihood” because we can minimize it. (Source: [CrossValidated](https://stats.stackexchange.com/questions/141087/why-do-we-minimize-the-negative-likelihood-if-it-is-equivalent-to-maximization-o).)

Thus, when you minimize the negative log likelihood, you are performing maximum likelihood estimation. Per [the Wikipedia article on MLE](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation),

> *maximum likelihood estimation (MLE) is a method of estimating the parameters of a probability distribution by maximizing a likelihood function, so that under the assumed statistical model the observed data is most probable. The point in the parameter space that maximizes the likelihood function is called the maximum likelihood estimate. […] From the point of view of Bayesian inference, MLE is a special case of maximum a posteriori estimation (MAP) that assumes a uniform prior distribution of the parameters.*

And here’s another summary from [Jonathan Gordon on Quora](https://www.quora.com/What-are-the-differences-between-maximum-likelihood-and-cross-entropy-as-a-loss-function):

> *Maximizing the (log) likelihood is equivalent to minimizing the binary cross entropy. There is literally no difference between the two objective functions, so there can be no difference between the resulting model or its characteristics.*
>
> *This of course, can be extended quite simply to the multiclass case using softmax cross-entropy and the so-called multinoulli likelihood, so there is no difference when doing this for multiclass cases as is typical in, say, neural networks.*
>
> *The difference between MLE and cross-entropy is that MLE represents a structured and principled approach to modeling and training, and binary/softmax cross-entropy simply represent special cases of that applied to problems that people typically care about.*

# **Entropy**

After that aside on maximum likelihood estimation, let’s delve more into the relationship between negative log likelihood and cross entropy. First, we’ll define entropy:



![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-12-08-82311.png)

# **Cross Entropy**



![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-12-08-082300.png)

Section references: [Wikipedia Cross entropy](https://en.wikipedia.org/wiki/Cross_entropy), [“Cross entropy and log likelihood” by Andrew Webb](http://www.awebb.info/blog/cross_entropy)

# **Kullback-Leibler (KL) Divergence**

The Kullback-Leibler (KL) divergence is often conceptualized as a measurement of how one probability distribution differs from a second probability distribution, *i.e.* as a measurement of the distance between two probability distributions. Technically speaking, KL divergence is not a true metric because it doesn’t obey the triangle inequality and *D_KL(g||f)* does not equal *D_KL(f||g)* — but still, intuitively it may seem like a more natural way of representing a loss, since we want the distribution our model learns to be very similar to the true distribution ( *i.e.* we want the KL divergence to be small — we want to minimize the KL divergence.)

Here’s the equation for KL divergence, which can be interpreted as the expected number of additional bits needed to communicate the value taken by random variable *X* (distributed as *g(x)*) if we use the optimal encoding for *f(x)* rather than the optimal encoding for *g(x):*

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-12-08-082305.png)

Additional ways to think about KL divergence *D_KL (g||f):*

- it’s the relative entropy of *g* with respect to *f*
- it’s a measure of the information gained when one revises one’s beliefs from the prior probability distribution *f* to the posterior probability distribution *g*
- it’s the amount of information lost when *f* is used to approximate *g*

In machine learning, *g* typically represents the true distribution of data, while *f* represents the model’s approximation of the distribution. Thus for our neural network we can write the KL divergence like this:

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-12-08-082302.png)

Notice that the second term (colored in blue) depends only on the data, which are fixed. Because this second term does NOT depend on the likelihood y-hat (the predicted probabilities), it also doesn’t depend on the parameters of the model. Therefore, the parameters that minimize the KL divergence are the same as the parameters that minimize the cross entropy and the negative log likelihood! This means we can minimize a cross-entropy loss function and get the same parameters that we would’ve gotten by minimizing the KL divergence.

Section references: [Wikipedia Kullback-Leibler divergence,](https://en.wikipedia.org/wiki/Kullback–Leibler_divergence) [“Cross entropy and log likelihood” by Andrew Webb](http://www.awebb.info/blog/cross_entropy)

# **What about “sigmoid cross entropy loss”?**

So far, we have focused on “softmax cross entropy loss” in the context of a multi *class* classification problem. However, when training a multi *label* classification model, in which more than one output class is possible, then a “sigmoid cross entropy loss” is used instead of a “softmax cross entropy loss.” Please see [this article](https://glassboxmedicine.com/2019/05/26/classification-sigmoid-vs-softmax/) for more background on multilabel vs. multiclass classification.

As we just saw, cross-entropy is defined between two probability distributions *f(x)* and *g(x).* But this doesn’t make sense in the context of a sigmoid applied at the output layer, since the sum of the output decimals won’t be 1, and therefore we don’t really have an output “distribution.”

Per [Michael Nielsen’s book, chapter 3 equation 63](http://neuralnetworksanddeeplearning.com/chap3.html), one valid way to think about the sigmoid cross entropy loss is as “a summed set of per-neuron cross-entropies, with the activation of each neuron being interpreted as part of a two-element probability distribution.”

So, instead of thinking of a probability distribution across all output neurons (which is completely fine in the softmax cross entropy case), for the sigmoid cross entropy case we will think about a bunch of probability distributions, where each neuron is conceptually representing one part of a two-element probability distribution.

For example, let’s say we feed the following picture to a multilabel image classification neural network which is trained with a sigmoid cross-entropy loss:



![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-12-08-082313.jpg)

Image Source: [Wikipedia](https://commons.wikimedia.org/wiki/File:Cat_and_dog.JPG).

Our network has output neurons corresponding to the classes cat, dog, couch, airplane, train, and car.

After applying a sigmoid function to the raw value of the cat neuron, we get 0.8 as our value. We can consider this 0.8 to be the probability of class “cat” and we can imagine an implicit probability value of 1–0.8 = 0.2 as the probability of class “NO cat.” This implicit probability value does NOT correspond to an actual neuron in the network. It’s only imagined/hypothetical.

Similarly, after applying a sigmoid function to the raw value of the dog neuron, we get 0.9 as our value. We can consider this 0.9 to be the probability of class “dog” and we can imagine an implicit probability value of 1–0.9 = 0.1 as the probability of class “NO dog.”

For the airplane neuron, we get a probability of 0.01 out. This means we have a 1–0.01 = 0.99 probability of “NO airplane” — and so on, for all the output neurons. Thus, each neuron has its own “cross entropy loss” and we just sum together the cross entropies of each neuron to get our total sigmoid cross entropy loss.

We can write out the sigmoid cross entropy loss for this network as follows:

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/2019-12-08-082306.png)

“Sigmoid cross entropy” is sometimes referred to as “binary cross-entropy.” [This article](https://peltarion.com/knowledge-center/documentation/modeling-view/build-an-ai-model/loss-functions/binary-crossentropy) discusses “binary cross-entropy” for multilabel classification problems and includes the equation.

# **Connections Between Logistic Regression, Neural Networks, Cross Entropy, and Negative Log Likelihood**

- If a neural network has no hidden layers and the raw output vector has a softmax applied, then that is equivalent to multinomial logistic regression
- if a neural network has no hidden layers and the raw output is a single value with a sigmoid applied (a logistic function) then this is logistic regression
- thus, logistic regression is just a special case of a neural network! ([refA](https://www.quora.com/Is-a-single-layer-feed-forward-neural-network-equivalent-to-a-logistic-regression-algorithm), [refB](https://www.datasciencecentral.com/profiles/blogs/logistic-regression-as-a-neural-network))
- if a neural network does have hidden layers and the raw output vector has a softmax applied, and it’s trained using a cross-entropy loss, then this is a “softmax cross entropy loss” which can be interpreted as a negative log likelihood because the softmax creates a probability distribution.
- if a neural network does have hidden layers and the raw output vector has element-wise sigmoids applied, and it’s trained using a cross-entropy loss, then this is a “sigmoid cross entropy loss” which CANNOT be interpreted as a negative log likelihood, because there is no probability distribution across all the examples. The outputs don’t sum to one. Here we need to use the interpretation provided in the previous section, in which we conceptualize the loss as a bunch of per-neuron cross entropies that are summed together.

Section reference: [Stats StackExchange](https://stats.stackexchange.com/questions/297749/how-meaningful-is-the-connection-between-mle-and-cross-entropy-in-deep-learning)

For additional info you can look at [the Wikipedia article on Cross entropy](https://en.wikipedia.org/wiki/Cross_entropy), specifically the final section which is entitled “Cross-entropy loss function and logistic regression.” This section describes how the typical loss function used in logistic regression is computed as the average of all cross-entropies in the sample (“sigmoid cross entropy loss” above.) The cross-entropy loss is sometimes called the “logistic loss” or the “log loss”, and the sigmoid function is also called the “logistic function.”

# **Cross Entropy Implementations**

In Pytorch, [there are several implementations for cross-entropy](https://github.com/rasbt/stat479-deep-learning-ss19/blob/master/other/pytorch-lossfunc-cheatsheet.md):

*Implementation A:*`torch.nn.functional.binary_cross_entropy` (see`torch.nn.BCELoss)`: the input values to this function have already had a sigmoid applied, e.g.

*Implementation B:*`torch.nn.functional.binary_cross_entropy_with_logits`(see `torch.nn.BCEWithLogitsLoss)`: "this loss combines a Sigmoid layer and the BCELoss in one single class. This version is more numerically stable than using a plain Sigmoid followed by a BCELoss as, by combining the operations into one layer, we take advantage of the log-sum-exp trick for numerical stability."

*Implementation C:*`torch.nn.functional.nll_loss`(see `torch.nn.NLLLoss`) : "the negative log likelihood loss. It is useful to train a classification problem with C classes. [...] The input given through a forward call is expected to contain log-probabilities of each class."

*Implementation D:* `torch.nn.functional.cross_entropy`(see `torch.nn.CrossEntropyLoss`): "this criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class. It is useful when training a classification problem with C classes. [...] The input is expected to contain raw, unnormalized scores for each class."

# **Conclusion**

There are fundamental relationships between negative log likelihood, cross entropy, KL divergence, neural networks, and logistic regression as we have discussed here. I hope you have enjoyed learning about the connections between these different models and losses!

# **About the Featured Image**

Since the topic of this post was connections, the featured image is a “connectome.” A connectome is “a comprehensive map of neural connections in the brain, and may be thought of as its ‘wiring diagram.’” Reference: [Wikipedia](https://en.wikipedia.org/wiki/Connectome). Featured Image Source: [The Human Connectome](https://en.wikipedia.org/wiki/Connectome#/media/File:The_Human_Connectome.png).

------

*Originally published at* [*http://glassboxmedicine.com*](https://glassboxmedicine.com/2019/12/07/connections-log-likelihood-cross-entropy-kl-divergence-logistic-regression-and-neural-networks/) *on December 7, 2019.*