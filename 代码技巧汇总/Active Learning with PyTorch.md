I have been building an Active Learning library with PyTorch to accompany my new book, [Human-in-the-Loop Machine Learning](https://www.manning.com/books/human-in-the-loop-machine-learning). It addresses one of the most important problems in technology: how do human and machines combine their intelligence to solve problems?

The code is open-source:

Most deployed Machine Learning models use Supervised Learning powered by human training data. Selecting the right data for human review is known as *Active Learning*. Almost every company invents (or reinvents) the same Active Learning strategies and too often they repeat the same avoidable errors.

Everyone gets taught how to build good Machine Learning algorithms in their college education, but not enough people are taught how to build in the human components. My book + code aims to plug that gap.

A confused robot that needs your human help

Within the [Knowledge Quadrant for Machine Learning](https://towardsdatascience.com/knowledge-quadrant-for-machine-learning-5f82ff979890?source=friends_link&sk=9672722cf46b0f050c1adbe9b09de096), Active Learning combines [Uncertainty Sampling](https://towardsdatascience.com/uncertainty-sampling-cheatsheet-ec57bc067c0b?source=friends_link&sk=e1ebcc288e0e0d5db72d2ea13151a7a0) which are your “known unknowns” (where you know your model is confused because of low confidence predictions) and [Diversity Sampling](https://towardsdatascience.com/https-towardsdatascience-com-diversity-sampling-cheatsheet-32619693c304?source=friends_link&sk=746d864e65a5e8aa3b09efdf654a5608) which are your “unknown unknowns” (the gaps in your model knowledge that are not so obvious):

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2020-04-14-153338.png)

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2020-04-14-153347.png)

Active Learning is the right column above: the Known Unknowns that you can address with Uncertainty Sampling, and the Unknown Unknowns that you can address with Diversity Sampling.

People often underestimate how easy it is to build an *ok* Active Learning system and also underestimate how hard it is to get state-of-the-art results. You can implement Active Learning strategies at very different levels of sophistication and I’ll cover each in more detail in this article:

1. **Quick but complete Human-in-the-Loop strategies**
2. **Fully baked Active Learning strategies with a few 1,000 lines of code**
3. **State-of-the-art Human-in-the-Loop strategies that require experimentation**

This is a similar breakdown to what every Machine Learning Scientist is already familiar with for the predictive algorithm side of the problem. There’s often a good-enough method to get started with relatively little code. You can often deploy a good model with the right data and only a few 1,000 line of code. State-of-the-art models require much more work and often novel research. The same is true for Active Learning.

My book and code supports each level of complexity. I get the readers started with a fully working system in the second chapter!

## 1. Quick but complete Human-in-the-Loop strategies

Chapter 2 of the book implements a complete Human-in-the-Loop system in only ~500 lines of code, all in the `active_learning_basics.py` file in the library. The use case is to classify messages as being “disaster-related” or “not disaster-related”.

This is a use case that I’ve worked in many times. I worked in post-conflict development for the UN before coming to the USA to complete a PhD focused on disaster response and health at Stanford. I’ve continued to work in disaster response in parallel to my Machine Learning career and I look for ways to being them together where-ever I can.

The takeaway here is: *create a complete Human-in-the-Loop Machine Learning system first, and then build on it.*

I often see data scientists hacking together Active Learning strategies: “Hactive Learning” (thanks to [Jennifer Prendki](https://www.linkedin.com/in/jennifer-prendki/) for co-inventing this term). In many companies, I have seen scientists adding one hand-built filter after another to take care of all the edge cases as they pop up. By the end of it, they might have written 1,000s of lines of codes to process their data files, with no documentation about how to string their dozen or more scripts together to reproduce the result. As a result, they end up with non-reproducible Hactive Learning systems that miss important data, at best, or amplify biases at worst.

The example in this chapter shows that you can build a more mathematically well-principled system with less code, provided that you design it right from the start.

## 2. Fully baked Active Learning strategies with a few 1,000 lines of code.

The code has examples of Uncertainty Sampling and Diversity Sampling that cover about 90% of what I have used in industry. These are in `uncertainty_sampling.py` and `diversity_sampling.py` in the library.

The code to implement most of them is only a few 1,000 lines long. You could use the library or copy the code into your own code base, testing it within your framework. Any Data Scientist who wants to have a complete set of Machine Learning tools should understand the major Active Learning algorithms, so it is good practice to implement them yourself. The core algorithms are pretty simple — here’s a code snippet for Entropy-based sampling:

```
prob_dist = tensor([0.0321, 0.6439, 0.0871, 0.2369])                            log_probs = prob_dist * torch.log2(prob_dist) 
raw_entropy = 0 - torch.sum(log_probs)            
normalized_entropy = raw_entropy / math.log2(prob_dist.numel())
```

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2020-04-14-153401.png)

Examples of different kinds of Uncertainty Sampling. The black dots each represent a different label. The left examples show a uniform 3-label division. The right examples show a more complex division. They show how Margin of Confidence and Ratio of Confidence favor pair-wise uncertainty while Entropy favors uncertainty across all labels equally.

Playing around with the different methods for Active Learning will also give you the intuition for which ones to start with for your projects. The graphic above is a visualization of the four most common Uncertainty Sampling methods. You can play around with this visualization at: http://robertmunro.com/uncertainty_sampling_example.html

The takeaway here is: *become familiar with the most common Active Learning strategies because they will be some of the most important tools in your Data Scientist tool-kit.*

Needless to say, it is hard to discover what you *don’t know that you don’t know*, and so the set of strategies for Diversity Sampling are more varied than for Uncertainty Sampling. The ones covered in the book are:

1. **Model-based Outliers:** sampling for low activation in your logits and hidden layers to find items that are confusing to your model because of lack of information
2. **Cluster-based Sampling:** using Unsupervised Machine Learning to sample data from all the meaningful trends in your data’s feature-space
3. **Representative Sampling:** sampling items that are the most representative of the target domain for your model, relative to your current training data
4. **Real-World Diversity:** using sampling strategies that increase fairness when trying to support real-world diversity

## **3. State-of-the-art Human-in-the-Loop strategies that require experimentation**

State-of-the-art Human-in-the-Loop strategies use almost every new development in Machine Learning: Transfer Learning, Interpretability, Bias Detection, Bayesian Deep Learning, and Probing Neural Models to extract internal representations.

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2020-04-14-153405.png)

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2020-04-14-153406.png)

Combining Uncertainty Sampling and Clustering to sample a diverse selection of unlabeled data items that are near the decision boundary.

Pushing the boundaries of Active Learning will often mean pushing the boundaries of other Machine Learning subfields. Sometimes, you can combine simpler methods, like the example above where you apply Uncertainty Sampling first and then Diversity Sampling. In other cases, you will experiment with [new ways to interpret your model](https://arxiv.org/abs/1808.05697):

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2020-04-14-153411.png)

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2020-04-14-153412.png)

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2020-04-14-153424.png)

Monte Carlo Dropout: random dropouts during prediction produce Gaussian variation for in-domain data, which can be useful if your model doesn’t give reliable confidence estimates for the predictions.

For most advanced Active Learning methods, you need to understand what’s going on inside your model. You need to probe it for levels of activation (and then quantize with validation data); apply masks to sample the right parts of your data; and/or to introduce noise to simulate Bayesian dropouts. PyTorch makes this incredibly simple with the ability to pass the activation of every neuron back to other processes:

```
def forward(self, feature_vec, return_all_layers=False):    hidden1 = self.linear1(feature_vec).clamp(min=0)
    output = self.linear2(hidden1)
    log_softmax = F.log_softmax(output, dim=1)    if return_all_layers:
        return [hidden1, output, log_softmax]        
    else:
        return log_softmax
```

The takeaway here is: *the building blocks for innovation in Active Learning already exist in PyTorch, so you can concentrate on innovating.*

Writing this book was the first time I’d coded to PyTorch. Having written to other libraries for years, it was a joy to implement so many strategies so quickly.

The most advanced techniques use Machine Learning to optimize the Active Learning strategies themselves. I will cover them in chapters that are still to come in the book!

# Completing the Loop: putting the Human in Human-in-the-Loop

Active Learning is just one piece of a complete Human-in-the-Loop Machine Learning model:

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2020-04-14-153417.png)

![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2020-04-14-153422.png)

Every other step, from incorporating other models via Transfer Learning to Evaluating Quality for Annotation, can similarly be implemented simply or by pushing the state-of-the-art.

The [Human-in-the-Loop Machine Learning](https://www.manning.com/books/human-in-the-loop-machine-learning) book is being published by Manning Publications as each chapter is being written, so stay tuned for more details about how each part of this puzzle fits together!

Robert Munro

October, 2019

*NOTE: to accompany this article, we are offering PyTorch users 40% off the book price! P*lease go to the [Human-in-the-Loop Machine Learning page](https://www.manning.com/books/human-in-the-loop-machine-learning) at Manning Publications, and check out with the code: *pytorch40*