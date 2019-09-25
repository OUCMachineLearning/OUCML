## **distribution-is-all-you-need**

**distribution-is-all-you-need** is the basic distribution probability tutorial for **most common distribution focused on Deep learning** using python library.

#### Overview of distribution probability

[![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-25-031409.png)](https://github.com/graykode/distribution-is-all-you-need/blob/master/overview.png)

- `conjugate` means it has relationship of **conjugate distributions**.

    > In [Bayesian probability](https://en.wikipedia.org/wiki/Bayesian_probability) theory, if the [posterior distributions](https://en.wikipedia.org/wiki/Posterior_probability) *p*(*θ* | *x*) are in the same [probability distribution family](https://en.wikipedia.org/wiki/List_of_probability_distributions) as the [prior probability distribution](https://en.wikipedia.org/wiki/Prior_probability_distribution) *p*(θ), the prior and posterior are then called **conjugate distributions,** and the prior is called a **conjugate prior** for the [likelihood function](https://en.wikipedia.org/wiki/Likelihood_function). [Conjugate prior, wikipedia](https://en.wikipedia.org/wiki/Conjugate_prior)

- `Multi-Class` means that Random Varivance are more than 2.

- `N Times` means that we also consider prior probability P(X).

- To learn more about probability, I recommend reading [pattern recognition and machine learning, Bishop 2006].

## distribution probabilities and features

1. Uniform distribution(continuous)

    ![image-20190925111628343](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-25-031628.png) 

    code

    - Uniform distribution has same probaility value on [a, b], easy probability.

[![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-25-031410.png)](https://github.com/graykode/distribution-is-all-you-need/blob/master/graph/uniform.png)

1. Bernoulli distribution(discrete)

    ,

     

    code

    - Bernoulli distribution is not considered about prior probability P(X). Therefore, if we optimize to the maximum likelihood, we will be vulnerable to overfitting.
    - We use **binary cross entropy** to classify binary classification. It has same form like taking a negative log of the bernoulli distribution.

[![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-25-031411.png)](https://github.com/graykode/distribution-is-all-you-need/blob/master/graph/bernoulli.png)

1. Binomial distribution(discrete)

    ,

     

    code

    - Binomial distribution with parameters n and p is the discrete probability distribution of the number of successes in a sequence of n independent experiments.
    - Binomial distribution is distribution considered prior probaility by specifying the number to be picked in advance.

[![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-25-031412.png)](https://github.com/graykode/distribution-is-all-you-need/blob/master/graph/binomial.png)

1. Multi-Bernoulli distribution, Categorical distribution(discrete)

    ,

     

    code

    - Multi-bernoulli called categorical distribution, is a probability expanded more than 2.
    - **cross entopy** has same form like taking a negative log of the Multi-Bernoulli distribution.

[![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-25-31420.png)](https://github.com/graykode/distribution-is-all-you-need/blob/master/graph/categorical.png)

1. Multinomial distribution(discrete)

    ,

     

    code

    - The multinomial distribution has the same relationship with the categorical distribution as the relationship between Bernoull and Binomial.

[![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-25-031413.png)](https://github.com/graykode/distribution-is-all-you-need/blob/master/graph/multinomial.png)

1. Beta distribution(continuous)

    ,

     

    code

    - Beta distribution is conjugate to the binomial and Bernoulli distributions.
    - Using conjucation, we can get the posterior distribution more easily using the prior distribution we know.
    - Uniform distiribution is same when beta distribution met special case(alpha=1, beta=1).

[![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-25-31414.png)](https://github.com/graykode/distribution-is-all-you-need/blob/master/graph/beta.png)

1. Dirichlet distribution(continuous)

    ,

     

    code

    - Dirichlet distribution is conjugate to the MultiNomial distributions.
    - If k=2, it will be Beta distribution.

[![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-25-031419.png)](https://github.com/graykode/distribution-is-all-you-need/blob/master/graph/dirichlet.png)

1. Gamma distribution(continuous)

    ,

     

    code

    - Gamma distribution will be beta distribution, if `Gamma(a,1) / Gamma(a,1) + Gamma(b,1)` is same with `Beta(a,b)`.
    - The exponential distribution and chi-squared distribution are special cases of the gamma distribution.

[![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-25-031414.png)](https://github.com/graykode/distribution-is-all-you-need/blob/master/graph/gamma.png)

1. Exponential distribution(continuous)

    ,

     

    code

    - Exponential distribution is special cases of the gamma distribution when alpha is 1.

[![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-25-031415.png)](https://github.com/graykode/distribution-is-all-you-need/blob/master/graph/exponential.png)

1. Gaussian distribution(continuous)

    ,

     

    code

    - Gaussian distribution is a very common continuous probability distribution

[![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-25-031416.png)](https://github.com/graykode/distribution-is-all-you-need/blob/master/graph/gaussian.png)

1. Normal distribution(continuous)

    ,

     

    code

    - Normal distribution is standarzed Gaussian distribution, it has 0 mean and 1 std.

[![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-25-031420.png)](https://github.com/graykode/distribution-is-all-you-need/blob/master/graph/normal.png)

1. Chi-squared distribution(continuous)

    ,

     

    code

    - Chi-square distribution with k degrees of freedom is the distribution of a sum of the squares of k independent standard normal random variables.
    - Chi-square distribution is special case of Beta distribution

[![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-25-031418.png)](https://github.com/graykode/distribution-is-all-you-need/blob/master/graph/chi-squared.png)

1. Student-t distribution(continuous)

    ,

     

    code

    - The t-distribution is symmetric and bell-shaped, like the normal distribution, but has heavier tails, meaning that it is more prone to producing values that fall far from its mean.

[![img](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-25-031417.png)](https://github.com/graykode/distribution-is-all-you-need/blob/master/graph/student_t.png)

## Author

If you would like to see the details about relationship of distribution probability, please refer to [this](https://en.wikipedia.org/wiki/Relationships_among_probability_distributions).

![image-20190925111634017](https://cy-1256894686.cos.ap-beijing.myqcloud.com/cy/2019-09-25-031634.png)

- Tae Hwan Jung [@graykode](https://github.com/graykode), Kyung Hee Univ CE(Undergraduate).
- Author Email : [nlkey2022@gmail.com](mailto:nlkey2022@gmail.com)
- **If you leave the source, you can use it freely.**

