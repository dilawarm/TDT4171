# Chapter 18 - Learning From Examples
_In which we describe agents that can improve their behavior through diligent study of their own experiences._
## Summary
This chapter has concentrated on inductive learning of functions from examples. The main points were as follows:
* Learning takes many forms, depending on the nature of the agent, the component to be improved, and the available feedback.
* If the available feedback provides the correct answer for example inputs, then the learning problem is called __supervised learning.__ The task is to learn a function. Learning a discreate-value funciton is called __classification;__ learning a continious function is called __regression.__
* Inductive learning involves finding a hypothesis that agrees well with the examples. __Ockham's razor__ suggests choosing the simplest consistent hypothesis. The difficulty of the task depends on the chosen representation.
* __Decision trees__ can represent all Boolean functions. The __information-gain__ heuristic provides an efficient method for finding a simple, consistent decision tree.
* The performance of a learning algorithm is measured by the __learning curve,__ which shows the prediction accuracy on the __test set__ as a function of the __training-set__-size.
* When there are multiple models to choose from, __cross-validation__ can be used to select a model that will generalize well.
* Sometimes not all errors are equal. A __loss function__ tells us how bad each error is; the goal is then to minimize loss over a validation set.
* __Computational learning theory__ analyzes the sample complexity and computational complexity of inductive learning. There is a tradeoff between the expressiveness of the hypothesis language and the ease of learning.
* __Linear regression__ is a widely used model. The optimal parameters of a linear regression model can be found by gradient descent search, or computed exactly.
* A linear classifier with a hard threshold - also known as a __perceptron__ - can be trained by a simple weight update rule to fit data that are __linearly separable.__ In other cases, the rule fails to converge.
* __Logistic regression__ replaces the perceptron's hard threshold with a soft threshold defined by a logistic function. Gradient descent works well even for noisy data that are not linearly separable.
* __Neural networks__ represent complex nonlinear functions with a network of linear-threshold units. A multilayer feed-forward neural network can represent any function, given enough units. The __back-propagation__ algorithm implements a gradient descent in parameter space to minimize the output error.
* __Nonparametric model__ use all the data to make each prediction, rather than trying to summarize the data first with a few parameters. Examples include __nearest neighbors__ and __locally weighted regression.__
* __Support vector machines__ find linear separators with __maximum margin__ to improve the generalization performance of the classifier. __Kernel methods__ implicitly transform the input data into a high-dimensional space where a linear separator may exist, even if the original data are non-separable.
* Ensemble methods such as __boosting__ often perform better than individual methods. In __online learning__ we can aggergate the opinions of experts to come arbitrarily close to the best expert's performance, even when the distribution of the data is constantly shifting.