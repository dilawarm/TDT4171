# Chapter 14 - Probabilistic Reasoning
_In which we explain how to build network models to reason under uncertainty according to the laws of probability theory._
## Representing the full joint distribution
The value of this entry is given by the formula
$$P(x_1,\dots,x_n)=\prod_{i=1}^n{\theta(x_i|\textit{parents}(X_i))},\tag{14.1}$$
where $\textit{parents}(X_i)$ denote the values of $\textit{Parents}(X_i)$ that appear in $x_1,\dots,x_n.$
Hence, we can rewrite Equation $(14.1)$ as
$$P(x_1,\dots,x_n)=\prod_{i=1}^n{P(x_i|\textit{parents}(X_i))}.\tag{14.2}$$
## Chain Rule
Comparing it with Equation $(14.2),$ we see that the specification of the joint distribution is equivalent to the general assertion that, for every variable $X_i$ in the network,
$$\mathbf{P}(X_i|X_{i-1},\dots,X_1)=\mathbf{P}(X_i|\textit{Parents}(X_i)),\tag{14.3}$$
provided that $\textit{Parents}(X_i)\subseteq\{X_{i-1},\dots,X_1\}.$
## Inference by enumeration
Hence, we have
$$P(b|j,m)=\alpha P(b)\sum_e{P(e)}\sum_a{P(a|b,e)P(j|a)P(m|a)}.\tag{14.4}$$
## Summary
This chapter has described __Bayesian networks__, as well-developed representation for uncertain knowledge. Bayesian networks play a role roughly analogous to that of propositional logic for definite knowledge.
* A Bayesian network is a directed acyclic graph whose nodes correspond to random variables; each node has a conditional distribution for the node, given its parents.
* Bayesian networks provide a concise way to represent __conditional independence__ relationships in the domain.
* A Bayesian network specifies a full joint distribution; each joint entry is defined as the product of the corresponding entries in the local conditional distributions. A Bayesian network is often exponentially smaller than an explicitly enumerated joint distribution.
* Inference in Bayesian networks means computing the probability distribution of a set of query variables, given a set of evidence variables.
* Various alternative systems for reasoning under uncertainty have been suggested. Generally speaking, __truth-functional__ systems are not well suited for such reasoning.
