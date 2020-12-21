# Chapter 13 - Quantifying Uncertainty
_In which we see how an agent can tame uncertainty with degrees of belief_

## Probability Model
The basic axioms of probability theory say that every world as a probability between $0$ and $1$ and tha the total probability of the set of possible worlds is $1$:
$$0\leq P(\omega)\leq 1\text{ for every }\omega\text{ and }\sum_{\omega\in\Omega}P(\omega)=1.\tag{13.1}$$
## Event
The probability associated with a proposition is defined to be the sum of the probabilities of the worlds in which it holds:
$$\text{For any proposition }\phi,\quad P(\phi)=\sum_{\omega\in\phi}P(\omega).\tag{13.2}$$
## Conditional/Posterior Probability
Mathematically speaking, conditional probabilities are defined in terms of unconditional probabilities as follows: for any proposition $a$ and $b,$ we have
$$P(a|b)=\frac{P(a\wedge b)}{P(b)},\tag{13.3}$$
which holds whenever $P(b)>0.$
## Inclusion-Exclusion Principle
We can also derive the well-known formula for the probability of a disjunction, sometimes called the __inclusion-exclusion principle:__
$$P(a\vee b)=P(a)+P(b)-P(a\wedge b)\tag{13.4}$$
## Kolmogrov's Axioms
Why, then, can an agent not hold the following set of beliefs (even though they violate Kolmogrov's axioms)?
$$\begin{aligned}
P(a)=0.4\quad\quad\quad P(a\wedge b)=0.0\\
P(b)=0.3\quad\quad\quad P(a\vee b)=0.8\\
\end{aligned}\tag{13.5}$$
## Marginalization
We can write the following general marginalization rule for any sets of variables $\mathbf{Y}$ and $\mathbf{Z}$:
$$\mathbf{P}(\mathbf{Y})=\sum_{\mathbf{z}\in\mathbf{Z}}{\mathbf{P}(\mathbf{Y},\mathbf{z})},\tag{13.6}$$
where $\sum_{\mathbf{z}\in\mathbf{Z}}$ means to sum over all the possible combinations of values of the set of variables $\mathbf{Z}.$

We just used the rule as
$$\mathbf{P}(\textit{Cavity})=\sum_{\mathbf{z}\in\{\textit{Catch, Toothache}\}}{\mathbf{P}(\textit{Cavity},\mathbf{z})},\tag{13.7}.$$
## Conditioning
A variant of this rule involves conditional probabilities instead of joint probabilities, using the product rule:
$$\mathbf{P}(\mathbf{Y})=\sum_{\mathbf{z}}{\mathbf{P}(\mathbf{Y}|\mathbf{z})P(\mathbf{z})}.\tag{13.8}$$
## Normalization
The query is $\mathbf{P}(X|\mathbf{e})$ and can be evaluated as
$$\mathbf{P}(X|\mathbf{e})=\alpha\mathbf{P}(X,\mathbf{e})=\alpha\sum_\mathbf{y}{\mathbf{P}(X,\mathbf{e},\mathbf{y})},\tag{13.9}$$
where the summation is over all possible $\mathbf{y}$s (i.e., all possible combinations of values of the unobserved variables $\mathbf{Y}$).
## Independence
The following assertion seems reasonable:
$$P(\textit{cloudy}|\textit{toothache, catch, cavity})=P(\textit{clody}).\tag{13.10}$$
Independence between propositions $a$ and $b$ can be written as
$$P(a|b)=P(a)\text{ or }P(b|a)=P(b)\text{ or }P(a\wedge b)=P(a)P(b).\tag{13.11}$$
## Bayes' Rule
$$P(b|a)=\frac{P(a|b)P(b)}{P(a)}.\tag{13.12}$$
We will also have occasion to use a more general version conditionalized on some background evidence $\mathbf{e}$:
$$\mathbf{P}(Y|X,\mathbf{e})=\frac{P(X|Y,\mathbf{e})\mathbf{P}(Y|\mathbf{e})}{\mathbf{P}(X|\mathbf{e})}.\tag{13.13}$$
## Causal Diagnostic
Letting $s$ be the proposition that the patient has a stiff neck and $m$ be the proposition that the patient has meningitis, we have
$$\begin{aligned}
P(s|m)&=0.7\\
P(m)&=1/50000\\
P(s|m)&=0.01\\
P(m|s)&=\frac{P(s|m)P(m)}{P(s)}=\frac{0.7\times1/50000}{0.01}=0.0014.
\end{aligned}\tag{13.14}$$
The general form of Bayes' rule with normalization is
$$\mathbf{P}(Y|X)=\alpha\mathbf{P}(X|Y)\mathbf{P}(Y),\tag{13.15}$$
where $\alpha$ is the normalization constant needed to make the entries in $\mathbf{P}(Y|X)$ sum to $1.$
## Using Bayes' rule: Combining evidence
We can try using Bayes’ rule to reformulate the problem:
$$\begin{aligned}
\mathbf{P}(\textit{Cavity}|\textit{toothache}\wedge\textit{catch})\\
&=\alpha\mathbf{P}(\textit{toothache}\wedge\textit{catch}|\textit{Cavity})\mathbf{P}({\textit{Cavity}}).\tag{13.16}
\end{aligned}$$
## Conditional Independence
Mathematically, this property is written as
$$\mathbf{P}(\textit{toothache}\wedge\textit{catch}|\textit{Cavity})=\mathbf{P}(\textit{toothache}|\textit{Cavity})\mathbf{P}(\textit{catch}|\textit{Cavity}).\tag{13.17}$$
We can plug it into Equation $(13.16)$ to obtain the probability of a cavity:
$$\begin{aligned}
\mathbf{P}(\textit{Cavity}|\textit{toothache}\wedge\textit{catch})\\
&=\alpha\mathbf{P}(\textit{toothache}|\textit{Cavity})\mathbf{P}(\textit{catch}|\textit{Cavity})\mathbf{P}({\textit{Cavity}}).\tag{13.18}
\end{aligned}$$
In the dentist domain, for example, it seems reasonable to assert the conditional independence of the variables _Toothache_ and _Catch_, given _Cavity_:
$$\mathbf{P}(\textit{Toothache, catch}|\textit{Cavity})=\mathbf{P}(\textit{Toothache}|\textit{Cavity})\mathbf{P}(\textit{Catch}|\textit{Cavity}).\tag{13.19}$$
## The Wumpus World Revisited
Each square contains a pit with probability $0.2,$ independently of other squares; hence,
$$\mathbf{P}(P_{1,1},\dots,P_{4,4})=\prod_{i,j=1,1}^{4,4}{\mathbf{P}(P_{i,j})}.\tag{13.20}$$
## Summary
This chapter has suggested probability theory as a suitable foundation for uncertain reasoning and provided a gentle introduction to its use.
* Uncertainty arises because of both laziness and ignorance. It is inescapebable in complex, nondeterministic, or partially observable environments.
* Probabilities express the agent's inability to reach a definite decision regarding the truth of a sentence. Probabilities summarize the agent's beliefs relative to the evidence.
* Decision theory combines the agent's beliefs and desires, defining the best action as the one that maximizes expected utility.
* Basic probability statements include __prior probabilities__ and __conditional probabilities__ over simple and complex propositions.
* The axioms of probability constrain the possible asssignment of probabilities to propositions. An agent that violates the axioms must behave irrationally in some cases.
* The __full disjoint probability distribution__ specifies the probability of each complete assignment of values to random variables. It is usually too large to create or use in its explicit form, but when it is available it can be used to answer queries simply by adding up entries for the possible worlds corresponding to the query propositions.
* __Absolute independence__ between subsets of random variables allows the full disjoint distribution to be factored into smaller joint distributions, greatly reducing its complexity. Absolute independence seldom occurs in practise.
* __Bayes' rule__ allows unknown probabilites to be computed from known conditional probabilities, usually in the casual direction. Applying Bayes' rule with many pieces of evidence runs into the same scaling problems as does the full joint distribution.
* __Conditional independence__ brought about by direct causal relationships in the domain mith allow the full disjoint distribution to be factored into smaller, conditional distributions. The __naive Bayes__ model assumes the conditional independence of all effect variables, given a single cause variable, and grows linearly with the number of effects.
* A wumpus-world agent can calculate probabilites for unobserved aspects of the world, thereby improving on the decisions of a purely logical agent. Conditional independence makes these calculations tractable.

