# TDT4171 - Artificial Intelligence Methods
## Uncertainty, Bayesian networks
### Exercise 1
#### Problem 1
__a)__ Having at most 2 siblings is equivalent of having 0, 1, or 2 siblings. Let the random variable $X$ denote the number of siblings. Using the addition rule, we get that the probability of a child having at most 2 siblings is
$$\begin{aligned}
P(X\leq2)&=\sum_{x=0}^2{P(X=x)}\\
&=0.15+0.49+0.27\\
&=\underline{\underline{0.91}}.
\end{aligned}$$
__b)__ Using the formula for conditional probability, the probability that a child has more than 2 siblings given that he has at least 1 sibling is
$$P(X>2|X\geq1)=\frac{P(X>2\wedge X\geq1)}{P(X\geq 1)}.$$
The intersection between $X>2$ and $X\geq1$ is $X>2$, so
$$\begin{aligned}
P(X>2\wedge X\geq1)&=P(X>2)\\
&=\left(\sum_{x=3}^4{P(X=x)}\right)+P(X\geq5)\\
&=0.06+0.02+0.01\\
&=0.09.
\end{aligned}$$
Further we have that
$$\begin{aligned}
P(X\geq1)&=\left(\sum_{x=1}^4{P(X=x)}\right)+P(X\geq5)\\
&=0.49+0.27+0.06+0.02+0.01\\
&=0.85.
\end{aligned}$$
Finally,
$$\begin{aligned}
P(X>2|X\geq1)&=\frac{P(X>2\wedge X\geq1)}{P(X\geq 1)}\\
&=\frac{0.09}{0.85}\\
&=\underline{\underline{\frac{9}{85}}}.
\end{aligned}$$
__c)__ To determine the probability that the friends combined have theee siblings, we have to consider all the combinations that give three siblings.
* Every friend has one sibling, so a total of 3 siblings:
$$P(X=1)^3.$$
* Only one of the friends have three siblings, the rest have zero siblings. Since there are three friends, this could happen in three ways:
$$3\cdot P(X=0)^2\cdot P(X=3).$$
* The first friend has zero siblings, the second friend has 1 sibling, and the third friend have 2 siblings. The number of combinations are $3! =6$:
$$6\cdot P(X=0)\cdot P(X=1)\cdot P(X=2).$$

The probability that the friends combined have theee siblings is:
$$\begin{aligned}
P(X=1)^3+3\cdot P(X=0)^2\cdot P(X=3)+6\cdot P(X=0)\cdot P(X=1)\cdot P(X=2)&=0.49^3+3\cdot0.15^2\cdot0.06+6\cdot0.15\cdot0.49\cdot0.27\\
&=0.240769\\
&\approx\underline{\underline{0.24}}.
\end{aligned}$$
__d)__ Let $E$ and $J$ denote the number of siblings Emma and Jacob have, respectively. We know that $E+J=3$. Using Bayes' Rule, the probability of Emma having no siblings is:
$$P(E=0|E+J=3)=\frac{P(E+J=3|E=0)\cdot P(E=0)}{P(E+J=3)}.$$
$E+J=3$ given that $E=0$ implies that $J=3$. So
$$P(E+J=3|E=0)=P(J=3)=0.06.$$
Emma and Jacob can have three siblings combined in the following ways:
* One of the friends have zero siblings, the other has three. 2 combinations:
$$2\cdot P(X=0)\cdot P(X=3).$$
* One of the friends have one sibling, the other has three. 2 combinations:
$$2\cdot P(X=1)\cdot P(X=2).$$

We get
$$\begin{aligned}
P(E+J=3)&=2\cdot P(X=0)\cdot P(X=3)+2\cdot P(X=1)\cdot P(X=2)\\
&=2\cdot0.15\cdot0.06+2\cdot0.49\cdot0.27\\
&=0.2826.
\end{aligned}$$
Finally,
$$\begin{aligned}
P(E=0|E+J=3)&=\frac{P(E+J=3|E=0)\cdot P(E=0)}{P(E+J=3)}\\
&=\frac{0.06\cdot0.15}{0.2826}\\
&=\underline{\underline{\frac{5}{157}}}.
\end{aligned}$$
