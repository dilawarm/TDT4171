# Chapter 16 - Making Simple Decisions
_In which we see how an agent make decisions so that it gets what it wants - on average, at least._
## Expected Utility
The __expected utility__ of an action given the evidence, $EU(a|\mathbf{e}),$ is just the average utility of the outcomes, weighted by the probability that the outcome occurs:
$$EU(a|\mathbf{e})=\sum_{s'}{P(\text{RESULT}(a)=s'|a,\mathbf{e})U(s')}\tag{16.1}$$
## Preferences lead to utility
It is easy to see, in fact, that an agent's behavior would not change if its utility function $U(S)$ were transformed according to
$$U'(S)=aU(S)+b,\tag{16.2}$$
where $a$ and $b$ are constants and $a>0;$ an affine transformation.
## Summary
This chapter shows how to combine utility theory with probability to enable an agent to select actions that will maximize its expected performance.
* __Probability theory__ describes what an agent should believe on the basis of evidence, __utility theory__ describes what an agent wants, and __decision theory__ puts the two together to describe what an agent should do.
* We can use decision theory to build a system that makes decision by considering all possible actions and choosing the one that leads to the best expected outcome. Such a system is known as a __rational agent.__
* Utility theory shows that an agent whose preferences between lotteries are consistent with a set of simple axioms can be described as possessing a utility function; furthermore, the agent selects actions as if maximizing its expected utility.
* __Multiattribute utility theory__ deals with utilities that depends on several distinct attributes of states. __Stochastic dominance__ is a particularly useful technique for making unambigious decisions, even without precise utility values for attributes.
* __Decision networks__ provide a simple formalism for expresssing and solving decision problems. They are a natural extension of Bayesian networks, containing decision and utility nodes in addition to chance nodes.
* Sometimes, solving a problem involves finding more information before making a decision. The __value of information__ is defined as the expected improvement in utilitty compared with making a decision without the information.
* __Expert systems__ that incorporate utility information have additional capabilities compared with pure inference systems. In addition to being able to make decisions, they can use the value of information to decide which questions to ask, if any; they can recommend contingency plans; and they can calculate the sensitivity of their decisions to small changes in probability and utility assessments.