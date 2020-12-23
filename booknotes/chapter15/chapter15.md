# Chapter 15 - Probabilistic Reasoning Over Time
_In which we try to interpret the present, understand the past, and perhaps predict the future, even when very little is crystal clear._
## Summary
This chapter has addressed the general problem of representing and reasoning about probabilistic temporal processes. The main points are as follows:
* The changing state of the world is handled by using a set of random variables to represent the state at each point in time.
* Representations can be designed to satisfy the __Markov property,__ so that the future is independent of the past given the present. Combined with the assumption that the process is __stationary__ - that is, the dynamics do not change over time - this greatly simplifies the representation.
* A temporal probability model can be thought of as containing a __transition model__ describing the state evolution and a __sensor model__ describing the observation process.
* The principal inference tasks in temporal models are __filtering, prediction, smoothing,__ and computing the __most likely explanation.__ Each of these can be achieved using simple, recursive algorithms whose run time is linear in the length of the sequence.
* Three families of temporal models were studied on more depth: __hidden Markov models, Kalman filters,__ and __dynamic Bayesian networks__ (which include the other two as special cases).
* Unless special assumptions are made, as in Kalman filters, exact interference with many state variables is intractable. In practise, the __particle filtering__ algorithm seems to be an effective approximation algorithm.
* When trying to keep track of many objects, uncertainty arises as to which observations belong to which object - the __data association__ problem. The number of association hypotheses is typically intractable large, but MCMC and particle filtering algorithms for data association work well in practise.