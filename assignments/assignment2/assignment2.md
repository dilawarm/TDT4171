# TDT4171 - Artificial Intelligence Methods
## Assignment 2 - Probabilistic Reasoning over Time
### Problem 1
__a)__ Formulating the problem given in the problem as a hidden Markov model:
![](hmm.jpg)
**The complete probability tables for the model:**
| $F_{t-1}$ | $P(F_t)$ |
| --------- | -------- |
| $t$       | $0.8$    |
| $f$       | $0.3$    |

| $F_t$ | $P(B_t)$ |
| ----- | -------- |
| $t$   | $0.75$   |
| $f$   | $0.2$    |

__b)__ Output from [assignment2.py](assignment2.py) after calculating
$$\mathbf{P}\left(X_t|\mathbf{e}_{1:t}\right),\quad\text{for}\quad t=1,\dots,6\tag{1}$$
```{sh}
Problem 1b)
Calculating P(X_t|e_1:t) for t = 1,...,6:
P(X_1|e_1:1) = [0.78947368 0.21052632]
P(X_2|e_1:2) = [0.86799277 0.13200723]
P(X_3|e_1:3) = [0.38966879 0.61033121]
P(X_4|e_1:4) = [0.74935019 0.25064981]
P(X_5|e_1:5) = [0.33650851 0.66349149]
P(X_6|e_1:6) = [0.72719639 0.27280361]
```
This operation is __filtering__ because we are calculating $\mathbf{P}(X_t|\mathbf{e}_{1:t})$ for every observation in our evidence vector. Filtering gives us the belief state of a rational agent. In this example, filtering will give us the probabilities for if we are going to find fish at day $t$ given the evidence up until day $t$.

__c)__ Output from [assignment2.py](assignment2.py) after calculating
$$\mathbf{P}\left(X_t|\mathbf{e}_{1:6}\right),\quad\text{for}\quad t=7,\dots,30\tag{2}$$
```{sh}
Calculating P(X_t|e_1:6) for t = 7,...,30:
P(X_7|e_1:6) = [0.66359819 0.33640181]
P(X_8|e_1:6) = [0.6317991 0.3682009]
P(X_9|e_1:6) = [0.61589955 0.38410045]
P(X_10|e_1:6) = [0.60794977 0.39205023]
P(X_11|e_1:6) = [0.60397489 0.39602511]
P(X_12|e_1:6) = [0.60198744 0.39801256]
P(X_13|e_1:6) = [0.60099372 0.39900628]
P(X_14|e_1:6) = [0.60049686 0.39950314]
P(X_15|e_1:6) = [0.60024843 0.39975157]
P(X_16|e_1:6) = [0.60012422 0.39987578]
P(X_17|e_1:6) = [0.60006211 0.39993789]
P(X_18|e_1:6) = [0.60003105 0.39996895]
P(X_19|e_1:6) = [0.60001553 0.39998447]
P(X_20|e_1:6) = [0.60000776 0.39999224]
P(X_21|e_1:6) = [0.60000388 0.39999612]
P(X_22|e_1:6) = [0.60000194 0.39999806]
P(X_23|e_1:6) = [0.60000097 0.39999903]
P(X_24|e_1:6) = [0.60000049 0.39999951]
P(X_25|e_1:6) = [0.60000024 0.39999976]
P(X_26|e_1:6) = [0.60000012 0.39999988]
P(X_27|e_1:6) = [0.60000006 0.39999994]
P(X_28|e_1:6) = [0.60000003 0.39999997]
P(X_29|e_1:6) = [0.60000002 0.39999998]
P(X_30|e_1:6) = [0.60000001 0.39999999]
```
This operation is called __prediction__ because we are calculating $\mathbf{P}(X_t|\mathbf{e}_{1:6})$ for $t$'s where we don't have any evidence. Prediction gives us an evaluation of possible action sequences. In this example, prediction gives us the probability for finding fish nearby from day 7 to 30 given our observations the first 6 days.
As $t$ increases, the distribution converges to $\left[\frac{3}{5},\frac{2}{5}\right].$ In other words,
$$\lim_{t\to\infty}{\mathbf{P}(X_t|\mathbf{e}_{1:6})}=\left[\frac{3}{5},\frac{2}{5}\right].$$

__d)__ Output from [assignment2.py](assignment2.py) after calculating
$$\mathbf{P}\left(X_t|\mathbf{e}_{1:6}\right),\quad\text{for}\quad t=0,\dots,5\tag{3}$$
```{sh}
Problem 1d)
Calculating P(X_t|e_1:6) for t = 0,...,5:
P(X_5|e_1:6) = [0.33650851 0.66349149]
P(X_4|e_1:6) = [0.68318883 0.31681117]
P(X_3|e_1:6) = [0.55901131 0.44098869]
P(X_2|e_1:6) = [0.864022 0.135978]
P(X_1|e_1:6) = [0.89384268 0.10615732]
P(X_0|e_1:6) = [0.74692134 0.25307866] 
```
This operation is called __smoothing__ because we are calculating $\mathbf{P}(X_t|\mathbf{e}_{1:6})$ for past $t$'s. Smoothing gives us better estimates of past states. In this example, smoothing will give us better estimates for finding fish nearby from day 0 to 5.

__e)__ Output from [assignment2.py](assignment2.py) after calculating
$$\argmax_{x_1,\dots,x_{t-1}}\mathbf{P}\left(x_1,\dots,x_{t-1},X_t|\mathbf{e}_{1:t}\right),\quad\text{for}\quad t=1,\dots,6\tag{4}$$
```{sh}
Problem 1e)
Calculating P(x_1,...,x_(t-1),X_t|e_1:t) for t = 1,...,6:
m_1:1 = [0.3  0.07] --> argmax([0.3  0.07]) = 0 --> True
m_1:2 = [0.18  0.012] --> argmax([0.18  0.012]) = 0 --> True
m_1:3 = [0.036  0.0288] --> argmax([0.036  0.0288]) = 0 --> True
m_1:4 = [0.0216   0.004032] --> argmax([0.0216   0.004032]) = 0 --> True
m_1:5 = [0.00432  0.003456] --> argmax([0.00432  0.003456]) = 0 --> True
m_1:6 = [0.002592   0.00048384] --> argmax([0.002592   0.00048384]) = 0 --> True
```
Here $m_{1:t}(i)$ gives the probability of the **most likely sequence** to state $i$. The most likely sequence tells us which states are most probable given our evidence. In this example, the most likely sequence tells us the most likely occurrences of fish being nearby the first 6 days given our evidence.

### Problem 2
