import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from matplotlib.figure import Figure
import pandas as pd

plt.rc("text", usetex=True)
plt.rc(
    "text.latex",
    preamble=r"\usepackage{mathtools,bm}"
    r"\newcommand{\myboldsymbol}[1]{\bm{\mathrm{#1}}}"
    r"\DeclareMathOperator*{\argmax}{arg\,max}",
)


# ---------------------------------- Plotting utilities
def get_plot() -> Tuple[Figure, np.ndarray]:
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(18, 12), constrained_layout=True)
    for ax in axs.reshape(-1):
        ax.grid(True)

    fig.suptitle("1 Hidden Markov Model", fontsize=30)
    return fig, axs


def post_processing_plot(fig: Figure, axs: np.ndarray) -> None:
    axs[1, 2].axis("off")
    for ax in axs.reshape(-1):
        for tick in ax.xaxis.get_major_ticks() + ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(15)
        ax.legend(loc="best", prop={"size": 15})
    fig.show()
    fig.savefig("result.pdf")


def plot(
    title: str, x_axis: np.ndarray, messages: np.ndarray, y_label: str, ax
) -> None:
    ax.set_title(title, fontsize=20)
    width = 0.35
    ax.set_ylim(0, 1)
    ax.bar(x_axis - width / 2, messages[0], width, label=r"$X_t$=true")
    ax.bar(x_axis + width / 2, messages[1], width, label=r"$X_t$=false")
    ax.set_xlabel(r"$t$", fontsize=20)
    ax.set_ylabel(y_label, fontsize=20)


# ---------------------------------- End


class Model:
    def __init__(
        self, prior: np.ndarray, T: np.ndarray, O: List[np.ndarray], E: List[int]
    ) -> None:
        """
        Representing a HMM model
        :param prior: Prior distribution
        :param T: Transition matrix
        :param O: Observation matrices
        :param E: Evidence
        """
        self.prior = prior
        self.T = T
        self.O = O
        self.E = E

    @staticmethod
    def _normalize(array: np.ndarray) -> np.ndarray:
        """Normalize distribution"""
        return array / array.sum(axis=0, keepdims=True)

    def _compute_f_next(self, f_prev: np.ndarray, index: int) -> np.ndarray:
        """Page 572, forward"""
        message = self.O[self.E[index]].dot(self.T.T.dot(f_prev))  # Compute
        return self._normalize(message)  # Normalize and return

    def compute_f_all(self) -> np.ndarray:
        """Filtering - compute sequence of forward messages"""
        f_messages = np.empty((2, len(self.E)))  # Array for storing values
        f_message = self.prior  # Set initial
        # Compute
        for i in range(len(self.E)):
            f_message = self._compute_f_next(f_message, i)
            f_messages[:, i, np.newaxis] = f_message
        return f_messages

    def _compute_f_next_no_e(self, f_prev: np.ndarray) -> np.ndarray:
        """Page 572, forward without new evidence"""
        message = self.T.T.dot(f_prev)  # Compute
        return self._normalize(message)  # Normalize and return

    def compute_f_all_no_e(self, end: int) -> np.ndarray:
        """Prediction - compute sequence of forward messages without new evidence"""
        n = end - len(self.E)  # n predictions
        f_messages_no_e = np.empty((2, n))  # Array for storing values

        # Set initial, last forward message with evidence on same time step
        f_messages = self.compute_f_all()
        f_message_no_e = np.array([[f_messages[0][-1]], [f_messages[1][-1]]])

        # Compute sequence
        for i in range(n):
            f_message_no_e = self._compute_f_next_no_e(f_message_no_e)
            f_messages_no_e[:, i, np.newaxis] = f_message_no_e
        return f_messages_no_e

    def _compute_b_next(self, b_prev: np.ndarray, index: int) -> np.ndarray:
        """Page 574, backward"""
        return self.T.dot(self.O[self.E[index]].dot(b_prev))  # Compute and return

    def compute_b_all(self) -> np.ndarray:
        """Compute sequence of backward messages"""
        b_messages = np.empty((2, len(self.E)))  # Array for storing values

        b_message = np.array([[1], [1]])  # Set initial
        # Compute sequence
        for i in reversed(range(len(self.E))):
            b_message = self._compute_b_next(b_message, i)
            b_messages[:, i, np.newaxis] = b_message
        return b_messages

    def compute_smoothing(self) -> np.ndarray:
        """Page 574, smoothing"""
        f_messages = self.compute_f_all()  # forward messages
        b_messages = self.compute_b_all()  # backward messages
        return self._normalize(
            np.multiply(
                np.concatenate([self.prior, f_messages], axis=1)[:, :-1], b_messages
            )
        )

    def _compute_mls(self, prev_max: np.ndarray, index: int) -> np.ndarray:
        """Page 576, most likely sequence"""
        print(prev_max)
        print(np.max(self.T.T * prev_max, axis=0))
        message = self.O[self.E[index]].dot(np.max(self.T.T * prev_max, axis=0))[
            :, np.newaxis
        ]  # Compute
        return message

    def compute_mls_all(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute sequence of most likely sequence messages"""
        max_vals = np.empty((2, len(self.E)))  # Array for storing probabilities
        backtracking_graph = np.empty(
            (2, len(self.E) - 1)
        )  # Array for storing sequence
        max_vals[:, 0, np.newaxis] = self._compute_f_next(self.prior, 0)  # Set initial
        # print(max_vals[:, 0, np.newaxis])
        # Compute sequence
        for i in range(1, len(self.E)):
            prev_max = max_vals[:, i - 1, np.newaxis]
            backtracking_graph[:, i - 1] = np.argmax(
                self.T.T * prev_max, axis=0
            )  # Compute backtracking graph
            max_vals[:, i, np.newaxis] = self._compute_mls(prev_max, i)

        # print(max_vals)
        return max_vals, backtracking_graph.astype(int)


def get_problem1_model() -> Model:
    """Define and return the HMM"""
    alpha = 0.6

    beta_1 = 0.7
    beta_0 = 0.4

    gamma_1 = 0.8
    gamma_0 = 0.3

    return Model(
        prior=np.array([[alpha], [1 - alpha]]),
        T=np.array([[beta_1, 1 - beta_1], [beta_0, 1 - beta_0]]),
        O=[
            np.array([[1 - gamma_1, 0], [0, 1 - gamma_0]]),
            np.array([[gamma_1, 0], [0, gamma_0]]),
        ],
        # 1 = Fish nearby, 0 = No fish nearby
        E=[1, 1],
    )


def problem_b(model: Model, ax) -> None:
    # Compute forward messages
    f_messages = model.compute_f_all()
    print(f"Problem 1b - Filtering")
    print(
        f"{pd.DataFrame(f_messages, index=[True, False], columns=list(range(1, f_messages.shape[1] + 1)))}\n"
    )
    # Plotting
    x_axis = np.arange(1, len(model.E) + 1)
    plot(
        "Problem 1b - Filtering",
        x_axis,
        f_messages,
        r"$\myboldsymbol{P}(X_t|\myboldsymbol{e}_{1:t})$",
        ax,
    )


def problem_c(model: Model, ax, end: int) -> None:
    # Compute forward messages without new evidence
    f_messages_no_e = model.compute_f_all_no_e(end)
    print(f_messages_no_e)
    start = len(model.E)
    print(f"Problem 1c - Prediction")
    print(
        f"{pd.DataFrame(f_messages_no_e, index=[True, False], columns=list(range(start + 1, end + 1)))}\n"
    )
    # Plotting
    x_axis = np.arange(start + 1, end + 1)
    plot(
        "Problem 1c - Prediction",
        x_axis,
        f_messages_no_e,
        fr"$\myboldsymbol{{P}}(X_t|\myboldsymbol{{e}}_{{1:{start}}})$",
        ax,
    )


def problem_d1(model: Model, ax) -> None:
    # Compute backward messages
    b_messages = model.compute_b_all()
    # Plotting
    x_axis = np.arange(1, len(model.E) + 1, 1)
    plot(
        "Problem 1d - Backward messages",
        x_axis,
        b_messages,
        fr"$\myboldsymbol{{P}}(\myboldsymbol{{e}}_{{t:{len(model.E)}}}|X_{{t-1}})$",
        ax,
    )


def problem_d2(model: Model, ax) -> None:
    # Compute smoothing messages
    s_messages = model.compute_smoothing()
    print(f"Problem 1d - Smoothing")
    print(
        f"{pd.DataFrame(s_messages, index=[True, False], columns=list(range(0, s_messages.shape[1])))}\n"
    )
    # Plotting
    x_axis = np.arange(0, s_messages.shape[1], 1)
    plot(
        "Problem 1d - Smoothing",
        x_axis,
        s_messages,
        fr"$\myboldsymbol{{P}}(X_t|\myboldsymbol{{e}}_{{1:{s_messages.shape[1]}}})$",
        ax,
    )


def problem_e(model: Model, ax) -> None:
    # Compute most likely sequence messages
    mls_messeges, sequence = model.compute_mls_all()
    print(f"Problem 1e - Most likely sequence probabilities")
    print(
        f"{pd.DataFrame(mls_messeges, index=[True, False], columns=list(range(1, mls_messeges.shape[1] + 1)))}\n"
    )
    print(f"Most likely sequence - backtracking graph")
    print(
        f"{pd.DataFrame(sequence, index=[True, False], columns=list(range(1, sequence.shape[1] + 1)))}\n"
    )
    # Plotting
    x_axis = np.arange(1, mls_messeges.shape[1] + 1, 1)
    plot(
        "Problem 1e - Most likely sequence",
        x_axis,
        mls_messeges,
        r"$\\argmax_{{x_1,...,x_{{t-1}}}}\myboldsymbol{{P}}(x_1,...,x_{{t-1}},X_t|\myboldsymbol{{e}}_{1:t})$",
        ax,
    )


def run():
    fig, axs = get_plot()
    model = get_problem1_model()
    # Problem 1b
    problem_b(model, axs[0, 0])
    # Problem 1c
    problem_c(model, axs[0, 1], 30)
    # Problem 1d - backward messages
    problem_d1(model, axs[0, 2])
    # Problem 1d - smoothing
    problem_d2(model, axs[1, 0])
    # Problem 1e
    problem_e(model, axs[1, 1])
    # post_processing_plot(fig, axs)


if __name__ == "__main__":
    run()
