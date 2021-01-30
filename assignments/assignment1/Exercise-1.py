from collections import defaultdict

import numpy as np


class Variable:
    def __init__(self, name, no_states, table, parents=[], no_parent_states=[]):
        """
        name (string): Name of the variable
        no_states (int): Number of states this variable can take
        table (list or Array of reals): Conditional probability table (see below)
        parents (list of strings): Name for each parent variable.
        no_parent_states (list of ints): Number of states that each parent variable can take.

        The table is a 2d array of size #events * #number_of_conditions.
        #number_of_conditions is the number of possible conditions (prod(no_parent_states))
        If the distribution is unconditional #number_of_conditions is 1.
        Each column represents a conditional distribution and sum to 1.

        Here is an example of a variable with 3 states and two parents cond0 and cond1,
        with 3 and 2 possible states respectively.
        +----------+----------+----------+----------+----------+----------+----------+
        |  cond0   | cond0(0) | cond0(1) | cond0(2) | cond0(0) | cond0(1) | cond0(2) |
        +----------+----------+----------+----------+----------+----------+----------+
        |  cond1   | cond1(0) | cond1(0) | cond1(0) | cond1(1) | cond1(1) | cond1(1) |
        +----------+----------+----------+----------+----------+----------+----------+
        | event(0) |  0.2000  |  0.2000  |  0.7000  |  0.0000  |  0.2000  |  0.4000  |
        +----------+----------+----------+----------+----------+----------+----------+
        | event(1) |  0.3000  |  0.8000  |  0.2000  |  0.0000  |  0.2000  |  0.4000  |
        +----------+----------+----------+----------+----------+----------+----------+
        | event(2) |  0.5000  |  0.0000  |  0.1000  |  1.0000  |  0.6000  |  0.2000  |
        +----------+----------+----------+----------+----------+----------+----------+

        To create this table you would use the following parameters:

        Variable('event', 3, [[0.2, 0.2, 0.7, 0.0, 0.2, 0.4],
                              [0.3, 0.8, 0.2, 0.0, 0.2, 0.4],
                              [0.5, 0.0, 0.1, 1.0, 0.6, 0.2]],
                 parents=['cond0', 'cond1'],
                 no_parent_states=[3, 2])
        """
        self.name = name
        self.no_states = no_states
        self.table = np.array(table)
        self.parents = parents
        self.no_parent_states = no_parent_states

        if self.table.shape[0] != self.no_states:
            raise ValueError(
                f"Number of states and number of rows in table must be equal. "
                f"Recieved {self.no_states} number of states, but table has "
                f"{self.table.shape[0]} number of rows."
            )

        if self.table.shape[1] != np.prod(no_parent_states):
            raise ValueError(
                "Number of table columns does not match number of parent states combinations."
            )

        if not np.allclose(self.table.sum(axis=0), 1):
            raise ValueError("All columns in table must sum to 1.")

        if len(parents) != len(no_parent_states):
            raise ValueError(
                "Number of parents must match number of length of list no_parent_states."
            )

    def __str__(self):
        """
        Pretty string for the table distribution
        For printing to display properly, don't use variable names with more than 7 characters
        """
        width = int(np.prod(self.no_parent_states))
        grid = np.meshgrid(*[range(i) for i in self.no_parent_states])
        s = ""
        for (i, e) in enumerate(self.parents):
            s += "+----------+" + "----------+" * width + "\n"
            gi = grid[i].reshape(-1)
            s += f"|{e:^10}|" + "|".join([f'{e + "("+str(j)+")":^10}' for j in gi])
            s += "|\n"

        for i in range(self.no_states):
            s += "+----------+" + "----------+" * width + "\n"
            state_name = self.name + f"({i})"
            s += f"|{state_name:^10}|" + "|".join(
                [f"{p:^10.4f}" for p in self.table[i]]
            )
            s += "|\n"

        s += "+----------+" + "----------+" * width + "\n"

        return s

    def probability(self, state, parentstates):
        """
        Returns probability of variable taking on a "state" given "parentstates"
        This method is a simple lookup in the conditional probability table, it does not calculate anything.

        Input:
            state: integer between 0 and no_states
            parentstates: dictionary of {'parent': state}
        Output:
            float with value between 0 and 1
        """
        if not isinstance(state, int):
            raise TypeError(
                f"Expected state to be of type int; got type {type(state)}."
            )
        if not isinstance(parentstates, dict):
            raise TypeError(
                f"Expected parentstates to be of type dict; got type {type(parentstates)}."
            )
        if state >= self.no_states:
            raise ValueError(
                f"Recieved state={state}; this variable's last state is {self.no_states - 1}."
            )
        if state < 0:
            raise ValueError(f"Recieved state={state}; state cannot be negative.")

        table_index = 0
        for variable in self.parents:
            if variable not in parentstates:
                raise ValueError(
                    f"Variable {variable.name} does not have a defined value in parentstates."
                )

            var_index = self.parents.index(variable)
            table_index += parentstates[variable] * np.prod(
                self.no_parent_states[:var_index]
            )

        return self.table[state, int(table_index)]


class BayesianNetwork:
    """
    Class representing a Bayesian network.
    Nodes can be accessed through self.variables['variable_name'].
    Each node is a Variable.

    Edges are stored in a dictionary. A node's children can be accessed by
    self.edges[variable]. Both the key and value in this dictionary is a Variable.
    """

    def __init__(self):
        self.edges = defaultdict(lambda: [])  # All nodes start out with 0 edges
        self.variables = {}  # Dictionary of "name":TabularDistribution

    def add_variable(self, variable):
        """
        Adds a variable to the network.
        """
        if not isinstance(variable, Variable):
            raise TypeError(f"Expected {Variable}; got {type(variable)}.")
        self.variables[variable.name] = variable

    def add_edge(self, from_variable, to_variable):
        """
        Adds an edge from one variable to another in the network. Both variables must have
        been added to the network before calling this method.
        """
        if from_variable not in self.variables.values():
            raise ValueError("Parent variable is not added to list of variables.")
        if to_variable not in self.variables.values():
            raise ValueError("Child variable is not added to list of variables.")
        self.edges[from_variable].append(to_variable)

    def sorted_nodes(self):
        """
        Kahn's algorithm for putting variables in lexicographical topological order.
        Returns: List of sorted variable names.
        """
        in_degree = {
            variable: 0 for variable in self.variables.values()
        }  # In-degree for all the variables, initially 0

        for variable in self.variables.values():
            for from_variable in self.edges:
                if variable in self.edges[from_variable]:
                    in_degree[variable] += 1

        queue = [
            variable for variable in self.variables.values() if in_degree[variable] == 0
        ]  # Contains all the variables with in-degree equal to 0, i.e., variables with no parents.

        topological_order = []

        while queue:
            variable = queue.pop(0)
            topological_order.append(variable)

            for to_variable in self.edges[variable]:
                in_degree[to_variable] -= 1
                # If the in_degree is 0, then it is safe to add variable to solution
                if in_degree[to_variable] == 0:
                    queue.append(to_variable)

        return [variable.name for variable in topological_order]


class InferenceByEnumeration:
    def __init__(self, bayesian_network):
        self.bayesian_network = bayesian_network
        self.topo_order = bayesian_network.sorted_nodes()

    def _enumeration_ask(self, X, evidence):
        """
        An implementation of the Enumeration-Ask algortihm as described in Problem 4 b)
        """
        Q = []  # A distribution over X, initially empty
        for x in range(self.bayesian_network.variables[X].no_states):
            evidence = evidence.copy()
            evidence[X] = x  # Extending evidence with X = x
            Q.append(self._enumerate_all(self.topo_order, evidence))
        Q = np.array(Q)
        return Q / Q.sum(
            axis=0, keepdims=1
        )  # normalizing the distribution over X so that the sum is 1.0

    def _enumerate_all(self, vars, evidence):
        """
        An implementation of the Enumerate-All algortihm as described in Problem 4 b)
        """
        if len(vars) == 0:
            return 1.0

        Y = vars[0]

        if Y in evidence:
            return self.bayesian_network.variables[Y].probability(
                evidence[Y], evidence
            ) * self._enumerate_all(vars[1:], evidence)
        else:
            extended_evidence = evidence.copy()
            probabilities = []
            for y in range(self.bayesian_network.variables[Y].no_states):
                extended_evidence[Y] = y  # Extending evicence with Y = y
                probabilities.append(
                    self.bayesian_network.variables[Y].probability(y, evidence)
                    * self._enumerate_all(vars[1:], extended_evidence)
                )
            return sum(probabilities)

    def query(self, var, evidence={}):
        """
        Wrapper around "_enumeration_ask" that returns a
        Tabular variable instead of a vector
        """
        q = self._enumeration_ask(var, evidence).reshape(-1, 1)
        return Variable(var, self.bayesian_network.variables[var].no_states, q)


def problem3c():
    d1 = Variable("A", 2, [[0.8], [0.2]])
    d2 = Variable("B", 2, [[0.5, 0.2], [0.5, 0.8]], parents=["A"], no_parent_states=[2])
    d3 = Variable("C", 2, [[0.1, 0.3], [0.9, 0.7]], parents=["B"], no_parent_states=[2])
    d4 = Variable("D", 2, [[0.6, 0.8], [0.4, 0.2]], parents=["B"], no_parent_states=[2])

    print(f"Probability distribution, P({d1.name})")
    print(d1)

    print(f"Probability distribution, P({d2.name} | {d1.name})")
    print(d2)

    print(f"Probability distribution, P({d3.name} | {d2.name})")
    print(d3)

    print(f"Probability distribution, P({d4.name} | {d2.name})")
    print(d4)

    bn = BayesianNetwork()

    bn.add_variable(d1)
    bn.add_variable(d2)
    bn.add_variable(d3)
    bn.add_variable(d4)
    bn.add_edge(d1, d2)
    bn.add_edge(d2, d3)
    bn.add_edge(d2, d4)

    inference = InferenceByEnumeration(bn)
    posterior = inference.query("C", {"D": 1})

    print(f"Probability distribution, P({d3.name} | !{d4.name})")
    print(posterior)


def monty_hall():
    chosen_by_guest = Variable("ChosenByGuest", 3, [[1 / 3], [1 / 3], [1 / 3]])
    prize = Variable("Prize", 3, [[1 / 3], [1 / 3], [1 / 3]])
    opened_by_host = Variable(
        "OpenedByHost",
        3,
        [
            [0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 0.0, 1.0, 0.5],
            [0.5, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.5],
            [0.5, 1.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0],
        ],
        parents=["Prize", "ChosenByGuest"],
        no_parent_states=[3, 3],
    )

    bn = BayesianNetwork()
    print(opened_by_host)

    bn.add_variable(chosen_by_guest)
    bn.add_variable(opened_by_host)
    bn.add_variable(prize)
    bn.add_edge(prize, opened_by_host)
    bn.add_edge(chosen_by_guest, opened_by_host)

    inference = InferenceByEnumeration(bn)
    posterior = inference.query(
        "Prize", {"ChosenByGuest": 0, "OpenedByHost": 2}
    )  # Using 0 and 2 because the states are zero-indexed.

    print(
        f"Probability distribution, P({prize.name} | {chosen_by_guest.name} = 1, {opened_by_host.name} = 3)"
    )
    print(posterior)


if __name__ == "__main__":
    problem3c()
    monty_hall()