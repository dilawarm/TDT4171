import pandas as pd
import math
import numpy as np
import pydot
import uuid
import operator


Y = "Survived"
DROP_ATTRIBUTES = ["Name", "Ticket", "Cabin"]
CATEGORICAL_ATTRIBUTES = ["Pclass", "Sex", "SibSp", "Parch", "Embarked"]
NUMERICAL_ATTRIBUTES = ["Age", "Fare"]

# Lambda function for creating dataset based on path and which predictors I want to drop.
create_dataset = (
    lambda path, drop_attributes: pd.read_csv(path)
    .dropna(axis=0)
    .drop(
        drop_attributes,
        axis=1,
    )
)

# Lambda function for retuning the plularity value in a dataset.
plurality_value = lambda examples: str(examples[Y].mode()[0])

# Lambda function for checking if a dataset only contains the same classifications.
same_classification = lambda examples: (
    examples.to_numpy()[0] == examples.to_numpy()
).all()

# Lambda function for calculating entropy.
entropy = lambda examples: -sum(
    (val / len(examples)) * math.log2(val / len(examples))
    for val in examples[Y].value_counts()
)


def categorical_importance(attribute, examples):

    """
    Function for caulcating importance for a categorical variable.

    attribute: An attribute (column) from dataset
    examples: Dataset where we are checking the importance of our attribute.
    """

    subset = examples[attribute].value_counts()
    subset_keys = subset.keys()
    return entropy(examples) - sum(
        (val / len(examples)) * entropy(examples[examples[attribute] == subset_keys[s]])
        for s, val in enumerate(subset)
    )


def numerical_importance(attribute, examples, split):

    """
    Function for caulcating importance for a continious (numerical) variable.

    attribute: An attribute (column) from dataset
    examples: Dataset where we are checking the importance of our attribute.
    """

    keys = examples[attribute].value_counts().keys()
    subsets = [
        examples[examples[attribute] <= split],
        examples[examples[attribute] > split],
    ]

    return entropy(examples) - sum(
        (len(subset) / len(examples)) * entropy(subset) for subset in subsets
    )


def decision_tree_learning(examples, attributes, parent_examples):

    """
    Function for leanring a decision tree.

    examples: Our training dataset
    attributes: Attributes we are considering for a split.
    parent_examples: The dataset used on the preceding node of the split we are considering.
    """

    if examples.empty:
        return plurality_value(parent_examples)
    elif same_classification(examples):
        return str(examples[Y].unique()[0])
    elif not attributes:
        return plurality_value(examples)
    else:
        attribute_importances = {}  # Store importance of each attribute
        decision_tree_splits = {}  # Storing splits for continous variables.
        for a in attributes:
            if a in CATEGORICAL_ATTRIBUTES:
                attribute_importances[a] = categorical_importance(a, examples)
            else:
                sorted_examples = examples.sort_values(by=a, axis=0)
                values = np.array(sorted_examples[a].unique())

                if len(values) == 1:
                    split = values[0]
                    decision_tree_splits[a] = split
                    attribute_importances[a] = numerical_importance(
                        a, sorted_examples, split
                    )

                else:
                    splits = (values[1:] + values[:-1]) / 2
                    split_importances = [
                        numerical_importance(a, sorted_examples, split)
                        for split in splits
                    ]
                    best_importance = max(split_importances)
                    split = splits[split_importances.index(best_importance)]
                    decision_tree_splits[a] = split
                    attribute_importances[a] = best_importance

        A = max(attribute_importances.items(), key=operator.itemgetter(1))[
            0
        ]  # Choosing the most important attribute
        attributes = [a for a in attributes if a != A]

        if A in CATEGORICAL_ATTRIBUTES:
            tree = {A: {}}
            values = train_data[A].unique()
            for value in values:
                exs = examples[examples[A] == value]
                if len(exs[Y].unique()) == 1:
                    tree[A][str(value)] = str(exs[Y].unique()[0])  # leaf node
                else:
                    subtree = decision_tree_learning(
                        examples=exs,
                        attributes=attributes,
                        parent_examples=examples,
                    )
                    tree[A][str(value)] = subtree

        else:
            split = decision_tree_splits[A]
            node = f"{A} <= {split}"
            tree = {node: {}}

            subsets = [
                examples[examples[A] <= split],
                examples[examples[A] > split],
            ]

            for i, subset in enumerate(subsets):
                if len(subset[Y].unique()) == 1:
                    tree[node][i] = str(subset[Y].unique()[0])  # leaf node
                else:
                    tree[node][i] = decision_tree_learning(
                        examples=subset,
                        attributes=attributes,
                        parent_examples=examples,
                    )

    return tree


def predict(decision_tree, datapoint):

    """
    Function for predicting outcome for a datapoint with our decision tree.

    decision_tree: Decision tree
    datapoint: Datapoint we want to predict for.
    """

    for node in decision_tree.keys():
        if "<=" in node:
            node_info = node.split(" <= ")
            attribute, split = node_info[0], float(node_info[1])
            value = datapoint[attribute]
            if value <= split:
                decision_tree = decision_tree[node][0]  # Go left
            else:
                decision_tree = decision_tree[node][1]  # Go right
        else:
            value = datapoint[node]
            decision_tree = decision_tree[node][str(value)]
        if type(decision_tree) == dict:
            prediction = predict(
                decision_tree, datapoint
            )  # Not finished with prediction, can still go deeper in the tree
        else:
            prediction = decision_tree  # Have reached leaf node, this is our prediction
    return prediction


def evaluate_model(decision_tree, data):

    """
    Function for evauluating accuracy of decision tree on a given dataset.

    decision_tree: Decision tree
    data: Data to evaluate for
    """

    predictions = [
        int(predict(decision_tree, datapoint)) for datapoint in data.to_dict("records")
    ]
    return sum(1 for x, y in zip(predictions, list(data[Y])) if x == y) / len(
        predictions
    )


def add_nodes(decision_tree, graph, parent=None):

    """
    Function for adding nodes to the graphical graph object.

    decision_tree: Desicion tree
    graph: pydot graph object
    parent: parent of the current node we are drawing. Starts as None.
    """

    for node in decision_tree.keys():
        label = str(node)

        if parent:
            from_name = parent.get_name().replace('"', "") + "_" + label

            if type(node) == int:
                from_node = pydot.Node(
                    from_name, shape="point"
                )  # node for continious variable
            else:
                from_node = pydot.Node(
                    from_name, label=label
                )  # node for categorical variable

            graph.add_node(from_node)
            graph.add_edge(pydot.Edge(parent, from_node))

            if type(decision_tree[node]) == dict:
                add_nodes(decision_tree[node], graph, from_node)  # can still add nodes
            else:
                # Add leaf node
                to_name = str(uuid.uuid4()) + "_" + str(decision_tree[node])
                label = str(decision_tree[node])

                to_node = pydot.Node(to_name, label=label, shape="box")
                graph.add_node(to_node)
                graph.add_edge(pydot.Edge(from_node, to_node))
        else:
            root_node = pydot.Node(label, label=label)  # Root node in the decision tree
            add_nodes(decision_tree[node], graph, root_node)


def draw_decision_tree(decision_tree, name):

    """
    Function for drawing decision tree.

    decision_tree. Decision tree
    name: Name of png file where we want to save the decision tree.
    """

    graph = pydot.Dot(graph_type="graph")
    add_nodes(decision_tree, graph)
    graph.write_png(name + ".png")


if __name__ == "__main__":
    # Problem 1a

    train_data = create_dataset(
        "data/train.csv", DROP_ATTRIBUTES + NUMERICAL_ATTRIBUTES
    )
    attributes = [c for c in train_data.columns if c != Y]
    decision_tree = decision_tree_learning(train_data, attributes, None)

    draw_decision_tree(decision_tree, "categorical_decision_tree")

    test_data = create_dataset("data/test.csv", DROP_ATTRIBUTES)
    accuracy = evaluate_model(decision_tree, test_data)
    print(f"Arruacy [categorical features] = {accuracy*100} %")

    ######################################################################

    # Problem 1b

    train_data = create_dataset("data/train.csv", DROP_ATTRIBUTES)
    attributes = [c for c in train_data.columns if c != Y]
    decision_tree = decision_tree_learning(train_data, attributes, None)

    draw_decision_tree(decision_tree, "decision_tree")

    test_data = create_dataset("data/test.csv", DROP_ATTRIBUTES)
    accuracy = evaluate_model(decision_tree, test_data)
    print(f"Arruacy [all features] = {accuracy*100} %")