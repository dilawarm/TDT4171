import pandas as pd
import math
import pydot
import uuid
import random

Y = "Survived"

create_categorical_dataset = (
    lambda path: pd.read_csv(path)
    .dropna(axis=0)
    .drop(["Name", "Age", "Ticket", "Fare", "Cabin"], axis=1)
)

plurality_value = lambda examples: str(examples[Y].mode()[0])

same_classification = lambda examples: (
    examples.to_numpy()[0] == examples.to_numpy()
).all()

entropy = lambda examples: -sum(
    (val / len(examples)) * math.log2(val / len(examples))
    for val in examples[Y].value_counts()
)


def importance(attribute, examples):
    subset = examples[attribute].value_counts()
    subset_keys = subset.keys()
    return entropy(examples) - sum(
        (val / len(examples)) * entropy(examples[examples[attribute] == subset_keys[s]])
        for s, val in enumerate(subset)
    )


def decision_tree_learning(examples, attributes, parent_examples):
    if examples.empty:
        return plurality_value(parent_examples)
    elif same_classification(examples):
        return str(examples[Y].unique()[0])
    elif not attributes:
        return plurality_value(examples)
    else:
        importances = [importance(a, examples) for a in attributes]
        A = attributes[importances.index(max(importances))]
        tree = {A: {}}
        values = categorical_data[A].unique()
        for value in values:
            exs = examples[examples[A] == value]
            if len(exs[Y].unique()) == 1:
                tree[A][str(value)] = str(exs[Y].unique()[0])
            else:
                subtree = decision_tree_learning(
                    examples=exs,
                    attributes=[a for a in attributes if a != A],
                    parent_examples=examples,
                )
                tree[A][str(value)] = subtree
    return tree


def predict(decision_tree, datapoint):
    for node in decision_tree.keys():
        value = datapoint[node]
        decision_tree = decision_tree[node][str(value)]
        if type(decision_tree) == dict:
            prediction = predict(decision_tree, datapoint)
        else:
            prediction = decision_tree
    return prediction


def evaluate_model(decision_tree, data):
    predictions = [
        int(predict(decision_tree, datapoint)) for datapoint in data.to_dict("records")
    ]
    return sum(1 for x, y in zip(predictions, list(data[Y])) if x == y) / len(
        predictions
    )


def add_nodes(dictionary, graph, parent=None):
    for node in dictionary.keys():
        name = str(node)
        if parent:

            from_name = parent.get_name().replace('"', "") + "_" + str(node)
            to_node = pydot.Node(from_name, label=name)
            graph.add_node(to_node)
            graph.add_edge(pydot.Edge(parent, to_node))

            if type(dictionary[node]) == dict:
                add_nodes(dictionary[node], graph, to_node)

            else:
                to_name = str(uuid.uuid4()) + "_" + str(dictionary[node])
                name = str(dictionary[node])

                node_to = pydot.Node(to_name, label=name, shape="box")
                graph.add_node(node_to)
                graph.add_edge(pydot.Edge(to_node, node_to))
        else:
            root_node = pydot.Node(name, label=name)
            add_nodes(dictionary[node], graph, root_node)


def draw_decision_tree(tree, name):
    graph = pydot.Dot(graph_type="graph")
    add_nodes(decision_tree, graph)
    graph.write_png(name + ".png")


if __name__ == "__main__":
    categorical_data = create_categorical_dataset("data/train.csv")
    attributes = [c for c in categorical_data.columns if c != Y]
    decision_tree = decision_tree_learning(categorical_data, attributes, [])
    draw_decision_tree(decision_tree, "categorical_decision_tree")

    test_data = create_categorical_dataset("data/test.csv")
    accuracy = evaluate_model(decision_tree, test_data)
    print(f"Arruacy [only categorical features]= {accuracy*100} %")
