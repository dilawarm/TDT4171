import numpy as np

normalize = lambda arr: arr / arr.sum()

forward = lambda sensor_model, transition_model, forward_message: normalize(
    np.matmul(np.matmul(sensor_model, transition_model.T), forward_message)
)

backward = lambda sensor_model, transition_model, backward_message: np.matmul(
    np.matmul(transition_model, sensor_model), backward_message
)


def get_sensor_model(evidence, sensor_model, index):
    if evidence[index]:
        return sensor_model
    else:
        return np.identity(n=sensor_model.shape[0]) - sensor_model


def filtering(evidence, prior, sensor_model, transition_model):
    filters = [prior]
    for i in range(len(evidence)):
        filters.append(
            forward(
                get_sensor_model(evidence, sensor_model, i),
                transition_model,
                filters[i],
            )
        )
    return filters[1:]


def prediction(evidence, prior, sensor_model, transition_model, start_t, end_t):
    last_filter = filtering(evidence, prior, sensor_model, transition_model)[-1]
    predictions = [last_filter]
    for t in range(end_t - start_t + 1):
        predictions.append(normalize(np.matmul(transition_model, predictions[t])))
    return predictions[1:]


def smoothing(evidence, prior, sensor_model, transition_model):
    filters = [prior] + filtering(evidence, prior, sensor_model, transition_model)
    smoothings = []
    back = np.array([1.0, 1.0])
    for i in range(len(filters) - 1, -1, -1):
        smoothings.append(normalize(filters[i] * back))
        back = backward(
            get_sensor_model(evidence, sensor_model, i - 1), transition_model, back
        )
    return smoothings


def viterbi(evidence, prior, sensor_model, transition_model):
    sensor_model = np.array([sensor_model[0][0], sensor_model[1][1]])
    most_likely_sequence = [prior]
    identity_vec = np.array([1.0, 1.0])
    sensor = None
    for i in range(len(evidence)):
        if evidence[i]:
            sensor = sensor_model
        else:
            sensor = identity_vec - sensor_model
        dist = sensor * np.max(transition_model * most_likely_sequence[i], axis=1)
        dist = normalize(dist) if i == 0 else dist
        most_likely_sequence.append(dist)

    return most_likely_sequence[1:]


def problem1():
    evidence = [True, True, False, True, False, True]
    prior = np.array([0.5, 0.5])
    sensor_model = np.array([[0.75, 0.0], [0.0, 0.2]])
    transition_model = np.array([[0.8, 0.3], [0.2, 0.7]])

    def problem1b():
        start_t, end_t = 1, 6
        print(f"Problem 1b)\nCalculating P(X_t|e_1:t) for t = {start_t},...,{end_t}:")
        filters = filtering(evidence, prior, sensor_model, transition_model)
        print(
            "\n".join(
                [f"P(X_{t+1}|e_1:{t+1}) = {filters[t]}" for t in range(len(filters))]
            ),
            "\n",
        )

    def problem1c():
        start_t, end_t = 7, 30
        print(f"Problem 1c)\nCalculating P(X_t|e_1:6) for t = {start_t},...,{end_t}:")
        predictions = prediction(
            evidence,
            prior,
            sensor_model,
            transition_model,
            start_t=start_t,
            end_t=end_t,
        )
        print(
            "\n".join(
                [
                    f"P(X_{t+start_t}|e_1:6) = {predictions[t]}"
                    for t in range(len(predictions))
                ]
            ),
            "\n",
        )

    def problem1d():
        start_t, end_t = 0, 5
        print(f"Problem 1d)\nCalculating P(X_t|e_1:6) for t = {start_t},...,{end_t}:")
        smoothings = smoothing(evidence, prior, sensor_model, transition_model)[::-1]
        print(
            "\n".join(
                [
                    f"P(X_{t}|e_1:6) = {smoothings[t]}"
                    for t in range(len(smoothings) - 2, -1, -1)
                ]
            ),
            "\n",
        )

    def problem1e():
        start_t, end_t = 1, 6
        print(
            f"Problem 1e)\nCalculating P(x_1,...,x_(t-1),X_t|e_1:t) for t = {start_t},...,{end_t}:"
        )
        most_likely_sequence = viterbi(evidence, prior, sensor_model, transition_model)
        print(
            "\n".join(
                [
                    f"m_1:{t+1} = {most_likely_sequence[t]} --> argmax({most_likely_sequence[t]}) = {np.argmax(most_likely_sequence[t])} --> {[True, False][np.argmax(most_likely_sequence[t])]}"
                    for t in range(len(most_likely_sequence))
                ]
            ),
            "\n",
        )

    problem1b()
    problem1c()
    problem1d()
    problem1e()


if __name__ == "__main__":
    problem1()