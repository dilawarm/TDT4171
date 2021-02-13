import numpy as np

normalize = lambda arr: arr / arr.sum()

forward = (
    lambda sensor_model_1, sensor_model_2, transition_model, forward_message: normalize(
        np.matmul(
            np.matmul(np.matmul(sensor_model_1, sensor_model_2), transition_model.T),
            forward_message,
        )
    )
)

backward = lambda sensor_model_1, sensor_model_2, transition_model, backward_message: np.matmul(
    np.matmul(transition_model, np.matmul(sensor_model_1, sensor_model_2)),
    backward_message,
)


def get_sensor_model(evidence, sensor_model, index, i):
    if evidence[index][i]:
        return sensor_model
    else:
        return np.identity(n=sensor_model.shape[0]) - sensor_model


def filtering(evidence, prior, sensor_model_1, sensor_model_2, transition_model):
    filters = [prior]
    for i in range(len(evidence)):
        filters.append(
            np.round(
                forward(
                    get_sensor_model(evidence, sensor_model_1, i, 0),
                    get_sensor_model(evidence, sensor_model_2, i, 1),
                    transition_model,
                    filters[i],
                ),
                4,
            )
        )
    return filters[1:]


def prediction(
    evidence,
    prior,
    sensor_model_1,
    sensor_model_2,
    transition_model,
    start_t,
    end_t,
    test_convergence=False,
    converged=None,
):
    if not test_convergence:
        last_filter = filtering(
            evidence, prior, sensor_model_1, sensor_model_2, transition_model
        )[-1]
        predictions = [last_filter]
    else:
        predictions = [converged]
    for t in range(end_t - start_t + 1):
        predictions.append(
            np.round(normalize(np.matmul(transition_model, predictions[t])), 4)
        )
    return predictions[1:]


def smoothing(evidence, prior, sensor_model_1, sensor_model_2, transition_model):
    filters = [prior] + filtering(
        evidence, prior, sensor_model_1, sensor_model_2, transition_model
    )
    smoothings = []
    back = np.array([1.0, 1.0])
    print(back)
    for i in range(len(filters) - 1, -1, -1):
        smoothings.append(np.round(normalize(filters[i] * back), 4))
        back = backward(
            get_sensor_model(evidence, sensor_model_1, i - 1, 0),
            get_sensor_model(evidence, sensor_model_2, i - 1, 1),
            transition_model,
            back,
        )
        print(np.round(back, 4))
    return smoothings


def problem2():
    evidence = [(True, True), (False, True), (False, False), (True, False)]
    prior = np.array([0.7, 0.3])
    sensor_model_1 = np.array([[0.7, 0.0], [0.0, 0.2]])
    sensor_model_2 = np.array([[0.3, 0.0], [0.0, 0.1]])
    transition_model = np.array([[0.8, 0.3], [0.2, 0.7]])

    def problem1b():
        start_t, end_t = 1, 4
        print(f"Problem 1b)\nCalculating P(X_t|e_1:t) for t = {start_t},...,{end_t}:")
        filters = filtering(
            evidence, prior, sensor_model_1, sensor_model_2, transition_model
        )
        print(
            "\n".join(
                [f"P(X_{t+1}|e_1:{t+1}) = {filters[t]}" for t in range(len(filters))]
            ),
            "\n",
        )

    def problem1c():
        start_t, end_t = 5, 8
        print(f"Problem 1c)\nCalculating P(X_t|e_1:4) for t = {start_t},...,{end_t}:")
        predictions = prediction(
            evidence,
            prior,
            sensor_model_1,
            sensor_model_2,
            transition_model,
            start_t=start_t,
            end_t=end_t,
        )
        print(
            "\n".join(
                [
                    f"P(X_{t+start_t}|e_1:4) = {predictions[t]}"
                    for t in range(len(predictions))
                ]
            ),
            "\n",
        )

    def problem1d():
        print("Problem 1d)")
        start_t, end_t = 5, 6
        converged = np.array([0.6, 0.4])
        predictions = prediction(
            evidence,
            prior,
            sensor_model_1,
            sensor_model_2,
            transition_model,
            start_t=start_t,
            end_t=end_t,
            test_convergence=True,
            converged=converged,
        )
        if predictions[0].all() == predictions[1].all():
            print("Verified.\n")
        else:
            print("Not verified.\n")

    def problem1e():
        start_t, end_t = 0, 3
        print(f"Problem 1e)\nCalculating P(X_t|e_1:4) for t = {start_t},...,{end_t}:")
        smoothings = smoothing(
            evidence, prior, sensor_model_1, sensor_model_2, transition_model
        )[::-1]
        print(
            "\n".join(
                [
                    f"P(X_{t}|e_1:4) = {smoothings[t]}"
                    for t in range(len(smoothings) - 1, -1, -1)
                ]
            ),
            "\n",
        )

    problem1b()
    problem1c()
    problem1d()
    problem1e()

    res = np.matmul(transition_model, np.array([0.6, 0.4]))
    print(1 / res.sum())


if __name__ == "__main__":
    problem2()