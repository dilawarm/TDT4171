import numpy as np


if __name__ == "__main__":
    grid = [[-0.2, -0.2, -0.2], [-0.2, 10, -7], [-0.2, -0.2, -2]]
    new_grid = [[None for _ in range(len(grid[0]))] for _ in range(len(grid))]
    probs = np.array([0.8, 0.1, 0.1])
    gamma = 0.9

    for i in range(len(grid)):
        for j in range(len(grid[0])):
            neighbors = []
            for a in ((-1, 0), (1, 0), (0, 1), (0, -1)):
                neighbor = []
                a1, a2 = a
                if (
                    a1 + i >= 0
                    and a1 + i < len(grid)
                    and a2 + j >= 0
                    and a2 + j < len(grid[0])
                ):
                    neighbor.append(grid[i + a1][j + a2])
                else:
                    neighbor.append(grid[i][j])
                if a1 != 0:
                    sides = [(0, -1), (0, 1)]
                    for s in sides:
                        s1, s2 = s
                        if (
                            s1 + i >= 0
                            and s1 + i < len(grid)
                            and s2 + j >= 0
                            and s2 + j < len(grid[0])
                        ):
                            neighbor.append(grid[i + s1][j + s2])
                        else:
                            neighbor.append(grid[i][j])
                else:
                    sides = [(-1, 0), (1, 0)]
                    for s in sides:
                        s1, s2 = s
                        if (
                            s1 + i >= 0
                            and s1 + i < len(grid)
                            and s2 + j >= 0
                            and s2 + j < len(grid[0])
                        ):
                            neighbor.append(grid[i + s1][j + s2])
                        else:
                            neighbor.append(grid[i][j])
                print(neighbor)
                neighbors.append(np.dot(probs, np.array(neighbor)))
            print(neighbors, end="\n\n")
            m = np.max(np.array(neighbors))
            new_grid[i][j] = grid[i][j] + gamma * m

    print(new_grid)
