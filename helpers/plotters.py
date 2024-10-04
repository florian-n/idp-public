import matplotlib.pyplot as plt


def plot_3d_system(system, n_molecules_per_axis, BOX_SIZE):
    x = [point[0] for point in system]
    y = [point[1] for point in system]
    z = [point[2] for point in system]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(x, y, z)

    ax.set_xlabel(f"{n_molecules_per_axis} molecules")
    ax.set_ylabel(f"{n_molecules_per_axis} molecules")
    ax.set_zlabel(f"{n_molecules_per_axis} molecules")

    plt.show()


def plot_3d_system_rigidbody(system):
    x = [point[0] for point in system.center]
    y = [point[1] for point in system.center]
    z = [point[2] for point in system.center]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(x, y, z)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.show()
