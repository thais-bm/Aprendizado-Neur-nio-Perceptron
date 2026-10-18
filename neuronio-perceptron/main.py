import numpy as np
from neuron import PerceptronNeuron

def run_part_one():
    """Run the first part of the perceptron neuron training and visualization."""
    dados = np.load("dados/dataAll20262.txt", allow_pickle=True).reshape(-1, 3)
    X_dados = dados[:, :2]
    y_dados = dados[:, 2]

    perceptron = PerceptronNeuron(learning_rate=0.1, limiar=0.0)
    perceptron.train(X_dados, y_dados)
    perceptron.plot_decision_boundary(X_dados, y_dados)


if __name__ == "__main__":
    run_part_one()