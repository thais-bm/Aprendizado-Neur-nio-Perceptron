import numpy as np
import matplotlib.pyplot as plt

class PerceptronNeuron:
    """A simple implementation of a perceptron Rosenblatt neuron."""


    def __init__(self, learning_rate: float = 0.1, limiar: float = 0.0):
        """Initialize the perceptron neuron with random weights and bias."""
        self.learning_rate = learning_rate
        self.limiar = limiar
        self.weights: np.ndarray = np.array([])
        self.total_adjustments: int = 0
        self.epochs: int = 0

    def activation_function(self, u: float | np.ndarray) -> int | np.ndarray:
        """The activation function for the perceptron neuron (step function)."""
        return np.where(u >= self.limiar, 1, 0)


    def train(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Train the perceptron neuron using the provided training data."""
        m = X.shape[0]

        # adicionar x0 = 1 para cada amostra de entrada (bias)
        bigger_X = np.c_[-np.ones(m), X]

        # inicializando os pesos 
        self.weights = np.random.uniform(-0.5, 0.5, size=3)
        print(f"Vetor inicial de pesos [w0, w1, w2]: {self.weights}")

        self.total_adjustments = 0
        self.epochs = 0

        while True:
            errors = 0
            self.epochs += 1

            for xi, yd in zip(bigger_X, Y):
                u = float(np.dot(self.weights, xi))
                y_pred = self.activation_function(u)
                error = yd - y_pred

                if error != 0:
                    self.weights += self.learning_rate * error * xi
                    errors += 1
                    self.total_adjustments += 1

            if errors == 0:
                break

        print(f"Vetor final de pesos   [w0, w1, w2]: {self.weights}")
        print(f"(a) Número total de ajustes no vetor de pesos: {self.total_adjustments}")
        print(f"(b) Número de épocas até a convergência:       {self.epochs}")


    def predict(self, inputs: np.ndarray) -> int:
        """Make a prediction based on the inputs."""
        pass

    def plot_decision_boundary(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Requisito 5(c): Gráfico com Classe 0 (vermelho), Classe 1 (azul) e reta separadora."""
        plt.figure(figsize=(9, 7))

        c0 = X[Y == 0]
        c1 = X[Y == 1]

        plt.scatter(c0[:, 0], c0[:, 1], color="red", label="Classe 0", alpha=0.6, s=25, edgecolors="k", linewidths=0.4)
        plt.scatter(c1[:, 0], c1[:, 1], color="blue", label="Classe 1", alpha=0.6, s=25, edgecolors="k", linewidths=0.4)

        # Equação da reta: -w0 + w1*x1 + w2*x2 = 0  =>  x2 = (w0 - w1*x1) / w2
        w0, w1, w2 = self.weights
        margem = 50.0
        x1_min, x1_max = X[:, 0].min() - margem, X[:, 0].max() + margem
        x2_min, x2_max = X[:, 1].min() - margem, X[:, 1].max() + margem

        x1_reta = np.linspace(x1_min, x1_max, 500)
        x2_reta = (w0 - w1 * x1_reta) / w2

        plt.plot(x1_reta, x2_reta, color="black", linewidth=2.2, label="Reta de Decisão (Perceptron)")

        plt.xlim(x1_min, x1_max)
        plt.ylim(x2_min, x2_max)
        plt.title(
            f"Classificação Linear - Perceptron de Rosenblatt\n"
            f"(Épocas: {self.epochs} | Total de ajustes: {self.total_adjustments})"
        )
        plt.xlabel("$x_1$")
        plt.ylabel("$x_2$")
        plt.legend(loc="upper right")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()