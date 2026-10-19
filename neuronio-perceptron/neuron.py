import numpy as np
import matplotlib.pyplot as plt

try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:  # sem Numba, usa-se o caminho em numpy puro
    _HAS_NUMBA = False


if _HAS_NUMBA:
    @njit(cache=True)
    def _train_jit(Xb, Y, w, eta, limiar, max_epochs, shuffle, seed):
        """Mesmo algoritmo do Perceptron, compilado. Retorna (épocas, ajustes)."""
        np.random.seed(seed)
        m = Xb.shape[0]
        epochs = 0
        adjustments = 0
        while max_epochs < 0 or epochs < max_epochs:
            epochs += 1
            errors = 0
            order = np.random.permutation(m) if shuffle else np.arange(m)
            for idx in range(m):
                i = order[idx]
                u = 0.0
                for j in range(Xb.shape[1]):
                    u += w[j] * Xb[i, j]
                pred = 1.0 if u >= limiar else 0.0
                err = Y[i] - pred
                if err != 0.0:
                    for j in range(Xb.shape[1]):
                        w[j] += eta * err * Xb[i, j]
                    errors += 1
                    adjustments += 1
            if errors == 0:
                break
        return epochs, adjustments


class PerceptronNeuron:
    """Implementação do neurônio Perceptron de Rosenblatt."""

    def __init__(self, learning_rate: float = 0.1, limiar: float = 0.0,
                 weight_range: tuple = (-0.5, 0.5)):
        self.learning_rate = learning_rate
        self.limiar = limiar
        self.weight_range = weight_range
        self.weights: np.ndarray = np.array([])
        self.total_adjustments: int = 0
        self.epochs: int = 0

    def activation_function(self, u):
        """Função degrau com limiar ϑ."""
        return np.where(u >= self.limiar, 1, 0)

    @staticmethod
    def _add_bias(X: np.ndarray) -> np.ndarray:
        # x0 = -1 (o peso w0 faz o papel do limiar/bias)
        return np.c_[-np.ones(X.shape[0]), X]

    def train(self, X: np.ndarray, Y: np.ndarray, max_epochs: int | None = None,
              shuffle: bool = False, rng: np.random.Generator | None = None,
              verbose: bool = True, use_jit: bool = True) -> None:
        """Treina o neurônio.

        Se max_epochs for None, executa até a convergência (zero erros em uma época).
        Caso contrário, executa exatamente max_epochs épocas (ou até convergir).
        Se shuffle=True, apresenta os exemplos em ordem aleatória a cada época.
        """
        rng = rng if rng is not None else np.random.default_rng()
        bigger_X = self._add_bias(X)
        m = bigger_X.shape[0]

        low, high = self.weight_range
        self.weights = rng.uniform(low, high, size=bigger_X.shape[1])
        if verbose:
            print(f"Vetor inicial de pesos [w0, w1, w2]: {self.weights}")

        self.total_adjustments = 0
        self.epochs = 0

        if use_jit and _HAS_NUMBA:
            w = self.weights.astype(np.float64).copy()
            epochs, adj = _train_jit(bigger_X.astype(np.float64), Y.astype(np.float64), w,
                                     float(self.learning_rate), float(self.limiar),
                                     -1 if max_epochs is None else int(max_epochs),
                                     bool(shuffle), int(rng.integers(0, 2**31 - 1)))
            self.weights, self.epochs, self.total_adjustments = w, int(epochs), int(adj)
            max_epochs = 0  # pula o laço em numpy abaixo

        while max_epochs is None or self.epochs < max_epochs:
            errors = 0
            self.epochs += 1
            order = rng.permutation(m) if shuffle else np.arange(m)
            Xe, Ye = bigger_X[order], Y[order]

            # Varredura sequencial equivalente ao laço exemplo-a-exemplo: os pesos só
            # mudam em erros, então localizamos o próximo erro com uma única operação
            # matricial e atualizamos os pesos nele (acelera muito o treino).
            pos = 0
            while pos < m:
                pred = self.activation_function(Xe[pos:] @ self.weights)
                wrong = np.flatnonzero(pred != Ye[pos:])
                if wrong.size == 0:
                    break
                k = pos + wrong[0]
                error = Ye[k] - pred[wrong[0]]
                self.weights = self.weights + self.learning_rate * error * Xe[k]
                errors += 1
                self.total_adjustments += 1
                pos = k + 1

            if errors == 0:
                break

        if verbose:
            print(f"Vetor final de pesos   [w0, w1, w2]: {self.weights}")
            print(f"(a) Número total de ajustes no vetor de pesos: {self.total_adjustments}")
            print(f"(b) Número de épocas até a convergência:       {self.epochs}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Prevê as classes (0/1) para os exemplos X de shape (n, 2)."""
        return self.activation_function(self._add_bias(X) @ self.weights).astype(int)

    def plot_decision_boundary(self, X: np.ndarray, Y: np.ndarray, ax=None,
                               title: str | None = None, show: bool = True) -> None:
        """Gráfico: classe 0 (vermelho), classe 1 (azul) e reta separadora."""
        own_fig = ax is None
        if own_fig:
            _, ax = plt.subplots(figsize=(9, 7))

        c0, c1 = X[Y == 0], X[Y == 1]
        ax.scatter(c0[:, 0], c0[:, 1], color="red", label="Classe 0", alpha=0.6,
                   s=25, edgecolors="k", linewidths=0.4)
        ax.scatter(c1[:, 0], c1[:, 1], color="blue", label="Classe 1", alpha=0.6,
                   s=25, edgecolors="k", linewidths=0.4)

        # -w0 + w1*x1 + w2*x2 = 0  =>  x2 = (w0 - w1*x1) / w2
        w0, w1, w2 = self.weights
        x1_lim = (X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
        x2_lim = (X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)
        x1_reta = np.linspace(*x1_lim, 500)
        if w2 != 0:
            ax.plot(x1_reta, (w0 - w1 * x1_reta) / w2, color="black", linewidth=2.2,
                    label="Reta de decisão")
        elif w1 != 0:
            ax.axvline(w0 / w1, color="black", linewidth=2.2, label="Reta de decisão")

        ax.set_xlim(*x1_lim)
        ax.set_ylim(*x2_lim)
        ax.set_title(title or f"Perceptron (épocas: {self.epochs} | ajustes: {self.total_adjustments})")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.legend(loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.5)
        if own_fig:
            plt.tight_layout()
            if show:
                plt.show()