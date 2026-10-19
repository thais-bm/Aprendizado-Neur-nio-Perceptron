import numpy as np
import matplotlib.pyplot as plt

# Estratégia didática: Se o Numba não existir, criamos um decorador @njit "falso"
# que não faz nada. Assim, usamos exatamente o mesmo algoritmo para os dois casos.
try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


@njit(cache=True)
def _train_perceptron(X, Y, weights, learning_rate, limiar, max_epochs, shuffle, seed):
    """
    Treinamento de forma estatica para usar o numba pra acelerar o loop de treinamento do Perceptron.
    """
    np.random.seed(seed)
    num_samples, num_features = X.shape
    
    epochs = 0
    total_adjustments = 0

    # Rodar para sempre ou ate o limite
    while max_epochs <= 0 or epochs < max_epochs:
        epochs += 1
        errors_in_epoch = 0
        
        # Define a ordem de apresentação dos exemplos
        if shuffle:
            order = np.random.permutation(num_samples)
        else:
            order = np.arange(num_samples)
            
        for idx in range(num_samples):
            i = order[idx]
            
            # Combinação linear (u = W * X)
            u = 0.0
            for j in range(num_features):
                u += weights[j] * X[i, j]
                
            # Função de Ativação (Degrau)
            pred = 1.0 if u >= limiar else 0.0
            
            # Cálculo do Erro
            err = Y[i] - pred
            
            # Atualização dos pesos (Apenas se houver erro)
            if err != 0.0:
                for j in range(num_features):
                    weights[j] += learning_rate * err * X[i, j]
                errors_in_epoch += 1
                total_adjustments += 1
                
        # Critério de parada: convergência
        if errors_in_epoch == 0:
            break
            
    return weights, epochs, total_adjustments


class PerceptronNeuron:
    """Implementação didática do neurônio Perceptron de Rosenblatt."""

    def __init__(self, learning_rate: float = 0.1, limiar: float = 0.0, 
                 weight_range: tuple = (-0.5, 0.5)): # <-- Restaurado
        self.learning_rate = learning_rate
        self.limiar = limiar
        self.weight_range = weight_range
        self.weights = np.array([])
        self.total_adjustments = 0
        self.epochs = 0

    @staticmethod
    def _add_bias(X: np.ndarray) -> np.ndarray:
        """Adiciona a entrada fixa x0 = -1 (bias/limiar)."""
        return np.column_stack((-np.ones(X.shape[0]), X))

    def train(self, X: np.ndarray, Y: np.ndarray, max_epochs: int | None = None,
              shuffle: bool = False, rng: np.random.Generator | None = None, 
              verbose: bool = True) -> None:
        
        X_bias = self._add_bias(X)
        
        # Usa o gerador fornecido ou cria um novo
        if rng is None:
            rng = np.random.default_rng()
            
        # Numba requer um valor inteiro para a seed internamente
        # Extraímos um inteiro do gerador para manter o determinismo atrelado ao rng original
        seed_for_numba = int(rng.integers(0, 2**31 - 1))
        
        # Inicialização dos pesos usando os limites definidos
        low, high = self.weight_range
        self.weights = rng.uniform(low, high, size=X_bias.shape[1])
        
        if verbose:
            print(f"Pesos iniciais: {self.weights}")

        # Se max_epochs for None, forçamos um valor negativo para a função C/Numba rodar sem fixar o número de épocas.
        safe_max_epochs = -1 if max_epochs is None else int(max_epochs)

        # Chama a função principal compilada via Numba
        w_final, epochs, adjs = _train_perceptron(
            X_bias.astype(np.float64),
            Y.astype(np.float64),
            self.weights.astype(np.float64).copy(),
            float(self.learning_rate),
            float(self.limiar),
            safe_max_epochs,
            bool(shuffle),
            seed_for_numba
        )
        
        self.weights = w_final
        self.epochs = epochs
        self.total_adjustments = adjs

        if verbose:
            print(f"Pesos finais:   {self.weights}")
            print(f"Total de ajustes: {self.total_adjustments} | Épocas: {self.epochs}")
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Realiza a inferência para novos dados."""
        X_bias = self._add_bias(X)
        u = np.dot(X_bias, self.weights)
        return np.where(u >= self.limiar, 1, 0)

    def plot_decision_boundary(self, X: np.ndarray, Y: np.ndarray, title: str = None) -> None:
        """Visualiza os dados (2D) e a reta de decisão do Perceptron."""
        plt.figure(figsize=(8, 6))

        # Plota as classes
        plt.scatter(X[Y == 0][:, 0], X[Y == 0][:, 1], color="red", label="Classe 0", edgecolors="k")
        plt.scatter(X[Y == 1][:, 0], X[Y == 1][:, 1], color="blue", label="Classe 1", edgecolors="k")

        # Define os limites do gráfico
        x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

        # Equação da reta: w0(-1) + w1*x1 + w2*x2 = 0  =>  x2 = (w0 - w1*x1) / w2
        w0, w1, w2 = self.weights
        x1_vals = np.array([x1_min, x1_max])
        
        if w2 != 0:
            x2_vals = (w0 - w1 * x1_vals) / w2
            plt.plot(x1_vals, x2_vals, color="black", linewidth=2, label="Reta de decisão")
        elif w1 != 0: # Caso a reta seja perfeitamente vertical
            plt.axvline(w0 / w1, color="black", linewidth=2, label="Reta de decisão")

        plt.xlim(x1_min, x1_max)
        plt.ylim(x2_min, x2_max)
        plt.title(title or f"Perceptron ({self.epochs} épocas, {self.total_adjustments} ajustes)")
        plt.xlabel("$x_1$")
        plt.ylabel("$x_2$")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.show()