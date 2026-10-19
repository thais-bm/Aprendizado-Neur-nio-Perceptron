import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable

from neuron import PerceptronNeuron

ID_EQUIPE = 0


def carregar(caminho: str) -> tuple[np.ndarray, np.ndarray]:
    """Carrega um arquivo de dados e retorna (X com shape (m, 2), y com shape (m,))."""
    dados = np.load(caminho, allow_pickle=True).reshape(-1, 3)
    return dados[:, :2], dados[:, 2].astype(int)


def run_part_one():
    """Parte I: problema linearmente separável (dataAll20262.txt)."""
    X, y = carregar("dados/dataAll20262.txt")

    perceptron = PerceptronNeuron(learning_rate=0.1, limiar=0.0, weight_range=(-0.5, 0.5))
    perceptron.train(X, y)
    perceptron.plot_decision_boundary(
        X, y,
        title=f"Parte I - Perceptron (épocas: {perceptron.epochs} | ajustes: {perceptron.total_adjustments})",
    )


def run_part_two(n_rep: int = 10):
    """Parte II: 6 configurações (η × intervalo dos pesos), n_rep repetições cada."""
    X, y = carregar(f"dados/data{ID_EQUIPE}20262.txt")

    taxas = [0.4, 0.1, 0.01]
    intervalos = [(-100, 100), (-0.5, 0.5)]
    rng = np.random.default_rng(ID_EQUIPE)

    tabela = PrettyTable([
        "Taxa de Aprendizado", "Intervalo de Pesos",
        "Qtde Ajustes (média ± dp)", "Min Épocas", "Max Épocas",
    ])
    resultados = {}

    for eta in taxas:
        for intervalo in intervalos:
            ajustes, epocas = [], []
            for _ in range(n_rep):
                neuronio = PerceptronNeuron(learning_rate=eta, limiar=0.0, weight_range=intervalo)
                neuronio.train(X, y, rng=rng, verbose=False)
                ajustes.append(neuronio.total_adjustments)
                epocas.append(neuronio.epochs)

            resultados[(eta, intervalo)] = (np.array(ajustes), np.array(epocas))
            tabela.add_row([
                f"η = {eta}", f"({intervalo[0]}, {intervalo[1]})",
                f"{np.mean(ajustes):.2f} ± {np.std(ajustes):.2f}", min(epocas), max(epocas),
            ])

    print(tabela)

    rotulos = [f"η={eta}\n{intervalo}" for eta, intervalo in resultados]
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].boxplot([r[0] for r in resultados.values()], tick_labels=rotulos)
    axs[0].set_title("Ajustes no vetor de pesos")
    axs[0].set_ylabel("Ajustes")
    axs[1].boxplot([r[1] for r in resultados.values()], tick_labels=rotulos)
    axs[1].set_title("Épocas até a convergência")
    axs[1].set_ylabel("Épocas")
    for ax in axs:
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.tick_params(axis="x", labelsize=8)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_part_one()
    run_part_two()