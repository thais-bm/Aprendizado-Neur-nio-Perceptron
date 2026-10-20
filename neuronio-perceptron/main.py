import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from neuron import PerceptronNeuron  

# ==========================================
# CONFIGURAÇÕES DA EQUIPE
# ==========================================
ID_EQUIPE = 0  

def carregar(filepath: str):
    """Função auxiliar para carregar os dados."""
    try:
        dados = np.load(filepath, allow_pickle=True)
    except (ValueError, OSError):
        dados = np.loadtxt(filepath)
    
    # Força o formato de 3 colunas
    dados = dados.reshape(-1, 3)
        
    return dados[:, :2], dados[:, 2]


def run_part_one() -> None:
    """Executa a Parte I do treinamento e visualização do Perceptron."""
    print("=======================================")
    print(" INICIANDO PARTE I")
    print("=======================================")
    
    X_dados, y_dados = carregar("dados/dataAll20262.txt")

    perceptron = PerceptronNeuron(learning_rate=0.1, limiar=0.0, weight_range=(-0.5, 0.5))
    perceptron.train(X_dados, y_dados, verbose=True)
    
    titulo = f"Parte I – Perceptron (épocas: {perceptron.epochs} | ajustes: {perceptron.total_adjustments})"
    perceptron.plot_decision_boundary(X_dados, y_dados, title=titulo)


def run_part_two() -> None:
    """Executa a Parte II conforme o enunciado do trabalho."""
    print("\n=======================================")
    print(" INICIANDO PARTE II (Experimentação)")
    print("=======================================")
    
    try:
        dados = np.loadtxt("dados/data020262.txt") # <-- Exemplo: Equipe 3
    except ValueError:
        dados = np.load("dados/data020262.txt", allow_pickle=True).reshape(-1, 3)
        
    X_dados = dados[:, :2]
    y_dados = dados[:, 2]

    etas = [0.4, 0.1, 0.01]
    intervalos = [(-100, 100), (-0.5, 0.5)]
    N_REP = 10
    
    resultados = {}

    # Configuração da Tabela exigida
    tabela = PrettyTable([
        "Taxa de Aprendizado", 
        "Intervalo de Pesos", 
        "Qtde Ajustes (média ± dp)", 
        "Min Épocas", 
        "Max Épocas"
    ])

    total_iteracoes = len(etas) * len(intervalos) * N_REP

    # Laço sequencial com barra de progresso
    with tqdm(total=total_iteracoes, desc="Executando Experimentos") as pbar:
        for eta in etas:
            for inter in intervalos:
                ajustes = []
                epocas = []
                
                for rep in range(N_REP):
                    # Garante reprodutibilidade mas varia a cada repetição
                    seed = hash((eta, inter, rep)) % (2**31 - 1)
                    rng = np.random.default_rng(seed)
                    
                    # O limiar é sempre fixo em 0.0. Mexemos no weight_range!
                    n = PerceptronNeuron(learning_rate=eta, limiar=0.0, weight_range=inter)
                    n.train(X_dados, y_dados, max_epochs=None, rng=rng, verbose=False)
                    
                    ajustes.append(n.total_adjustments)
                    epocas.append(n.epochs)
                    
                    pbar.update(1)
                
                arr_ajustes = np.array(ajustes)
                arr_epocas = np.array(epocas)
                resultados[(eta, inter)] = (arr_ajustes, arr_epocas)
                
                # Formatando as strings para a tabela
                str_ajustes = f"{np.mean(arr_ajustes):.2f} ± {np.std(arr_ajustes):.2f}"
                str_inter = f"({inter[0]}, +{inter[1]})" if inter[1] > 0 else f"({inter[0]}, {inter[1]})"
                
                # Preenchendo a tabela
                tabela.add_row([
                    f"η = {eta}", 
                    str_inter, 
                    str_ajustes, 
                    np.min(arr_epocas), 
                    np.max(arr_epocas)
                ])

    print("\nResultados Parte II:")
    print(tabela)

    # ==========================================
    # PLOTAGEM DOS BOXPLOTS
    # ==========================================
    chaves = list(resultados.keys())
    rotulos = [f"η={e}\n{i}" for (e, i) in chaves]
    
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    
    dados_ajustes = [resultados[k][0] for k in chaves]
    axs[0].boxplot(dados_ajustes, tick_labels=rotulos, patch_artist=True, boxprops=dict(facecolor="lightblue"))
    axs[0].set_title("Ajustes no vetor de pesos")
    axs[0].set_ylabel("Quantidade de Ajustes")
    
    dados_epocas = [resultados[k][1] for k in chaves]
    axs[1].boxplot(dados_epocas, tick_labels=rotulos, patch_artist=True, boxprops=dict(facecolor="lightgreen"))
    axs[1].set_title("Épocas até a convergência")
    axs[1].set_ylabel("Quantidade de Épocas")
    
    for ax in axs: 
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.tick_params(axis="x", labelsize=9)
        
    plt.tight_layout() 
    plt.show()


def run_part_three() -> None:
    """Executa a Parte III: validação Holdout em problema não linear."""
    print("\n=======================================")
    print(" INICIANDO PARTE III (Holdout)")
    print("=======================================")

    X, y = carregar("dados/dataHoldout20262.txt")

    # Gráfico inicial para evidenciar a não separabilidade linear.
    plt.figure(figsize=(8, 5))
    plt.scatter(X[y == 0, 0], X[y == 0, 1],
                color="red", label="Classe 0", edgecolors="k", s=25)
    plt.scatter(X[y == 1, 0], X[y == 1, 1],
                color="blue", label="Classe 1", edgecolors="k", s=25)
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.title("Parte III – Dados Holdout: problema não-linearmente separável")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

    # Divisão 70/30 com semente igual ao identificador da equipe.
    rng_split = np.random.default_rng(ID_EQUIPE)
    indices = rng_split.permutation(len(X))
    n_treino = int(0.70 * len(X))

    idx_treino = indices[:n_treino]
    idx_teste = indices[n_treino:]

    X_treino, y_treino = X[idx_treino], y[idx_treino]
    X_teste, y_teste = X[idx_teste], y[idx_teste]

    print(f"Exemplos de treinamento: {len(X_treino)}")
    print(f"Exemplos de teste: {len(X_teste)}")

    # Parâmetros recomendados: eta=0.1 e pesos ~ U(-0.5, +0.5).
    # O treinamento ocorre por exatamente 100 épocas e embaralha
    # os exemplos a cada época.
    perceptron = PerceptronNeuron(
        learning_rate=0.1,
        limiar=0.0,
        weight_range=(-0.5, 0.5)
    )

    rng_treino = np.random.default_rng(ID_EQUIPE)
    perceptron.train(
        X_treino,
        y_treino,
        max_epochs=100,
        shuffle=True,
        rng=rng_treino,
        verbose=True
    )

    y_pred_treino = perceptron.predict(X_treino)
    y_pred_teste = perceptron.predict(X_teste)

    # 1. Matriz de confusão no conjunto de teste.
    cm = confusion_matrix(y_teste, y_pred_teste)
    print("\nMatriz de confusão – conjunto de teste:")
    print(cm)

    # 2 e 3. Métricas solicitadas para o conjunto de treinamento.
    print("\nMétricas – conjunto de treinamento:")
    print(f"Acurácia:   {accuracy_score(y_treino, y_pred_treino):.4f}")
    print(f"Precisão:   {precision_score(y_treino, y_pred_treino, zero_division=0):.4f}")
    print(f"Revocação:  {recall_score(y_treino, y_pred_treino, zero_division=0):.4f}")
    print(f"F-Score:    {f1_score(y_treino, y_pred_treino, zero_division=0):.4f}")

    # Métricas do teste
    print("\nMétricas – conjunto de teste:")
    print(f"Acurácia:   {accuracy_score(y_teste, y_pred_teste):.4f}")
    print(f"Precisão:   {precision_score(y_teste, y_pred_teste, zero_division=0):.4f}")
    print(f"Revocação:  {recall_score(y_teste, y_pred_teste, zero_division=0):.4f}")
    print(f"F-Score:    {f1_score(y_teste, y_pred_teste, zero_division=0):.4f}")

    # Dois gráficos lado a lado com a mesma reta aprendida.
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    def plot_holdout(ax, X_plot, y_plot, titulo):
        ax.scatter(X_plot[y_plot == 0, 0], X_plot[y_plot == 0, 1],
                   color="red", label="Classe 0", edgecolors="k", s=25)
        ax.scatter(X_plot[y_plot == 1, 0], X_plot[y_plot == 1, 1],
                   color="blue", label="Classe 1", edgecolors="k", s=25)

        x_min, x_max = X[:, 0].min() - 5, X[:, 0].max() + 5
        y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
        w0, w1, w2 = perceptron.weights
        x_vals = np.array([x_min, x_max])

        if w2 != 0:
            y_vals = (w0 - w1 * x_vals) / w2
            ax.plot(x_vals, y_vals, color="black",
                    linewidth=2, label="Reta de decisão")
        elif w1 != 0:
            ax.axvline(w0 / w1, color="black",
                       linewidth=2, label="Reta de decisão")

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_title(titulo)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)

    plot_holdout(axs[0], X_treino, y_treino,
                 "Dados de treinamento + reta do Perceptron")
    plot_holdout(axs[1], X_teste, y_teste,
                 "Dados de teste + reta do Perceptron")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Execute a parte que desejar
    run_part_one()
    run_part_two()
    run_part_three()