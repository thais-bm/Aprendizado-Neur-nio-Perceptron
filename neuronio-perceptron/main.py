import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from tqdm.auto import tqdm
from neuron import PerceptronNeuron  # Certifique-se de que o arquivo salvo se chama neuron.py

# ==========================================
# CONFIGURAÇÕES DA EQUIPE
# ==========================================
ID_EQUIPE = 0  # <-- Altere aqui para o ID da sua equipe

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
    
    # Título conforme você especificou anteriormente
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

    # AS 6 CONFIGURAÇÕES EXATAS DO ENUNCIADO
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

if __name__ == "__main__":
    # Execute a parte que desejar
    run_part_one()
    run_part_two()