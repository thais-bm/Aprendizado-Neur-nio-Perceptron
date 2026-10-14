# Aprendizado Supervisionado no Neurônio Perceptron

Repositório dedicado ao desenvolvimento do **Projeto Prático 2.1** da disciplina de **Redes Neurais Artificiais (2026.2)** da **Universidade do Estado do Amazonas (UEA)**, ministrada pela Profa. Dra. Elloá B. Guedes.

O objetivo deste projeto consiste em implementar, avaliar e analisar o algoritmo de treinamento supervisionado do neurônio **Perceptron de Rosenblatt** aplicado a problemas de classificação binária no $\mathbb{R}^2$.

---

## Integrantes da Dupla

| Nome | Matrícula | Último Dígito |
| :--- | :--- | :---: |
| JULIO CLEITON SEABRA MENDES | 2415310015 | $5$ |
| THAIS CAROLINA BRAGA DE MAGALHAES | 2415310037 | $7$ |

### Identificador do Conjunto de Dados da Equipe:
O identificador de dados atribuído à dupla é determinado pela regra:
$$\text{ID} = (5 + 7) \pmod 4$$
* **Identificador da Equipe:** `data[0]20262.txt`

---

## 📌 Escopo do Projeto

O projeto é estruturado em três etapas principais:

### 1. Parte I — Problema Linearmente Separável
- **Dataset:** `dataAll20262.txt`
- **Configuração:**
  - Função de ativação degrau com limiar $\vartheta = 0$;
  - Taxa de aprendizado $\eta = 0.1$;
  - Vetor de pesos inicializados com distribuição uniforme $w_i \sim \mathcal{U}(-0.5, +0.5)$;
  - Critério de parada: convergência total (zero erros de classificação em todos os exemplos).
- **Métricas & Visualização:**
  - Número de épocas até a convergência e quantidade de ajustes no vetor de pesos;
  - Gráfico de dispersão com os exemplos rotulados (Classe 0 em **vermelho**, Classe 1 em **azul**) e a reta de separação aprendida.

### 2. Parte II — Experimentação Sistemática
- **Dataset:** `data[0]20262.txt`
- **Planejamento Experimental:**
  - Grade de hiperparâmetros: $\eta \in \{0.4, 0.1, 0.01\}$ combinado aos intervalos de pesos $\mathcal{I} \in \{(-100, +100), (-0.5, +0.5)\}$, totalizando 6 configurações;
  - Cada configuração é executada por **10 repetições** independentes;
  - Coleta da média e desvio padrão de ajustes no vetor de pesos, além do mínimo e máximo de épocas até convergência;
  - Análise comparativa através de tabelas e diagramas de caixa (*boxplots*).

### 3. Parte III — Validação Holdout em Problema Não-Linearmente Separável
- **Dataset:** `dataHoldout20262.txt`
- **Configuração & Metodologia:**
  - Demonstração gráfica da não-linearidade do conjunto de dados;
  - Divisão Holdout em 70% (Treinamento) e 30% (Teste) fixando a semente aleatória com o `ID` da equipe;
  - Treinamento por 100 épocas com permutação aleatória dos exemplos a cada época;
  - Avaliação no conjunto de testes via Matriz de Confusão, Acurácia, Precisão, Revocação e F1-Score (utilizando `sklearn.metrics`);
  - Gráficos lado a lado exibindo a fronteira de decisão final nos conjuntos de treino e teste.

---

## 🛠️ Tecnologias e Dependências

O projeto é gerenciado através do [**uv**](https://docs.astral.sh/uv/), um gerenciador de pacotes e ambientes virtuais ultrarrápido para Python.

- **Python:** $\ge 3.11$
- **NumPy:** Operações vetoriais e matriciais obrigatórias (sem uso de laços/listas nativas para álgebra linear).
- **Matplotlib:** Geração das fronteiras de decisão, gráficos de dispersão e boxplots.
- **Pandas:** Tabulação e manipulação de resultados consolidados.
- **Scikit-Learn:** Exclusivamente para cálculo de métricas de desempenho e partições (`sklearn.metrics`, `train_test_split`).

---

## 🚀 Como Executar o Projeto

### Pré-requisitos
Certifique-se de ter o **uv** instalado na sua máquina:
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Instalação do Ambiente

1. Clone o repositório:
```bash
git clone https://github.com/SEU-USUARIO/neuronio-perceptron.git
cd neuronio-perceptron
```

2. Crie o ambiente virtual e sincronize as dependências com base no arquivo `uv.lock`:
```bash
uv sync
```

### Execução

Para rodar os scripts ou notebooks interativos utilizando o ambiente virtual gerenciado pelo `uv`:

- **Executar script principal:**
```bash
uv run python main.py
```

- **Abrir ambiente Jupyter Notebook / JupyterLab:**
```bash
uv run --with jupyter jupyter lab
# ou
uv run --with jupyter jupyter notebook
```

---

## 📂 Estrutura do Repositório

```text
├── dados/
│   ├── data020262.txt
│   ├── data120262.txt
│   ├── data220262.txt
│   ├── data320262.txt
│   ├── dataAll20262.txt
│   └── dataHoldout20262.txt
├── .python-version
├── pyproject.toml
├── uv.lock
├── main.py
└── README.md
```

---

## 📄 Licença

Este projeto é desenvolvido para fins acadêmicos e está sob a licença [MIT](LICENSE).