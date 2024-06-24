
## Índice

- [Equipe](#equipe)
- [Introdução](#introdução)
- [Descrição do Descritor Implementado](#descrição-do-Descritor-Implementado)



# Projeto Final de Processamento de Imagens

## Equipe
- Luiz Henrique da Silva Araujo 2348101
- Roberto Luiz Pereira Raposo 2350653
- Ricardo Takeshi Outi Miyamoto 2348144



# Introdução
#### Classificação de Imagens de COVID-19 Usando Padrões Binários Locais (LBP) e Máquina de Vetores de Suporte (SVM)

Este projeto demonstra um pipeline simples de aprendizado de máquina para classificar imagens de COVID-19 usando Padrões Binários Locais (LBP) para extração de características e uma Máquina de Vetores de Suporte (SVM) para classificação.


# Descrição do Descritor Implementado

### Descritor de Padrões Binários Locais (LBP)
Nome do Descritor: Local Binary Patterns (LBP)

#### Tipo de Características Extraídas:

- Textura: O LBP é um descritor baseado em textura. Ele captura informações de textura local das imagens.
#### Como Funciona:

- Padrão Binário Local: Para cada pixel da imagem, o LBP considera um vizinho de pixels em uma determinada vizinhança (geralmente uma janela de 3x3).

- Comparação Binária: O valor do pixel central é comparado com o valor dos pixels vizinhos. Se o valor do pixel vizinho for maior ou igual ao do pixel central, um bit (1) é definido, caso contrário, um bit (0) é definido.

- Códigos Binários: Esses bits são combinados para formar um código binário, que é então convertido em um valor decimal.

- Histograma: A imagem resultante de valores decimais (LBP) é usada para calcular um histograma, que conta a frequência de cada padrão binário. Este histograma é normalizado para que a soma de seus valores seja 1.

#### Características Extraídas:
- Padrões Locais de Textura: O LBP captura padrões locais de textura que são robustos a mudanças de iluminação. Cada valor no histograma representa a ocorrência de um padrão específico de textura na imagem.

- Invariância a Iluminação: Devido à forma como os padrões são calculados, o LBP é relativamente insensível a variações de iluminação, tornando-o útil para aplicações em condições de iluminação variáveis.

#### Aplicações:
- Classificação de Imagens: O LBP é frequentemente usado em tarefas de classificação de imagens onde a textura é uma característica importante, como na análise de superfícies, detecção de faces e, como neste caso, na distinção entre imagens de pulmões com e sem COVID-19.

- Reconhecimento de Padrões: Devido à sua robustez e simplicidade, o LBP é uma escolha comum em problemas de reconhecimento de padrões.

#### Vantagens:
- Simplicidade: O cálculo do LBP é simples e eficiente.
- Robustez: O LBP é robusto a variações de iluminação e contraste.
- Eficiência Computacional: A extração de características usando LBP é rápida e requer poucos recursos computacionais.

#### Limitações:
- Dependência de Resolução: O LBP pode ser sensível a variações na resolução da imagem.
- Informação Espacial Limitada: Como o LBP captura apenas informações de textura local, pode perder algumas informações espaciais mais amplas. 

No código, a função extract_lbp_features implementa a extração do LBP da seguinte forma:

- Cálculo do LBP: A função local_binary_pattern da biblioteca skimage é usada para calcular o LBP da imagem.
- Histograma: Um histograma dos valores do LBP é calculado usando np.histogram.
- Normalização: O histograma é normalizado para que a soma de seus valores seja 1.
Este histograma de LBP é então usado como vetor de características para a classificação usando uma Máquina de Vetores de Suporte (SVM).

# Repositório do Projeto
> https://github.com/RobertoLuiz99/Projeto-de-Processamento-de-Imagens


# Classificador e Acurácia

### Classificador

Neste projeto, usamos uma Máquina de Vetores de Suporte (SVM) para classificar as imagens com base nas características extraídas pelos Padrões Binários Locais (LBP). A SVM é um classificador poderoso e eficiente, especialmente adequado para problemas de classificação binária. 

**Configurações do Classificador:**
- **Kernel**: Linear
- **Divisão dos Dados**: Os dados são divididos em conjuntos de treino e teste na proporção de 80% para treino e 20% para teste usando a função `train_test_split` da biblioteca `scikit-learn`.

O pipeline de classificação envolve as seguintes etapas:
1. **Extração de Características**: As características LBP são extraídas de cada imagem.
2. **Treinamento do Classificador**: A SVM é treinada com as características extraídas das imagens do conjunto de treino.
3. **Predição**: O classificador treinado é usado para prever as classes das imagens do conjunto de teste.

# Pré-requisitos

- Python
- OpenCV
- NumPy
- Scikit-image
- Scikit-learn

## Conjunto de Dados

O conjunto de dados deve estar organizado na seguinte estrutura de diretórios:

```
dataset/
├── covid/
│   ├── imagem1.png
│   ├── imagem2.png
│   └── ...
└── normal/
    ├── imagem1.png
    ├── imagem2.png
    └── ...
```

Cada pasta contém imagens em escala de cinza para a respectiva classe.

# Instruções de Uso


1. Clone o repositório:

    ```bash
    git clone https://github.com/RobertoLuiz99/Projeto-de-Processamento-de-Imagens.git
    cd covid-classificacao
    ```

2. Instale as dependências necessárias:

    ```bash
    pip install opencv-python numpy scikit-image scikit-learn
    ```

## Uso

1. Coloque seu conjunto de dados no diretório `dataset` seguindo a estrutura mencionada acima.

2. Execute o script:

    ```bash
    python deteccaoCovid.py
    ```

O script processará as imagens, extrairá as características LBP, treinará um classificador SVM e exibirá a matriz de confusão e a acurácia.

## Resultados

O script imprime a matriz de confusão e a acurácia do modelo após treinar e testar no conjunto de dados.

```
Matriz de Confusão:
[[VP, FP]
 [FN, VN]]
Acurácia: XX.XX%
```