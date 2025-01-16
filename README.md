# Projeto Python IA: Inteligência Artificial e Previsões

## Case: Score de Crédito dos Clientes

Este projeto utiliza técnicas de inteligência artificial para prever o score de crédito dos clientes de um banco. O objetivo é desenvolver um modelo que analise as informações disponíveis dos clientes e determine automaticamente se o score de crédito é "Ruim", "Ok" ou "Bom".

---

## Descrição do Projeto

### Contexto
Você foi contratado por um banco para desenvolver um modelo preditivo que auxilie na classificação do score de crédito dos clientes. A base de dados contém informações diversas sobre os clientes, e o desafio é criar um modelo eficiente para realizar essas previsões.

### Base de Dados
O projeto utiliza duas bases de dados:
1. **clientes.csv**: Informações históricas dos clientes e seus scores de crédito.
2. **novos_clientes.csv**: Dados de novos clientes para os quais o score deve ser previsto.

### Tecnologias Utilizadas
- Linguagem: Python
- Bibliotecas:
  - pandas
  - scikit-learn

### Instalação de Dependências
```bash
!pip install pandas scikit-learn
```

---

## Passo a Passo do Projeto

### 1. Entendimento do Problema e da Base de Dados
A base de dados é carregada e analisada com a biblioteca `pandas`. Aqui verificamos o tipo de dados e realizamos transformações iniciais:
```python
import pandas as pd
tabela = pd.read_csv('clientes.csv')
display(tabela)
display(tabela.info())
```

### 2. Preparação dos Dados
Os dados categóricos foram transformados em números com `LabelEncoder` para que os modelos possam processá-los:
```python
from sklearn.preprocessing import LabelEncoder as le

# Exemplo de transformação
codificador_profissao = le()
tabela['profissao'] = codificador_profissao.fit_transform(tabela['profissao'])
```

### 3. Treinamento do Modelo
Dividimos os dados em conjuntos de treino e teste:
```python
from sklearn.model_selection import train_test_split

y = tabela['score_credito']
x = tabela.drop(columns=['score_credito', "id_cliente"])

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3)
```

Dois algoritmos foram treinados:
- Random Forest Classifier
- K-Nearest Neighbors (KNN)

#### Exemplo de Treinamento:
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

modelo_arvoredecisao = RandomForestClassifier()
modelo_knn = KNeighborsClassifier()

modelo_arvoredecisao.fit(x_treino, y_treino)
modelo_knn.fit(x_treino, y_treino)
```

### 4. Avaliação do Modelo
A precisão dos modelos foi avaliada usando a métrica `accuracy_score`:
```python
from sklearn.metrics import accuracy_score

previsao_arvoredecisao = modelo_arvoredecisao.predict(x_teste)
previsao_knn = modelo_knn.predict(x_teste)

print("Precisão Random Forest:", accuracy_score(y_teste, previsao_arvoredecisao))
print("Precisão KNN:", accuracy_score(y_teste, previsao_knn))
```

### 5. Previsão para Novos Clientes
O modelo selecionado (Random Forest) foi utilizado para prever os scores de novos clientes:
```python
tabela_novos_clientes = pd.read_csv('novos_clientes.csv')

# Transformações
# ...
previsao_novos_clientes = modelo_arvoredecisao.predict(tabela_novos_clientes)

display(previsao_novos_clientes)
```

---

## Resultados
O modelo foi capaz de prever os scores de crédito com alta precisão, mostrando-se eficaz para o caso de uso proposto. O Random Forest foi escolhido como o melhor modelo para a tarefa.

---

## Links e Recursos
- [Arquivos do Projeto](https://drive.google.com/drive/folders/1FbDqVq4XLvU85VBlVIMJ73p9oOu6u2-J?usp=drive_link)
- [Documentação do Scikit-learn](https://scikit-learn.org/stable/)

---

## Considerações Finais
Este projeto é um excelente ponto de partida para explorar modelos de classificação e suas aplicações no setor financeiro. Com aprimoramentos e mais dados, a solução pode ser ainda mais robusta e generalizável.
