import mysql.connector
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import sklearn

# Verificar a versão do scikit-learn para garantir que está atualizado
print(f"Versão do scikit-learn: {sklearn.__version__}")

# Conectar ao MySQL
conn = mysql.connector.connect(
    host="127.0.0.1",  # Substitua pelo seu host
    user="root",  # Substitua pelo seu usuário
    password="1234",  # Substitua pela sua senha
    database="loja_virtual"  # Substitua pelo nome do seu banco de dados
)

# Consultar os dados da tabela clientes
query = "SELECT * FROM clientes;"
df = pd.read_sql(query, conn)

# Fechar a conexão
conn.close()

# Exibir as primeiras linhas dos dados para garantir que o DataFrame foi carregado corretamente
print(df.head())

# Estatísticas básicas
print("\nEstatísticas básicas dos dados:")
print(df.describe())

# Analisando a distribuição das idades
plt.figure(figsize=(8, 6))
sns.histplot(df['idade'], kde=True, color='blue')
plt.title('Distribuição das Idades dos Clientes')
plt.xlabel('Idade')
plt.ylabel('Frequência')
plt.show()

# Distribuição do total de compras
plt.figure(figsize=(8, 6))
sns.histplot(df['total_compras'], kde=True, color='green')
plt.title('Distribuição do Total de Compras dos Clientes')
plt.xlabel('Total de Compras')
plt.ylabel('Frequência')
plt.show()

# Analisando a correlação entre as variáveis
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlação entre as Variáveis')
plt.show()

# Criando uma variável para a recência da última compra
df['data_ultima_compra'] = pd.to_datetime(df['data_ultima_compra'])
df['dias_ultima_compra'] = (datetime.now() - df['data_ultima_compra']).dt.days

# Vamos também criar uma variável target, indicando se o cliente fez compras nos últimos 30 dias
df['comprou_novamente'] = (df['dias_ultima_compra'] <= 30).astype(int)

# Exibir as primeiras linhas do DataFrame atualizado
print("\nPrimeiras linhas com a variável target 'comprou_novamente':")
print(df[['nome', 'dias_ultima_compra', 'comprou_novamente']].head())

# Definindo as variáveis independentes (X) e a variável dependente (y)
X = df[['idade', 'total_compras', 'quantidade_compras', 'dias_ultima_compra']]
y = df['comprou_novamente']

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Treinando o modelo de regressão logística
modelo = LogisticRegression(max_iter=1000)  # Definindo max_iter para evitar erro de convergência
modelo.fit(X_train, y_train)

# Prevendo no conjunto de teste
y_pred = modelo.predict(X_test)

# Avaliando o modelo
print("\nAvaliação do Modelo:")
print(classification_report(y_test, y_pred))  # Relatório de precisão, recall e f1-score
print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, y_pred))  # Matriz de confusão

# Exibindo os coeficientes do modelo
coeficientes = pd.DataFrame(modelo.coef_[0], X.columns, columns=['Coeficiente'])
print("\nCoeficientes do Modelo:")
print(coeficientes)











