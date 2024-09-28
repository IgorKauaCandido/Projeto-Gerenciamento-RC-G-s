import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def apply_pca(data, n_components=2):
   
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

   
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(data_scaled)

    columns = [f'PC{i+1}' for i in range(n_components)]
    pca_df = pd.DataFrame(data=pca_data, columns=columns)

    
    explained_variance = pca.explained_variance_ratio_

    return pca_df, pca, explained_variance

iris = load_iris()
iris_data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_target = iris.target
iris_target_names = iris.target_names


X_train, X_test, y_train, y_test = train_test_split(iris_data, iris_target, test_size=0.3, random_state=42)

pca_result_train, pca_model, variance = apply_pca(X_train, n_components=2)


pca_result_train['true_class'] = pd.Categorical.from_codes(y_train, iris_target_names)

scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_test)
pca_result_test = pca_model.transform(X_test_scaled)

pca_result_test_df = pd.DataFrame(data=pca_result_test, columns=['PC1', 'PC2'])
pca_result_test_df['true_class'] = pd.Categorical.from_codes(y_test, iris_target_names)

print("Componentes Principais (Treinamento):\n", pca_result_train.head())
print("Variância Explicada:", variance)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
for iris_class in iris_target_names:
    subset = pca_result_train[pca_result_train['true_class'] == iris_class]
    plt.scatter(subset['PC1'], subset['PC2'], label=f'Classe: {iris_class}')

plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title('PCA do Conjunto de Dados Iris (Treinamento)')
plt.legend()


plt.subplot(1, 2, 2)
for iris_class in iris_target_names:
    subset = pca_result_test_df[pca_result_test_df['true_class'] == iris_class]
    plt.scatter(subset['PC1'], subset['PC2'], label=f'Classe: {iris_class}')

plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title('PCA do Conjunto de Dados Iris (Teste)')
plt.legend()

plt.tight_layout()
plt.show()
 este codigo eteve o intuito de construir um sistema de ia com pca para demonstração 