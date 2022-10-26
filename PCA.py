import matplotlib

matplotlib.use('TkAgg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

sns.set()

# Base de datos
datos = pd.read_csv('datos_depurados.csv')

df = pd.DataFrame(datos)
df.City = df.City.astype('category')
df.type = df.type.astype('category')
df2 = df.iloc[:, [1, 2, 3, 4, 5, 6, 7]]


# Normalizar datos

def mean_norm(df_input):
    return df_input.apply(lambda x: (x - x.mean()) / x.std(), axis=0)


df_mean_norm = mean_norm(df2)

# PCA

pca = PCA(n_components=7)
pca.fit(df_mean_norm)
x_pca = pca.transform(df_mean_norm)
expl = pca.explained_variance_ratio_

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')

# Realizar PCA con 3 componentes

pca2 = PCA(n_components=3)
pca2.fit(df_mean_norm)
pca2.transform(df_mean_norm)

scores_pca = pca2.transform(df_mean_norm)

wcss = []

for i in range(1, 21):
    kmeans_pca = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans_pca.fit(scores_pca)
    wcss.append(kmeans_pca.inertia_)

plt.figure(figsize=(10, 8))
plt.plot(range(1, 21), wcss, marker='o', linestyle='--')
plt.xlabel('Number of Cluster')
plt.ylabel('WCSS')
plt.title('K-means with PCA Clustering')



# Implementar K-means

k_means_pca2 = KMeans(n_clusters=3,init='k-means++',random_state=42)
k_means_pca2.fit(scores_pca)




# ANALISIS DE RESULTADOS

# Eiquetar componentes

df_segm_pca_kmeans = pd.concat([df_mean_norm.reset_index(drop=True), pd.DataFrame(scores_pca)], axis=1)
df_segm_pca_kmeans.columns.values[-3:] = ['Component 1', 'Component 2', 'Component 3']

# The last Column we add contains the pca K-means clustering labels

df_segm_pca_kmeans['Segment K-means PCA'] = k_means_pca2.labels_
df_segm_pca_kmeans['Segment'] = df_segm_pca_kmeans['Segment K-means PCA'].map({0: 'First',
                                                                               1: 'Second',
                                                                               2: 'Third'})

plt.figure(figsize=(10,8))
sns.scatterplot(x=df_segm_pca_kmeans['Component 2'], y=df_segm_pca_kmeans['Component 1'], hue=df_segm_pca_kmeans['Segment'], palette=['g', 'r', 'c'])
plt.title('Clusters by PCA Components')
#plt.show()




#Contar observaciones por Cluster

labels = k_means_pca2.predict(scores_pca)
# Getting the cluster centers
C = k_means_pca2.cluster_centers_
colores = ['g', 'r', 'c']
asignar = []
for row in labels:
    asignar.append(colores[row])

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(scores_pca[:, 0], scores_pca[:, 1], scores_pca[:, 2], c=asignar, s=60)
ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='*', c=colores, s=1000)


copy = pd.DataFrame()
copy['City']= datos['City'].values
copy['label'] = labels
cantidadGrupo = pd.DataFrame()
cantidadGrupo['color']= colores
cantidadGrupo['cantidad']=copy.groupby('label').size()



# Mirar observaciones de algun cluster por alguna variable
group_referrer_index = copy['label'] == 0
group_referrals = copy[group_referrer_index]


diversidadGrupo = pd.DataFrame()
diversidadGrupo['City'] = ['Bogotá','Medellín']
diversidadGrupo['Cantidad'] = group_referrals.groupby('City').size()


###Inmuebles mas cerca de los centroides

#Posicion en el array de los inmuebles
closest,_= pairwise_distances_argmin_min(k_means_pca2.cluster_centers_,scores_pca)
print(closest)


#Numero de inmuebles
inmuebles= datos['Unnamed: 0'].values

for row in closest:
    print(inmuebles[row])

#Clasificar nuevas muestras

x_new=np.array([[1.22,-0.03,0.59]])
new_labels= k_means_pca2.predict(x_new)

