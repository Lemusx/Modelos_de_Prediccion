
# coding: utf-8

# # Algoritmo de clasificacion de canciones
# 
# 26 Febrero 2018
# 
# Alonso Jesús Aguirre Tobar
# 
# ## Desafio
# 
# 1. Analiza los dataset data_todotipo.csv y data_reggaeton.csv. ¿Qué puedes decir de los datos,
# distribuciones, missing, etc? ¿En qué se diferencian? Entregable: texto/imágenes.
# 2. Consolida los dos datasets en uno solo y genera una marca para las canciones que sean
# reggaeton en base a los datasets. Entregable: csv con dataset y código (si es que usaste).
# 3. Entrena uno o varios modelos (usando el/los algoritmo(s) que prefieras) para detectar qué
# canciones son reggaeton. Entregable: modelo en cualquier formato y código (si es que
# usaste).
# 4. Evalúa tu modelo. ¿Qué performance tiene? ¿Qué métricas usas para evaluar esa
# performance? ¿Por qué elegiste ese algoritmo en particular? ¿Cómo podrías mejorar la
# performance​ ? Entregable: texto/imágenes.
# 5. Aplica tu modelo sobre el dataset “data_test.csv”, agregándole a cada registro dos nuevos
# campos: marca_reggaeton (un 1 cuando es reggaetón, un 0 cuando no lo es) y
# probabilidad_reggaeton (probabilidad de que la canción sea reggaeton). ¿Cómo elegiste
# cuáles marcar? ¿De qué depende que predigas la existencia de mayor o menor cantidad de
# reggaetones? Entregable: texto/imágenes.
# 
# ## Informacion del dataset
# 
# Como dice el título, vamos a hacer un detector de reggaetones. Para eso armamos un dataset usando
# la API de Spotify, que nos da información detallada de cada canción (documentación). Para cada
# canción se tienen los siguientes parámetros:  acousticness, danceability, duration_ms, energy,
# instrumentalness, key, liveness, loudness, mode, speechiness, tempo y  time_signature.
# Encontrarás tres archivos,
# ● data_todotipo.csv tiene canciones NO reggaeton.
# ● data_reggaeton.csv tiene SOLO canciones reggaeton.
# ● data_test.csv será el dataset para testear la solución.

# ### Importar las librerías requeridas

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_predict


# ### Pregunta 1 y 2
# 
# Se leen los datos y se revisa los missing values, datos NaN (elimine la columna 'id_new' puesto que tiene la misma correlacion con 'Unnamed: 0')

# In[2]:


df = pd.read_csv('data_todotipo.csv')
df = df.drop(columns='id_new')
df.isnull().sum()


# Una vez identificadas las columnas que contienen valores NaN se reemplazaron por la media o mediana dependiendo del caso (el criterio para elegir una fue utilizar df.describe(), donde me entregaba la desviacion estandar, asi las columnas con mayor desviacion estandar se reemplazaron los NaN por la media. En cambio las columnas que tenian poca desviacion estandar se reemplazaron los NaN por la mediana).

# In[3]:


df['popularity'].fillna(df['popularity'].mean(), inplace = True)
df['danceability'].fillna(df['danceability'].median(), inplace = True)
df['energy'].fillna(df['energy'].median(), inplace = True)
df['key'].fillna(df['key'].mean(), inplace = True)
df['loudness'].fillna(df['loudness'].mean(), inplace = True)
df['mode'].fillna(df['mode'].median(), inplace = True)
df['speechiness'].fillna(df['speechiness'].median(), inplace = True)
df['acousticness'].fillna(df['acousticness'].median(), inplace = True)
df['instrumentalness'].fillna(df['instrumentalness'].median(), inplace = True)
df['liveness'].fillna(df['liveness'].median(), inplace = True)
df['valence'].fillna(df['valence'].median(), inplace = True)
df['tempo'].fillna(df['tempo'].mean(), inplace = True)
df['duration'].fillna(df['duration'].mean(), inplace = True)
df['time_signature'].fillna(df['time_signature'].median(), inplace = True)
df['class'] = np.array([0]*len(df))


# Se agrego una columna 'class' para clasificar si es (1) o no es (0) una cancion de Reggaeton, puesto que el primer archivo es exclusivamente de canciones que no son de reggaeton; se creo un vector con solo '0'. Verificamos ademas los valores nulos del mismo DataFrame

# In[4]:


df.isnull().sum()


# In[5]:


df.describe()


# La cantidad de datos es 2230, y la media de la columna 'class' es 0, es logico puesto que es una columna de ceros.

# Se lee el siguiente archivo y se agrega la coluna que permite clasificar si es o no es una cancion de reggaeton, puesto que el archivo es unicamente canciones de reggaeton; se creo un arreglo de valores 1 con longitud del DataFrame

# In[6]:


df2 = pd.read_csv('data_reggaeton.csv')
df2['class'] = np.array([1]*len(df2))


# In[7]:


df.isnull().sum()


# No hay valores nulos, entonces seguimos con la exploracion de datos

# In[8]:


df2.describe()


# La cantidad de canciones es de 70.
# Se combinaron los dos dataset, y se eliminaron las columnas que eran exclusivas de cada dataset. Se guardo el archivo en formato csv (sera adjuntado en el mail)

# In[9]:


df3 = pd.merge(df, df2, how='outer')
df3 = df3.drop(columns='id_new')
df3 = df3.drop(columns='time_signature')
np.savetxt("output.csv", df3, fmt="%f", delimiter=",")


# Para analizar los graficos se contrarrestaron en un mismo grafico las canciones que son y no son de Reggaeton, si bien es cierto es dificil distinguir la distribucion de las canciones de reggaeton por su notoria diferencia de numero de canciones, en general es muy utilizada esta forma de exponer los datos porque compara directamente las distintas distribuciones de datos.

# In[10]:


figure = plt.figure(figsize=(15,8))
plt.hist([df3[df3['class']==1]['popularity'],df3[df3['class']==0]['popularity']], stacked=True, color = ['g','r'],
         bins = 30,label = ['Reggaeton','No-Reggaeton'])
plt.xlabel('Popularity')
plt.ylabel('Numero de canciones')
plt.legend()


# In[11]:


figure = plt.figure(figsize=(15,8))
plt.hist([df3[df3['class']==1]['danceability'],df3[df3['class']==0]['danceability']], stacked=True, color = ['g','r'],
         bins = 30,label = ['Reggaeton','No-Reggaeton'])
plt.xlabel('Danceability')
plt.ylabel('Numero de canciones')
plt.legend()


# In[12]:


figure = plt.figure(figsize=(15,8))
plt.hist([df3[df3['class']==1]['energy'],df3[df3['class']==0]['energy']], stacked=True, color = ['g','r'],
         bins = 30,label = ['Reggaeton','No-Reggaeton'])
plt.xlabel('Energy')
plt.ylabel('Numero de canciones')
plt.legend()


# In[13]:


figure = plt.figure(figsize=(15,8))
plt.hist([df3[df3['class']==1]['key'],df3[df3['class']==0]['key']], stacked=True, color = ['g','r'],
         bins = 30,label = ['Reggaeton','No-Reggaeton'])
plt.xlabel('Key')
plt.ylabel('Numero de canciones')
plt.legend()


# In[14]:


figure = plt.figure(figsize=(15,8))
plt.hist([df3[df3['class']==1]['loudness'],df3[df3['class']==0]['loudness']], stacked=True, color = ['g','r'],
         bins = 30,label = ['Reggaeton','No-Reggaeton'])
plt.xlabel('Loudness')
plt.ylabel('Numero de canciones')
plt.legend()


# In[15]:


figure = plt.figure(figsize=(15,8))
plt.hist([df3[df3['class']==1]['mode'],df3[df3['class']==0]['mode']], stacked=True, color = ['g','r'],
         bins = 30,label = ['Reggaeton','No-Reggaeton'])
plt.xlabel('Mode')
plt.ylabel('Numero de canciones')
plt.legend()


# In[16]:


figure = plt.figure(figsize=(15,8))
plt.hist([df3[df3['class']==1]['speechiness'],df3[df3['class']==0]['speechiness']], stacked=True, color = ['g','r'],
         bins = 30,label = ['Reggaeton','No-Reggaeton'])
plt.xlabel('Speechiness')
plt.ylabel('Numero de canciones')
plt.legend()


# In[17]:


figure = plt.figure(figsize=(15,8))
plt.hist([df3[df3['class']==1]['acousticness'],df3[df3['class']==0]['acousticness']], stacked=True, color = ['g','r'],
         bins = 30,label = ['Reggaeton','No-Reggaeton'])
plt.xlabel('Acouctisness')
plt.ylabel('Numero de canciones')
plt.legend()


# In[18]:


figure = plt.figure(figsize=(15,8))
plt.hist([df3[df3['class']==1]['instrumentalness'],df3[df3['class']==0]['instrumentalness']], stacked=True, color = ['g','r'],
         bins = 30,label = ['Reggaeton','No-Reggaeton'])
plt.xlabel('Instrumentalness')
plt.ylabel('Numero de canciones')
plt.legend()


# In[19]:


figure = plt.figure(figsize=(15,8))
plt.hist([df3[df3['class']==1]['liveness'],df3[df3['class']==0]['liveness']], stacked=True, color = ['g','r'],
         bins = 30,label = ['Reggaeton','No-Reggaeton'])
plt.xlabel('Liveness')
plt.ylabel('Numero de canciones')
plt.legend()


# In[20]:


figure = plt.figure(figsize=(15,8))
plt.hist([df3[df3['class']==1]['valence'],df3[df3['class']==0]['valence']], stacked=True, color = ['g','r'],
         bins = 30,label = ['Reggaeton','No-Reggaeton'])
plt.xlabel('Valence')
plt.ylabel('Numero de canciones')
plt.legend()


# In[21]:


figure = plt.figure(figsize=(15,8))
plt.hist([df3[df3['class']==1]['tempo'],df3[df3['class']==0]['tempo']], stacked=True, color = ['g','r'],
         bins = 30,label = ['Reggaeton','No-Reggaeton'])
plt.xlabel('Tempo')
plt.ylabel('Numero de canciones')
plt.legend()


# In[22]:


figure = plt.figure(figsize=(15,8))
plt.hist([df3[df3['class']==1]['duration'],df3[df3['class']==0]['duration']], stacked=True, color = ['g','r'],
         bins = 30,label = ['Reggaeton','No-Reggaeton'])
plt.xlabel('Duration')
plt.ylabel('Numero de canciones')
plt.legend()


# ### Diferenciacion de los dataset
# 
# Dado los graficos de mas arriba se aprecia que ciertas features se distribuyen de forma distinta, esto quiere decir que esas features son predominantes para clasificar si una cancion es o no una cancion de reggaeton... una forma rapida pero bastante simple es comparar la media con su incertidumbre correspondiente, determinada por la desviacion estandar y el numero de observaciones. Estos datos estadisticos se visualizan usando df.describe(), si bien es cierto la media de por ejemplo 'popularity' se distingue con claridad, pero la incertidumbre complica la comparativa dada la gran diferencia de observaciones, esto provoca que las canciones de reggaeton tengan una alta incertidumbre. Es por esta razon que se busca un modelo predictivo, para que no solo compare una sola feature, sino que con un algoritmo incluya todas las variantes en juego, asi tener una mayor precision a la hora de clasificar las canciones.

# ### Pregunta 3 y 4
# 
# Creamos los modelos predictivos, para se uso el dataset 'df3' que contiene los dataset combinados. Definimos las variables que seran entrenadas,  escalamos los valores de X para tener una variable aleatoria, y una proyeccion de los datos. Usamos KFold y Cross Validation para evitar un sobre-entrenamiento de datos.

# In[23]:


X = np.array(df3.drop('class', axis=1))
X = StandardScaler().fit_transform(X)
y = np.array(df3['class'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
def accuracy(model):
    kf = KFold(n_splits=5)
    crv = cross_val_score(model, X_train, y_train, cv=kf,scoring='accuracy')
   
    return crv.mean()*100

accuracy(LogisticRegression())


# Se obtuvo la presicion del modelo de Logistic Regression, e incluimos ademas los modelos Random Forest y Support Vector Machine, para compararlos. Se uso ademas la matriz confusion, dado que entrega diversas herramientas para comparar la precision de cada modelo en base a la veracidad de los datos.

# In[24]:


clf = LogisticRegression()
clf.fit(X_train,y_train)
clf.score(X_test, y_test)*100

con = confusion_matrix(y_test, clf.predict(X_test))
print(classification_report(y_test, clf.predict(X_test)))




# In[25]:


accuracy(RandomForestClassifier())


# In[26]:


clf = RandomForestClassifier()
clf.fit(X_train,y_train)
clf.score(X_test, y_test)*100

con = confusion_matrix(y_test, clf.predict(X_test))
print(classification_report(y_test, clf.predict(X_test)))




# Para usar el modelo de SVM analizamos las features que sean mas determinantes 

# In[27]:


clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
C = clf.feature_importances_
for i in range(len((C))):
    print(df.columns[i], ': ', C[i]*100)


# In[28]:


X = np.array(df3[['Unnamed: 0','instrumentalness','valence','tempo']])
X = StandardScaler().fit_transform(X)
y = np.array(df3['class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
accuracy(SVC(kernel='linear', C=0.1))


# In[29]:


clf = SVC(kernel='linear', C=0.1)
clf.fit(X_train,y_train)
clf.score(X_test, y_test)*100

con = confusion_matrix(y_test, clf.predict(X_test))
print(classification_report(y_test, clf.predict(X_test)))


# Se elegio el modelo Logistic Regression, porque se adapto mejor al comportamiento de los datos. Se que hay muchas formas de comparar modelo, de hecho con la experiencia se puede saber la tendencia de los datos, y asi determinar el modelo a utilizar, sin embargo es fundamental una comparacion de la presicion de cada modelo, porque no todos los problemas son intuitivos.

# ### Pregunta 5
# 
# Dado el modelo predictivo elegido se usa para clasificar un nuevo data set. Para se utilizo el escalamiento que entreno los datos de  'df3' (La combinacion de los dos primeros dataset). Se ingreso el nuevo dataset como 'dft', y se ajustaron los datos del nuevo dataset con el escalamiento de entrenamiento. 

# In[30]:


X = np.array(df3.drop('class', axis=1))
y = np.array(df3['class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

dft = pd.read_csv('data_test.csv')
dft = dft.drop(columns='id_new')
dft = dft.drop(columns='time_signature')

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
new_data = np.array(dft)
new_data_scaled = scaler.transform(new_data)

def accuracy(model):
    kf = KFold(n_splits=5)
    crv = cross_val_score(model, X_train, y_train, cv=kf,scoring='accuracy')
   
    return crv.mean()*100


clf = LogisticRegression()
clf.fit(X_train,y_train)
clf.predict(new_data_scaled)


# Por utlimo se creo una columna 'new_class' que representa la clasificacion que es predecida al evaluar nuestro modelo. Esta columna se adjunto al nuevo dataset 'dft'

# In[31]:


dft['new_class'] = clf.predict(new_data_scaled)
dft.head()


# In[32]:


dft.describe()


# Los datos estadisticos determinan la media de la columna 'new_class', dando un 20%, o sea que 10 de 50 canciones son de reggaeton.

# ### Conclusiones
# 
# Se realizo un analisis completo de una combinacion de dos dataset, estos dataset clasificaban si una cancion es o no de reggaeton. La combinacion de los dataset se utilizo para diferenciar los dataset y crear un modelo en base a entrenamiento de datos, la precision de los distintos modelos fue buena, alrededor de un 96%. Para determinar que modelo elegir se compararon de diferentes formas su precision encontrando un mejor rendimiento con el modelo de Logistic Regression. Se guardo con una variable el escalamiento usado para el entrenamiento de datos, con ese escalamiento se clasifico  las canciones del nuevo dataset, encontrando ¿ que 10 de 50 canciones son de reggaeton.
# 
# Realice este analisis y explique paso por paso las herramientas utilizadas de forma tecnica, tengo claro que uando trabaje en su empresa tendre que resumir y ajustar el lenguaje para un entendimiento simple y claro para la mejor toma de desicion de la gerencia. 

# ### Referencias
# 
# 1. https://archive.ics.uci.edu/ml/datasets/pima+indians+diabetes
# 2. https://www.ncbi.nlm.nihgov/pmc/articles/PMC2245318/pdf/procascamc00018-0276.pdf
# 3. http://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-f-measure-metrics
# 4. https://www.analyticsvidhya.com/blog/2017/09/common-machine-learning-algorithms/
# 5. https://www.analyticsvidhya.com/blog/2014/10/support-vector-machine-simplified/
# 6. https://www.analyticsvidhya.com/blog/2017/09/understaing-support-vector-machine-example-code/
# 7. https://relopezbriega.github.io/blog/2015/10/10/machine-learning-con-python/
