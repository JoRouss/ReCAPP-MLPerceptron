# ReCAPP-MLPerceptron
Activité pédagogique d'apprentissage machine sur une partie des données du projet ReCAPP.  

Outils utilisés:  
- Python: le langage et l’interpréteur  
- virtualenv: librairie python pour isoler les dépendances des autres projets sur le même ordinateur  
- Jupyter: outil de prototype, permet d’exécuter des blocs de Python avec interface web  
- numpy, pandas: librairies de manipulation de données, imaginez Excel mais en code  
- scikit-learn: librairie d’apprentissage machine, simple mais CPU seulement  

# Crédits
Cette activitée est une adaptation d'un chapitre d'une formation dispensée par Olivier Leclerc (https://github.com/oleclerc)  
Les images ont été filmées par Margaret Kraenzel, enseignante en biologie au Cégep de Matane, et David Pelletier, enseignant en biologie au Cégep de Rimouski, dans les bassins du musée Exploramer à Sainte-Anne-des-Monts.

# Environnement
 Suivez les instructions de [Environnement.md](Environnement.md)  pour créer un environnement virtuel pour le projet.

# Structure de projet
- Téléchargez la structure de dossiers "ReCAPP_structure".  
- Placez-la au même niveau que l'environnement que vous venez de créé.

![Structure](/assets/Picture7.png "Structure")

# Données
- Téléchargez le dossier "Data". 
Dans "ReCAPP_structure\data\processed", placez les dossiers "0" et "1" que contient le dossier "Data".  

![Data](/assets/Picture8.png "Data")

Les dossiers contiennent des images traitées et découpées de lançons (0) et d'éperlans (1).

Les images sont tirées des annotations (les cadres) faites sur les images du projet ReCAPP provenant de Exploramer.  

![Annotations](/assets/Picture1.png "Annotations")

Cette laborieuse étape d'annotation a été faite au préalable et est omise dans ce projet pour sauver du temps. Le découpage a lui aussi déjà été fait pour éviter plusieurs gigabites de données.

# Création du notebook
Ouvrez Jupyter tel qu'expliqué précédemment.  

Dans Jupyter, sous "ReCAPP_structure\notebooks", cliquez sur "New", puis sur "Python 3".  
![New notebook](/assets/Picture2.png "New notebook")

Ouvrez le notebook nouvellement créé.

# Chargement des données
Il suffit de copier/coller les blocs qui vont suivre dans le notebook et de les exécuter.

```python
import numpy
from PIL import Image
from sklearn import datasets

image_path = "../data/processed"
# Charge la structure, mais pas les images. Nous allons devoir les redimensionner uniformément
recapp_dataset = datasets.load_files(image_path, load_content=False)

# Affichage de la structure
print(recapp_dataset.keys())

# Taille des images voulues
uniformized_image_size = 64

# On recrée la clé 'data'
recapp_dataset['data'] = []
for image_path in recapp_dataset['filenames']:
    with Image.open(image_path) as im: # Ouverture de l'image
        resized = im.resize((uniformized_image_size, uniformized_image_size)) # Redimension
        image_as_array = numpy.asarray(resized).reshape(-1) # Conversion en 1 seul vecteur de couleurs
        recapp_dataset['data'].append(image_as_array) # Ajout à la liste de 'data'

# On crée des raccourcis pour rendre le code plus lisible et facile à écrire
data = recapp_dataset['data']
labels = recapp_dataset['target']

print("done!")
```

# Multi-Layer Perceptron
Architechture de base où tous les neurones sont interconnectés.  
C'est la raison pour laquelle nous avons redimensionné nos images uniformément.

Il en résulte une perte de précision.

```python
from sklearn.neural_network import MLPClassifier
import pandas as pd

nn = MLPClassifier(random_state=42, hidden_layer_sizes=(50,25), learning_rate_init=0.0000003, max_iter=12, verbose=2)

nn.fit(data, labels)

pd.DataFrame(nn.loss_curve_)[3:].plot()
```

# Matrice de confusion
Elle permet d'observer les résultats de notre entrainement.

```python
import sklearn.metrics as metrics

metrics.plot_confusion_matrix(nn, data, labels)
```

![Matrice de confusion](/assets/Picture5.png "Matrice de confusion")

Comme on peut le constater, notre IA a énormément de difficulté à identifier des éperlans (1).  
- Lorsque l'IA s'est fait montrer un lançon, elle a eu 5699 bonnes et 1795 mauvaises réponses.
- Lorsque l'IA s'est fait présenter un éperlan, elle a eu 272 bonnes et 845 mauvaises réponses.


Probable cause: On a 7494 lançons et 1117 éperlans. Si le réseau prédit toujours lançon, il aura raison dans 87% des cas.

Pour y remédier, on va sur-échantillonner nos images d'éperlans.

# Sur échantillonnage des éperlans
## Séparation par classe
On doit séparer nos données pour augmenter la quantité d'éperlans.

```python
data_lancons = []
data_eperlans = []
labels_lancons = []
labels_eperlans = []

for i in range(len(data)):
    current_label = labels[i]
    current_data = data[i]
    if current_label == 0:
        data_lancons.append(current_data)
        labels_lancons.append(current_label)
    elif current_label == 1:
        data_eperlans.append(current_data)
        labels_eperlans.append(current_label)
    else:
        print("Invalid class")
    
print("Lancons:", len(data_lancons), "Éperlans:", len(data_eperlans))
```

## Sklearn resample
La fonction "resample" de Sklearn nous permet d'automatiquement augmenter la quantitée d'éperlans pour obtenir un nombre équivalent à celui de lançons.

```python
from sklearn.utils import resample

(data_eperlans_upsampled, labels_eperlans_upsampled) = resample(data_eperlans, labels_eperlans, n_samples=len(data_lancons), random_state=42)

#Nos nouvelles données
data_balanced = data_lancons + data_eperlans_upsampled
labels_balanced = labels_lancons + labels_eperlans_upsampled

print("Images total:", len(data_balanced))
```

## Nouvel entrainement avec les nouvelles données
```python
from sklearn.neural_network import MLPClassifier
import sklearn.metrics as metrics
import pandas as pd

nn = MLPClassifier(random_state=42, hidden_layer_sizes=(50,25), learning_rate_init=0.0000003, max_iter=12, verbose=2)

#On utilise nos nouvelles listes
nn.fit(data_balanced, labels_balanced)

pd.DataFrame(nn.loss_curve_)[3:].plot()

metrics.plot_confusion_matrix(nn, data, labels)
```

![Resultats3](/assets/Picture6.png "Resultats3")

Le résultat est pire! Voyons comment on pourrait améliorer notre résultat.

# Améliorations
Voici quelques paramètres qui peuvent être modifiés qui impacteront les résultats obtenus.  

- Nombre d'itérations: Faites passer "max-iter" à 40.  

- Taille du lot: On peut jouer avec la taille du lot, mais notre mémoire graphique contraint sa taille maximale.  
Expérimentez en ajoutant batch_size=5000.  

- Taille des images: On peut augmenter la précision de nos images en changeant "uniformized_image_size".  

- Nombre de neurones: "hidden_layer_sizes".  

- Learning rate: Vitesse d'adaptation à l'erreur. Essayez avec learning_rate_init=0.00005.  

```python
from sklearn.neural_network import MLPClassifier
import sklearn.metrics as metrics
import pandas as pd

#Certains paramètres ont été modifiés
nn = MLPClassifier(random_state=42, hidden_layer_sizes=(50,25), learning_rate_init=0.00005, max_iter=40, verbose=2, batch_size=5000)

nn.fit(data_balanced, labels_balanced)

pd.DataFrame(nn.loss_curve_)[3:].plot()
metrics.plot_confusion_matrix(nn, data_balanced, labels_balanced)
```

![Resultats1](/assets/Picture3.png "Resultats1")

Beaucoup mieux! Cependant, on ne s'est pas gardé de données pour tester... Notre IA a déjà vu toutes nos photos.

# Validation
On veut se garder environ 20% de nos données pour pouvoir tester notre IA sur des images qu'il n'a jamais vu.

```python
from sklearn.model_selection import train_test_split

#Séparation automatique des données d'entrainement et de test
train_data, test_data, train_labels, test_labels = train_test_split(data_balanced, labels_balanced, random_state=42)
```

Nouveau code d'apprentissage:

```python
from sklearn.neural_network import MLPClassifier
import sklearn.metrics as metrics
import pandas as pd

nn = MLPClassifier(random_state=42, hidden_layer_sizes=(50,25), learning_rate_init=0.00005, max_iter=40, verbose=2, batch_size=5000)

#Entrainement fait sur les données d'entrainement
nn.fit(train_data, train_labels)

pd.DataFrame(nn.loss_curve_)[3:].plot()
#Résultats sur les données d'entrainement
metrics.plot_confusion_matrix(nn, train_data, train_labels)
#Résultats sur les données de test
metrics.plot_confusion_matrix(nn, test_data, test_labels)
```

Résultat final:

![Resultats2](/assets/Picture4.png "Resultats2")

On peut dire que notre IA performe bien autant sur les données d'entrainement que sur les données de test.

À vous d'essayer de faire mieux!