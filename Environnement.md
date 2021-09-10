# Python 3.6
https://www.python.org/downloads/release/python-368/

Installez python à partir du lien précédent.

# Environnement virtuel
- Ouvrez un invite de commande et placez-vous à la racine de votre répertoire de projet. (Un dossier vide, ex: "Projet ReCAPP")
- Exécutez les ligne de commande suivantes:

python -m pip install --user -U virtualenv  
python -m virtualenv ReCAPP_environnement

- Activez l'environnement avec la commande suivante:

.\ReCAPP_environnement\Scripts\activate

- Installez les dépendances que nous allons utiliser à l'aide de la commande suivante (l'opération peut prendre une minute):

python -m pip install -U jupyter matplotlib numpy pandas scipy scikit-learn imageio

- Ouvrez Jupyter avec la commande suivante:

jupyter notebook

À l'avenir, pour ouvrir Jupyter, vous devrez lancer la console, activer l'environnement et lancer Jupyter avec les deux commandes ci-haut.
