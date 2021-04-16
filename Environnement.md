# Python 3+
py --version

https://www.python.org/downloads/

# PIP
py -m pip install --upgrade pip

# Tensorflow
py -m pip install --user --upgrade tensorflow

py -m pip show tensorflow

# Keras
py -m pip install keras

py -m pip install keras-applications

py -m pip list | select-string 'Keras'

On devrait retrouver Keras, Keras-Applications et Keras-Preprocessing.

