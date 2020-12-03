# Deep learning : age guessing

_Par Anatole Zhâ, Nils Lapôtre & Marion Hurteau_

## Prérequis

- PyTorch
- Numpy
- Pandas
- Skimage
- MathPlotLib

Le dataset FaceAgesDataset, qui contient 23 000 photos nommées avec l'age de la personne et d'autres informations.

## Lancement
`python3 ./project.py`

## Doc

Le fichier `project.py` est la base du projet. Il va construire le CNN, utiliser la classe FaceAgesDataset (dans `FaceAgesDataset.py`) pour construire le dataset, puis exécuter le CNN et enregistrer le réseau dans le fichier `project.dat`.

Le fichier `images-test/createCSV.py` est appelé par le programme principal et crée le csv nécessaire pour alimenter le CNN.

Le modèle peut être testé avec le script `testCNN.py`. (Pas encore totalement fonctionnel)