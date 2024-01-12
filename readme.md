#Entraînement d'un Modèle pour le Jeu Diamant

Entraînement d'un Modèle pour le Jeu Diamant
Ce document fournit une explication détaillée du code Python fourni pour l'entraînement d'un modèle d'apprentissage profond (Deep Q-Network) pour jouer au jeu Diamant. Le code utilise la bibliothèque PyTorch pour la mise en œuvre du modèle de réseau neuronal et de l'agent d'apprentissage par renforcement.

## Prérequis

Avant d'exécuter le code, assurez-vous d'avoir les bibliothèques suivantes installées :

- PyTorch
- Numpy
- Matplotlib
- pandas

## Architecture du Code

### Modèle de Jeu (DiamantGame)

La classe DiamantGame représente le jeu Diamant. Elle comprend des méthodes pour initialiser, réinitialiser et jouer des tours dans le jeu. Le jeu implémente un certain nombre de cartes avec des trésors, des reliques et des dangers. L'objectif est de collecter des diamants tout en évitant les dangers.

### Réseau Neuronal (DQN)

La classe DQN définit le modèle de réseau neuronal utilisé pour l'apprentissage profond. Le réseau est composé de trois couches linéaires avec des fonctions d'activation ReLU.

### Agent d'Apprentissage (DQNAgent)

La classe DQNAgent encapsule le modèle et l'optimiseur. Elle gère également la mémoire pour l'expérience de rejouer (experience replay) et implémente la sélection d'action basée sur la politique ε-greedy.

### Entraînement de l'Agent (train_agent)

La fonction train_agent initialise le jeu et l'agent, puis effectue l'entraînement de l'agent sur un nombre spécifié d'épisodes. Elle renvoie les récompenses moyennes par épisode et le facteur de risque moyen.

### Fonction de Démarrage (start)

La fonction start appelle train_agent, sauvegarde le modèle entraîné et génère des graphiques pour visualiser les performances de l'agent au fil du temps.

## Paramètres et Configuration

- state_dim: La dimension de l'état du jeu.
- action_dim: La dimension de l'action (décider de continuer ou quitter le jeu).
- epsilon_start, epsilon_end, epsilon_decay: Les paramètres pour l'exploration ε-greedy.
- BATCH_SIZE: La taille du lot pour l'apprentissage par mini-lots.
- episodes: Le nombre total d'épisodes d'entraînement.

## Exécution du Code

Pour exécuter le code, appelez simplement la fonction start() à la fin du script. Cela entraînera l'agent, sauvegardera le modèle, et affichera des graphiques illustrant les performances de l'agent au fil du temps.

N'oubliez pas d'ajuster les paramètres en fonction de vos besoins spécifiques, tels que le nombre d'épisodes d'entraînement. Vous pouvez également personnaliser le modèle du jeu (DiamantGame) et ajuster l'architecture du réseau neuronal (DQN) selon vos besoins.

## Intégration avec le Jeu Diamant

Dans cette section, nous avons étendu le code pour inclure une API FastAPI qui peut être exposée pour être intégrée au jeu Diamant. Cette API permet au jeu d'envoyer l'état actuel du jeu à un modèle pré-entraîné pour obtenir des prédictions sur la prochaine action à prendre.

###  FastAPI et Définition des Données (main.py)

Nous avons importé la bibliothèque FastAPI et créé une instance d'application (app). La classe State est définie pour valider les données envoyées à l'API, représentant l'état actuel du jeu avec le nombre de diamants collectés et les cartes déjà jouées.

### Chargement du Modèle

Le modèle pré-entraîné est chargé à l'aide de la fonction load_model() provenant du script d'entraînement.

```python
model = load_model()
model.eval()
```

### Endpoint FastAPI pour les Prédictions

Nous avons créé un endpoint FastAPI /predict/ qui accepte des données d'état du jeu via une requête POST. Les données d'état sont validées à l'aide de la classe State, puis converties en un tableau pour être utilisées avec le modèle.

Le modèle prédit l'action à prendre en fonction de l'état actuel du jeu, et la réponse est renvoyée sous la forme d'un objet JSON contenant l'action prédite.

API @app.post("/predict/") async def predict(state_model: State) -> Coroutine[Any, Any, dict[str, Any]]

### Exécution de l'API

```python
uvicorn API:app --reload