# Projet 10 — Système de Recommandation Multimédia
## Tâche 1 : Données, Algorithmes Deep Learning et Stratégie de Collecte

**Module :** Deep Learning (M2 Data Science 5)  
**Étudiants :** Fedi Berrima & Nadia Akhsay  
**Année :** 2026–2027  

---

### 1. C'est quoi l'objectif du projet ?

Le but est de recommander des films à des utilisateurs en combinant **trois types d'informations différentes (multimodalité)** :

| Modalité | Donnée | Rôle dans le Deep Learning |
| :--- | :--- | :--- |
| **🖼️ Image** | L'Affiche du film (poster) | Capture l'ambiance visuelle, les couleurs, le style graphique et le casting visuel. |
| **📝 Texte** | Le Synopsis (résumé) | Comprend les thèmes de l'histoire, le scénario et l'univers du film. |
| **📊 Données numériques** | Métadonnées | Genres (action, horreur...), durée, popularité, année de sortie. |

> **Pourquoi du Deep Learning ? (Le problème du Cold Start)**  
> Les algorithmes classiques ne savent recommander que les films qui ont déjà beaucoup de notes. Si un nouveau film sort, il a 0 note (**démarrage à froid / Cold Start**). Grâce au Deep Learning multimodal, le modèle regarde l'affiche, lit le synopsis et regarde le genre : il peut donc recommander un film sans aucune note historique !

---

### 2. Les Sources de Données

On utilise trois sources pour construire notre jeu de données :

| Source | Ce qu'on récupère | Pourquoi on l'utilise |
| :--- | :--- | :--- |
| **MovieLens** (`latest-small`) | 100 000 notes d'utilisateurs, 9 700 films, fichier `links.csv` | Notre référence pour savoir qui aime quoi. Mais MovieLens n'a ni affiches ni texte. |
| **TMDB** (API The Movie Database) | Affiches officielles (images JPG), synopsis (texte), durée, popularité | On interroge son API pour enrichir chaque film MovieLens avec son affiche et son résumé. |
| **Kaggle** (The Movies Dataset) | Métadonnées de 45 000 films | Données de secours si un film n'est pas trouvé sur l'API TMDB. |

---

### 3. L'Architecture Deep Learning (Comment ça marche ?)

Chaque modalité passe dans son propre réseau de neurones, puis on fusionne le tout :

* **Branche Image (CNN) :** On prend un réseau **ResNet-50** pré-entraîné. L'image de l'affiche est redimensionnée en $224 \times 224$ pixels. Le réseau extrait l'ambiance visuelle sous forme d'un vecteur de 128 valeurs.
* **Branche Texte (NLP) :** On utilise un transformeur léger **Sentence-BERT** (MiniLM). Il transforme le synopsis en un vecteur de 128 valeurs qui représente le sens de l'histoire.
* **Branche Métadonnées :** Un petit réseau dense (MLP) qui traite les genres, la durée et la popularité (64 valeurs).
* **Couche de Fusion :** On colle les 3 vecteurs ensemble ($128 + 128 + 64 = 320$ valeurs) puis on passe dans des couches denses pour obtenir le vecteur final du film.
* **Score final :** On compare ce vecteur avec les préférences de l'utilisateur pour prédire sa note de 0.5 à 5 étoiles.

---

### 4. Gestion du Cold Start et Évaluation

* **💡 Comment on teste le Cold Start (démarrage à froid) ?**  
  On met de côté **5 % des films** qu'on cache complètement au modèle pendant l'entraînement. Pendant le test, le modèle est forcé de recommander ces films **uniquement grâce à leur affiche, leur synopsis et leurs genres**.
* **Découpage des données :**  
  * **80 %** pour l'entraînement (notes les plus anciennes).
  * **10 %** pour la validation.
  * **10 %** pour le test final (notes les plus récentes).
* **Métriques pour évaluer le modèle :**  
  * **RMSE** : mesure l'erreur sur la note (ex: si le film vaut 4 et qu'on prédit 4.2).
  * **NDCG@K et Recall@K** : vérifie si les films qu'on recommande dans le Top 5 ou Top 10 sont vraiment ceux que l'utilisateur aime.
* **Modèles de comparaison (Baselines) :**  
  On compare notre réseau multimodal avec :
  1. La popularité simple (recommander les films les plus vus).
  2. La factorisation de matrice classique (SVD).
