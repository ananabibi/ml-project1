import numpy as np

def compute_pca(X, num_components):
    """
    Applique l'Analyse en Composantes Principales (PCA) sur les données X.
    """
    # Centrer les données (on ramène a l'origine
    X_meaned = X - np.mean(X, axis=0)
    
    # matrice de covariance
    # rowvar=False -> colonnes sont les features
    cov_mat = np.cov(X_meaned, rowvar=False)
    
    # valeurs propres et vecteurs propres matric cov
    # ->quantité de variance expliquée par chaque axe
    eigenvalues, eigenvectors = np.linalg.eigh(cov_mat)
    
    # Trier les vecteurs propres par ordre décroissant des valeurs propres (we obtain PCA that way)
    sorted_index = np.argsort(eigenvalues)[::-1] #inverse l'ordre pour avoir les plus grandes valeurs en premier
    sorted_eigenvectors = eigenvectors[:, sorted_index]
    
    eigenvector_subset = sorted_eigenvectors[:, 0:num_components]
    
    #réduire la dimension
    X_reduced = np.dot(X_meaned, eigenvector_subset)
    
    return X_reduced, eigenvector_subset

def sigmoid(z):
    """
    Fonction d'activation sigmoïde qui transforme une valeur réelle en probabilité (entre 0 et 1).
    """
    # Formule mathématique standard de la sigmoïde
    return 1 / (1 + np.exp(-z))

def compute_loss(y, y_pred, weights):
    """
    Calcule l'erreur (Binary Cross-Entropy Loss) pondérée.
    """
    # Ajouter un tout petit nombre (epsilon) pour éviter de faire log(0) qui donne une erreur
    epsilon = 1e-15 
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # Formule de la log-loss avec intégration des poids pour pénaliser plus fort les erreurs sur la classe minoritaire
    loss = -np.mean(weights * (y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)))
    return loss

def logistic_regression_train(X, y, lr=0.01, epochs=1000, minority_weight=2.0):
    """
    Entraîne le modèle avec une descente de gradient et des poids pour déséquilibres de classes.
    """
    # Récupérer le nombre d'échantillons et de features
    n_samples, n_features = X.shape
    
    # Initialiser les poids (w) à zéro et le biais (b) à zéro
    w = np.zeros(n_features)
    b = 0

    # Créer un tableau de poids : si y=1 (patient malade), on applique le minority_weight, sinon 1.0
    sample_weights = np.where(y == 1, minority_weight, 1.0)

    # Boucle d'entraînement (descente de gradient)
    for epoch in range(epochs):
        # Calculer le modèle linéaire (w * x + b)
        linear_model = np.dot(X, w) + b
        
        # Transformer en probabilités avec la sigmoïde
        y_pred = sigmoid(linear_model)

        # Calculer l'erreur de prédiction
        error = y_pred - y
        
        # Multiplier l'erreur par nos poids pour forcer l'algorithme à se concentrer sur les cas malades
        weighted_error = error * sample_weights
        
        # Calculer les gradients (dérivées) pour w et b
        dw = (1 / n_samples) * np.dot(X.T, weighted_error)
        db = (1 / n_samples) * np.sum(weighted_error)

        # Mettre à jour les poids (w) et le biais (b) en soustrayant le gradient multiplié par le learning rate (lr)
        w -= lr * dw
        b -= lr * db

    return w, b

def logistic_regression_predict(X, w, b, threshold=0.5):
    """
    Prédit les classes (0 ou 1) pour de nouvelles données X.
    """
    # Appliquer le modèle avec les poids entraînés
    linear_model = np.dot(X, w) + b
    y_pred = sigmoid(linear_model)
    
    # Si la probabilité est supérieure au seuil (0.5), retourner 1 (malade), sinon 0 (sain)
    return (y_pred >= threshold).astype(int)