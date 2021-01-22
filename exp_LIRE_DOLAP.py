"""
    This is an attempt to produce a simple Numpy based version of the whole code
    This program implements
        - a new approach for RS explanation based on LIME. Contrary to LIME-RS
            - interpretable space is no more binary
            - we are closer to LIME philosophy since our perturbed entries in interpretable space do not
            directly coincidate with pre-existing points
            - as a consequence we propose a new out-of-sample prediction method to retrieve the prediction for all
            perturbed points as a quadratic error minimization problem
            - we change the definition of locality to better embrace potential decision boundaries
        - a new robustness measure for RS explanation
    What changes:
        - only a single approach implemented: LIRE with varying percentage of training points
    either from cluster or from perturbed points
        - no more torch representation, and no more use of Adam optimizer to determine out-of-sample predictions
"""
import time
import numpy as np
from scipy.spatial.distance import cosine as cosine_dist
import pandas as pd
import os.path
import pickle
import umap
from scipy.sparse.linalg import svds
import scipy
from tqdm import tqdm
from sklearn import linear_model
from sklearn.cluster import KMeans
import torch
import datetime
import random
from OOSPredictors import OOS_pred_slice
from models import LinearRecommender, train
from config import Config
from utility import read_sparse
from loss import LocalLossMAE_v3 as LocalLoss
import math
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_samples,silhouette_score
import hdbscan

"""
The main is right at the bottom
"""

def make_tensor(array):
    """
    Function that produces a torch tensor with an array as input
    """
    return torch.tensor(array, device=Config.device(), dtype=Config.precision())


def perturbations_gaussian(original_user, fake_users: int, std=0.47, proba=0.1):
    """
    Function that does the gaussian perturbation and therefore yield perturbated points that is supposedly close to the
    instance to explain
    """
    if(isinstance(original_user,scipy.sparse.csr.csr_matrix)):
        original_user = original_user.toarray()
    else:
        original_user = original_user.reshape(1, len(original_user))
    # Comes from a scipy sparse matrix
    nb_dim = original_user.shape[1]
    users = np.tile(original_user, (fake_users, 1))

    noise = np.random.normal(np.zeros(nb_dim), global_variance/2, (fake_users, nb_dim))
    rd_mask = np.random.binomial(1, proba, (fake_users, nb_dim))
    noise = noise * rd_mask * (users != 0.)
    users = users + noise
    return np.clip(users, 0., 5.)


def make_black_box_slice(U, sigma, Vt, means, indexes):
    """
    Generate black box predictions for multiple users at once with a matrice factorisation predictor.
    """
    return (U[indexes] @ sigma @ Vt) + np.tile(means[indexes].reshape(len(indexes), 1), (1, Vt.shape[1]))

def make_black_white_box_slice(weights,X):
    """
    Generate black box prediction with a simple white box : a weights dot product.
    """

    return np.dot(X,weights)

def make_double_black_white_box_slice(weights1,weights2,X,dist,threshold):
    """
    Generate black box prediction with double white boxes predictors. If the condition is met the first predictor
    is used; Otherwise it's the second one.
    """

    return np.where(dist < threshold, np.dot(X, weights1), np.dot(X, weights2))


def explain(user_id:int, item_id:int, n_coeff:int, sigma, Vt, user_means, all_user_ratings, cluster_labels, train_set_size:int, pert_ratio:float=0.5, mode:str= "lars",movies=None,iid_map=None):
    """

    :param user_id: user for which an explanation is expected
    :param item_id: item for which an explanation is expected
    :param n_coeff: number of coefficients for the explanation
    :param sigma: intensity of latent factors
    :param Vt: item latent space
    :param all_user_ratings: matrix containing all predicted user ratings
    :param cluster_labels: vector indicating for each user its cluster label
    :param train_set_size: size of the train set (perturbation + neighbors from clusters) to train local surrogate
    :param pert_ratio: perturbation ratio
    :return: a vector representing the explanation
    """

    if(isinstance(all_user_ratings,scipy.sparse.csr.csr_matrix)):
        all_user_ratings = all_user_ratings.toarray()
    else:
        all_user_ratings = np.nan_to_num(all_user_ratings)

    # 1. Generate a train set for local surrogate model
    X_train = np.zeros((train_set_size, Vt.shape[1]))   # 2D array
    y_train = np.zeros(train_set_size)                  # 1D array

    pert_nb = int(train_set_size * pert_ratio)      # nb of perturbed entries
    cluster_nb = train_set_size - pert_nb           # nb of real neighbors
    if pert_nb > 0:                                 # generate perturbed training set part
        # generate perturbed users
        X_train[0:pert_nb, :] = perturbations_gaussian(all_user_ratings[user_id], pert_nb)
        X_train[0:pert_nb, item_id] = 0
        #Make the predictions for those
        #for k in range(pert_nb):
        #    y_train[k] = OOS_pred_smart(torch.tensor(X_train[k], device=device, dtype=torch_precision), sigma_t, Vt_t, U[user_id])[item_id].item()
        what_is_this_bug = OOS_pred_slice(make_tensor(X_train[range(pert_nb)]), sigma_t, Vt_t,item_id).cpu().numpy()[:, item_id]

        y_train[range(pert_nb)] = what_is_this_bug

    if cluster_nb > 0:
        # generate neighbors training set part
        cluster_index = cluster_labels[user_id]
        # retrieve the cluster index of user "user_id"
        neighbors_index = np.where(cluster_labels == cluster_index)[0]
        print("nombre d'element dans cluster : ",len(neighbors_index))
        neighbors_index = neighbors_index[neighbors_index != user_id]
        if(len(neighbors_index)>1):
            neighbors_index = np.random.choice(neighbors_index, cluster_nb)
            X_train[pert_nb:train_set_size, :] = all_user_ratings[neighbors_index, :]
            X_train[pert_nb:train_set_size, item_id] = 0

            predictor_slice = make_black_box_slice(U, sigma, Vt, user_means, neighbors_index)
            y_train[pert_nb:train_set_size] = predictor_slice[:, item_id]

    X_user_id = all_user_ratings[user_id].copy()
    X_user_id[item_id] = 0

    # Check the real prediction
    y_predictor_slice = make_black_box_slice(U, sigma, Vt, user_means, np.array([user_id]))
    y_predictor_slice = y_predictor_slice.transpose()[item_id]

    # 2. Now run a LARS linear regression model on the train set to generate the most parcimonious explanation
    if mode == "lars":
        #todo : A voir si déterministe, potentiellement re-optimiser
        reg = linear_model.Lars(fit_intercept=False, n_nonzero_coefs=n_coeff,eps=90)
        reg.fit(X_train, y_train)
        coef = reg.coef_
        # Predict the value with the surrogate
        pred = reg.predict(X_user_id.reshape(1, -1))
    # Or a classic lime style regression
    elif mode == "lime":
        models_ = []
        errors_ = []
        # A few runs to avoid bad starts
        for _ in range(1):
            model = LinearRecommender(X_train.shape[1])
            local_loss = LocalLoss(make_tensor(all_user_ratings[user_id]), sigma=1, alpha=0.01)
            train(model, make_tensor(X_train), make_tensor(y_train), local_loss, 1000, verbose=True)
            pred = model(make_tensor(X_user_id)).item()
            models_.append(model)
            errors_.append(abs(pred - y_predictor_slice)[0])
            models_.append(model)
            if abs(pred - y_predictor_slice)[0] < 0.1:  # Good enough
                break
        best = models_[np.argmin(errors_)]
        coef = best.omega.detach().cpu().numpy()
        pred = best(make_tensor(X_user_id)).item()
    else:
        raise NotImplementedError("No mode " + mode + " exists !")

    #Attention création des listes pour les clés et les valeurs du dico iid_map à chaque appel d'explain => pas opti
    movielens_ids = list(iid_map.keys())
    matrice_ids = list(iid_map.values())

    explication_matrice_ids = np.argsort(coef)[-10:][::-1]
    explication_movielens_ids = [movielens_ids[matrice_ids.index(matrice_id)] for matrice_id in explication_matrice_ids]
    item_id_to_explained_movielens = movielens_ids[matrice_ids.index(item_id)]
    coefs = coef[explication_matrice_ids]
    explication_movies = [(movies.loc[movies['movieId'] == id_movie  ]['genres'].values.tolist()[0],coef) for id_movie,coef in zip(explication_movielens_ids,coefs)]
    movie_to_explain = movies.loc[movies['movieId'] == item_id_to_explained_movielens]['genres'].values.tolist()


    print("Local prediction : ",pred)
    print("Black-box prediction :", y_predictor_slice[0])
    print("explication : ",explication_matrice_ids)
    print("coeff value of explication :",coef[explication_matrice_ids])
    f = open("temp/result_premier_exp_premier_draft.csv", "a")
    line = str(pred)+';'+str(y_predictor_slice[0])+';'+str(explication_matrice_ids)+';'+str(coef[explication_matrice_ids])+';'+str(abs(pred - y_predictor_slice)[0])+';'+ str(explication_movies)+';'+str(movie_to_explain)
    line = line.replace("\n","")
    f.write(line)
    f.close()

    return coef, abs(pred - y_predictor_slice)[0]


def explain_white_black(user_id:int, item_id:int, n_coeff:int, sigma, Vt, user_means, all_user_ratings, cluster_labels, train_set_size:int, pert_ratio:float=0.5, mode:str= "lars",movies=None,iid_map=None,weights_white_black=None):
    """

    The explain function used in the single white box experiment; It differs from the vanilla "explain" because the
    blackbox predictions aren't generated the same way. It is not a matrice factorization but a simple dot product
    between random weights and the user representation (user row in the matrice).
    :param user_id: the user id for who we want an explanation
    :param item_id: the item id to explain
    :param n_coeff: the number of relevant coefs
    :param sigma: intensity of latent factors
    :param Vt: item latent space
    :param user_means: the ratings average per users
    :param all_user_ratings: the rating matrix
    :param cluster_labels: the cluster labels
    :param train_set_size: the size of the train set in other word the number of points used
    for the whitebox training
    :param pert_ratio: the number of point within the train set that are generated with the gaussian noise; this ratio
    directly implies the number
    :param mode: either 'lars' or 'lime'
    :param movies: the movie dataframe containing the movielens100k dataset
    :param iid_map: a dictionnary that permits to switch between the item id in the matrix
    and the item id in the movielens dataset.
    :param weights_white_black: the weights representing whitebox
    :return:
    """

    if(isinstance(all_user_ratings,scipy.sparse.csr.csr_matrix)):
        all_user_ratings = all_user_ratings.toarray()
    else:
        all_user_ratings = np.nan_to_num(all_user_ratings)

    # 1. Generate a train set for local surrogate model
    X_train = np.zeros((train_set_size, Vt.shape[1]))   # 2D array
    y_train = np.zeros(train_set_size)                  # 1D array

    pert_nb = int(train_set_size * pert_ratio)      # nb of perturbed entries
    cluster_nb = train_set_size - pert_nb           # nb of real neighbors
    if pert_nb > 0:                                 # generate perturbed training set part
        # generate perturbed users
        X_train[0:pert_nb, :] = perturbations_gaussian(all_user_ratings[user_id], pert_nb)
        X_train[0:pert_nb, item_id] = 0
        #Make the predictions for those
        #for k in range(pert_nb):
        #    y_train[k] = OOS_pred_smart(torch.tensor(X_train[k], device=device, dtype=torch_precision), sigma_t, Vt_t, U[user_id])[item_id].item()
        #y_train[range(pert_nb)] = OOS_pred_slice(make_tensor(X_train[range(pert_nb)]), sigma_t, Vt_t).cpu().numpy()[:, item_id]
        #y_train[range(pert_nb)] = make_black_white_box_slice(weights_white_black,X_train)

    if cluster_nb > 0:
        # generate neighbors training set part
        cluster_index = cluster_labels[user_id]
        # retrieve the cluster index of user "user_id"
        neighbors_index = np.where(cluster_labels == cluster_index)[0]
        neighbors_index = neighbors_index[neighbors_index != user_id]
        neighbors_index = np.random.choice(neighbors_index, cluster_nb)
        X_train[pert_nb:train_set_size, :] = all_user_ratings[neighbors_index, :]
        X_train[pert_nb:train_set_size, item_id] = 0

        #predictor_slice = make_black_box_slice(U, sigma, Vt, user_means, neighbors_index)
    y_train = make_black_white_box_slice(weights_white_black,X_train)

    X_user_id = all_user_ratings[user_id].copy()
    X_user_id[item_id] = 0

    # Check the real prediction
    y_predictor_slice = make_black_box_slice(U, sigma, Vt, user_means, np.array([user_id]))
    y_predictor_slice = y_predictor_slice.transpose()[item_id]

    # 2. Now run a LARS linear regression model on the train set to generate the most parcimonious explanation
    if mode == "lars":
        #todo : A voir si déterministe, potentiellement re-optimiser
        reg = linear_model.Lars(fit_intercept=False, n_nonzero_coefs=n_coeff)
        reg.fit(X_train, y_train)
        coef = reg.coef_
        # Predict the value with the surrogate
        pred = reg.predict(X_user_id.reshape(1, -1))
    # Or a classic lime style regression
    elif mode == "lime":
        models_ = []
        errors_ = []
        # A few runs to avoid bad starts
        for _ in range(1):
            model = LinearRecommender(X_train.shape[1])
            local_loss = LocalLoss(make_tensor(all_user_ratings[user_id]), sigma=1, alpha=0.01)
            train(model, make_tensor(X_train), make_tensor(y_train), local_loss, 1000, verbose=True)
            pred = model(make_tensor(X_user_id)).item()
            models_.append(model)
            errors_.append(abs(pred - y_predictor_slice)[0])
            models_.append(model)
            if abs(pred - y_predictor_slice)[0] < 0.1:  # Good enough
                break
        best = models_[np.argmin(errors_)]
        coef = best.omega.detach().cpu().numpy()
        pred = best(make_tensor(X_user_id)).item()
    else:
        raise NotImplementedError("No mode " + mode + " exists !")

    #Attention création des listes pour les clés et les valeurs du dico iid_map à chaque appel d'explain => pas opti
    movielens_ids = list(iid_map.keys())
    matrice_ids = list(iid_map.values())

    explication_matrice_ids = np.argsort(coef)[-10:][::-1]
    explication_movielens_ids = [movielens_ids[matrice_ids.index(matrice_id)] for matrice_id in explication_matrice_ids]
    item_id_to_explained_movielens = movielens_ids[matrice_ids.index(item_id)]
    coefs = coef[explication_matrice_ids]

    explication_movies = [(movies.loc[movies['movieId'] == id_movie  ]['genres'].values.tolist()[0],coef) for id_movie,coef in zip(explication_movielens_ids,coefs)]
    movie_to_explain = movies.loc[movies['movieId'] == item_id_to_explained_movielens]['genres'].values.tolist()


    print("Local prediction : ",pred)
    print("Black-box prediction :", y_predictor_slice[0])
    print("explication : ",explication_matrice_ids)
    print("coeff value of explication :",coef[explication_matrice_ids])

    return coef, abs(pred - y_predictor_slice)[0],explication_matrice_ids,coefs


def explain_double_white_black(user_id:int, item_id:int, n_coeff:int, sigma, Vt, user_means, all_user_ratings, cluster_labels, train_set_size:int, pert_ratio:float=0.5, mode:str= "lars",movies=None,iid_map=None,weights_white_black1=None,weights_white_black2=None):
    """
    The explain function used in the double white box experiment; It differs from the vanilla "explain" because the
    blackbox predictions aren't generated the same way. It is not a matrice factorization but a conditional dot product
    between random weights and the user representation (user row in the matrice). Based on the condition, it's either
    the weights from the first whitebox that are used or the second ones.
    :param user_id: the user id for who we want an explanation
    :param item_id: the item id to explain
    :param n_coeff: the number of relevant coefs
    :param sigma: intensity of latent factors
    :param Vt: item latent space
    :param user_means: the ratings average per users
    :param all_user_ratings: the rating matrix
    :param cluster_labels: the cluster labels
    :param train_set_size: the size of the train set in other word the number of points used
    for the whitebox training
    :param pert_ratio: the number of point within the train set that are generated with the gaussian noise; this ratio
    directly implies the number
    :param mode: either 'lars' or 'lime'
    :param movies: the movie dataframe containing the movielens100k dataset
    :param iid_map: a dictionnary that permits to switch between the item id in the matrix
    and the item id in the movielens dataset.
    :param weights_white_black1: the weights representing the first whitebox
    :param weights_white_black2: the weights representing the second whitebox
    :return: an explanation : a list of items coeffs; The more relevant ones. They are the feature used in the whitebox
    """
    k_nearest = 10
    if(isinstance(all_user_ratings,scipy.sparse.csr.csr_matrix)):
        all_user_ratings = all_user_ratings.toarray()
    else:
        all_user_ratings = np.nan_to_num(all_user_ratings)

    # 1. Generate a train set for local surrogate model
    X_train = np.zeros((train_set_size, Vt.shape[1]))   # 2D array
    y_train = np.zeros(train_set_size)                  # 1D array
    X_user_id = all_user_ratings[user_id].copy()
    pert_nb = int(train_set_size * pert_ratio)      # nb of perturbed entries
    cluster_nb = train_set_size - pert_nb           # nb of real neighbors
    if pert_nb > 0:                                 # generate perturbed training set part
        # generate perturbed users
        X_train[0:pert_nb, :] = perturbations_gaussian(all_user_ratings[user_id], pert_nb)
        X_train[0:pert_nb, item_id] = 0
        #Make the predictions for those
        #for k in range(pert_nb):
        #    y_train[k] = OOS_pred_smart(torch.tensor(X_train[k], device=device, dtype=torch_precision), sigma_t, Vt_t, U[user_id])[item_id].item()
        #y_train[range(pert_nb)] = OOS_pred_slice(make_tensor(X_train[range(pert_nb)]), sigma_t, Vt_t).cpu().numpy()[:, item_id]
        #y_train[range(pert_nb)] = make_black_white_box_slice(weights_white_black,X_train)


    cluster_index = cluster_labels[user_id]
    # retrieve the cluster index of user "user_id"
    neighbors_index = np.where(cluster_labels == cluster_index)[0]
    neighbors_index = neighbors_index[neighbors_index != user_id]
    k_nearest = min(len(neighbors_index) - 1, k_nearest - 1)
    neighbors_representation = all_actual_ratings[neighbors_index, :].toarray()
    if(len(neighbors_index > 0)):
        dist = pairwise_distances(np.asarray([X_user_id]), neighbors_representation)
        sortedDist = sorted(dist[0], reverse=False)
        threshold = sortedDist[k_nearest]
    else:
        threshold = 0

    if cluster_nb > 0:
        # generate neighbors training set part

        neighbors_index = np.random.choice(neighbors_index, cluster_nb)

        X_train[pert_nb:train_set_size, :] = all_user_ratings[neighbors_index, :]
        X_train[pert_nb:train_set_size, item_id] = 0

        #predictor_slice = make_black_box_slice(U, sigma, Vt, user_means, neighbors_index)
    dist = pairwise_distances(np.asarray([X_user_id]), X_train)
    y_train = make_double_black_white_box_slice(weights_white_black1,weights_white_black2,X_train,dist,threshold)[0]

    #X_user_id = all_user_ratings[user_id].copy()
    X_user_id[item_id] = 0

    # Check the real prediction
    y_predictor_slice = make_black_box_slice(U, sigma, Vt, user_means, np.array([user_id]))
    y_predictor_slice = y_predictor_slice.transpose()[item_id]

    # 2. Now run a LARS linear regression model on the train set to generate the most parcimonious explanation
    if mode == "lars":
        #todo : A voir si déterministe, potentiellement re-optimiser
        reg = linear_model.Lars(fit_intercept=False, n_nonzero_coefs=n_coeff)
        reg.fit(X_train, y_train)
        coef = reg.coef_
        # Predict the value with the surrogate
        pred = reg.predict(X_user_id.reshape(1, -1))
    # Or a classic lime style regression
    elif mode == "lime":
        models_ = []
        errors_ = []
        # A few runs to avoid bad starts
        for _ in range(1):
            model = LinearRecommender(X_train.shape[1])
            local_loss = LocalLoss(make_tensor(all_user_ratings[user_id]), sigma=1, alpha=0.01)
            train(model, make_tensor(X_train), make_tensor(y_train), local_loss, 1000, verbose=True)
            pred = model(make_tensor(X_user_id)).item()
            models_.append(model)
            errors_.append(abs(pred - y_predictor_slice)[0])
            models_.append(model)
            if abs(pred - y_predictor_slice)[0] < 0.1:  # Good enough
                break
        best = models_[np.argmin(errors_)]
        coef = best.omega.detach().cpu().numpy()
        pred = best(make_tensor(X_user_id)).item()
    else:
        raise NotImplementedError("No mode " + mode + " exists !")

    #Attention création des listes pour les clés et les valeurs du dico iid_map à chaque appel d'explain => pas opti
    movielens_ids = list(iid_map.keys())
    matrice_ids = list(iid_map.values())

    explication_matrice_ids = np.argsort(coef)[-10:][::-1]
    explication_movielens_ids = [movielens_ids[matrice_ids.index(matrice_id)] for matrice_id in explication_matrice_ids]
    item_id_to_explained_movielens = movielens_ids[matrice_ids.index(item_id)]
    coefs = coef[explication_matrice_ids]

    explication_movies = [(movies.loc[movies['movieId'] == id_movie  ]['genres'].values.tolist()[0],coef) for id_movie,coef in zip(explication_movielens_ids,coefs)]
    movie_to_explain = movies.loc[movies['movieId'] == item_id_to_explained_movielens]['genres'].values.tolist()


    print("Local prediction : ",pred)
    print("Black-box prediction :", y_predictor_slice[0])
    print("explication : ",explication_matrice_ids)
    print("coeff value of explication :",coef[explication_matrice_ids])

    return coef, abs(pred - y_predictor_slice)[0],explication_matrice_ids,coefs



def DCG2(labels_ordered,K):
    '''
    Allows to calculate the Discounted Cumulative Gain
    :param Labels_ordered: The items labels (gains, or ground truth) sorted like the predicted ranking
    :param K: The number of item to take into account within the ranking
    :return: The Discounted Cumulative Gain
    '''
    DCG = 0
    for i in range(1,len(labels_ordered)+1):
        if(i < K):
            y = labels_ordered[i-1]
            DCG = DCG + ( y / (math.log2(i + 1)))

    return DCG



def experiment_test_top_recommendation(U, sigma, Vt, user_means, labels, all_actual_ratings,iid_map):
    """
    This is the top/flop/random experiment that is described in the dolap paper 2021. There are matrice_idd_top
    , matrice_idd_flop and matrice_idd variables. Use matrice_idd_flop in the "explain" function to yield result for the
    flop configuration; matrice_idd_top for the top configuration and matrice_idd for the random configuration.
    :param U: users latent space
    :param sigma: intensity of latent factors
    :param Vt: items latent space
    :param user_means: the ratings average per users
    :param labels: the cluster ids per users
    :param all_actual_ratings: the rating matrix
    :param iid_map: a dictionnary that permits to switch between the item id in the matrix
    and the item id in the movielens dataset.
    :return: a result file in /temp/ with the blackbox prediction, the white box prediction, the mae and
    other contextual informations..
    """
    #reset le fichier en mettant le header
    f = open("temp/result_premier_exp_premier_draft.csv", "w")
    f.write('local_prediction;blackbox_prediction;explication_matrice_ids;coef_explication;mae_local_blackbox;movies_explication;movie_to_explain;user_mean;user_std;user_min;user_max;number_of_rated_item_by_user;top20film_genres_user;elapsed_time;pert_ratio\n')
    f.close()

    n_coeff = 10
    train_set_size = 1000
    pert_ratio = 1

    toBeExplained = [(int(x[0]), int(x[1]),x[2]) for x in
                     pd.read_csv('ml-latest-small/ratings.csv', sep=",").values.tolist()]

    toBeExplained = random.sample(toBeExplained, 50)
    movies = pd.read_csv('ml-latest-small/movies.csv', sep=",",header=0)
    #Les clé d'iid map sont les ids de movielens et les valeurs sont les ids de la matrice.


    #random.shuffle(toBeExplained)
    for couple_uid_iid in toBeExplained:
        movielens_uid = couple_uid_iid[0]
        #movielens_iid = couple_uid_iid[1]
        matrice_uid = movielens_uid - 1
        matrice_iid = iid_map[couple_uid_iid[1]]

        black_box_predictions_for_user = make_black_box_slice(U, sigma, Vt, user_means, [matrice_uid])[0]
        user_representation = all_actual_ratings[matrice_uid,:].toarray()[0]
        #Si plusieurs tops ou flop alors on prend le 1er qui vient : d'ou le [0][0] en fin d'instruction
        matrice_iid_top = np.where(black_box_predictions_for_user == np.max(black_box_predictions_for_user))[0][0]
        matrice_iid_flop = np.where(black_box_predictions_for_user == np.min(black_box_predictions_for_user))[0][0]

        #TODO:PAS OPTI DU TOUT SE TROUVE EN DOUBLON DANS EXPLAIN EN PLUS
        movielens_ids = list(iid_map.keys())
        matrice_ids = list(iid_map.values())
        top20_user_matrice_ids = np.argsort(user_representation)[-20:][::-1]
        top20_user_movielens_ids = [movielens_ids[matrice_ids.index(matrice_id)] for matrice_id in
                                     top20_user_matrice_ids]

        top20_user_genres = [movies.loc[movies['movieId'] == id_movie]['genres'].values.tolist()[0] for
                              id_movie in top20_user_movielens_ids]

        number_of_rated_item_by_user = np.count_nonzero(user_representation)
        user_non_zero = user_representation[np.nonzero(user_representation)]
        user_mean = user_means[matrice_uid]




        user_std = np.std(user_non_zero)
        user_min = np.min(user_non_zero)
        user_max = np.max(user_non_zero)
        start_time = time.time()
        base_exp, mae = explain(matrice_uid, matrice_iid, n_coeff, sigma, Vt, user_means, all_actual_ratings, labels, train_set_size, pert_ratio,mode='lars',movies=movies,iid_map=iid_map)
        elapsed_time = (time.time() - start_time)
        print("--- %s seconds ---" % elapsed_time)

        f = open("temp/result_premier_exp_premier_draft.csv", "a")
        line = ';'+str(user_mean)+';'+str(user_std)+';'+str(user_min)+';'+str(user_max)+';'+str(number_of_rated_item_by_user)+';'+str(top20_user_genres)+';'+str(elapsed_time)+';'+str(pert_ratio)
        line = line.replace("\n", "")
        line = line+'\n'
        f.write(line)
        f.close()
        print("Mae = ", mae)
        print("True rating = ", couple_uid_iid[2])
        print("--------------------------")


def experiment_white_black_box(U, sigma, Vt, user_means, labels, all_actual_ratings,iid_map):
    '''
    This is the single white box experiments described in the Dolap paper 2021
    :param U: users latent space
    :param sigma: intensity of latent factors
    :param Vt: items latent space
    :param user_means: the ratings average per users
    :param labels: the cluster ids per users
    :param all_actual_ratings: the rating matrix
    :param iid_map: a dictionnary that permits to switch between the item id in the matrix
    and the item id in the movielens dataset.
    :return: a file with the results : precision/recalls, ndcg@3,@5 etc..
    '''

    f = open("temp/deuxieme_exp_result.csv", "w")
    f.write('precision_recall_exp;precision_recall_baseline;exp_ndcg_at_3;baseline_ndcg_at_3;exp_ndcg_at_5;baseline_ndcg_at_5;baseline_ndcg_at_10;exp_ndcg_at_10;number_item_rated_by_user\n')
    f.close()
    n_coeff = 10
    train_set_size = 1000
    pert_ratio = 1

    toBeExplained = [(int(x[0]), int(x[1]),x[2]) for x in
                     pd.read_csv('ml-latest-small/ratings.csv', sep=",").values.tolist()]


    toBeExplained = random.sample(toBeExplained,50)
    movies = pd.read_csv('ml-latest-small/movies.csv', sep=",",header=0)
    #Les clé d'iid map sont les ids de movielens et les valeurs sont les ids de la matrice.


    #random.shuffle(toBeExplained)
    for couple_uid_iid in toBeExplained:



        movielens_uid = couple_uid_iid[0]
        #movielens_iid = couple_uid_iid[1]
        matrice_uid = movielens_uid - 1
        matrice_iid = iid_map[couple_uid_iid[1]]

        black_box_predictions_for_user = make_black_box_slice(U, sigma, Vt, user_means, [matrice_uid])[0]
        user_representation = all_actual_ratings[matrice_uid,:].toarray()[0]
        item_matrices_ids_rated_by_user = np.nonzero(user_representation)[0]

        weights_black_white = [0] * 9724
        non_zero_index = list(np.random.choice(item_matrices_ids_rated_by_user,10,replace=False))
        weights_alone_ordered = []
        random_baseline_indexs = list(np.random.choice(item_matrices_ids_rated_by_user,10,replace=False))
        random_baseline_coefs = []
        for index in non_zero_index:
            weight = np.random.rand()
            baseline_weights = np.random.rand()
            random_baseline_coefs.append(baseline_weights)
            weights_alone_ordered.append(weight)
            weights_black_white[index] = weight

        #weights_alone_ordered = softmax(weights_alone_ordered)
        print("white_black_box weights : ",non_zero_index)
        print("weight ordered",weights_alone_ordered)


        #Si plusieurs tops alors on prend le 1er qui vient : d'ou le [0][0] en fin d'instruction
        matrice_iid_top = np.where(black_box_predictions_for_user == np.max(black_box_predictions_for_user))[0][0]
        matrice_iid_flop = np.where(black_box_predictions_for_user == np.min(black_box_predictions_for_user))[0][0]

        #TODO:PAS OPTI DU TOUT SE TROUVE EN DOUBLON DANS EXPLAIN EN PLUS
        movielens_ids = list(iid_map.keys())
        matrice_ids = list(iid_map.values())
        top20_user_matrice_ids = np.argsort(user_representation)[-20:][::-1]
        top20_user_movielens_ids = [movielens_ids[matrice_ids.index(matrice_id)] for matrice_id in
                                     top20_user_matrice_ids]


        top20_user_genres = [movies.loc[movies['movieId'] == id_movie]['genres'].values.tolist()[0] for
                              id_movie in top20_user_movielens_ids]


        number_of_rated_item_by_user = np.count_nonzero(user_representation)
        user_non_zero = user_representation[np.nonzero(user_representation)]
        user_mean = user_means[matrice_uid]





        user_std = np.std(user_non_zero)
        user_min = np.min(user_non_zero)
        user_max = np.max(user_non_zero)

        base_exp, mae, exp_matrice_ids,coefs_ordered = explain_white_black(matrice_uid, matrice_iid, n_coeff, sigma, Vt, user_means, all_actual_ratings, labels, train_set_size, pert_ratio,mode='lars',movies=movies,iid_map=iid_map,weights_white_black=weights_black_white)

        baseline_ranking = [weights_alone_ordered[non_zero_index.index(x)] if x in  non_zero_index else 0  for _,x in sorted(zip(random_baseline_coefs,random_baseline_indexs),reverse=True)]
        exp_ranking = [weights_alone_ordered[non_zero_index.index(x)] if x in  non_zero_index else 0  for _,x in sorted(zip(coefs_ordered,exp_matrice_ids),reverse=True)]
        ideal_ranking = sorted(weights_alone_ordered, reverse=True)

        ideal_dcg_10 = DCG2(ideal_ranking, 10)
        ideal_dcg_5 = DCG2(ideal_ranking, 5)
        ideal_dcg_3 = DCG2(ideal_ranking, 3)
        precision_recall_baseline = len(np.intersect1d(non_zero_index,random_baseline_indexs))/len(non_zero_index)
        precision_recall_exp = len(np.intersect1d(non_zero_index,exp_matrice_ids))/len(non_zero_index)

        baseline_ndcg_at_10 = DCG2(baseline_ranking,10) / ideal_dcg_10
        baseline_ndcg_at_5 = DCG2(baseline_ranking,5) / ideal_dcg_5
        baseline_ndcg_at_3 = DCG2(baseline_ranking,3) / ideal_dcg_3

        exp_ndcg_at_10 = DCG2(exp_ranking,10) / ideal_dcg_10
        exp_ndcg_at_5 = DCG2(exp_ranking,5) / ideal_dcg_5
        exp_ndcg_at_3 = DCG2(exp_ranking,3) / ideal_dcg_3


        f = open("temp/deuxieme_exp_result.csv", "a")
        line = str(precision_recall_exp)+';'+str(precision_recall_baseline)+';'+str(exp_ndcg_at_3)+';'+str(baseline_ndcg_at_3)+';'+str(exp_ndcg_at_5)+';'+str(baseline_ndcg_at_5)+';'+str(baseline_ndcg_at_10)+';'+str(exp_ndcg_at_10)+';'+str(number_of_rated_item_by_user)
        line = line.replace("\n", "")
        line = line+'\n'
        f.write(line)
        f.close()
        print("the baseline ranking = ",baseline_ranking)
        print("precision = recall = ",len(np.intersect1d(non_zero_index,exp_matrice_ids))/len(non_zero_index))
        print("Mae = ", mae)
        print("True rating = ", couple_uid_iid[2])
        print("ndcg baseline random = ",DCG2(baseline_ranking,10)/ideal_dcg_10)
        print("ndcg lire = ", DCG2(exp_ranking,10) / ideal_dcg_10)
        print("--------------------------")


def experiment_double_white_black_box(U, sigma, Vt, user_means, labels, all_actual_ratings,iid_map ):
    '''
    This is the double white box experiments described in the Dolap paper 2021
    :param U: users latent space
    :param sigma: intensity of latent factors
    :param Vt: items latent space
    :param user_means: the ratings average per users
    :param labels: the cluster ids per users
    :param all_actual_ratings: the rating matrix
    :param iid_map: a dictionnary that permits to switch between the item id in the matrix
    and the item id in the movielens dataset.
    :return: a file with the results : precision/recalls, ndcg@3,@5 etc..
    '''

    f = open("temp/double_whitebox_exp_result.csv", "w")
    f.write('precision_recall_exp;precision_recall_baseline;exp_ndcg_at_3;baseline_ndcg_at_3;exp_ndcg_at_5;baseline_ndcg_at_5;baseline_ndcg_at_10;exp_ndcg_at_10;number_item_rated_by_user;precision_recall_in_k_neighborhood;precision_recall_out_k_neighborhood;precision_recall_baseline_in_k_neighborhood;precision_recall_baseline_out_k_neighborhood\n')
    f.close()
    n_coeff = 10
    train_set_size = 1000
    pert_ratio = 0

    toBeExplained = [(int(x[0]), int(x[1]),x[2]) for x in
                     pd.read_csv('ml-latest-small/ratings.csv', sep=",").values.tolist()]


    toBeExplained = random.sample(toBeExplained,50)
    movies = pd.read_csv('ml-latest-small/movies.csv', sep=",",header=0)
    #Les clé d'iid map sont les ids de movielens et les valeurs sont les ids de la matrice.


    #random.shuffle(toBeExplained)
    for couple_uid_iid in toBeExplained:



        movielens_uid = couple_uid_iid[0]
        #movielens_iid = couple_uid_iid[1]
        matrice_uid = movielens_uid - 1
        matrice_iid = iid_map[couple_uid_iid[1]]

        black_box_predictions_for_user = make_black_box_slice(U, sigma, Vt, user_means, [matrice_uid])[0]
        user_representation = all_actual_ratings[matrice_uid,:].toarray()[0]
        item_matrices_ids_rated_by_user = np.nonzero(user_representation)[0]

        weights1 = [0] * 9724
        weights2 = [0] * 9724
        non_zero_index = list(np.random.choice(item_matrices_ids_rated_by_user,10,replace=False))
        weights_alone_ordered = []
        random_baseline_indexs = list(np.random.choice(item_matrices_ids_rated_by_user, 10, replace=False))
        random_baseline_coefs = []
        cpt = 0
        for index in non_zero_index:
            baseline_weights = np.random.rand()
            random_baseline_coefs.append(baseline_weights)
            weight = np.random.rand()
            weights_alone_ordered.append(weight)
            if(cpt >4):
                weights2[index] = weight
            else:
                weights1[index] = weight
            cpt = cpt + 1

        #weights_alone_ordered = softmax(weights_alone_ordered)
        print("white_black_box weights 1 : ",np.nonzero(weights1))
        print("white_black_box weights 2 : ", np.nonzero(weights2))
        print("index chosen = ",non_zero_index)
        print("weight ordered",weights_alone_ordered)



        #Si plusieurs tops alors on prend le 1er qui vient : d'ou le [0][0] en fin d'instruction
        matrice_iid_top = np.where(black_box_predictions_for_user == np.max(black_box_predictions_for_user))[0][0]
        matrice_iid_flop = np.where(black_box_predictions_for_user == np.min(black_box_predictions_for_user))[0][0]

        #TODO:PAS OPTI DU TOUT SE TROUVE EN DOUBLON DANS EXPLAIN EN PLUS
        movielens_ids = list(iid_map.keys())
        matrice_ids = list(iid_map.values())
        top20_user_matrice_ids = np.argsort(user_representation)[-20:][::-1]
        top20_user_movielens_ids = [movielens_ids[matrice_ids.index(matrice_id)] for matrice_id in
                                     top20_user_matrice_ids]




        number_of_rated_item_by_user = np.count_nonzero(user_representation)
        user_non_zero = user_representation[np.nonzero(user_representation)]


        base_exp, mae, exp_matrice_ids,coefs_ordered = explain_double_white_black(matrice_uid, matrice_iid, n_coeff, sigma, Vt, user_means, all_actual_ratings, labels, train_set_size, pert_ratio,mode='lars',movies=movies,iid_map=iid_map,weights_white_black1=weights1,weights_white_black2=weights2)

        baseline_ranking = [weights_alone_ordered[non_zero_index.index(x)] if x in  non_zero_index else 0  for _,x in sorted(zip(random_baseline_coefs,random_baseline_indexs),reverse=True)]
        exp_ranking = [weights_alone_ordered[non_zero_index.index(x)] if x in  non_zero_index else 0  for _,x in sorted(zip(coefs_ordered,exp_matrice_ids),reverse=True)]
        ideal_ranking = sorted(weights_alone_ordered, reverse=True)

        ideal_dcg_10 = DCG2(ideal_ranking, 10)
        ideal_dcg_5 = DCG2(ideal_ranking, 5)
        ideal_dcg_3 = DCG2(ideal_ranking, 3)
        precision_recall_baseline = len(np.intersect1d(non_zero_index,random_baseline_indexs))/len(non_zero_index)
        precision_recall_baseline_in_k_neighborhood = len(np.intersect1d(non_zero_index[:5],random_baseline_indexs))/len(non_zero_index[:5])
        precision_recall_baseline_out_k_neighborhood = len(np.intersect1d(non_zero_index[5:],random_baseline_indexs))/len(non_zero_index[5:])

        precision_recall_exp = len(np.intersect1d(non_zero_index,exp_matrice_ids))/len(non_zero_index)

        precision_recall_in_k_neighborhood = len(np.intersect1d(non_zero_index[:5],exp_matrice_ids))/len(non_zero_index[:5])
        precision_recall_out_k_neighborhood = len(np.intersect1d(non_zero_index[5:],exp_matrice_ids))/len(non_zero_index[5:])

        precision_recall_in_k_neighborhood = len(np.intersect1d(non_zero_index[:5], exp_matrice_ids)) / len(
            non_zero_index[:5])
        precision_recall_out_k_neighborhood = len(np.intersect1d(non_zero_index[5:], exp_matrice_ids)) / len(
            non_zero_index[5:])

        baseline_ndcg_at_10 = DCG2(baseline_ranking,10) / ideal_dcg_10
        baseline_ndcg_at_5 = DCG2(baseline_ranking,5) / ideal_dcg_5
        baseline_ndcg_at_3 = DCG2(baseline_ranking,3) / ideal_dcg_3

        exp_ndcg_at_10 = DCG2(exp_ranking,10) / ideal_dcg_10
        exp_ndcg_at_5 = DCG2(exp_ranking,5) / ideal_dcg_5
        exp_ndcg_at_3 = DCG2(exp_ranking,3) / ideal_dcg_3


        f = open("temp/double_whitebox_exp_result.csv", "a")
        line = str(precision_recall_exp)+';'+str(precision_recall_baseline)+';'+str(exp_ndcg_at_3)+';'+str(baseline_ndcg_at_3)+';'+str(exp_ndcg_at_5)+';'+str(baseline_ndcg_at_5)+';'+str(baseline_ndcg_at_10)+';'+str(exp_ndcg_at_10)+';'+str(number_of_rated_item_by_user)+';'+str(precision_recall_in_k_neighborhood)+';'+str(precision_recall_out_k_neighborhood)+';'+str(precision_recall_baseline_in_k_neighborhood)+';'+str(precision_recall_baseline_out_k_neighborhood)
        line = line.replace("\n", "")
        line = line+'\n'
        f.write(line)
        f.close()
        print("the baseline ranking = ",baseline_ranking)
        print("precision = recall = ",len(np.intersect1d(non_zero_index,exp_matrice_ids))/len(non_zero_index))
        print("Mae = ", mae)
        print("True rating = ", couple_uid_iid[2])
        print("ndcg baseline random = ",DCG2(baseline_ranking,10)/ideal_dcg_10)
        print("ndcg lire = ", DCG2(exp_ranking,10) / ideal_dcg_10)
        print("--------------------------")

def robustness_score_tab(user_id, item_id, n_coeff, sigma, Vt, user_means, all_user_ratings, cluster_labels, train_set_size:int, pert_ratio:float=0.5, k_neighbors=[5, 10, 15]):

    base_exp, mae = explain(user_id, item_id, n_coeff, sigma, Vt, user_means, all_user_ratings, cluster_labels, train_set_size, pert_ratio)
    max_neighbors = np.max(k_neighbors)
    # get user_id cluster neighbors
    cluster_index = cluster_labels[user_id]  # retrieve the cluster index of user "user_id"
    neighbors_index = np.where(cluster_labels == cluster_index)[0]
    neighbors_index = neighbors_index[neighbors_index != user_id]
    neighbors_index = np.random.choice(neighbors_index, max_neighbors)      # look for max # of neighbors

    # objective is now to compute several robustness score for different values of k in k-NN
    dist_to_neighbors = {}      # structure to sort neighbors based on their increasing distance to user_id
    rob_to_neighbors = {}       # structure that contain the local "robustness" score of each neighbor to user_id
    for id in neighbors_index:
        # todo: here we can retrieve the absolute error on neighbors
        exp_id, _ = explain(id, item_id, n_coeff, sigma, Vt, user_means, all_user_ratings, cluster_labels, train_set_size, pert_ratio)
        if (isinstance(all_user_ratings, scipy.sparse.csr._cs_matrix)):
            dist_to_neighbors[id] = cosine_dist(np.nan_to_num(all_user_ratings[user_id].toarray()), np.nan_to_num(all_user_ratings[id].toarray()))
        else:
            dist_to_neighbors[id] = cosine_dist(np.nan_to_num(all_user_ratings[user_id]), np.nan_to_num(all_user_ratings[id]))
        rob_to_neighbors[id] = cosine_dist(exp_id, base_exp) / dist_to_neighbors[id]
        if np.isnan(cosine_dist(exp_id, base_exp)):
            print('error on explanations distance')

    # sort dict values by preserving key-value relation
    sorted_dict = {k: v for k, v in sorted(dist_to_neighbors.items(), key=lambda item: item[1])} # need Python 3.6

    sorted_dist = np.zeros(max_neighbors)       # all sorted distances to user_id
    sorted_rob = np.zeros(max_neighbors)        # all robustness to user_id explanation corresponding
                                                # to sorted distance value
                                                # at index i, sorted_dist contains the i+1th distance to user_id
                                                # that corresponds to id = key
                                                # in this case, sorted_rob[i] contains robustness of id = key
    cpt = 0
    for key in sorted_dict.keys():              # checked! Keys respect the order of elements in dict
        sorted_dist[cpt] = sorted_dict[key]
        sorted_rob[cpt] = rob_to_neighbors[key]
        cpt += 1

    # finally, we compute the max(rob)@5,10,15 or any number of neighbors specified in k_neigbors
    res = np.empty(len(k_neighbors))
    cpt = 0
    for k in k_neighbors:
        res[cpt] = np.max(sorted_rob[0:k])
        cpt += 1

    # todo : check output
    return res, mae, sorted_dict.keys(), sorted_dist

def exp_check_UMAP(n_coeff, sigma, Vt, all_user_ratings, cluster_labels, train_set_size=50,n_dim_UMAP = [3,10,15],
                   min_dist_UMAP=[0.1,0.01,0.001], n_neighbors_UMAP=[10, 30, 50], pert_ratio:float=0, k_neighbors=[5, 10, 15]):
    """
    Run test to evaluate the sensitivity of our method to UMAP dimensionality reduction
    :param users: numpy array of user ids
    :param items: numpy array of item ids
    :param n_coeff: number of coefficients of interpretable features for the explanation
    :param sigma: influence of each latent dimension
    :param Vt: latent item space
    :param all_user_ratings: initial rating matrix
    :param cluster_labels: user clustering, defines neighborhood
    :param train_set_size: number of training instance to learn surrogate model
    :param n_dim_UMAP: numpy array of reducted number of dimensions
    :param n_neighbors_UMAP: numpy array of number of neighbors to preserve local topology in UMAP
    :param training_set_size:
    :param pert_ratio:
    :param k_neighbors:
    :return:
    """
    items = np.random.choice(range(Vt.shape[1]), 5)
    users = np.random.choice(range(U.shape[0]), 10)
    columns = ['clustering_algorithm','n_cluster','robustness@5','robustness@10','robustness@15','mae','user_id','item_id','silhoutte_score_by_cluster','silhouette_score_all']
    result = []
    for n_dim in n_dim_UMAP:
        for min_dist in min_dist_UMAP:
            for n_neighbor in n_neighbors_UMAP:
                reducer = umap.UMAP(n_components=n_dim, n_neighbors=n_neighbor, random_state=12,
                                    min_dist=min_dist)  # metric='cosine'
                embedding = reducer.fit_transform(all_actual_ratings)
                for clustering_algorithm in ['kmeans','hdbscan']:

                    if (clustering_algorithm == 'kmeans'):

                        hyperparameters_clustering = [5,10,15]
                        for hyperparameter in hyperparameters_clustering:
                            hyperparameter_clustering = hyperparameter
                            clusterer = KMeans(n_clusters = hyperparameter)
                            clusterer.fit(embedding)
                            labels = clusterer.labels_

                            #Calcul of silhouette scores by sample, cluster and all
                            X = all_user_ratings.toarray()
                            silh_samp = silhouette_samples(X=X, labels=labels, metric='cosine')
                            silh_score = silhouette_score(X=X, labels=labels, metric='cosine')
                            df_temp = pd.DataFrame(silh_samp,columns=['silh_samp'])
                            df_temp['labels'] = labels
                            silh_clust = [df_temp.loc[df_temp['labels']==label]['silh_samp'].mean() for label in set(labels)]

                            np.savetxt("labels_" + str(n_dim) + "_" + str(n_neighbor) + ".gz",
                                       labels)  # personalize output filename

                            # robustness measure
                            for user_id in users:
                                for item_id in items:
                                    # todo : save in a file somewhere, which format?

                                    res, mae, keys, distances = robustness_score_tab(user_id, item_id, n_coeff, sigma,
                                                                                     Vt, user_means, all_user_ratings,
                                                                                     labels, train_set_size, pert_ratio,
                                                                                     k_neighbors)
                                    l = [clustering_algorithm, hyperparameter_clustering, res[0], res[1], res[2], mae,
                                         user_id, item_id, silh_score, silh_clust]
                                    print("user :", user_id, " item_id", item_id, " line added :", l)
                                    result.append(l)
                    else:
                        hyperparameter_clustering = None
                        clusterer = hdbscan.HDBSCAN()

                        clusterer.fit(embedding)

                        labels = clusterer.labels_
                        # Calcul of silhouette scores by sample, cluster and all
                        X = all_user_ratings.toarray()
                        silh_samp = silhouette_samples(X=X, labels=labels, metric='cosine')
                        silh_score = silhouette_score(X=X, labels=labels, metric='cosine')
                        df_temp = pd.DataFrame(silh_samp, columns=['silh_samp'])
                        df_temp['labels'] = labels
                        silh_clust = [df_temp.loc[df_temp['labels'] == label]['silh_samp'].mean() for label in set(labels)]

                        np.savetxt("labels_" + str(n_dim) + "_" + str(n_neighbor) +".gz", labels)   # personalize output filename

                        # robustness measure
                        for user_id in users:
                            for item_id in items:
                                # todo : save in a file somewhere, which format?
                                res,mae, keys, distances = robustness_score_tab(user_id, item_id, n_coeff, sigma, Vt, user_means, all_user_ratings,
                                                 labels, train_set_size, pert_ratio, k_neighbors)
                                l = [clustering_algorithm,hyperparameter_clustering,res[0],res[1],res[2],mae,user_id,item_id,silh_score,silh_clust]
                                print("user :",user_id," item_id",item_id, " line added :",l)
                                result.append(l)

    df = pd.DataFrame(result, columns=columns)
    df.to_csv('res/clustering_result_parameters_search_v2.csv')



def vars(a, axis=None):
    """ Variance of sparse matrix a
    var = mean(a**2) - mean(a)**2
    """
    a_squared = a.copy()
    a_squared.data **= 2
    return a_squared.mean(axis) - np.square(a.mean(axis))


def stds(a, axis=None):
    """ Standard deviation of sparse matrix a
    std = sqrt(var(a))
    """
    return np.sqrt(vars(a, axis))


if __name__ == '__main__':
    U = None
    sigma = None
    Vt = None
    all_user_predicted_ratings = None
    OUTFILE = "res/edbt/exp_edbt_"+datetime.datetime.now().strftime("%j_%H_%M")+".csv"
    TEMP = "temp/"# Fodler for precomputed black box data
    SIZE = "small"#Size of dataset small/big 100k or 20M

    print('--- Configuring Torch')
    Config.set_device_gpu()
    print("Running tensor computations on", Config.device())

    print("--- Loading Ratings ---")
    if SIZE == "small":
        all_actual_ratings, iid_map = read_sparse("./ml-latest-small/ratings.csv")
        TAG = "_small_"
        print("[WARNING] Using 100K SMALL dataset !")
    else:
        all_actual_ratings, iid_map = read_sparse("./ml-20m/ratings.csv")
        TAG = ""

    # Loading data and setting all matrices
    if os.path.isfile(TEMP + "flag" + TAG + ".flag"):
        print("-- LOAD MODE ---")
        U = np.loadtxt(TEMP + "U" + TAG + ".gz")
        sigma = np.loadtxt(TEMP + "sigma" + TAG + ".gz")
        Vt = np.loadtxt(TEMP + "Vt" + TAG + ".gz")
        labels = np.loadtxt(TEMP + "labels" + TAG + ".gz")
        user_means = np.loadtxt(TEMP + "user_means" + TAG + ".gz")
        iid_map = pickle.load(open(TEMP + "iid_map" + TAG + ".p", "rb"))

    # No data found computing black box and clusters results
    else:
        print('--- COMPUTE MODE ---')
        print("  De-Mean")
        user_means = [None] * all_actual_ratings.shape[0]
        all_actual_ratings_demean = scipy.sparse.dok_matrix(all_actual_ratings.shape)
        for line, col in tqdm(all_actual_ratings.todok().keys()):
            if user_means[line] is None:
                user = all_actual_ratings[line].toarray()
                user[user == 0.] = np.nan
                user_means[line] = np.nanmean(user)
            all_actual_ratings_demean[line, col] = all_actual_ratings[line, col] - user_means[line]
        user_means = np.array(user_means)
        user_means = user_means.reshape(all_actual_ratings.shape[0],1)

        print("  Running SVD")
        U, sigma, Vt = svds(all_actual_ratings_demean.tocsr(), k=50,solver='lobpcg',which='LM', maxiter=1000)
        sigma = np.diag(sigma)

        # saving matrices
        np.savetxt(TEMP + "U" + TAG + ".gz", U)
        np.savetxt(TEMP + "sigma" + TAG + ".gz", sigma)
        np.savetxt(TEMP + "Vt" + TAG + ".gz", Vt)
        np.savetxt(TEMP + "user_means" + TAG + ".gz", user_means)
        user_means = np.loadtxt(TEMP + "user_means" + TAG + ".gz")# Dirty fix to avoid a shape issue
        pickle.dump(iid_map, open(TEMP + "iid_map" + TAG + ".p", "wb"))

        print("Running UMAP")
        reducer = umap.UMAP(n_components=3, n_neighbors=30, min_dist=0.01, low_memory=True)  # metric='cosine'
        embedding = reducer.fit_transform(all_actual_ratings)
        print("Running clustering")
        clusterer = KMeans(n_clusters=75)
        clusterer.fit(embedding)
        labels = clusterer.labels_
        np.savetxt(TEMP + "labels" + TAG + ".gz", labels)
        with open(TEMP + "flag" + TAG + ".flag", mode="w") as f:
            f.write("1")


    # Load sigma and Vt in memory for torch (possibly on the GPU)
    sigma_t = make_tensor(sigma)
    Vt_t = make_tensor(Vt)

    global_variance = stds(all_actual_ratings, axis=0)

    experiment_double_white_black_box(U, sigma, Vt, user_means, labels, all_actual_ratings,iid_map)
    #exp_check_UMAP(10, sigma, Vt, all_actual_ratings, None, train_set_size=50, n_dim_UMAP=[3],
    #                   min_dist_UMAP=[0.01], n_neighbors_UMAP=[30], pert_ratio=0., k_neighbors=[5, 10, 15])


