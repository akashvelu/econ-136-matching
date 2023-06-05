import numpy as np
import pickle
from numpy.linalg import norm


def generate_embeddings(N, D, dist="std", store=True):
    """
    Generates embeddings of dimension N x D from the
    specified distribution
    """
    if dist == "std":
        emb = np.random.standard_normal((N, D))
    if dist == "unif":
        # by default between -1 and 1
        emb = np.random.uniform(low=-1, high=1, size=(N, D))
    if dist == "bimodal":
        # by default from N(-1, 1) and N(1, 1)
        emb1 = np.random.normal(loc=-1, scale=1, size=(N // 2, D))
        emb2 = np.random.normal(loc=1, scale=1, size=(N - N // 2, D))
        emb = np.concatenate([emb1, emb2], axis=0)
        emb = np.random.shuffle(emb)

    if store:
        pickle.dump(emb, open(f"data/{dist}_{N}x{D}.pkl", "wb"))

    return emb


def cosine_similarity(v1, v2):
    """
    Gets cosine similarity of vectors v1 and v2
    """
    return np.dot(v1, v2) / (norm(v1) * norm(v2))


def get_cosine_similarities(m1, m2, filename=None):
    """
    Return a cosine similarity matrix between m1 and m2
    """
    M, N = m1.shape[0], m2.shape[0]
    assert m1.shape[1] == m2.shape[1]
    similarity_matrix = np.zeros((M, N))

    for i in range(M):
        for j in range(N):
            similarity_matrix[i][j] = cosine_similarity(m1[i], m2[j])

    if filename:
        pickle.dump(similarity_matrix, open(filename, "wb"))

    return similarity_matrix


def get_oracle_pref(sim_mat, filename=None):
    """
    Finds the oracle matches from the columns for each row index.
    Sorted by most preferred to least.
    To find closest row matches for each column index, pass in the
    transposed matrix
    """
    pref_mat = (-sim_mat).argsort(axis=-1)

    if filename:
        pickle.dump(pref_mat, open(filename, "wb"))

    return pref_mat


def get_permutation(v, k):
    """
    Generates a permutation of vector v but only permutes
    the first k indices. The remaining indices are kept intact
    """
    x = np.random.permutation(v[:k])
    return np.concatenate([x, v[k:]])
