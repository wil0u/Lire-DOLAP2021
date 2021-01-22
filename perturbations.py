import numpy as np
import torch
from torch import nn as nn

from config import Config


def perturbations_uniform(ux, fake_users, proba=0.25):
    """
    :param ux: initial user
    :param fake_users: number of perturbed copies from the initial user
    :param proba: probability of changing a value
    :return: a matrix of perturbed fake users
    """
    nb_dim = ux.size()[0]
    users = ux.expand(fake_users, nb_dim).detach().clone()
    rd_mask = torch.zeros(fake_users, nb_dim, device=Config.device()).uniform_() > (proba)

    #print(rd_mask * users)

    return rd_mask * users


def perturbations_3(ux):
    """
        Generate all possible perturbed users from ux by removing each time a single feature dimension
        :param ux: initial user
        :return: a matrix of perturbed fake users
    """
    nb_dim = ux.size()[0]
    tmp = torch.tensor(ux[ux > 0])

    nb_non_zero_dim = tmp.size()[0]
    users = ux.expand(nb_non_zero_dim, nb_dim).detach().clone()

    # for loop are bad in Python!
    row = 0
    col = 0
    for v in (ux > 0):
        if v:
            users[row][col] = 0
            row +=1
        col += 1


    return users


def perturbations_swap(ux, fake_users: int):

    """
    Generate random perturbed users from ux by exchanging 2 feature dimensions
    :param ux: initial user
    :param fake_users: number of perturbed instances based on user
    :return: a set of perturbed users
    """

    nb_dim = ux.size()[0]
    users = ux.expand(fake_users, nb_dim).detach().clone()
    #print(users)

    #tmp = torch.tensor(ux[ux > 0])
    # nb_non_zero_dim = tmp.size()[0]
    # print(nb_non_zero_dim)

    # small loop can't hurt
    non_zeros = np.argwhere((ux > 0.).numpy()).flatten()
    zeros = np.argwhere((ux == 0.).numpy()).flatten()

    for i in range(fake_users):
        a = non_zeros[np.random.randint(low=0, high=non_zeros.size)]
        b = zeros[np.random.randint(low=0, high=zeros.size)]
        users[i][a] = ux[b].detach().clone()
        users[i][b] = ux[a].detach().clone()

    return users


def perturbations_gaussian(ux, fake_users: int, std=2, proba=0.1):
    nb_dim = ux.size()[0]
    users = ux.expand(fake_users, nb_dim).detach().clone()

    perturbation = nn.init.normal_(torch.zeros(fake_users, nb_dim, device=Config.device()), 0, std)
    rd_mask = torch.zeros(fake_users, nb_dim, device=Config.device()).uniform_() > (1. - proba)
    perturbation = perturbation * rd_mask * (users != 0.)
    users = users + perturbation
    # users[users > 5.] = 5. #seems to introduce detrimental bias in the training set
    return torch.abs(users)