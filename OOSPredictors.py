from scipy.optimize import least_squares
import numpy as np
from torch import optim, nn
import torch


# Scipy OOS
def OOS_pred_SciPy(perturbation, sigma, Vt):
    def prepare(perturbation):
        """
        construct residual computation for least square optimization
        :param perturbation: oos user signature
        :return: method to compute residual vector for oos user "perturbation"
        """
        def pred_fn(user_vec_lat):
            """
            Compute an upgrade to the latent representation as a residual
            :param user_vec_lat: current latent space representation of perturbation oos user
            :return:
            """
            umean = user_vec_lat.sum() / (user_vec_lat != 0.).sum()
            umask = perturbation != 0.
            pred = (user_vec_lat @ sigma @ Vt) + umean
            return (perturbation - pred) * umask

        return pred_fn

    res = least_squares(prepare(perturbation), np.ones(sigma.shape[0])) # latent space dimension

    moy = perturbation.sum() / (perturbation != 0.).sum()

    return (res.x @ sigma @ Vt) + moy


# Init on original point vector
def OOS_pred_smart(user, s, v, init_vec, epochs=50):
    umean = user.sum() / (user != 0.).sum()
    umask = user != 0.

    unew = nn.Parameter(torch.tensor(init_vec, device=user.device, dtype=user.dtype, requires_grad=True))
    opt = optim.Adagrad([unew], 1)

    for epoch in range(epochs):
        pred = unew @ s @ v + umean
        loss = ( torch.sum(torch.pow(((user - pred) * umask), 2)) / torch.sum(umask) )
        loss.backward()
        opt.step()
        opt.zero_grad()

    return torch.clamp((unew @ s @ v + umean).detach(), 0., 5.)

# Single OOS pred with torch
def OOS_pred_single(user, s, v,epochs=50):
    umean = user.sum() / (user != 0.).sum()
    umask = user != 0.

    unew = nn.Parameter(torch.zeros(s.size()[0], device=user.device, dtype=user.dtype, requires_grad=True))
    opt = optim.Adagrad([unew])


    for epoch in range(epochs):
        pred = unew @ s @ v + umean
        loss = torch.sum(torch.pow(((user - pred) * umask), 2)) / torch.sum(umask)
        loss.backward()
        opt.step()
        opt.zero_grad()

    return (unew @ s @ v + umean).detach()


def OOS_pred_slice(users, s, v,item_id, epochs=120, init_vec=None):
    umean = users.sum(axis=1) / (users != 0.).sum(axis=1)
    umask = users != 0.
    if init_vec:
        unew = nn.Parameter(torch.tensor(init_vec, device=users.device, dtype=users.dtype, requires_grad=True))
    else:
        #unew = nn.Parameter(torch.ones(users.size()[0], s.size()[0], device=users.device, dtype=users.dtype, requires_grad=True))
        unew = nn.Parameter(
            torch.ones(users.size()[0], s.size()[0], device=users.device, dtype=users.dtype, requires_grad=True))

    #opt = optim.Adadelta([unew],lr=10,rho=0.9,weight_decay=1)
    opt = optim.Adagrad([unew],lr=0.1)
    #opt = optim.SGD([unew], lr=1, momentum=0.8)

    for epoch in range(epochs):

        pred = unew @ s @ v + umean.expand(v.size()[1], users.size()[0]).transpose(0, 1) * (users == 0.) + users
        #print(pred[0][item_id])
        loss = torch.sum(torch.pow((users - pred), 2)) / torch.numel(users)
        loss.backward()
        #print(loss)

        opt.step()
        opt.zero_grad()
    return ((unew @ s @ v + umean.expand(v.size()[1], users.size()[0]).transpose(0, 1)) * (users == 0.) + users).detach().clamp(0., 5.)