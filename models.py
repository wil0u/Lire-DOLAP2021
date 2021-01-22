import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from config import Config


class LinearRecommender(nn.Module):
    def __init__(self, in_shape, decay=0.3):
        super(LinearRecommender, self).__init__()
        self.dims = in_shape  # number of dimensions (eg movies)
        self.decay = torch.tensor(decay, device=Config.device())
        self.omega = nn.Parameter(
            nn.init.normal_(torch.zeros((self.dims,), device=Config.device()), mean=.0, std=0.2))

    def forward(self, x):
        # init output
        if len(x.size()) > 1:
            y = (torch.sum(x * self.omega.expand(x.size()[0], self.dims), axis=1) / torch.sum(self.omega))
        else:
            y = (torch.sum(x * self.omega) / torch.sum(self.omega))
        # for each user
        # for u in range(x.size()[0]):
        # compute sim to all other users
        # sim = torch.sum(x[u]*x, 1) / ( torch.sqrt(torch.sum((x[u])**2)) * torch.sqrt(torch.sum((x)**2, 1)) )
        # sim = torch.exp(-0.5*torch.pow((sim-1)/self.decay ,2))
        # norm = torch.sum(sim)

        # y[u] =  torch.sum( (torch.sum(x * self.omega.expand(x.size()[0], self.dims), axis = 1) /  torch.sum(self.omega)) * sim ) / norm

        return y

    def predict(self, X):
        return (torch.sum(torch.tensor(X) * self.omega.expand(X.shape[0], self.dims), axis=1) / torch.sum(self.omega)).detach().numpy()

# args :
# 0 : number of users
# 1 : x
# 2 : y
# 3 : x' original space projection of x
# 4 : ref user
# 5 : kernel width
def linear_recommender(omega, *args):
    ex = np.array([[1]*args[0]])
    omega = omega.reshape((1, omega.shape[0]))

    pred = np.sum(args[1] * (ex.T @ omega)) / np.sum(omega)

    err = np.abs(pred - args[2])

    sim = np.sum(args[4] * args[3], 1) / (
            np.sqrt(np.sum((args[4]) ** 2)) * np.sqrt(np.sum((args[3]) ** 2, 1)))
    sim = np.exp(-0.5 * np.power((sim - 1) / args[5], 2))

    return np.mean(err * sim)


# Training loop
# target_user_vec -> user targeted for explaination (as a 'real' ratings vector)
def train(model, X, y, loss_criterion, epochs, learning_rate=0.01, verbose=True, clamp=False):
    #optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):  # loop over the dataset multiple times
        if verbose:
            print("epoch", epoch, end='')
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(X)
        loss = loss_criterion(X, model.omega, outputs, y)
        loss.backward()
        optimizer.step()

        # Clamp wheights to 0 no less
        if clamp:
            model.omega.data[model.omega < 0.] = 0.

        # print statistics
        if verbose:
            print(" | loss", loss.item())


def get_fx(user, s, v, films_nb, epochs=20):
    umean = user.sum() / (user != 0.).sum()
    umask = user != 0.

    unew = nn.Parameter(torch.zeros(1, s.size()[0], device=user.device, requires_grad=True))

    opt = optim.Adagrad([unew])

    for epoch in range(epochs):
        pred = unew @ s @ v + umean.expand(films_nb)
        loss = torch.sum(torch.pow(((user - pred) * umask), 2)) / torch.sum(umask)
        loss.backward()
        opt.step()
        opt.zero_grad()

    return ((unew @ s @ v + umean.expand(films_nb)) * (user == 0.) + user).detach()

# Quick Out of sample prediction for matrix factorization
# WITH REDUCED SET OF DIMENSIONS
# Minimizes the error reconstructing it's ratings
# TODO -> same as get_OOS_pred
def get_OOS_pred_inner(user, s, v, films_nb, epochs=20):
    umean = user.sum(axis=1) / (user != 0.).sum(axis=1)
    umask = user != 0.

    unew = nn.Parameter(torch.zeros(user.size()[0], s.size()[0], device=user.device, requires_grad=True))
    opt = optim.Adagrad([unew])

    for epoch in range(epochs):
        # Warning: nb_films-1 as we work in a space where a movie was removed to be explained
        pred = unew @ s @ v + umean.expand(films_nb - 1, user.size()[0]).transpose(0, 1)
        loss = torch.sum(torch.pow(((user - pred) * umask), 2)) / torch.sum(umask)
        loss.backward()
        opt.step()
        opt.zero_grad()

    # Warning: nb_films-1 as we work in a space where a movie was removed to be explained
    return ((unew @ s @ v + umean.expand(films_nb - 1, user.size()[0]).transpose(0, 1)) * (user == 0.) + user).detach()


# Quick Out of sample prediction for matrix factorization, we try to get the user's vector in the latent space
# that minimizes the error reconstructing it's ratings
def get_OOS_pred(user, s, v, films_nb, epochs=20):
    # print("  --- --- ---")
    umean = user.sum(axis=1) / (user != 0.).sum(axis=1)
    umask = user != 0.

    unew = nn.Parameter(torch.zeros(user.size()[0], s.size()[0], device=user.device, requires_grad=True))
    opt = optim.Adagrad([unew])

    for epoch in range(epochs):
        pred = unew @ s @ v + umean.expand(films_nb, user.size()[0]).transpose(0, 1)
        loss = torch.sum(torch.pow(((user - pred) * umask), 2)) / torch.sum(umask)
        loss.backward()
        opt.step()
        opt.zero_grad()
        # unew.data -= 0.0007 * unew.grad.data
        # unew.grad.data.zero_()
        # if epoch == 0 or epoch % 2 == 0 : print("  ", loss.item())

    return ((unew @ s @ v + umean.expand(films_nb, user.size()[0]).transpose(0, 1)) * (user == 0.) + user).detach()








if __name__ == '__main__':
    from utility import load_data
    U, sigma, Vt, all_actual_ratings, all_user_predicted_ratings, movies_df, ratings_df, films_nb = load_data()
    print("films", films_nb)

    from scipy.optimize import least_squares

    for i in range(250, 300):
        # Actual ratings
        test = np.nan_to_num(all_actual_ratings[i])

        def prepare(perturbation):
            def pred_fn(user_vec_lat):
                umean = user_vec_lat.sum() / (user_vec_lat != 0.).sum()
                umask = perturbation != 0.
                pred = (user_vec_lat @ sigma @ Vt) + umean
                return (perturbation - pred) * umask
            return pred_fn

        res = least_squares(prepare(test), np.ones(20), loss="soft_l1")
        #print(res)
        #print(U[50])
        moy = test.sum() / (test != 0.).sum()
        real_pred = (U[i] @ sigma @ Vt) + moy
        ls_pred = (res.x @ sigma @ Vt) + moy
        print("mae least_squares", np.average(np.abs(real_pred - ls_pred)))


