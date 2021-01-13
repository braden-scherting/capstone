import numpy as np
import random
import torch
import copy
from torch import nn
from torch import distributions
from torch.nn.parameter import Parameter
from torch.optim.lr_scheduler import StepLR

class CondNVP(nn.Module):

    """
    Implementation of conditional real NVP for conditional density estimation
    and sampling

    References (BibTeX)
    ___________________


    Dinh et al., 2016:
    @article{dinh2016density,
      title={Density estimation using real nvp},
      author={Dinh, Laurent and Sohl-Dickstein, Jascha and Bengio, Samy},
      journal={arXiv preprint arXiv:1605.08803},
      year={2016}
    }

    Code adapted from:
    @misc{Ashukha2018,
      author = {Arsenii, Ashukha},
      title = {Real NVP PyTorch a Minimal Working Example},
      year = {2018},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\ url{https://github.com/ars-ashuha/real-nvp-pytorch}},
    }


    y_{1:d}, \theta = x_{1:d}, \theta
    y_{d+1:D} = x_{d+1:D} * \exp(s(x_{1:d}, \theta)) + t(x_{1:d}, \theta)
    &
    x_{1:d}, \theta = y_{1:d}, \theta
    x_{d+1:D} = (y_{d+1:D} - t(y_{1:d}, \theta)) * \exp(-s(x_{1:d}, \theta))

    e.g. for dim(data)=3 and dim(parameters)=2, binary masks are:
    [1, 0, 1, 1, 1]
    [0, 1, 0, 1, 1]
    [1, 0, 1, 1, 1]
    [0, 1, 0, 1, 1]

    Parameters
    ----------
    mask   : torch.tensor
             Binary masks for coupling layers.
    nets   : see get_nvp(...) for details
             Scale function.
    nett   : see get_nvp(...) for details
             Translation function.
    latent : torch.distributions object
             latent distribution of dim(data)
    p_dim  : int
             Number of parameters. """

    def __init__(self, mask, nets, nett, latent, p_dim, x_dim, seed):
        super(CondNVP, self).__init__()
        
        # Binary mask for coupling layers (mask=1 for parameters)
        self.mask = nn.Parameter(mask, requires_grad=False)

        # NN scale and translation functions
        self.t = torch.nn.ModuleList([nett() for _ in range(len(mask))])
        self.s = torch.nn.ModuleList([nets() for _ in range(len(mask))])

        # Latent space, dim(latent) = dim(data)
        self.latent = latent

        # Dimension of parameter space
        self.p_dim = p_dim
        
        # Dimension of data space
        self.x_dim = x_dim
        
        self.seed = seed
        
    def g(self, z):
        # g: Z --> X
        # latent space --> data space
        x = z
        # For each coupling layer, do:
        for i in range(len(self.t)):
     
            # Compute transformations (eq. 4-5; Dinh et al.)
            x_ = x*self.mask[i]
            s = self.s[i](x_)*(1 - self.mask[i])
            t = self.t[i](x_)*(1 - self.mask[i])

            # Apply transformations (eq. 9; Dinh et al.)
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
        return x

    def f(self, x):
        # f: X --> Z; same as g(), just backwards
        # data space --> latent space
        log_det_J = x.new_zeros(x.shape[0])
        z = x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1 - self.mask[i])
            t = self.t[i](z_) * (1 - self.mask[i])
            z = z_ + (1 - self.mask[i]) * (z - t) * torch.exp(-s)
            
            # Because params mask=1, they make 0 contribution to prob.
            log_det_J -= s.sum(dim=1)
        return z, log_det_J
    
    def log_prob(self, x):
        z, logp = self.f(x)

        # Drop parameter dimensions when computing prior prob. 
        return self.latent.log_prob(z[:,:-self.p_dim]) + logp
    
    def sample(self, *params, batch_size, random_state=None):
        # Sample from latent for data dimensions
        if random_state is None:
            z_cond = self.latent.sample((batch_size, 1))
        else:
            z_cond = random_state.multivariate_normal(np.zeros(self.x_dim),
                                                      np.eye(self.x_dim), (batch_size, 1))
            z_cond = torch.from_numpy(z_cond.astype(np.float32))
        
        # Append arbitrary (observed) parameter values
        for param in params:
            z_cond = torch.cat((z_cond, param*torch.ones((batch_size,1,1))), dim=2)
        
        # Transform to data space
        x = self.g(z_cond)
        return x

    def train_nvp(self, data, batch_size=10, max_epochs=100, val_split=0.1, threshold=20, lr_step=20, lr_stepsize=0.5):
        """ Training loop for Real NVP as implemented above; not great, but seems to work
        
        Parameters
        __________
        nvp : CondNVP object as defined above
        data : numpy array (examples, x_dim + p_dim)
        batch_size: (int) batch size for 1 gradient update
        max_epochs: (int) Number of epochs if no early-stopping
        v : (float) validation split fraction
        threshold : (int) number of epochs without validation loss improvement before early-stop
        lr_step : Number of epochs between learning rate reductions
        lr_stepsize : When decaying, lr = lr_stepsize*lr

        Returns
        _______
        nothing... but the CondNVP object should be trained and samplable

        """
        torch.manual_seed(self.seed)
        rand_state = np.random.RandomState(self.seed)

        n_batches = int(len(data) / batch_size)

        optimizer = torch.optim.Adam([p for p in self.parameters() if p.requires_grad==True], lr=1e-3)
        
        # You may need to change these args for different problems; good defaults
        scheduler = StepLR(optimizer, lr_step, lr_stepsize)

        # Train/Validation split
        rand_state.shuffle(data)

        val_x = data[:int(val_split*len(data))]
        train_x = data[int(val_split*len(data)):]

        val_loss = -self.log_prob(torch.from_numpy(val_x.astype(np.float32))).mean()
        best_state = copy.deepcopy(self.state_dict())
        not_improved = 0

        for epoch in range(max_epochs):
            for b in range(n_batches):
                batch = torch.from_numpy(train_x[b*batch_size:(b+1)*batch_size].astype(np.float32))
                loss = -self.log_prob(batch).mean()
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
            scheduler.step()
            rand_state.shuffle(train_x)

            # Update early-stopping criterion
            check = -self.log_prob(torch.from_numpy(val_x.astype(np.float32))).mean()
            if check < val_loss:
                not_improved = 0
                val_loss = check
                best_state = copy.deepcopy(self.state_dict())
            else:
                not_improved += 1

            if epoch % 10 == 1:
                # print(scheduler.get_lr())
                self.load_state_dict(best_state)
                print(f"Epoch {epoch}; train loss = {-self.log_prob(torch.from_numpy(train_x.astype(np.float32))).mean():.3f}")
                print(f"Epoch {epoch}; val loss = {val_loss:.3f}")

            # Enforce early-stopping according to criterion
            if not_improved > threshold:
                print(f"Early stop after epoch: {epoch}")
                break

        self.load_state_dict(best_state)
        
        print(f"Final val loss: {-self.log_prob(torch.from_numpy(val_x.astype(np.float32))).mean():.3f}")



def get_nvp(x_dim, p_dim, units=128, mask_depth=4, seed=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    xp_dim = x_dim + p_dim

    # Scale and translation functions
    nets = lambda: nn.Sequential(nn.Linear(xp_dim, units), nn.LeakyReLU(), nn.Linear(units, units), 
                                 nn.LeakyReLU(), nn.Linear(units, xp_dim), nn.Tanh())

    nett = lambda: nn.Sequential(nn.Linear(xp_dim, units), nn.LeakyReLU(), nn.Linear(units, units), 
                                 nn.LeakyReLU(), nn.Linear(units, xp_dim))

    # parameter mask
    p_mask = p_dim*[1]

    # checkerboard mask for data
    mask1, mask2 = x_dim*[1], x_dim*[1]
    m_ = round(x_dim / 2.1) * [0]
    mask1[1::2] = m_
    mask2 = [mask2[i] - mask1[i] for i in range(x_dim)]

    mask = torch.from_numpy(np.array([[mask1 + p_mask],  
                                      [mask2 + p_mask]] * mask_depth).astype(np.float32))

    latent = distributions.MultivariateNormal(torch.zeros(x_dim), torch.eye(x_dim))

    return CondNVP(mask, nets, nett, latent, p_dim, x_dim, seed)



def simple_train(nvp, data, batch_size=10, epochs=60, seed=None):
    torch.manual_seed(seed)
    
    n_batches = int(len(data) / batch_size)
    
    optimizer = torch.optim.Adam([p for p in nvp.parameters() if p.requires_grad==True], lr=1e-3)
    scheduler = StepLR(optimizer, 10, 0.75)
    
    for epoch in range(epochs):
        for b in range(n_batches):
            batch = torch.from_numpy(data[b*batch_size:(b+1)*batch_size].astype(np.float32))
            loss = -nvp.log_prob(batch).mean()
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
        scheduler.step()
        np.random.shuffle(data)
        # print(scheduler.get_lr())
        if epoch % 10 == 1:
            # print(scheduler.get_lr())
            print(f"Epoch {epoch}; full loss = {-nvp.log_prob(torch.from_numpy(data.astype(np.float32))).mean()}")
    print(f"Final loss: {-nvp.log_prob(torch.from_numpy(data.astype(np.float32))).mean()}")









