import torch.nn as nn # import RNN, Module, Sequential, GRU, LSTM
import torch
from torch.nn import functional as F
from functools import partial
import numpy as np


# TODO: models should infer observations as distributions (to account for uncertainty).
# TODO: make sure about what "prediction" refers to: based on data up until current time step, current time step estimation, or next time step prediction
# TODO: learn contexts representation as another network


class SimpleRNN(nn.Module):
    def __init__(self, x_dim, output_dim, hidden_dim, n_layers, loss_func=nn.BCELoss, device=torch.device('cpu')):
        super().__init__()

        self.name = 'rnn'
        self.x_dim = x_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.rnn = nn.GRU(self.x_dim, self.hidden_dim, self.n_layers, batch_first=False, device=DEVICE) # expect x of size (N_batch, seq_len, input_dim)
        self.output_layer = nn.Linear(self.hidden_dim, self.output_dim, device=DEVICE)

        self.loss_func = loss_func

    def forward(self, x, hx=None):
        output_seq, h_t = self.rnn(x, hx)  # output_seq is the sequence of final hidden states after the last layer / h_last is the final hidden state of each layer
        output = self.output_layer(output_seq)
        return output, h_t
    
    def loss(self, out_estim, y_target, loss_func=nn.BCELoss, **kwargs):
        if loss_func is None: loss_func=self.loss_func
        if kwargs: loss_func = partial(loss_func, **kwargs) # In case more arguements are passed, depending on the loss_func considered
        if self.output_dim>1: 
            # If the model is learning sufficient stats of the distribution the estimation rather than the estimation,
            # then out_estim contains 2 variables: mu and var
            return loss_func(out_estim[:,:,0], y_target, out_estim[:,:,1])
        else:
            return loss_func(out_estim, y_target)


class VAE(nn.Module):
    
    def __init__(self, x_dim, output_dim, latent_dim, hidden_units_dim=None): # activation_fn?
        """Variational Autoencoder
        Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        https://arxiv.org/abs/1312.6114

        Note (# TODO): hidden_units intended for later addition of deeper layers in encoder and decoder 

        Parameters
        ----------
        x_dim : _type_
            _description_
        latent_dim : _type_
            _description_
        """

        super().__init__()

        self.name = 'vae'
        self.x_dim = x_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.hidden_units_dim = hidden_units_dim

        # Encoder and decoder
        self.encoder = self.build_encoder(self.x_dim, self.latent_dim, self.hidden_units_dim)
        self.decoder = self.build_decoder(self.latent_dim, self.output_dim, self.hidden_units_dim)

        # Reparametrization functions
        self.fc_mu = self.build_fc_mu(self.latent_dim, self.latent_dim)
        self.fc_logvar = self.build_fc_logvar(self.latent_dim, self.latent_dim)

    def build_decoder(self, in_dim, out_dim, hidden_units_dim=None):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU()
        )

    def build_encoder(self, in_dim, out_dim, hidden_units_dim=None):
        # if self.variational: in_dim = in_dim * 2 # Because needs to expand the reparemtrization # UNSURE, CHECK; IF UNCOMMENT, DIVIDE mu and var's output dim by 2
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU()
        )
    
    def build_fc_mu(self, in_dim, out_dim):
        return nn.Linear(in_dim, out_dim)
    
    def build_fc_logvar(self, in_dim, out_dim):
        return nn.Linear(in_dim, out_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std # TODO: check if should be point-wise multiplication?
    

    def forward(self, x):
        # Encode (infer)
        z = self.encoder(x, self.x_dim, self.hidden_units_dim)

        # Reparametrize the latent variable
        mu = self.fc_mu(z)
        logvar = self.fc_var(z)
        z = self.reparameterize(mu, logvar)

        # Decode (reconstruct / generate)
        x_ = self.decoder(z, self.hidden_units_dim, self.x_dim)

        return x_, mu, logvar


    def loss(self, x_output, x, mu, logvar, loss_func=nn.BCELoss, **kwargs):
        return self.vae_loss_function(self, x_output, x, mu, logvar, loss_func=loss_func, **kwargs)


    def vae_loss_function(self, x_output, x, mu, logvar, loss_func=nn.BCELoss, **kwargs):
        # (from pytorch's VAE implementation: https://github.com/pytorch/examples/blob/main/vae/main.py#L85)
        #Reconstruction + KL divergence losses summed over all elements and batch
        
        if kwargs: loss_func = partial(loss_func, **kwargs) # In case more arguements are passed, depending on the loss_func considered
        
        if self.output_dim>1: 
            recon_loss = loss_func(x_output[:, :, 0], x.view(-1, self.x_dim), x_output[:, :, 1], **kwargs) # TODO: verify if want to keep BCE over MSE
        else:
            x_output = torch.sigmoid(x_output)
            recon_loss = loss_func(x_output, x.view(-1, self.x_dim), **kwargs)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return recon_loss + kl_loss
    


class VRNN(VAE): # Or rather, inherit from VAE

    def __init__(self, x_dim, output_dim, latent_dim, phi_x_dim, phi_z_dim, phi_prior_dim, rnn_hidden_states_dim, rnn_n_layers, hidden_units_dim=None):
        """_summary_

        Parameters
        ----------
        x_dim : _type_
            _description_
        hidden_dim : _type_
            The length of the time sequence
        latent_dim : _type_
            _description_
        n_layers : _type_
            _description_
        bias : bool, optional
            _description_, by default False
        """

        super().__init__(x_dim, output_dim, latent_dim, hidden_units_dim=hidden_units_dim)
        
        self.name = 'vrnn'
        self.rnn_hidden_states_dim = rnn_hidden_states_dim
        self.rnn_n_layers = rnn_n_layers
        self.phi_x_dim = phi_x_dim # just h_dim most of the times
        self.phi_z_dim = phi_z_dim # h_dim
        self.phi_prior_dim = phi_prior_dim # h_dim

        # Encoder and decoder
        self.encoder = self.build_encoder(self.phi_x_dim + self.rnn_hidden_states_dim, self.latent_dim) 
        self.decoder = self.build_decoder(self.phi_z_dim + self.rnn_hidden_states_dim, self.output_dim) # shared with standard RNN (S.4 Models)


        # Feature extractors 
        self.phi_x = self.build_feature_extractor(self.x_dim, self.phi_x_dim)  # shared with standard RNN (S.4 Models)
        self.phi_z = self.build_feature_extractor(self.latent_dim, self.phi_z_dim)
        self.phi_prior = self.build_feature_extractor(self.rnn_hidden_states_dim, self.phi_prior_dim) # rnn_hidden_states_dim	phi_prior_dim
        self.prior_mu = self.build_fc_mu(self.phi_prior_dim, self.latent_dim)
        self.prior_logvar = self.build_fc_logvar(self.phi_prior_dim, self.latent_dim)


        # RNN part
        self.recurrence = SimpleRNN(self.phi_x_dim + self.phi_z_dim, self.rnn_hidden_states_dim,  self.rnn_n_layers)

        # self.rnn = nn.GRU(self.x_dim + self.latent_dim, self.rnn_hidden_states_dim, self.rnn_n_layers, batch_first=True, device=device)
        # self.output_layer = nn.Linear(self.rnn_hidden_states_dim, x_dim, device=device)


    def build_feature_extractor(self, in_dim, out_dim):
        module = nn.Sequential( 
            nn.Linear(in_dim, out_dim), 
            nn.ReLU()
        )
        return module


    # def reccurrence(self, x, hx=None):
    #     # Defined like SimpleRNN's forward
    #     # output_seq is the sequence of final hidden states after the last layer / h_last is the final hidden state of each layer
    #     output_seq, h_t = self.rnn(x, hx)  
    #     output = self.output_layer(output_seq)
    #     return output, h_t
   

    def forward(self, x):

        seq_len, batch_size, x_dim = x.size()
        if x_dim!=self.x_dim: raise ValueError("Incorrect dimensions for input x")

        hidden_states = torch.zeros(self.rnn_n_layers, batch_size, self.rnn_hidden_states_dim)
        h_prev = hidden_states
        h_t = h_prev
        # h = torch.zeros(B, self.rnn_hidden_states_dim, device=x.device)
        outputs = []
        rnn_outputs = []
        prior_mus = []
        prior_logvars = []

        for t in range(seq_len):            
            x_t = x[t, :, :].unsqueeze(0) # x_t has shape (1, batch_size, x_dim)
            
            # Encode 
            z_t = self.encoder(torch.cat([self.phi_x(x_t), h_prev], dim=-1)) # Eq. 9

            # Reparametrize the latent variable
            mu_seq = self.fc_mu(z_t)
            logvar_seq = self.fc_logvar(z_t)
            z_t = self.reparameterize(mu_seq, logvar_seq)

            # Prior
            z_prior = self.phi_prior(h_prev) # Eq. 5
            prior_mu = self.prior_mu(z_prior)
            prior_logvar = self.prior_logvar(z_prior)
            prior_mus.append(prior_mu)
            prior_logvars.append(prior_logvar)


            # Decode
            x_t_ = self.decoder(torch.cat([self.phi_z(z_t), h_prev], dim=-1)) # Eq. 6
            outputs.append(x_t_)
            
            # Recurrence
            rnn_output, h_t = self.recurrence(torch.cat([self.phi_x(x_t), self.phi_z(z_t)], dim=-1), h_prev) # Eq. 7
            rnn_outputs.append(rnn_output)
            h_prev = h_t


        # Stack across time
        outputs = torch.stack(outputs, dim=0)             # [T, B, x_dim]
        mu_seq = torch.stack(mu_seq, dim=0)               # [T, B, latent_dim]
        logvar_seq = torch.stack(logvar_seq, dim=0)       # [T, B, latent_dim]
        prior_mus = torch.stack(prior_mus, dim=0)         # [T, B, latent_dim]
        prior_logvars = torch.stack(prior_logvars, dim=0) # [T, B, latent_dim]
                

        return outputs, mu_seq, logvar_seq, prior_mus, prior_logvars
    
    
    def loss(self, x_output, x, mu, logvar, mu_prior, logvar_prior, loss_func=nn.BCELoss, **kwargs):
        return self.vrnn_loss(x_output, x, mu, logvar, mu_prior, logvar_prior, loss_func=loss_func, **kwargs)
    

    def vrnn_loss(self, x_output, x, mu, logvar, mu_prior, logvar_prior, loss_func=nn.BCELoss, **kwargs): # KL divergence
        
        if kwargs: loss_func = partial(loss_func, **kwargs) # In case more arguements are passed, depending on the loss_func considered
        
        if self.output_dim>1: 
            recon_loss = loss_func(x_output[:, :, 0], x.view(-1, self.x_dim), x_output[:, :, 1], **kwargs) # TODO: verify if want to keep BCE over MSE
        else:
            x_output = torch.sigmoid(x_output)
            recon_loss = loss_func(x_output, x.view(-1, self.x_dim), **kwargs)

        # KL divergence between two Gaussians per timestep & batch
        # KL(q||p) closed form between N(mu, var) and N(mu_prior, logvar_prior)
        kl_element = (
            logvar_prior - logvar
            + (logvar.exp() + (mu - mu_prior).pow(2)) / logvar_prior.exp()
            - 1
        )

        kl_loss = 0.5 * torch.sum(kl_element) # TODO: check if dim=1 is needed for torch.sum(..., dim=1)

        return recon_loss + kl_loss


if __name__ == '__main__':

    DEVICE = torch.device('cpu')


    # Define model dimensions
    batch_size=1
    # Input and output dimensions
    input_dim   = 1 # number of features in each observation: 1 --> need to write x = x.unsqueeze(-1)  at one point
    dim_out_obs = 2 # from goin: learn the sufficient statistics mu and var
    dim_out_ctx = 3 # from goin. nb_max_ctx + 1 = 2+1
    # RNN bit
    output_dim      = input_dim
    rnn_hidden_dim  = 64 # from goin. could also take varying values like [2**n for n in range(1, 9)]
    rnn_n_layers    = 1 # from goin
    # VRNN specific
    latent_dim      = 16 # needs to be < input_dim :thinks: --> in a VAE, yes, but in a VRNN, needs to be <input_dim + rnn_hidden_dim
    phi_x_dim       = rnn_hidden_dim
    phi_z_dim       = rnn_hidden_dim
    phi_prior_dim   = rnn_hidden_dim

        
    # Define model
    rnn     = SimpleRNN(x_dim=input_dim, hidden_dim=rnn_hidden_dim, n_layers=rnn_n_layers)
    vrnn    = VRNN(x_dim=input_dim, latent_dim=latent_dim, phi_x_dim=phi_x_dim, phi_z_dim=phi_z_dim, phi_prior_dim=phi_prior_dim, rnn_hidden_states_dim=rnn_hidden_dim, rnn_n_layers=rnn_n_layers)
    
    y = [1., 2., 1., 2.]
    data = torch.tensor(y).view(len(y), batch_size, input_dim)
    rnn(data)
    vrnn(data)
    pass