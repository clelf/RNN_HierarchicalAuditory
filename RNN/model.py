import torch.nn as nn # import RNN, Module, Sequential, GRU, LSTM
import torch
from torch.nn import functional as F
from functools import partial
import numpy as np

# TODO: handle autograd
# TODO: learn contexts representation as another network (later)


class SimpleRNN(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.name = 'rnn'
        self.input_dim = config['input_dim']
        self.output_dim = config['output_dim']
        self.hidden_dim = config['hidden_dim']
        self.n_layers = config['n_layers']
        self.device = config.get('device', torch.device('cpu'))

        self.rnn = nn.GRU(self.input_dim, self.hidden_dim, self.n_layers, batch_first=True, device=self.device) # expect x of size (batch_size, seq_len, input_dim)
        self.output_layer = nn.Linear(self.hidden_dim, self.output_dim, device=self.device)


    def init_hidden(self, batch_size):
        return torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device)

    def forward(self, x, hx=None, return_hidden=False):
        if hx is None:
            batch_size = x.size(0)         # batch_size, seq_len, input_dim = x.size()
            hx = self.init_hidden(batch_size)

        output_last, hx_new = self.rnn(x, hx)  # output_last is the sequence of final hidden states after the last layer / hx_new is the final hidden state of each layer
        output_seq = self.output_layer(output_last)

        if return_hidden:
            return output_seq, hx_new
        return output_seq
    
    def loss(self, target, x_output, loss_func): # torch.nn.GaussianNLLLoss()
        out_estim_mu    = x_output[..., [0]]
        out_estim_var   = F.softplus(x_output[..., [1]]) + 1e-6 # Ensure the variance is positive
        
        return loss_func(out_estim_mu, target, out_estim_var)


class VAE(nn.Module):
    
    def __init__(self, config): # activation_fn?
        """Variational Autoencoder
        Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        https://arxiv.org/abs/1312.6114

        Note (# TODO): hidden_units intended for later addition of deeper layers in encoder and decoder 

        Parameters
        ----------
        config : dict
            Configuration dictionary with keys:
            - input_dim: Input dimension
            - output_dim: Output dimension
            - latent_dim: Latent dimension
            - hidden_units_dim (optional): Hidden units dimension (defaults to latent_dim)
            - device (optional): Device to use (defaults to 'cpu')
        """

        super().__init__()

        self.name = 'vae'
        self.input_dim = config['input_dim']
        self.output_dim = config['output_dim']
        self.latent_dim = config['latent_dim']
        self.hidden_units_dim = config.get('hidden_units_dim', self.latent_dim)
        self.device = config.get('device', torch.device('cpu'))

        # Encoder: input_dim -> hidden_units_dim (intermediate representation)
        self.encoder = self.build_encoder(self.input_dim, self.hidden_units_dim)
        # Decoder: latent_dim -> output_dim
        self.decoder = self.build_decoder(self.latent_dim, self.output_dim)

        # Reparametrization: hidden_units_dim -> latent_dim
        self.fc_mu_latent = self.build_fc_mu(self.hidden_units_dim, self.latent_dim)
        self.fc_logvar_latent = self.build_fc_logvar(self.hidden_units_dim, self.latent_dim)

    def build_decoder(self, in_dim, out_dim):
        # No activation on output layer - mean can be any real value,
        # variance is handled by softplus in loss function
        return nn.Linear(in_dim, out_dim)

    def build_encoder(self, in_dim, out_dim):
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
        return mu + eps * std 
    

    def forward(self, x):
        # Encode (infer): x -> hidden representation
        h = self.encoder(x)

        # Reparametrize the latent variable: hidden -> latent
        mu_latent = self.fc_mu_latent(h)
        logvar_latent = self.fc_logvar_latent(h)
        z = self.reparameterize(mu_latent, logvar_latent)

        # Decode (reconstruct / generate): latent -> output
        x_ = self.decoder(z)

        return x_, mu_latent, logvar_latent


    def loss(self, x_target, forward_output, loss_func):
        x_output, mu_latent, logvar_latent = forward_output
        return self.vae_loss(x_target, x_output, mu_latent, logvar_latent, loss_func=loss_func)


    def vae_loss(self, x_target, x_output, mu_latent, logvar_latent, loss_func):
        # (from pytorch's VAE implementation: https://github.com/pytorch/examples/blob/main/vae/main.py#L85)
        #Reconstruction + KL divergence losses summed over all elements and batch

        # TODO: get rid of sigmoid and softplus if taken care of in decoder and encoder
        out_estim_mu    = x_output[:, :, [0]]
        out_estim_var   = F.softplus(x_output[:, :, [1]]) + 1e-6 # Ensure the variance is stricly positive
        recon_loss = loss_func(out_estim_mu, x_target, out_estim_var)
        
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # - 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = - 0.5 * torch.sum(1 + logvar_latent - mu_latent.pow(2) - logvar_latent.exp())

        return recon_loss + kl_loss
    


class VRNN(VAE): 
    def __init__(self, config):
        # input_dim, output_dim, latent_dim, hidden_units_dim=None
        super().__init__(config)
        
        self.name = 'vrnn'
        self.rnn_hidden_states_dim = config['rnn_hidden_states_dim']
        self.rnn_n_layers = config['rnn_n_layers']
        self.phi_x_dim = config['phi_x_dim'] # just h_dim most of the times
        self.phi_z_dim = config['phi_z_dim'] # h_dim
        self.phi_prior_dim = config['phi_prior_dim'] # h_dim

        # NOTE: in VRNN paper all phi functions have 4 layers combined with ReLU

        # Encoder and decoder
        # Encoder maps to hidden_units_dim, then fc_mu/fc_logvar map to latent_dim
        self.encoder = self.build_encoder(self.phi_x_dim + self.rnn_hidden_states_dim, self.hidden_units_dim) 
        self.decoder = self.build_decoder(self.phi_z_dim + self.rnn_hidden_states_dim, self.output_dim) # shared with standard RNN (S.4 Models)

        # Feature extractors 
        self.phi_x = self.build_feature_extractor(self.input_dim, self.phi_x_dim)  # shared with standard RNN (S.4 Models)
        self.phi_z = self.build_feature_extractor(self.latent_dim, self.phi_z_dim)
        self.phi_prior = self.build_feature_extractor(self.rnn_hidden_states_dim, self.phi_prior_dim)
        self.mu_prior = self.build_fc_mu(self.phi_prior_dim, self.latent_dim)
        self.logvar_prior = self.build_fc_logvar(self.phi_prior_dim, self.latent_dim)


        # RNN part
        # self.recurrence = SimpleRNN(input_dim=self.phi_x_dim + self.phi_z_dim, output_dim=..., hidden_dim=self.rnn_hidden_states_dim, n_layers=self.rnn_n_layers, batch_size=batch_size)
        self.rnn = nn.GRU(input_size=self.phi_x_dim + self.phi_z_dim, hidden_size=self.rnn_hidden_states_dim, num_layers=self.rnn_n_layers, batch_first=True, device=self.device)
        # self.output_layer = nn.Linear(self.rnn_hidden_states_dim, input_dim, device=device)


    def build_feature_extractor(self, in_dim, out_dim):
        module = nn.Sequential( 
            nn.Linear(in_dim, out_dim), 
            nn.ReLU()
        )
        return module


    def forward(self, x):
        
        batch_size, seq_len, input_dim = x.size()
        if input_dim!=self.input_dim: raise ValueError("Incorrect dimensions for input x")
        h_prev  = torch.zeros(self.rnn_n_layers, batch_size, self.rnn_hidden_states_dim, device=self.device) # self.n_layers, batch_size, self.hidden_dim
        outputs = []
        mus_latent = []
        logvars_latent = []
        mus_prior = []
        logvars_prior = []

        for t in range(seq_len):            
            x_t = x[:, t, :] # x_t has shape (batch_size, input_dim)
            
            # Feature extraction (compute once, use twice)
            phi_x_t = self.phi_x(x_t)
            
            # Encode 
            z_t = self.encoder(torch.cat([phi_x_t, h_prev[-1]], dim=-1)) # Eq. 9

            # Reparametrize the latent variable
            mu_latent = self.fc_mu_latent(z_t)
            logvar_latent = self.fc_logvar_latent(z_t)
            mus_latent.append(mu_latent)
            logvars_latent.append(logvar_latent)
            z_t = self.reparameterize(mu_latent, logvar_latent)

            # Prior
            z_prior = self.phi_prior(h_prev[-1]) # Eq. 5
            mu_prior = self.mu_prior(z_prior)
            logvar_prior = self.logvar_prior(z_prior)
            mus_prior.append(mu_prior)
            logvars_prior.append(logvar_prior)

            # Feature extraction for z
            phi_z_t = self.phi_z(z_t)

            # Decode
            x_t_out = self.decoder(torch.cat([phi_z_t, h_prev[-1]], dim=-1)) # Eq. 6
            outputs.append(x_t_out)
            
            # Recurrence
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], dim=-1).unsqueeze(1), h_prev) # Eq. 7 # since we're considering (B, T, feature_dim) with T=1, unsqueeze extra dimension at dim=1 to make up for T=1
            h_prev = h

        # self.rnn_hidden_states = self.recurrence.hidden_states

        # Stack across time
        outputs = torch.stack(outputs, dim=1)             # [B, T, input_dim]
        mus_latent = torch.stack(mus_latent, dim=1)               # [B, T, latent_dim]
        logvars_latent = torch.stack(logvars_latent, dim=1)       # [B, T, latent_dim]
        mus_prior = torch.stack(mus_prior, dim=1)         # [B, T, latent_dim]
        logvars_prior = torch.stack(logvars_prior, dim=1) # [B, T, latent_dim]      

        return outputs, mus_latent, logvars_latent, mus_prior, logvars_prior
    
    
    def loss(self, x_target, forward_output, loss_func):
        x_output, mu_latent, logvar_latent, mu_prior, logvar_prior = forward_output
        return self.vrnn_loss(x_target, x_output, mu_latent, logvar_latent, mu_prior, logvar_prior, loss_func=loss_func)
    

    def vrnn_loss(self, x_target, x_output, mu_latent, logvar_latent, mu_prior, logvar_prior, loss_func): # KL divergence
        out_estim_mu = x_output[:, :, [0]]
        out_estim_var = F.softplus(x_output[:, :, [1]]) + 1e-6

        recon_loss = loss_func(out_estim_mu, x_target, out_estim_var) # From Gemini2.5: recon_loss = 0.5 * torch.sum(decoder_logvar + (x - decoder_mu)**2 / torch.exp(decoder_logvar))
        
        # KL divergence between two Gaussians per timestep & batch
        # KL(q||p) closed form between N(mu_latent, var_latent) and N(mu_prior, logvar_prior)
        kl_element = (
            logvar_prior - logvar_latent
            + (logvar_latent.exp() + (mu_latent - mu_prior).pow(2)) / logvar_prior.exp()
            - 1
        )

        # KL loss - use mean to match the scale of recon_loss (which uses reduction='mean')
        kl_loss = 0.5 * torch.mean(kl_element)

        return recon_loss + kl_loss # Eq. 11 : - KL divergence + log posterior


class ModuleNetwork(nn.Module):
    """A multi-module neural network architecture with observation and context modules."""
    
    def __init__(self, config):
        super(ModuleNetwork, self).__init__()

        self.name = 'module_network'  # For Objective class to identify model type

        # Initialize modules (observation module, context module, rule module)
        self.observation_module = SimpleRNN({
            'input_dim': config['observation_module']['input_dim'],
            'output_dim': config['observation_module']['output_dim'],
            'hidden_dim': config['observation_module']['rnn_hidden_dim'],
            'n_layers': config['observation_module']['rnn_n_layers'],
            'device': config.get('device', torch.device('cpu'))
        }) # can be replaced with VRNN
        
        self.context_module = SimpleRNN({
            'input_dim': config['context_module']['input_dim'],
            'output_dim': config['context_module']['output_dim'],
            'hidden_dim': config['context_module']['rnn_hidden_dim'],
            'n_layers': config['context_module']['rnn_n_layers'],
            'device': config.get('device', torch.device('cpu'))
        })
        # self.rule_module = ...
        
        
        self.readout_obs2ctx = self.inter_module_readout(
            in_dim=config['observation_module']['output_dim'],
            out_dim=config['context_module']['input_dim'],
            bottleneck_dim=config['observation_module']['bottleneck_dim']
        )
        
        self.readout_ctx2obs = self.inter_module_readout(
            in_dim=config['context_module']['output_dim'],
            out_dim=config['observation_module']['input_dim'],
            bottleneck_dim=config['context_module']['bottleneck_dim']
        )

        # self.readout_obs2ctx = nn.Sequential(
        #     nn.Linear(config['observation_module']['output_dim'],
        #               config['observation_module']['bottleneck_dim']),
        #     nn.ReLU(),
        #     nn.Linear(config['observation_module']['bottleneck_dim'],
        #               config['context_module']['input_dim']),
        # )
        # self.readout_ctx2obs = nn.Sequential(
        #     nn.Linear(config['context_module']['output_dim'],
        #               config['context_module']['bottleneck_dim']),
        #     nn.ReLU(),
        #     nn.Linear(config['context_module']['bottleneck_dim'],
        #               config['observation_module']['input_dim']),
        # )

    def inter_module_readout(self, in_dim, out_dim, bottleneck_dim):
        return nn.Sequential(
            nn.Linear(in_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, out_dim),
        )


    def forward(self, x):
        # First pass: observation module processes input
        obs_output, obs_hx = self.observation_module(x, return_hidden=True)  # get hidden state
        # obs_output, obs_hx = self.observation_module(x, hx=obs_hx, return_hidden=True) # Example of "thinking"
        
        # Compress and send to context module
        enc_output = self.readout_obs2ctx(obs_output)
        context_output = self.context_module(enc_output)  # context module has its own independent hidden state
            
        # Feedback: compress context output
        enc_feedback = self.readout_ctx2obs(context_output)
        
        # Second pass: combine context feedback and previous hidden state
        informed_obs_output = self.observation_module(enc_feedback, hx=obs_hx)
        
        return informed_obs_output, context_output
    






if __name__ == '__main__':

    DEVICE = torch.device('cpu')

    # THIS IS JUST FOR DEBUG TESTING

    # Define model dimensions
    batch_size=1
    # Input and output dimensions
    input_dim   = 1 # number of features in each observation: 1 --> need to write x = x.unsqueeze(-1)  at one point
    dim_out_obs = 2 # from goin: learn the sufficient statistics mu and var
    # RNN bit
    output_dim      = input_dim
    rnn_hidden_dim  = 64 # from goin. could also take varying values like [2**n for n in range(1, 9)]
    rnn_n_layers    = 1 # from goin
    # VRNN specific
    latent_dim      = 16 # needs to be < input_dim :thinks: --> in a VAE, yes, but in a VRNN, needs to be <input_dim + rnn_hidden_dim
    phi_x_dim       = rnn_hidden_dim
    phi_z_dim       = rnn_hidden_dim
    phi_prior_dim   = rnn_hidden_dim

    # Create config dictionaries
    rnn_config = {
        'input_dim': input_dim,
        'output_dim': dim_out_obs,
        'hidden_dim': rnn_hidden_dim,
        'n_layers': rnn_n_layers,
        'device': DEVICE
    }
    
    vrnn_config = {
        'input_dim': input_dim,
        'output_dim': dim_out_obs,
        'latent_dim': latent_dim,
        'phi_x_dim': phi_x_dim,
        'phi_z_dim': phi_z_dim,
        'phi_prior_dim': phi_prior_dim,
        'rnn_hidden_states_dim': rnn_hidden_dim,
        'rnn_n_layers': rnn_n_layers,
        'device': DEVICE
    }
        
    # Define model
    rnn     = SimpleRNN(rnn_config)
    vrnn    = VRNN(vrnn_config)
    
    y = [1., 2., 1., 2.]
    # Shape should be (batch_size, seq_len, input_dim) since batch_first=True
    data = torch.tensor(y).view(batch_size, len(y), input_dim)
    rnn(data)
    vrnn(data)
    pass