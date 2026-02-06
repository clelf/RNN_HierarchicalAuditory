import torch.nn as nn # import RNN, Module, Sequential, GRU, LSTM
import torch
from torch.nn import functional as F

"""A module containing objective functions for RNN, VRNN, VAE, and ModuleNetwork models.

The Objective class provides a unified interface for computing losses across different
model architectures. It automatically detects the model type and applies the appropriate
loss function.

Supported model types:
- 'rnn' (SimpleRNN): GaussianNLL reconstruction loss
- 'vrnn' (VRNN): GaussianNLL reconstruction + KL divergence loss
- 'vae' (VAE): GaussianNLL reconstruction + KL divergence loss  
- 'module_network' (ModuleNetwork): Combines losses from observation and context modules
"""


class Objective:
    """Class handling the objective function for a model to be trained and tested.
    
    This class provides a model-aware loss computation interface. It detects the model
    type (RNN, VRNN, VAE, or ModuleNetwork) and applies the appropriate loss function.
    
    For ModuleNetwork models, it handles the composite loss from multiple sub-modules,
    where each sub-module can be either an RNN or VRNN.
    
    Parameters
    ----------
    loss_func : callable
        The base loss function to use for reconstruction (e.g., torch.nn.GaussianNLLLoss).
    
    Examples
    --------
    >>> objective = Objective(torch.nn.GaussianNLLLoss(reduction='mean'))
    >>> loss = objective.loss(model, target, model_output)
    """
    
    def __init__(self, loss_func, loss_func_ctx=None):
        self.loss_func = loss_func
        self.loss_func_ctx = loss_func_ctx # will be None if not used
    
    
    def loss(self, model, target, model_output, target_ctx=None, kappa=0.5, learning_objective='obs_ctx'):
        """Compute the loss for a given model and its output.
        
        Parameters
        ----------
        model : nn.Module
            The model (SimpleRNN, VRNN, VAE, or ModuleNetwork).
        target : torch.Tensor
            The target tensor to compare against.
        model_output : tuple or torch.Tensor
            The output from the model's forward pass.
            - For RNN: tensor of shape (batch, seq_len, output_dim)
            - For VRNN: tuple (outputs, mus_latent, logvars_latent, mus_prior, logvars_prior)
            - For VAE: tuple (x_output, mu_latent, logvar_latent)
            - For ModuleNetwork: tuple (obs_output, ctx_output)
        target_ctx : torch.Tensor, optional
            Target context labels for ModuleNetwork (default: None)
        kappa : float
            Weighting factor for ModuleNetwork combined loss (default: 0.5)
        learning_objective : str
            Learning objective for ModuleNetwork: 'obs', 'ctx', or 'obs_ctx' (default)
        
        Returns
        -------
        torch.Tensor
            The computed loss value.
        """
        model_name = model.name
        # In case of multi-module network, get the type of modules used
        if model_name == 'module_network':
            module_type = model.observation_module.name

        # Return appropriate loss based on model type
        if model_name == 'rnn':
            return self._rnn_loss(target, model_output)
        elif model_name == 'vrnn':
            return self._vrnn_loss(target, model_output)
        elif model_name == 'vae':
            return self._vae_loss(target, model_output)
        elif model_name == 'module_network':
            return self._module_network_loss(module_type, target_obs=target, target_ctx=target_ctx, 
                                             model_output=model_output, learning_objective=learning_objective, 
                                             kappa=kappa)
        else:
            raise ValueError(f"Unsupported model type: {model_name}")
    
    def _rnn_loss(self, target, x_output, loss_func=None):
        """Compute GaussianNLL loss for RNN models.
        
        The model outputs mean and (log)variance, which are processed with softplus
        to ensure positive variance.
        """
        out_estim_mu = x_output[..., [0]]
        out_estim_var = F.softplus(x_output[..., [1]]) + 1e-6
        
        if loss_func is None:
            loss_func = self.loss_func
        return loss_func(out_estim_mu, target, out_estim_var)
    
    def _reconstruction_loss(self, target, x_output, loss_func=None):
        """Compute reconstruction loss with softplus variance transformation."""
        out_estim_mu = x_output[:, :, [0]]
        out_estim_var = F.softplus(x_output[:, :, [1]]) + 1e-6

        if loss_func is None:
            loss_func = self.loss_func
        recon_loss = loss_func(out_estim_mu, target, out_estim_var)
        return recon_loss
    
    def _vrnn_loss(self, target, forward_output, loss_func=None):
        """Compute VRNN loss: reconstruction + KL divergence.
        
        The VRNN forward pass returns:
        (outputs, mus_latent, logvars_latent, mus_prior, logvars_prior)
        """
        x_output, mu_latent, logvar_latent, mu_prior, logvar_prior = forward_output
        
        # Reconstruction loss
        recon_loss = self._reconstruction_loss(target, x_output, loss_func=loss_func)
        
        # KL divergence between two Gaussians per timestep & batch
        # KL(q||p) closed form between N(mu_latent, var_latent) and N(mu_prior, logvar_prior)
        kl_element = (
            logvar_prior - logvar_latent
            + (logvar_latent.exp() + (mu_latent - mu_prior).pow(2)) / logvar_prior.exp()
            - 1
        )
        
        # KL loss - use mean to match the scale of recon_loss (which uses reduction='mean')
        kl_loss = 0.5 * torch.mean(kl_element) # NOTE: mean instead of sum to match recon_loss scale with reduction='mean'
        
        return recon_loss + kl_loss
    
    def _vae_loss(self, target, forward_output, loss_func=None):
        """Compute VAE loss: reconstruction + KL divergence.
        
        The VAE forward pass returns: (x_output, mu_latent, logvar_latent)
        """
        x_output, mu_latent, logvar_latent = forward_output
        
        # Reconstruction loss
        recon_loss = self._reconstruction_loss(target, x_output, loss_func=loss_func)
        
        # KL divergence: see Appendix B from VAE paper
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # - 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + logvar_latent - mu_latent.pow(2) - logvar_latent.exp()) # NOTE: probably mean to match recon_loss scale with reduction='mean'
        
        return recon_loss + kl_loss
    
    def _module_network_loss(self, module_type, target_obs, target_ctx, model_output, learning_objective='obs_ctx', kappa=0.5):
        """Compute ModuleNetwork loss based on learning objective.
        
        The ModuleNetwork forward pass returns: (obs_output, ctx_output), from which 
        each can return either RNN or VRNN outputs. (see respective loss functions)

        Parameters
        ----------
        module_type : str
            The type of modules used ('rnn' or 'vrnn').
        target_obs : torch.Tensor
            The target tensor for the observation module.
        target_ctx : torch.Tensor
            The target tensor for the context module (class labels for classification).
        model_output : tuple
            The output from the ModuleNetwork's forward pass (obs_output, ctx_output).
        learning_objective : str
            Learning objective determining which loss components to use:
            - 'obs': Train observation module only (hidden process prediction)
            - 'ctx': Train context module only (context inference)
            - 'obs_ctx': Train both modules with combined loss (default)
        kappa : float, optional
            Weighting factor for combining observation and context losses (default=0.5).
            Only used when learning_objective='obs_ctx'.
            
        Returns
        -------
        torch.Tensor
            The computed loss value.
        """

        obs_output, ctx_output = model_output
        
        # Observation module's loss (GaussianNLL for predicting next observation)
        if learning_objective in ['obs', 'obs_ctx']:
            if module_type == 'rnn':
                obs_loss = self._rnn_loss(target_obs, obs_output)
            elif module_type == 'vrnn':
                obs_loss = self._vrnn_loss(target_obs, obs_output)
        
        # Context module's loss (CrossEntropy for context classification)
        if learning_objective in ['ctx', 'obs_ctx']:
            # ctx_output shape: (batch, seq_len, n_contexts) - logits for each context class
            # target_ctx shape: (batch, seq_len) - integer class labels
            # Reshape for CrossEntropyLoss: (batch*seq_len, n_contexts) vs (batch*seq_len,)
            batch_size, seq_len, n_contexts = ctx_output.shape
            ctx_output_flat = ctx_output.reshape(-1, n_contexts)
            target_ctx_flat = target_ctx.reshape(-1)
            ctx_loss = self.loss_func_ctx(ctx_output_flat, target_ctx_flat)
        
        if learning_objective == 'obs':
            loss = obs_loss
        elif learning_objective == 'ctx':
            loss = ctx_loss
        elif learning_objective == 'obs_ctx':
            # Combine losses
            loss = kappa * obs_loss + (1 - kappa) * ctx_loss
        
        return loss


# Standalone helper functions (kept for backward compatibility)

def reconstruction_loss(x_target, x_output, loss_func):
    """Compute reconstruction loss with softplus variance transformation."""
    out_estim_mu = x_output[:, :, [0]]
    out_estim_var = F.softplus(x_output[:, :, [1]]) + 1e-6

    recon_loss = loss_func(out_estim_mu, x_target, out_estim_var)
    return recon_loss


def kl_loss_vrnn(mu_latent, logvar_latent, mu_prior, logvar_prior):
    """KL divergence between two Gaussians for VRNN."""
    kl_element = (
        logvar_prior - logvar_latent
        + (logvar_latent.exp() + (mu_latent - mu_prior).pow(2)) / logvar_prior.exp()
        - 1
    )
    kl_loss = 0.5 * torch.mean(kl_element)
    return kl_loss


def kl_loss_vae(mu_latent, logvar_latent):
    """KL divergence for VAE (prior is standard normal)."""
    kl_loss = -0.5 * torch.mean(1 + logvar_latent - mu_latent.pow(2) - logvar_latent.exp())
    return kl_loss