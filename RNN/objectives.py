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
    
    def __init__(self, loss_func_obs, loss_func_ctx=None):
        self.loss_func_obs = loss_func_obs
        self.loss_func_ctx = loss_func_ctx # will be None if not used
    
    
    def loss(self, model, target, model_output, target_ctx=None, target_dpos=None, target_rule=None, kappa=0.5, learning_objective='all'):
        """Compute the loss for a given model and its output.
        
        Parameters
        ----------
        model : nn.Module
            The model (SimpleRNN, VRNN, VAE, ModuleNetwork or PopulationNetwork).
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
            Learning objective for ModuleNetwork: 'obs', 'ctx', or 'all' (default)
        
        Returns
        -------
        torch.Tensor
            The computed loss value.
        """
        model_name = model.name
        # In case of multi-module network, get the type of modules used
        if model_name == 'module_network' or model_name == 'population_network':
            module_type = model.observation_module.name

        # Return appropriate loss based on model type
        if model_name == 'rnn':
            return self._regression_loss(target, model_output, model_type='rnn')
        elif model_name == 'vrnn':
            return self._regression_loss(target, model_output, model_type='vrnn')
        elif model_name == 'vae':
            return self._vae_loss(target, model_output)
        elif model_name == 'module_network':
            return self._module_network_loss(module_type, target_obs=target, target_ctx=target_ctx, 
                                             model_output=model_output, learning_objective=learning_objective, 
                                             kappa=kappa)
        elif model_name == 'population_network':
            return self._population_network_loss(module_type, target_obs=target, target_ctx=target_ctx,
                                                 target_dpos=target_dpos, target_rule=target_rule,
                                                 model_output=model_output, learning_objective=learning_objective)
        else:
            raise ValueError(f"Unsupported model type: {model_name}")
    
    def _regression_loss(self, target, output, model_type='rnn', loss_func=None):
        """Compute regression loss (reconstruction ± KL divergence).
        
        For RNN-based modules:
            - Assuming output stores mean and (log)variance
            - Applies GaussianNLL loss
        
        For VRNN-based modules:
            - Output is tuple: (x_output, mu_latent, logvar_latent, mu_prior, logvar_prior)
            - Applies GaussianNLL reconstruction + KL regularization on latent
        
        Parameters
        ----------
        target : torch.Tensor
            Target values for regression
        output : torch.Tensor or tuple
            Model output (tensor for RNN, tuple for VRNN)
        model_type : str
            'rnn' or 'vrnn'
        loss_func : callable, optional
            Loss function (defaults to self.loss_func_obs)
        """
        if loss_func is None:
            loss_func = self.loss_func_obs
        
        if model_type == 'rnn':
            # RNN output: shape (..., 2) with [mean, logvar]
            return self._reconstruction_loss(target, output, loss_func=loss_func)
        
        elif model_type == 'vrnn':
            # VRNN output: tuple (x_output, mu_latent, logvar_latent, mu_prior, logvar_prior)
            x_output, mu_latent, logvar_latent, mu_prior, logvar_prior = output
            
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
            kl_loss = 0.5 * torch.mean(kl_element)
            
            return recon_loss + kl_loss
        
        else:
            raise ValueError(f"Unsupported model_type for regression: {model_type}")
    
    def _classification_loss(self, target, output):
        """Compute classification loss.
        
        Handles both flat and sequence-based inputs:
        - Flat: (batch*seq_len, n_classes) vs (batch*seq_len,)
        - Sequence: (batch, seq_len, n_classes) vs (batch, seq_len)
        
        If output is 3D (batch, seq_len, n_classes), reshapes to (batch*seq_len, n_classes)
        and target from (batch, seq_len) to (batch*seq_len,).
        
        Parameters
        ----------
        target : torch.Tensor
            Target class labels, shape (batch, seq_len) or (batch*seq_len,)
        output : torch.Tensor
            Model logits, shape (batch, seq_len, n_classes) or (batch*seq_len, n_classes)
        
        Returns
        -------
        torch.Tensor
            Classification loss
        """
        # Handle 3D inputs (batch, seq_len, n_classes) by flattening
        if output.ndim == 3:
            batch_size, seq_len, n_classes = output.shape
            output = output.reshape(-1, n_classes)
            target = target.reshape(-1)        
        return self.loss_func_ctx(output, target)
    
    def _reconstruction_loss(self, target, x_output, loss_func=None):
        """Compute reconstruction loss with softplus variance transformation."""
        out_estim_mu = x_output[:, :, [0]]
        out_estim_var = F.softplus(x_output[:, :, [1]]) + 1e-6

        if loss_func is None:
            loss_func = self.loss_func_obs
        recon_loss = loss_func(out_estim_mu, target, out_estim_var)
        return recon_loss
    
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
    
    def _module_network_loss(self, module_type, target_obs, target_ctx, model_output, learning_objective='all', kappa=0.5):
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
            - 'all': Train both modules with combined loss (default)
        kappa : float, optional
            Weighting factor for combining observation and context losses (default=0.5).
            Only used when learning_objective='all'.
            
        Returns
        -------
        torch.Tensor
            The computed loss value.
        """

        obs_output, ctx_output = model_output
        
        # Observation module's loss. Regression task: reconstruction (+ KL)
        if learning_objective in ['obs', 'all']:
            obs_loss = self._regression_loss(target_obs, obs_output, model_type=module_type)
        
        # Context module's loss (Classification task: cross-entropy only, no KL)
        # KL divergence is only for regression tasks with continuous latents
        if learning_objective in ['ctx', 'all']:
            ctx_loss = self._classification_loss(target_ctx, ctx_output)
        
        
        if learning_objective == 'obs':
            loss = obs_loss
        elif learning_objective == 'ctx':
            loss = ctx_loss
        elif learning_objective == 'all':
            # Combine losses
            loss = kappa * obs_loss + (1 - kappa) * ctx_loss
        
        return loss
    

    def _population_network_loss(self, module_type, target_obs, target_ctx, target_dpos, target_rule, model_output, learning_objective='all'):
        # one loss per module, no competition between them
        
        # Get outputs
        obs_output, ctx_output, dpos_output, rule_output = model_output

        # Observation loss. Regression task: reconstruction (+ KL)
        obs_loss = self._regression_loss(target_obs, obs_output, model_type=module_type)

        # Context loss (classification task: context inference)
        ctx_loss = self._classification_loss(target_ctx, ctx_output)

        # Deviant position loss (classification task: deviant position inference)
        dpos_loss = self._classification_loss(target_dpos, dpos_output)

        # Rule loss (classification task: rule inference)
        rule_loss = self._classification_loss(target_rule, rule_output)

        loss = obs_loss + ctx_loss + dpos_loss + rule_loss
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