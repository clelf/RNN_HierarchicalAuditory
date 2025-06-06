from model import SimpleRNN, VAE, VRNN
import os
from pathlib import Path
import torch
from paradigm.audit_gm import NonHierachicalAuditGM
from time import time
import numpy as np
import matplotlib.pyplot as plt

DEVICE = torch.device('cpu')

def train(model, optimizer, learning_rate, lr_scheduler, gm_name, model_config, data_config):

    # Load model parameters
    loss_function = model_config["loss_function"]
    optimizer = model_config["optimizer"]
    num_epochs = model_config["num_epochs"]
    n_batches = model_config["n_batches"]
    lr = model_config["learning_rate"]

    # Define data generative model
    if gm_name == 'NonHierarchicalGM': gm = NonHierachicalAuditGM(data_config)
    elif gm_name == 'HierarchicalGM': gm = NonHierachicalAuditGM(data_config)
    else: raise ValueError("Invalid GM name")

    # Prepare to save the results
    save_path = Path(f'./training_results/{model.name}/')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # TODO CHECK-LIST
    # - a loss tolerance?
    # - an early stopping in case of overfitting
    # - dropout?
    # - set requires_grad / autograd
    # x training + validation in eval mode


    for epoch in range(num_epochs):
        for batch in range(n_batches):

            train_losses = []
            train_log_losses = []

            # TRAIN
            model.train()
            optimizer.zero_grad()

            # Generate data (outside of train loop maybe?)
            y, _, c = gm.generate_batch(n_trials, batch_size)

            # Call model
            x = torch.tensor(y, dtype=torch.float, requires_grad=False).to(DEVICE)
            model_output = model(x) # can contain output(, mu_latent, logvar_latent(, mu_prior, logvar_prior))

            # Compute loss
            loss = model.loss(y, model_output, loss_function=loss_function)

            # Learning
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            train_losses.append(float(loss.detach().cpu().numpy()))


            # Logging and reporting
            if batch % batch_res == batch_res-1:
                train_log_losses.append(float(loss.detach().cpu().numpy()))
                lr = lr_scheduler.get_last_lr()[0]
                loss_log(save_path, batch, batch_res, n_batches, tt, lr, loss.detach().cpu().numpy(), train_log_losses, plot=True)
                tt = time.time()
            
        avg_train_loss = np.mean(train_losses)


        # VALID
        model.eval()
        with torch.no_grad():
            for batch in n_batches:

                valid_losses = []

                # Generate data (outside of train loop maybe?)
                y, _, c = gm.generate_batch(n_trials, batch_size)

                # Call model
                x = torch.tensor(y, dtype=torch.float, requires_grad=False).to(DEVICE)
                output = model(x) # can contain output(, mu_latent, logvar_latent(, mu_prior, logvar_prior))

                # Compute loss
                loss = model.loss(y, output, loss_function=loss_function)
                valid_losses.append(float(loss.detach().cpu().numpy()))

                # Save valid samples
                if batch % batch_res == batch_res-1:
                    plot_samples(estim_mu=)
            
            
            avg_valid_loss = np.mean(valid_losses)
    
    print(f"Epoch: {epoch + 1}, Training Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}")

 


def test():
    return NotImplementedError

	
def loss_log(logpath, batch, batch_res, n_batches, tt, lr, loss, losses, plot):
    """ Auxiliary function that stores the loss in a textfile

    Parameters
    ----------
    logpath   : str (optional)
        Path of the textfile where to log the loss.
    batch     : int
        Current batch number
    batch_res : int
        Every how many batches the loss is written down to the loss history
    n_batches : int
        Number of batches
    tt        : 
        Time at which the training for the current batch stated (as measured by time.time()) 
    lr        : float
        current learning rate
    loss      : scalar torch.tensor
        total loss

    """

    loss_np = loss.detach().item()
    sprint  = f'Batch {batch+1:>2}/{n_batches}; Time = {time.time()-tt:.1f}s; '
    sprint += f'Loss = {loss_np:.3f} ('
    slog = f'{loss_np/batch_res:.2f}'

    sprint += f'LR = {lr:.2g})'
        
    with open(logpath, 'a') as f:
        f.write(f'{slog}\n')


    if plot: 
        plot_losses(losses, x_label='Batch steps', y_label='Loss', label='Train losses', save_path=logpath)



def plot_losses(losses, x_label, y_label, label, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label=label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def plot_samples(estim_mu, estim_sigma, x, ax):
    ax.plot(range(len(x)), x, color='tab:blue')
    ax.plot(range(len(x)), estim_mu, color='k')
    ax.fill_between(range(len(x)), estim_mu-estim_sigma, estim_mu+estim_sigma, color='k', alpha=0.2)

if __name__=='__main__':


    # Define experiment parameters (non-hierarchical)
    n_trials    = 5000 # Single tones
    batch_res   = 10   # Store and report loss every batch_res batches
    batch_size  = 128  # batch_size = N_batch too # TODO: check this
    gm_name = 'NonHierarchicalGM'

    # Define model dimensions
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

        
    # Define model
    rnn     = SimpleRNN(x_dim=input_dim, output_dim=dim_out_obs, hidden_dim=rnn_hidden_dim, n_layers=rnn_n_layers)
    vrnn    = VRNN(x_dim=input_dim, output_dim=dim_out_obs, latent_dim=latent_dim, phi_x_dim=phi_x_dim, phi_z_dim=phi_z_dim, phi_prior_dim=phi_prior_dim, rnn_hidden_states_dim=rnn_hidden_dim, rnn_n_layers=rnn_n_layers)

    
    
    # Define training parameters
    num_epochs      = 150
    batch_res       = 10   # Store and report loss every batch_res batches
    batch_size      = 128
    n_batches       = 20
    learning_rate   = 5e-4
    weight_decay    = 1e-5 
    loss_function   = torch.nn.GaussianNNL # GaussianNNL


    for model in [rnn, vrnn]:
        optimizer       = torch.optim.Adam(model.parameters(), lr=learning_rate)
        lr_scheduler    = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=np.prod((num_epochs, n_batches)))    
        train(model, optimizer=optimizer, lr=learning_rate, lr_scheduler=lr_scheduler)


        test(...)
