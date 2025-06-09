from model import SimpleRNN, VAE, VRNN
import os
from pathlib import Path
import torch
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from paradigm.audit_gm import NonHierachicalAuditGM
import time
import numpy as np
import matplotlib.pyplot as plt

DEVICE = torch.device('cpu')


# TODO:
# - delete possiblity of learning an estimation instead of a distribution
# - save model weights
# - load params
# TODO: handle device switch, autograd...




def train(model, n_batches, num_epochs, optimizer, lr_scheduler, gm_name, data_config):

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
        
        train_losses = []
        train_log_losses = []

        tt = time.time()


        for batch in range(n_batches):
            
            # TRAIN
            model.train()
            optimizer.zero_grad()

            # Generate data (outside of train loop maybe?)
            y, _, c = gm.generate_batch(N_batch=batch_size)
            x = torch.tensor(y, dtype=torch.float, requires_grad=False).unsqueeze(2).to(DEVICE)

            # Call model
            model_output = model(x) # can contain output(, mu_latent, logvar_latent(, mu_prior, logvar_prior))

            # Compute loss
            loss = model.loss(x, model_output, loss_func=loss_function)

            # Learning
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            train_losses.append(float(loss.detach().cpu().item()))


            # Logging and reporting
            if batch % batch_res == batch_res-1:
                train_log_losses.append(float(loss.detach().cpu().item()))
                lr = lr_scheduler.get_last_lr()[0]
                loss_log(f'{save_path}/loss-{epoch}-{batch}', batch, batch_res, n_batches, tt, lr, loss.detach().cpu().item(), train_log_losses, plot=False)
                tt = time.time()
            
        avg_train_loss = np.mean(train_losses)


        # VALID
        model.eval()
        with torch.no_grad():
            for batch in range(n_batches):

                valid_losses = []

                # Generate data (outside of train loop maybe?)
                y, _, c = gm.generate_batch(N_batch=batch_size)
                x = torch.tensor(y, dtype=torch.float, requires_grad=False).unsqueeze(2).to(DEVICE)

                # Call model
                model_output = model(x) # can contain output(, mu_latent, logvar_latent(, mu_prior, logvar_prior))

                # Compute loss
                loss = model.loss(x, model_output, loss_func=loss_function)
                valid_losses.append(float(loss.detach().item()))

                # Save valid samples
                if batch % batch_res == batch_res-1:
                    plot_samples(y, model_output, f'{save_path}/epoch-{epoch}_examples.png')
            
            
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

    logfilename = f'{logpath}.txt'
    with open(logfilename, 'a') as f:
        f.write(f'{slog}\n')


    if plot:
        plotfile = f'{logpath}.png'
        plot_losses(losses, x_label='Batch steps', y_label='Loss', label='Train losses', save_path=plotfile)



def plot_losses(losses, x_label, y_label, label, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label=label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def plot_samples(output, obs, ax, save_path):
    if isinstance(output, tuple): output = output[0]
    if output.size(2) == 2: 
        estim_mu, estim_sigma = output[:,:0], torch.sqrt(output[:,:1])
        ax.plot(range(len(obs)), obs, color='tab:blue')
        ax.plot(range(len(obs)), estim_mu, color='k')
        ax.fill_between(range(len(obs)), estim_mu-estim_sigma, estim_mu+estim_sigma, color='k', alpha=0.2)
        plt.savefig(save_path)
        plt.close()
    elif output.size(2) == 1:
        ax.plot(range(len(obs)), obs, color='tab:blue')
        ax.plot(range(len(obs)), output.squeeze(), color='k')
        plt.savefig(save_path)
        plt.close()




if __name__=='__main__':

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

    
    # Define training parameters
    num_epochs      = 2 # 150
    batch_res       = 10    # Store and report loss every batch_res batches
    batch_size      = 2 # 128   # batch_size = N_batch too # TODO: check this
    n_batches       = 2 # 20
    learning_rate   = 5e-4
    weight_decay    = 1e-5 
    loss_function   = torch.nn.GaussianNLLLoss() # GaussianNLL


    # Define models
    rnn     = SimpleRNN(x_dim=input_dim, output_dim=dim_out_obs, hidden_dim=rnn_hidden_dim, n_layers=rnn_n_layers, batch_size=batch_size, device=DEVICE)
    vrnn    = VRNN(x_dim=input_dim, output_dim=dim_out_obs, latent_dim=latent_dim, phi_x_dim=phi_x_dim, phi_z_dim=phi_z_dim, phi_prior_dim=phi_prior_dim, rnn_hidden_states_dim=rnn_hidden_dim, rnn_n_layers=rnn_n_layers, batch_size=batch_size, device=DEVICE)


    # Define experiment parameters (non-hierarchical)
    n_trials    = 50 # 5000 # Single tones
    gm_name = 'NonHierarchicalGM'

    config_NH = {
        "N_batch": batch_size,
        "N_blocks": 1,
        "N_tones": n_trials,
        "mu_rho_ctx": 0.9,
        "si_rho_ctx": 0.05,
        "tones_values": [1455, 1500, 1600],
        "mu_tau": 4,
        "si_tau": 1,
        "si_lim": 5,
        "si_q": 2,  # process noise
        "si_r": 2,  # measurement noise
    }

    for model in [rnn, vrnn]:
        optimizer       = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        lr_scheduler    = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=np.prod((num_epochs, n_batches)))    
        
        train(model, n_batches=n_batches, num_epochs=num_epochs,optimizer=optimizer, lr_scheduler=lr_scheduler, gm_name=gm_name, data_config=config_NH)


        # test(...)
