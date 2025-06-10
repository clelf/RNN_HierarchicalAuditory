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
FREQ_MIN = 1400
FREQ_MAX = 1650


# TODO:
# - delete possiblity of learning an estimation instead of a distribution
# - save model weights
# - load params
# TODO: handle device switch, autograd...


class MinMaxScaler():
    def __init__(self):
        self.freq_min = FREQ_MIN
        self.freq_max = FREQ_MAX

    def fit_normalize(self, x, freq_min=None, freq_max=None, margin=None):
        if freq_min is None:
            self.freq_min = x.min()
        if freq_max is None:
            self.freq_max = x.max()
        if margin is not None:
            self.freq_min -= margin
            self.freq_max += margin
        return self.normalize(x)

    def normalize(self, x):
        return (x - self.freq_min) / (self.freq_max - self.freq_min)
    
    def denormalize(self, x_norm):
        return x_norm * (self.freq_max - self.freq_min) + self.freq_min




def train(model, n_batches, num_epochs, optimizer, lr_scheduler, gm_name, data_config):

    # Define data generative model
    if gm_name == 'NonHierarchicalGM': gm = NonHierachicalAuditGM(data_config)
    elif gm_name == 'HierarchicalGM': gm = NonHierachicalAuditGM(data_config)
    else: raise ValueError("Invalid GM name")

    # Prepare to save the results
    save_path = Path(f'/home/clevyfidel/Documents/Workspace/RNN_paradigm/training_results/{model.name}/')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # TODO CHECK-LIST
    # - a loss tolerance?
    # - an early stopping in case of overfitting (ABSOLUTELY according to Seyma) --> include IF i observe overfitting
    # - dropout?
    # - assess need to change activation function or not (hand in hand with changing the models' inner layers' architecture)
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
            _, _, y = gm.generate_batch(N_batch=batch_size)
            y = torch.tensor(y, dtype=torch.float, requires_grad=False).unsqueeze(2).to(DEVICE)

            # Transform data
            minmax = MinMaxScaler()
            x = minmax.fit_normalize(y, margin=data_config["si_r"]+data_config["si_q"])

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
                loss_log(f'{save_path}/loss-{epoch}-{batch}', batch, batch_res, n_batches, tt, lr, loss.detach().cpu().item(), train_log_losses, plot=True)
                tt = time.time()
            
        avg_train_loss = np.mean(train_losses)


        # VALID
        model.eval()
        with torch.no_grad():
            for batch in range(n_batches):

                valid_losses = []

                # Generate data (outside of train loop maybe?)
                _, _, y = gm.generate_batch(N_batch=batch_size)
                y = torch.tensor(y, dtype=torch.float, requires_grad=False).unsqueeze(2).to(DEVICE)

                # Transform data
                x = minmax.normalize(y)

                # Call model
                model_output = model(x) # can contain output(, mu_latent, logvar_latent(, mu_prior, logvar_prior))

                # Compute loss
                loss = model.loss(x, model_output, loss_func=loss_function)
                valid_losses.append(float(loss.detach().item()))

                # Save valid samples
                if batch % batch_res == batch_res-1:
                    plot_samples(y, minmax.denormalize(model_output), save_path=f'{save_path}/epoch-{epoch}_examples')
            
            
            avg_valid_loss = np.mean(valid_losses)
    
    print(f"Model: {model.name}, Epoch: {epoch + 1}, Training Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}")
    

    return minmax
 


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
    loss      : scalar torch.tensor (needs to be detached)
        total loss

    """

    sprint  = f'Batch {batch+1:>2}/{n_batches}; Time = {time.time()-tt:.1f}s; '
    sprint += f'Loss = {loss:.3f} ('
    sprint += f'{loss/batch_res:.2f}'
    sprint += f'LR = {lr:.2g})'

    logfilename = f'{logpath}.txt'
    with open(logfilename, 'a') as f:
        f.write(f'{sprint}\n')

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


def plot_samples(obs, output, save_path):
    obs = obs.squeeze()
    if isinstance(output, tuple): output = output[0]
            
    # for some 8 samples out of the whole batch of length obs.size(0)
    for i in range(min(8, obs.size(0))):
        # Plot observation
        plt.plot(range(len(obs[i])), obs[i], color='tab:blue', label='y_true')

        # Plot estimation
        if output.size(2) == 2:
            # As a distribution (with uncertainty)
            estim_mu, estim_sigma = output[i,:,0], torch.sqrt(output[i,:,1])
            plt.plot(range(len(obs[i])), estim_mu, color='k', label='y_hat')
            plt.fill_between(range(len(obs[i])), estim_mu-estim_sigma, estim_mu+estim_sigma, color='k', alpha=0.2)
        
        elif output.size(2) == 1:
            # As a point estimation
            plt.plot(range(len(obs[i])), output[i].squeeze(), color='k', label='y_hat')

        plt.legend()
        plt.savefig(f'{save_path}_s{i}.png')
        plt.close()


def save_weights(self, modelpath, epoch=None):
    """ Saves the current statedict of the nn.Module to modelpath. Warning: initial states ar
        not saved in the current version (to do).

    Parameters
    ----------
    modelpath : str
        Path where to save the statedict. If not a full path, it assumes the file is in
        ./models.
    epoch     : int (optional)
        If provided, it appends 'e{epoch}' to modelpath; useful to save the training progress.
    """

    if not os.path.exists('./models'):
        os.mkdir('./models')

    if epoch is not None:
        modelpath = f'{modelpath.split(".")[0]}_e{epoch:02d}'
    if '.pt' not in modelpath:
        modelpath += '.pt'
    if modelpath[0] != '/':
        if './models' not in modelpath:
            modelpath = './models/' + modelpath

    torch.save(self.model.state_dict(), modelpath)





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
    num_epochs      = 10 # 150
    batch_res       = 10    # Store and report loss every batch_res batches
    batch_size      = 30 # 128   # batch_size = N_batch too # TODO: check this
    n_batches       = 20 # 20
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
