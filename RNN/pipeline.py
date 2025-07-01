from model import SimpleRNN, VAE, VRNN
import os
from pathlib import Path
import torch
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from paradigm.audit_gm import NonHierachicalAuditGM, HierarchicalAuditGM
import time
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import functional as F
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from Kalman.kalman import kalman_tau, plot_estim, kalman_batch


DEVICE = torch.device('cpu')
FREQ_MIN = 1400
FREQ_MAX = 1650


# TODO CHECK-LIST
# - a loss tolerance?
# - an early stopping in case of overfitting (ABSOLUTELY according to Seyma) --> include IF i observe overfitting
# - dropout?
# - assess need to change activation function or not (hand in hand with changing the models' inner layers' architecture)
# - set requires_grad / autograd
# x training + validation in eval mode


# TODO:
# - delete possiblity of learning an estimation instead of a distribution
# - save model weights !!
# - load params
# TODO: handle device switch, autograd...

# TODO: 
# - assess performance especially for test 
# - send outputs to 2nd (V)RNN for contexts evaluation


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
    
    def denormalize_mean(self, mu_norm):
        return mu_norm * (self.freq_max - self.freq_min) + self.freq_min
    
    def denormalize_var(self, var_norm):
        return var_norm * (self.freq_max - self.freq_min)**2





def train(model, n_batches, num_epochs, batch_res, epoch_res, optimizer, lr, lr_id, gm_name, data_config, kalman=False):
    
    # Define data generative model
    if gm_name == 'NonHierarchicalGM': gm = NonHierachicalAuditGM(data_config)
    elif gm_name == 'HierarchicalGM': gm = HierarchicalAuditGM(data_config)
    else: raise ValueError("Invalid GM name")

    # Prepare to save the results
    lr_title = f"Learning rate: {lr:>6.0e}"
    save_path = Path(f'/home/clevyfidel/Documents/Workspace/RNN_paradigm/training_results/N_ctx_{data_config["N_ctx"]}/{model.name}/')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        os.makedirs(save_path / 'samples/')

    train_losses_report = []
    train_steps         = []
    valid_losses_report = []
    valid_steps         = []
    valid_mse_report    = []
    valid_sigma_report  = []

    # Define loss instance
    loss_function   = torch.nn.GaussianNLLLoss(reduction='sum') # GaussianNLL

    for epoch in range(num_epochs):

        ### TRAINING
        train_losses = []
        tt = time.time()
        model.train()

        for batch in range(n_batches):

            optimizer.zero_grad()

            # Generate data (outside of train loop maybe?)
            _, _, y = gm.generate_batch(N_batch=batch_size)
            y = torch.tensor(y, dtype=torch.float, requires_grad=False).unsqueeze(2).to(DEVICE)

            # Transform data
            minmax_train = MinMaxScaler()
            x = minmax_train.fit_normalize(y, margin=data_config["si_r"]+data_config["si_q"])

            # Call model
            model_output = model(x) # can contain output(, mu_latent, logvar_latent(, mu_prior, logvar_prior))

            # Compute loss
            loss = model.loss(x, model_output, loss_func=loss_function)

            # Learning
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))

            # Logging and reporting
            if batch % batch_res == batch_res-1:
                train_losses_report.append(float(loss.detach().cpu().item()))
                train_steps.append(epoch * n_batches + batch)

                loss_report = np.mean(train_losses[-batch_res:])
                sprint=f'LR: {lr:>6.0e}; epoch: {epoch:0>3}; batch: {batch:>3}; loss: {loss:>7.4f}; batch loss: {loss_report:7.4f}; time: {time.time()-tt:>.2f}; training step: {epoch * n_batches + batch}'
                logfilename = f'{save_path}/training_loss_lr{lr_id}.txt'
                with open(logfilename, 'a') as f:
                    f.write(f'{sprint}\n')
                tt = time.time()
            
        avg_train_loss = np.mean(train_losses) # average loss for current epoch


        ### VALIDATION
        valid_losses = []
        valid_mse = []
        valid_sigma = []

        model.eval()
        with torch.no_grad():
            for batch in range(n_batches):

                # Generate data (outside of train loop maybe?)
                _, _, y, pars = gm.generate_batch(N_batch=batch_size, return_pars=True)
                y = torch.tensor(y, dtype=torch.float, requires_grad=False).unsqueeze(2).to(DEVICE)

                # Transform data
                minmax_valid = MinMaxScaler()
                x = minmax_valid.normalize(y)

                # Call model
                model_output = model(x) # can contain output(, mu_latent, logvar_latent(, mu_prior, logvar_prior))

                # Compute loss
                loss = model.loss(x, model_output, loss_func=loss_function)
                valid_losses.append(float(loss.detach().item()))

                # Get estimated distribution
                if model.name=='vrnn': model_output_dist = model_output[0]
                else: model_output_dist = model_output
                estim_mu    = minmax_valid.denormalize_mean(F.sigmoid(model_output_dist[...,0]))
                estim_var   = minmax_valid.denormalize_var(F.softplus(model_output_dist[..., 1]) + 1e-6)
                estim_sigma = torch.sqrt(estim_var)
                valid_sigma.append(estim_sigma)
                x_denorm    = minmax_valid.denormalize_mean(x).squeeze()


                # Evaluate MSE to true value
                mse = ((estim_mu-x_denorm)**2).mean()
                valid_mse.append(mse)

                # Compare with Kalman
                kalman_mu, kalman_sigma = kalman_batch(x_denorm, pars, C=1, Q=data_config["si_q"], R=data_config["si_r"], x0s=x_denorm[...,0]) # COMPUTING KALMAN MSE DOESNT MAKE ANY CONCEPTUAL SENSE


            # Save valid samples
            if epoch % epoch_res == epoch_res-1:                         
                plot_samples(x_denorm, estim_mu, estim_sigma, save_path=f'{save_path}/samples/lr{lr_id}-epoch-{epoch:0>3}_samples', title=lr_title, kalman_mu=kalman_mu, kalman_sigma=kalman_sigma)

            avg_valid_loss = np.mean(valid_losses)
            avg_valid_mse = np.mean(valid_mse)
            avg_valid_sigma = np.mean(valid_sigma)
        
        valid_mse_report.append(avg_valid_mse)
        valid_losses_report.append(avg_valid_loss) # avg loss for epoch (over batches)
        valid_steps.append(epoch*n_batches+n_batches)

        valid_sigma_report.append(avg_valid_sigma)

        print(f"Model: {model.name:>4}, LR: {lr:>6.0e}, Epoch: {epoch:>3}, Training Loss: {avg_train_loss:>7.4f}, Valid Loss: {avg_valid_loss:>7.4f}")
        sprint=f'LR: {lr:>6.0e}; epoch: {epoch:>3}; batch: {batch:>3}; loss: {loss:>7.4f}; batch loss: {loss_report:7.4f}; MSE: {avg_valid_mse:>7.4f}; time: {time.time()-tt:>.2f}; training step: {epoch * n_batches + batch}'
        logfilename = save_path / f'training_log_lr{lr_id}.txt'
        with open(logfilename, 'a') as f:
            f.write(f'{sprint}\n')

    lossplotfile = save_path/f'loss_trainvalid_lr{lr_id}.png'
    plot_losses(train_steps, valid_steps, train_losses_report, valid_losses_report, x_label='Training steps', y_label='Loss', title=lr_title, save_path=lossplotfile)
    plot_var(valid_steps, valid_sigma_report, title=lr_title, save_path=save_path/f'variance_valid_lr{lr_id}.png')

    # Save model's weights
    torch.save(model.state_dict(), f'{save_path}/lr{lr_id}_weights.pth')

    return minmax_train
 


def test():
    return NotImplementedError


def plot_var(valid_steps, valid_sigma, title, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(valid_steps, valid_sigma, label="valid std")
    plt.xlabel("Training steps")
    plt.ylabel("Valid variance (std)")
    plt.legend()
    plt.title(title)
    plt.savefig(save_path)
    plt.close()

def plot_losses(train_steps, valid_steps, train_losses_report, valid_losses_report, x_label, y_label, title, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(train_steps, train_losses_report, label="train loss")
    plt.plot(valid_steps, valid_losses_report, label="valid loss")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.title(title)
    plt.savefig(save_path)
    plt.close()


def plot_samples(obs, estim_mu, estim_sigma, save_path, title=None, kalman_mu=None, kalman_sigma=None):
    obs = obs.squeeze() # for safety
            
    # for some 8 samples out of the whole batch of length obs.size(0)
    N = min(8, obs.size(0))
    for id, i in enumerate(np.random.choice(range(N), size=(N,), replace=False)):
            
        plt.figure(figsize=(20, 6))
        
        # Plot observation
        plt.plot(range(len(obs[i])), obs[i], color='tab:blue', label='y_obs')

        # Plot estimation as a distribution (with uncertainty)
        plt.plot(range(len(estim_mu[i])), estim_mu[i], color='k', label='y_hat')
        plt.fill_between(range(len(estim_mu[i])), estim_mu[i]-estim_sigma[i], estim_mu[i]+estim_sigma[i], color='k', alpha=0.2)
        
        # Plot Kalman estimation if provided
        if kalman_mu is not None and kalman_sigma is not None:
            plt.plot(range(len(kalman_mu[i])), kalman_mu[i], label='y_kal', color='green', linewidth=2)
            plt.fill_between(range(len(kalman_mu[i])), kalman_mu[i]-kalman_sigma[i], kalman_mu[i]+kalman_sigma[i], color='green', alpha=0.2)

        plt.legend()
        plt.title(title)
        plt.savefig(f'{save_path}_s{id}.png')
        plt.close()




if __name__=='__main__':

    # Define model dimensions
    # Input and output dimensions
    input_dim   = 1 # number of features in each observation: 1 --> need to write x = x.unsqueeze(-1)  at one point
    dim_out_obs = 2 # learn the sufficient statistics mu and var
    # RNN bit
    output_dim      = input_dim
    rnn_hidden_dim  = 8 # 64 # from goin. could also take varying values like [2**n for n in range(1, 9)] # In VRNN paper: 2000
    rnn_n_layers    = 1 # from goin
    # VRNN specific
    latent_dim      = 4 # 16 # needs to be < input_dim --> in a VAE, yes, but in a VRNN, needs to be <input_dim + rnn_hidden_dim
    phi_x_dim       = rnn_hidden_dim
    phi_z_dim       = rnn_hidden_dim
    phi_prior_dim   = rnn_hidden_dim

    
    # Define training parameters
    num_epochs      = 50 # 150
    epoch_res       = 10
    batch_res       = 10    # Store and report loss every batch_res batches
    batch_size      = 32 # 128   # batch_size = N_batch too # TODO: check this
    n_batches       = 20 # 20
    learning_rate   = 5e-4
    weight_decay    = 1e-5 


    # Define models
    rnn     = SimpleRNN(x_dim=input_dim, output_dim=dim_out_obs, hidden_dim=rnn_hidden_dim, n_layers=rnn_n_layers, batch_size=batch_size, device=DEVICE)
    vrnn    = VRNN(x_dim=input_dim, output_dim=dim_out_obs, latent_dim=latent_dim, phi_x_dim=phi_x_dim, phi_z_dim=phi_z_dim, phi_prior_dim=phi_prior_dim, rnn_hidden_states_dim=rnn_hidden_dim, rnn_n_layers=rnn_n_layers, batch_size=batch_size, device=DEVICE)


    # Define experiment parameters (non-hierarchical)
    n_trials    = 150 # 5000 # Single tones
    gm_name = 'NonHierarchicalGM'

    config_NH = {
        "N_ctx": 1,
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

    for lr_id, learning_rate in enumerate([0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.000005, 0.000001]): # TODO: delete 0.1, 0.5
        if lr_id in [5, 6, 7, 8, 9, 10]:
            for model in [vrnn]:
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
                train(model, n_batches=n_batches, num_epochs=num_epochs, batch_res=batch_res, epoch_res=epoch_res, optimizer=optimizer, 
                    lr=learning_rate, lr_id=lr_id, gm_name=gm_name, data_config=config_NH, kalman=True)


            # test(...)
