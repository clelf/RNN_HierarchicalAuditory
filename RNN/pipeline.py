from model import SimpleRNN, VAE, VRNN
import os
from pathlib import Path
import torch
from paradigm.audit_gm import NonHierachicalAuditGM

DEVICE = torch.device('cpu')

def train(model, gm_name, model_config, data_config):

    # Load model parameters
    loss_function = model_config["loss_function"]
    optimizer = model_config["optimizer"]
    num_epochs = model_config["num_epochs"]
    n_batches = model_config["n_batches"]
    lr = model_config["learning_rate"]

    # Define data generative model
    if gm_name == 'NonHierachicalGM': gm = NonHierachicalAuditGM(data_config)

    # Prepare to save the results
    save_path = Path(f'./model_results/{model.name}/')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # TODO CHECK-LIST
    # - a loss tolerance?
    # - dropout?
    # - set requires_grad / autograd
    # - training + validation in eval mode
    # - an early stopping in case of overfitting

    train_losses = []

    for epoch in range(num_epochs):
        for batch in range(n_batches):

            # TRAIN
            model.train()
            optimizer.zero_grad()

            # Generate data (outside of train loop maybe?)
            y, _, c = gm.generate_batch(n_trials, batch_size)

            # Call model
            x = torch.tensor(y, dtype=torch.float, requires_grad=False).to(DEVICE)
            obs_mean, obs_var, ... = model(x)

            # Compute loss
            loss = model.loss(estim, target, loss_function=loss_function)

            # Learning
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().numpy()))


            # VALID
            model.eval()
            with torch.no_grad():
                # Similar
                pass 



def test():
    return NotImplementedError



if __name__=='__main__':

    # Define training parameters
    num_epochs      = 150
    batch_res       = 10   # Store and report loss every batch_res batches
    batch_size      = 128
    learning_rate   = 5e-4
    optimizer       = torch.optim.Adam

    # Define experiment parameters (non-hierarchical)
    n_trials    = 5000 # Single tones
    batch_res   = 10   # Store and report loss every batch_res batches
    batch_size  = 128  # batch_size = N_batch too # TODO: check this
    gm_name = 'NonHierachicalGM'

    # Define model dimensions
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

    for model in [rnn, vrnn]:    
        train(model, optimizer=optimizer, lr=learning_rate)


        test(...)
