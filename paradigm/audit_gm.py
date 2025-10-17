import numpy as np
import time
import sys
import os
import scipy.stats as ss
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.ticker import MaxNLocator
import random


##### UTILS

def reshape_batch_variable(var):
    if isinstance(var[0], dict):
        # Recursively process each dict key
        return {key: np.stack([batch[key] for batch in var], axis=0) for key in var[0]}
    else:
        # Direct array
        return np.stack([x for x in var], axis=0)
    

##### GENERATIVE MODEL CLASSES

class AuditGenerativeModel:
    """Generic class for building a generative model.
    Not meant to be used on its own but provide reusable methods for contexts, process and observation states generation.
    NonHierachicalGenerativeGM and HierachicalGenerativeGM build on top of it.

    Attributes
    ----------
        N_samples: int; number of data batches to generate
        N_blocks: int; number of blocks a batch contains
        N_tones: int; number of tones per block (usually 8)
        tones_values: list-like; a set of tone frequencies to sample a pair from, to assign to the pair of standard and deviant tones
        mu_tau, si_tau, si_lim: float; define the linear Gaussian dynamics of the standard and deviant processes
        si_q: float; variance of the noise in the process LGD
        si_r: float; variance of the noise in the observation LGD

    """

    def __init__(self, params):

        self.N_samples = params["N_samples"]
        self.N_blocks = params["N_blocks"]
        self.N_tones = params["N_tones"]
        if "tones_values" in params.keys():
            self.tones_values = params["tones_values"]
        if "mu_tau" in params.keys():
            self.mu_tau = params["mu_tau"]  # Std and dvt process
        self.si_tau = params["si_tau"]  # Std and dvt process
        self.si_lim = params["si_lim"]  # Std and dvt process
        # self.si_q = params["si_q"] # Obsolete TODO: replace by below si_stat and later define si_q_dev and si_q_std
        if "si_stat" in params.keys():
            self.si_stat = params["si_stat"]
        if "si_r" in params.keys():    
            self.si_r = params["si_r"]

        if "N_ctx" not in params.keys(): self.N_ctx = 2 # context refers to being a std / dvt
        else: self.N_ctx = params["N_ctx"]

        if "si_d_coef" in params.keys() and "si_stat" in params.keys() and "mu_d" in params.keys():
            self.mu_d = params["mu_d"]
            si_d_ub = (4 - self.mu_d)/3
            si_d_lb = (self.mu_d - 0.1)/3
            self.si_d = si_d_lb + params["si_d_coef"] * (si_d_ub - si_d_lb)

        # NOTE: this only happens in contexts where N_ctx is also == 1
        if "params_testing" in params.keys():
            self.params_testing = True
            self.mu_tau_set, self.si_stat_set,self.si_r_set = None, None, None
            if "mu_tau_bounds" in params.keys():
                self.mu_tau_bounds = params["mu_tau_bounds"]
                self.mu_tau_set = 10 ** np.random.uniform(
                    low=np.log10(self.mu_tau_bounds["low"]),
                    high=np.log10(self.mu_tau_bounds["high"]),
                    size=self.N_samples
                )
            if "si_stat_bounds" in params.keys():
                self.si_stat_bounds = params["si_stat_bounds"]
                self.si_stat_set = 10 ** np.random.uniform(
                    low=np.log10(self.si_stat_bounds["low"]),
                    high=np.log10(self.si_stat_bounds["high"]),
                    size=self.N_samples
                )
            if "si_r_bounds" in params.keys():
                self.si_r_bounds = params["si_r_bounds"]
                self.si_r_set = 10 ** np.random.uniform(
                    low=np.log10(self.si_r_bounds["low"]),
                    high=np.log10(self.si_r_bounds["high"]),
                    size=self.N_samples
                )
            if "si_d" in params.keys() and params["si_d"] and "mu_d" in params.keys():
                self.mu_d = params["mu_d"]
                si_d_ub = (4 - self.mu_d)/3
                si_d_lb = (self.mu_d - 0.1)/3
                self.si_d_set = np.random.uniform(si_d_lb, si_d_ub, self.N_samples)
            
        else:
            self.params_testing = False


    # Auxiliary samplers from goin.coin.GenerativeModel
    def _sample_N_(self, mu, si, size=1):
        """Samples from a normal distribution

        Parameters
        ----------
        mu : float
            Mean of the normal distribution
        si : float
            Standard deviation of the normal distribution
        size  : int or tuple of int (optional)
            Size of samples

        Returns
        -------
        np.array
            samples
        """

        return np.array(ss.norm.rvs(mu, si, size))

    def _sample_TN_(self, a, b, mu, si, size=1):
        """Samples from a truncated normal distribution

        Parameters
        ----------
        a  : float
            low truncation point
        b  : float
            high truncation point
        mu : float
            Mean of the normal distribution before truncation (i.e, location)
        si : float
            Standard deviation of the normal distribution before truncation (i.e, size)
        size  : int or tuple of int (optional)
            Size of samples

        Returns
        -------
        np.array
            samples
        """

        return np.array(ss.truncnorm.rvs((a - mu) / si, (b - mu) / si, mu, si, size))
    
    def _sample_logN_(self, min, mu, si, size=1):
        """Samples from a log-normal distribution
        Parameters
        ----------
        min : float
            Minimum value (location parameter of the log-normal distribution)
        mu : float
            Mean of the normal distribution before exponentiation
        si : float
            Standard deviation of the normal distribution before exponentiation (i.e, size)
        size  : int or tuple of int (optional)
            Size of samples
        Returns
        -------
        np.array
            samples
        """
        
        return np.array(ss.lognorm.rvs(s=si, loc=min, scale=mu, size=size))
    
    def _sample_halfN(self, mu, si, size=1):
        """Samples from a half-normal distribution
        Parameters
        ----------
        mu : float
            Mean of the normal distribution before absolute value (i.e, location)
        si : float
            Standard deviation of the normal distribution before absolute value (i.e, size)
        size  : int or tuple of int (optional)
            Size of samples
        Returns
        -------
        np.array
            samples
        """
        
        return np.array(ss.halfnorm.rvs(loc=mu, scale=si, size=size))

    def _sample_biN_(self, mu, si, size=1):
        """Samples from a bimodal normal distribution
        Parameters
        ----------
        mu : float
            Mean of one of the two normal distributions (i.e, location)
        si : float
            Standard deviation of each of the normal distributions (i.e, size)
        size  : int or tuple of int (optional)
            Size of samples
        Returns
        -------
        np.array
            samples
        """
        
        # return np.array(np.random.choice([-1, 1], size=size) * ss.norm.rvs(mu, si, size))
        return np.random.choice([-1, 1], size=size) * self._sample_N_(mu, si, size)

    def sample_uniform_choice(self, set_values):
        """ Sample a choice of one value from a set of possible values
        """
        return np.random.choice(set_values)

    def sample_uniform_set(self, set_values, N=2):
        """ Sample a set of N value choices without replacement from a set of possible values
        Note: len(values) should be > N
        """
        set = np.random.choice(set_values, size=(N,), replace=False)
        return set

    def sample_next_markov_state(
        self, current_state, states_values, states_trans_matrix
    ):
        return np.random.choice(states_values, p=states_trans_matrix[current_state])

    def sample_pi(self, N, mu_rho, si_rho):
        """A transition matrix with a sticky diagonal, controlled by the concentration parameter rho.
        rho is comprised between 0 and 1 and is is sampled from a truncated normal distribution of 
        mean mu_rho and standard deviation si_rho

        Parameters
        ----------
        mu_rho : float
            Mean parameter of the normal distribution to sample 
        si_rho : float
            _description_

        Returns
        -------
        np.array (N, N)
            Transition matrix
        """

        if N>1:
            # Sample parameters
            rho = self._sample_TN_(0, 1, mu_rho, si_rho).item()
            eps = [np.random.uniform() for n in range(N)]

            # Delta has a zero diagonal and the rest of the elements of a row (for a rule) are partitions from 1 using the corresponding eps[row] (parameter for that rule), controlling for the sum to be 1
            delta = np.array([[(0 if i == j else (eps[i] * (1 - eps[i]) ** j if (j < i and j < N - 2) else (eps[i] * (1 - eps[i]) ** (j - 1) if i < j < N - 1 else 1 - sum([eps[i] * (1 - eps[i]) ** k for k in range(N - 2)])))) for j in range(N)] for i in range(N)])
            
            # Transition matrix
            pi = rho * np.eye(N) + (1 - rho) * delta
        else:
            # if N==1:
            pi = np.eye(N)
        return pi

    def sample_contexts(self, N, N_ctx, mu_rho_ctx, si_rho_ctx, return_pi=False):
        """Samples a 1D sequence of N events that can each be associated with a context out of of N_ctx values in range(N_ctx), and evolve
        through a Markov-chain manner with a transition matrix of parameters mu_rho_ctx and si_rho_ctx.

        Parameters
        ----------


        Returns
        -------
        list
            List of contexts for each of the N events
        """

        # Sample contexts transition matrix is sampled initially from a parametric distribution
        pi_ctx = self.sample_pi(N_ctx, mu_rho_ctx, si_rho_ctx)
        # pi_rules_0   = np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]]) # A very sticky transition matrix

        # Sequence of N contexts
        ctx = np.zeros(N, dtype=np.int64)

        # Initilize context (assign to 0, randomly, or from the distribution from which the transition probas also come from)
        ctx[0] = 0 # TODO: decide if going for this?
        # rules[0] = np.random.choice(N_rules)

        for s in range(1, N):
            # Markov chain
            ctx[s] = self.sample_next_markov_state(
                current_state=ctx[s - 1],
                states_values=range(N_ctx),
                states_trans_matrix=pi_ctx,
            )

        if return_pi:
            return ctx, pi_ctx
        else:
            return ctx
        
    def sample_states_OBSOLETE(self, contexts, return_pars=False):
        """Generates a dictionary of data sequence for each context (std or dvt) dynamics given a sequence of contexts

        Here contexts is the sequence of tone-by-tone boolean value standing for {std: 0, dvt: 1} that has been
        hierarchically defined prior to the call of sample_states


        Parameters
        ----------
        contexts : integer np.array
            2-dimensional sequence of contexts filled with 0 or 1 (std or dvt), of size (N_blocks, N_tones)
        return_pars: bool
            also returns the time constant and sationary value (previously: retention and drift) parameters for each state at each block


        Returns
        -------
        states : dict
            dictionary encoding the hidden state values (one-dimensional np.array) for each
            context c (keys).
        pars (optional):
            tau: time constant parameter for each context (only if return_pars set to True)
            lim: stationary value parameter for each context
       
        # Note that time constant and sationary value (previously: retention and drift) are sampled in every call 

        """

        # Initialize arrays of params
        tau, lim, si_stat, si_q = np.zeros((self.N_ctx, self.N_blocks)), np.zeros((self.N_ctx, self.N_blocks)), np.zeros((self.N_blocks,)), np.zeros((self.N_ctx, self.N_blocks))  # self.N_ctx normally = 2 as for len({std, dvt})

        # Sample params for each block
        for b in range(self.N_blocks):
            # Sample one pair of std/dvt lim values for each block
            lim_Cs = self.sample_uniform_set(self.tones_values, N=self.N_ctx) 
            
            # Sample stationary variance for all (both) processes
            if self.params_testing:
                si_stat[b] = self.si_stat
            else:
                si_stat[b]   = self._sample_logN_(0, self.si_stat, 0.2).item() # TODO: not sure, check if this is a good idea

            for c in range(self.N_ctx):  # 2 contexts: std or dvt
                # Parameter that is not necessarily tested
                lim[c, b]       = self._sample_N_(lim_Cs[c], self.si_lim).item()
                if self.params_testing:
                    # In that case, parameters have already been sampled, no need to sample more
                    tau[c, b] = self.mu_tau
                    lim[c, b] = lim_Cs[c]
                else:
                    # Sample dynamics params for each context (std and dvt)
                    # tau[c, b]       = self._sample_TN_(1, 50, self.mu_tau, self.si_tau).item()  # A high boundary (actually 50 was not sufficient)
                    tau[c, b]       = self._sample_logN_(1, self.mu_tau, self.si_tau).item()

                # Compute si_q from them
                si_q[c, b]  = si_stat[b] * ((2 * tau[c, b] - 1) ** 0.5) / tau[c, b]

        # Initialize states
        states = dict([(int(c), np.zeros(contexts.shape)) for c in range(self.N_ctx)])

        # States dynamics
        for c in range(self.N_ctx):  # self.N_ctx == 2

            # Initialize first state with a sample from distribution of mean and std the LGD stationary values
            # states[c][:,0] = self._sample_N_(d[c]/(1-a[c]), self.si_q/((1-a[c]**2)**.5), (contexts.shape[0], 1)) # Obsolete (alternative LGD formulation)
            # si_stat = self.si_q * tau[c, :] / ((2 * tau[c, :] - 1) ** 0.5)            # Obsolete
            # states[c][:, 0] = self._sample_N_(lim[c, :], si_stat, (contexts.shape[0],)) # Obsolete
            states[c][:, 0] = self._sample_N_(lim[c, :], self.si_stat, (contexts.shape[0],))

            for b in range(self.N_blocks):

                # Sample noise
                w = self._sample_N_(0, si_q[c, b], contexts.shape)

                # Here the states exist independently of the contexts
                for t in range(1, contexts.shape[1]): # contexts.shape[1] == N_tones
                    # states[c][:,t] = a[c] * states[c][:,t-1] + d[c] + w[:,t-1]
                    states[c][b, t] = states[c][b, t - 1] + 1 / tau[c, b] * (lim[c, b] - states[c][b, t - 1]) + w[b, t - 1]

        if return_pars:
            # return states, a, d
            return states, (tau.squeeze(), lim.squeeze(), si_stat.squeeze(), si_q.squeeze()) # states, (tau, mu_lim, si_stat, si_q)
        else:
            return states

    def sample_states(self, contexts, return_pars=False):
        """Generates a dictionary of data sequence for each context (std or dvt) dynamics given a sequence of contexts

        Here contexts is the sequence of tone-by-tone boolean value standing for {std: 0, dvt: 1} that has been
        hierarchically defined prior to the call of sample_states

        Parameter sampling strategy:
        - tau and si_stat are sampled once per run.
        - lim is sampled per block.

        Processes dynamics:
        - states[1] (dvt) is updated at every block, and follows the same across the entire block
        - states[0] (std) is updated at every timestep.


        Parameters
        ----------
        contexts : integer np.array
            2-dimensional sequence of contexts filled with 0 or 1 (std or dvt), of size (N_blocks, N_tones)
        return_pars: bool
            also returns the time constant and sationary value (previously: retention and drift) parameters for each state at each block


        Returns
        -------
        states : dict
            dictionary encoding the hidden state values (one-dimensional np.array) for each
            context c (keys).
        pars (optional):
            tau: time constant parameter for each context (only if return_pars set to True)
            lim: stationary value parameter for each context
       

        """

        # Sample tau and si_stat for all of the run's blocks
        if self.params_testing:
            tau = np.array([self.mu_tau]) # for compatibility with later, since when self.params_testing, N_ctx can only be 1
            si_stat = self.si_stat
        else:
            # NOTE: for mu_tau=64, si_tau=0.5 the distributions covers well the range of values from 1 to 256
            tau = self._sample_logN_(min=1, mu=self.mu_tau, si=self.si_tau, size=self.N_ctx) # size = (N_ctx,) # TODO: should ensure tau_dev = (1/N_tones)*tau_std and tau_dev >= 1
            si_stat = self._sample_logN_(min=0, mu=self.si_stat, si=0.2).item() # std and dvt share the same stationary variance

        # Compute si_q (for both processes)
        si_q = si_stat * ((2 * tau - 1) ** 0.5) / tau
        
        # lim_Cs = self.sample_uniform_set(self.tones_values, N=self.N_ctx)
        # for c in range(self.N_ctx):
        #     lim[c] = self._sample_N_(lim_Cs[c], self.si_lim)
        
        # Sample lim once for the entire run
        lim = np.zeros((self.N_ctx,))        
        
        if self.N_ctx == 1:
            lim[0] = self._sample_N_(0, 1).item()

        # Sample d
        if self.N_ctx == 2:
            d = self._sample_biN_(self.mu_d, self.si_d).item()
            lim[0] = self._sample_N_(0.5, 0.5).item()
            if np.sign(lim[0]) == np.sign(d): lim[0] *= -1
            lim[1] = lim[0] + d * si_stat
        
        # Initialize states
        states = dict([(int(c), np.zeros(contexts.shape)) for c in range(self.N_ctx)])

        # --- STD process: update at every timestep ---
        for b in range(self.N_blocks):
            # Initial state for std process
            states[0][b, 0] = self._sample_N_(lim[0], si_stat).item()
            # Sample noise for std process
            w_std = self._sample_N_(0, si_q[0], contexts.shape[1])
            for t in range(1, contexts.shape[1]):
                # LGD update at individual tone level: x[t] = x[t-1] + 1/tau * (lim - x[t-1]) + noise
                states[0][b, t] = states[0][b, t - 1] + 1 / tau[0] * (lim[0] - states[0][b, t - 1]) + w_std[t - 1]

        # --- DVT process: update only at deviant position in each block ---
        if self.N_ctx > 1:
            for b in range(self.N_blocks):
                # Initial state for dvt process
                if b == 0:
                    # Sample the first value around the process' stationary value
                    states[1][b, :] = self._sample_N_(lim[1], si_stat, size=1)
                else:
                    # LGD update at block level: x[b] = x[b-1] + 1/tau * (lim - x[b-1]) + noise
                    w_dvt = self._sample_N_(0, si_q[1], size=1)
                    states[1][b, :] = states[1][b - 1, 0] + 1 / tau[1] * (lim[1] - states[1][b - 1, 0]) + w_dvt

        if return_pars:
            # return states, a, d
            return states, (tau.squeeze(), lim.squeeze(), si_stat, si_q.squeeze()) # states, (tau, mu_lim, si_stat, si_q)
        else:
            return states

    def sample_observations(self, contexts, states):
        """Generates a single data sequence y_t given a sequence of contexts c_t and a sequence of
        states x_t^c

        Parameters
        ----------
        contexts : integer np.array
            2-dimensional sequence of contexts filled with 0 or 1 (std or dvt), of size (N_blocks, N_tones)
        states : dict
            dictionary encoding the hidden state values (one-dimensional np.array) for each
            context c (keys).

        Returns
        -------
        y  : np.array
            2-dimensional sequence of observations of size (N_blocks, N_tones)
        """

        obs = np.zeros(contexts.shape)
        v = self._sample_N_(0, self.si_r, contexts.shape)

        for (b, t), c in np.ndenumerate(contexts):
            # Picking the state corresponding to current context c and adding normal noise
            obs[b, t] = states[c][b, t] + v[b, t]
        
        return obs

    def plot_contexts_states_obs(self, Cs, ys, x_stds, x_dvts, T, pars, figsize=(10, 6)):
        """For a non-hierarchical situation (only contexts std/dvt, no rules)

        Parameters
        ----------
        Cs : _type_
            sequence of contexts
        ys : _type_
            observations
        x_stds : _type_
            states of std
        x_dvts : _type_
            states of dvt
        """

        fig, ax1 = plt.subplots(figsize=figsize)
        ax1.plot(x_stds, label="x_std", color="green", linestyle="dotted", linewidth=2)
        ax1.plot(x_dvts, label="x_dvt", color="blue", linestyle="dotted", linewidth=2)
        ax1.plot(ys, label="y", color="red", linestyle="dashed", linewidth=2)
        ax1.set_ylabel("y")

        ax2 = ax1.twinx()
        ax2.plot(range(T), Cs, "o", color="black", label="context")
        ax2.set_ylabel("context")
        ax2.set_yticks(ticks=[0, 1], labels=["std", "dvt"])

        # Plot horizontal lines for lim_std and lim_dvt
        ax1.hlines(pars[1][0], xmin=0, xmax=len(x_stds)-1, color="green", linestyle="-", alpha=0.5, label="lim_std")
        ax1.hlines(pars[1][1], xmin=0, xmax=len(x_dvts)-1, color="blue", linestyle="-", alpha=0.5, label="lim_dvt")

        # Fill margin between lim ± si_stat for both processes
        ax1.fill_between(
            range(len(x_stds)),
            pars[1][0] - pars[2],
            pars[1][0] + pars[2],
            color="green",
            alpha=0.2,
            label="lim_std ± si_stat"
        )
        ax1.fill_between(
            range(len(x_dvts)),
            pars[1][1] - pars[2],
            pars[1][1] + pars[2],
            color="blue",
            alpha=0.2,
            label="lim_dvt ± si_stat"
        )

        fig.legend()

        fig.tight_layout()
        plt.show()

    def generate_batch(self, N_samples=None, return_pars=False):
        """Calls generate_run N_samples times and concatenates the return obsjects as (N_samples, *object_size) size objects
        I.e. generates N_samples samples / single sequences

        Parameters
        ----------
        N_samples : int, optional
            number of samples in one batch; if None takes value defined upon init of instance, by default None
        return_pars : bool, optional
            to return the hidden states (individual std and dvt) dynamics parameters tau and lim for each block in each batch, by default False

        Returns
        -------
        objects as in generate_run
            rules, rules_long, dpos, timbres, timbres_long, contexts, states, obs(, pars) (im the case of HGM) // contexts, states, obs(, pars) (in the case of N-HGM)
        """

        # Store latent rules and timbres, states and observations from N_samples batches
        # TODO: find a better way to store batches

        if N_samples is None:
            N_samples = self.N_samples

        batch = []

        for samp in tqdm(range(N_samples), desc="Generating sequences", leave=False):
            # Generate a batch of N_blocks sequences, sampling parameters and generating the paradigm's observations
            # *res == rules, rules_long, dpos, timbres, timbres_long, contexts, states, obs(, pars) (HGM) // contexts, states, obs(, pars) (NHGM)
            if self.params_testing:
                # sample a set of params
                if self.mu_tau_set is not None:
                    self.mu_tau = self.mu_tau_set[samp]
                if self.si_stat_set is not None:
                    self.si_stat = self.si_stat_set[samp]
                if self.si_r_set is not None:
                    self.si_r = self.si_r_set[samp]
                if self.si_d:
                    self.si_d = self.si_d_set[samp]
            res = self.generate_run(return_pars=return_pars)
            batch.append([*res])

        # Reorganize data as objects of size (N_samples, {obj_len}, 1) rather than a N_samples-long list of objects of size ({obj_len}, 1)
        # ! Except for the dictionary variable like states, that should keep the keys separate
        # res_reshaped = [np.stack([x for x in var_list], axis=0) for var_list in zip(*batches)]
        res_reshaped = tuple(reshape_batch_variable(var_list) for var_list in zip(*batch))

        return res_reshaped # TODO: should return pars here if params_testing as they're not sampled further down the pipeline



class NonHierachicalAuditGM(AuditGenerativeModel):
    """A generative model that only presents one level of context for a tone: to be a standard or a deviant tone
    Since data is not clustered in blocks defined by rules, there is only one "block" (N_block = 1)
    """

    def __init__(self, params):
        """
        Parameters
        ----------
        mu_rho_ctx :
            Mean of the truncated normal distribution to sample rho, the concentration (sticky) parameter of the transition matrix of contexts
        si_rho_ctx :
            Std of rho
        """

        super().__init__(params)
        self.N_blocks = 1
        self.mu_rho_ctx = params["mu_rho_ctx"]
        self.si_rho_ctx = params["si_rho_ctx"]

    def generate_run(
        self, return_pars=False
    ):
        """Generate data for one run of experiment: contexts, hidden states dynamics, observation

        Returns
        -------
        contexts:
            List of whether a tone is considered a dvt or a std --> contexts[t] = (current tone == dvt) (length = N_tones*N_blocks)
        states:
            List ynamics of both std (states[0]) and dvt (states[1]) at each "time step" (length = N_tones*N_blocks)
        obs:
            Observed tone at each time step (length = N_tones*N_blocks)
        pars: optional
            Time constant and sationary value parameters for each state at each block
        """

        # Get std/dvt contexts
        contexts = self.sample_contexts(
            N=self.N_tones, N_ctx=self.N_ctx, mu_rho_ctx=self.mu_rho_ctx, si_rho_ctx=self.si_rho_ctx
        )
        contexts = contexts.reshape((self.N_blocks, self.N_tones))

        # Sample states and observations
        if return_pars:
            states, pars = self.sample_states(contexts, return_pars)
        else:
            states = self.sample_states(contexts, return_pars)

        obs = self.sample_observations(contexts, states)

        # Flatten rules_long, contexts, (states, ) timbres and obs
        contexts = contexts.flatten()
        states = dict([(key, states[key].flatten()) for key in states.keys()])
        obs = obs.flatten()

        if return_pars:
            # Append self.si_r to the pars element of res (res[-1])
            pars = tuple((*pars, self.si_r))
            return contexts, states, obs, pars
        else:
            return contexts, states, obs


class HierarchicalAuditGM(AuditGenerativeModel):

    def __init__(self, params):

        super().__init__(params)

        self.N_blocks = params["N_blocks"]
        self.rules_dpos_set = params["rules_dpos_set"]
        self.N_rules = len(self.rules_dpos_set)
        self.mu_rho_rules = params["mu_rho_rules"]
        self.si_rho_rules = params["si_rho_rules"]
        self.mu_rho_timbres = params["mu_rho_timbres"]
        self.si_rho_timbres = params["si_rho_timbres"]

    def sample_rules(
        self, N_blocks, N_rules, mu_rho_rules, si_rho_rules, return_pi=False
    ):
        """Sample rules for a run consisting in a sequence of blocks of tones (each sequence being associated with one rule).
        Rules evolve in a Markov chain manner.

        Parameters
        ----------
        N_blocks : int
            Number of blocks of tones
        N_rules : int
            Number of rules
        mu_rho_rules : float
            _description_
        si_rho_rules : float
            _description_

        Returns
        -------
        np.array (N_blocks,)
            Sequence of rules associated with blocks for each block of N_blocks blocks
        """

        return self.sample_contexts(N=N_blocks, N_ctx=N_rules, mu_rho_ctx=mu_rho_rules, si_rho_ctx=si_rho_rules, return_pi=return_pi)

    def sample_timbres(self, rules_seq, N_timbres, mu_rho_timbres, si_rho_timbres):
        """Sample timbres, mediated by a Markov chain transition process too

        TODO: to check if correct

        Parameters
        ----------
        rules_seq : np.array
            _description_
        N_timbres : int
            _description_
        mu_rho_timbres : float
            _description_
        si_rho_timbres : float
            _description_

        Returns
        -------
        list
            _description_
        """

        # Sample timbres transition (emission from rule) matrix
        pi_timbre = self.sample_pi(N_timbres, mu_rho_timbres, si_rho_timbres)

        # timbres = np.array([np.random.choice(range(N_timbres), p=pi_timbre[seq]) for seq in rules_seq])
        timbres = np.array(
            [
                self.sample_next_markov_state(
                    current_state=seq,
                    states_values=range(N_timbres),
                    states_trans_matrix=pi_timbre,
                )
                for seq in rules_seq
            ]
        )

        return timbres

    def sample_dpos(self, rules, rules_dpos_set):
        """Sample positions of the deviant tones for each block of tones

        Parameters
        ----------
        rules : np.array
            Sequence of blocks rules_
        rules_dpos_set : np,array (N_rules, 3)
            Mapping of the 3 indexes of possible deviant positions for each of the rules

        Returns
        -------
        list
            List of deviant position indexes for each block of tones
        """

        # Given the trial's rule stored in rules and the possible positions for deviant associated with it,
        # sample the position where the deviant will be located in this trial
        # return [np.random.choice(rules_dpos_set[rule]) for rule in rules]
        return np.array(
            [self.sample_uniform_choice(rules_dpos_set[rule]) for rule in rules]
        )

    def get_contexts(self, dpos, N_blocks, N_tones):
        contexts = np.zeros((N_blocks, N_tones), dtype=np.int64)
        for i, pos in enumerate(dpos):
            contexts[i, pos] = 1
        return contexts

    def generate_run(
        self,
        return_pars = False
    ):
        """Generate data for one run of experiment: rules, dvt positions, timbres, contexts (std or dvt),
        states (hidden states dynamics), observations

        Returns
        -------
        rules:
            List of the rules that apply to each block of N_tones tones --> rules[b] = rule for current block (length = N_blocks)
        rules_long:
            List of rules that apply to each tone in the whole list of tones --> rules_long[t] is the same for every 8 consecutive t (length = N_tones*N_blocks)
        dpos:
            Positions of the dvt within each block (length = N_blocks)
        timbres:
            List of timbres associated with each block (NOTE: dynamics correct but physical values TBD, not implemented ATM) (length = N_blocks)
        timbres_long:
            Same as in rules_long, per tone list of timbres (length = N_tones*N_blocks)
        contexts:
            List of whether a tone is considered a dvt or a std --> contexts[t] = (current tone == dvt) (length = N_tones*N_blocks)
        states:
            List ynamics of both std (states[0]) and dvt (states[1]) at each "time step" (length = N_tones*N_blocks)
        obs:
            Observed tone at each time step (length = N_tones*N_blocks)
        pars: optional
            Time constant and sationary value parameters for each state at each block
        """

        # Sample sequence of rules ids
        rules = self.sample_rules(self.N_blocks, self.N_rules, self.mu_rho_rules, self.si_rho_rules)
        # Store latent rules in a per-tone array # This is equivalent to matlab's repmat
        rules_long = np.tile(rules[:, np.newaxis], (1, self.N_tones))

        # Sample timbres (here we consider that there are as many different timbres as there are different rules -- self.N_rules)
        timbres = self.sample_timbres(rules, self.N_rules, self.mu_rho_timbres, self.si_rho_timbres)
        # Store timbres in a per-tone array # This is equivalent to matlab's repmat
        timbres_long = np.tile(timbres[:, np.newaxis], (1, self.N_tones))

        # Sample deviant position
        dpos = self.sample_dpos(rules, self.rules_dpos_set)

        # Get contexts
        contexts = self.get_contexts(dpos, self.N_blocks, self.N_tones)

        # Sample states and observations
        # Sample states and observations
        if return_pars:
            states, pars = self.sample_states(contexts, return_pars)
        else:
            states = self.sample_states(contexts, return_pars)
        obs = self.sample_observations(contexts, states)

        # Flatten rules_long, contexts, (states, ) timbres and obs
        rules_long = rules_long.flatten()
        timbres_long = timbres_long.flatten()
        contexts = contexts.flatten()
        states = dict([(key, states[key].flatten()) for key in states.keys()])
        obs = obs.flatten()

        if return_pars:
            return rules, rules_long, dpos, timbres, timbres_long, contexts, states, obs, pars
        else:
            return rules, rules_long, dpos, timbres, timbres_long, contexts, states, obs

    def plot_contexts_rules_states_obs(self, x_stds, x_dvts, ys, Cs, rules, dpos, pars):
        """For the hierachical evolution of rules and contexts (NOTE: timbres not included in this viz atm)

        Parameters
        ----------
        x_stds : _type_
            _description_
        x_dvts : _type_
            _description_
        ys : _type_
            _description_
        Cs : _type_
            _description_
        rules : _type_
            _description_
        """

        # Visualize tone frequencies
        fig, ax1 = plt.subplots(figsize=(20, 6))
        ax1.plot(
            x_stds,
            label="x_std",
            color="green",
            marker="o",
            markersize=4,
            linestyle="dotted",
            linewidth=2,
            alpha=0.5,
        )
        ax1.plot(
            x_dvts,
            label="x_dvt",
            color="blue",
            marker="o",
            markersize=4,
            linestyle="dotted",
            linewidth=2,
            alpha=0.5,
        )
        ax1.plot(
            ys,
            label="y",
            color="k",
            marker="o",
            markersize=4,
            linestyle="dashed",
            linewidth=2,
        )
        ax1.set_ylabel("y")

        ax2 = ax1.twinx()
        ax2.plot(Cs, "o", color="black", label="context", markersize=2)
        ax2.set_ylabel("context")
        ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax2.set_yticks(ticks=[0, 1], labels=["std", "dvt"])

        rules_cmap = {0: "tab:blue", 1: "tab:red", 2: "tab:orange"}
        for i, rule in enumerate(rules):
            plt.axvspan(
                i * self.N_tones,
                i * self.N_tones + self.N_tones,
                facecolor=rules_cmap[rule],
                alpha=0.25,
            )

        for i in range(self.N_blocks):
            plt.axvline(i * self.N_tones, color="tab:gray", linewidth=0.9)

        # Add rule and deviant position texts above the plot
        text_y_position = 1.15  # Position above the plot
        for i in range(self.N_blocks):
            ax2.text(
            x=i * self.N_tones + 0.35 * self.N_tones,
            y=text_y_position,
            s=f"rule {rules[i]}",
            color=rules_cmap[rules[i]],
            transform=ax2.transData,
            ha="center",
            )
            ax2.text(
            x=i * self.N_tones + 0.35 * self.N_tones,
            y=text_y_position - 0.075,
            s=f"dvt {dpos[i]}",
            color=rules_cmap[rules[i]],
            transform=ax2.transData,
            ha="center",
            )

        # Plot horizontal lines for lim_std and lim_dvt
        ax1.hlines(pars[1][0], xmin=0, xmax=len(x_stds)-1, color="green", linestyle="-", alpha=0.5, label="lim_std")
        ax1.hlines(pars[1][1], xmin=0, xmax=len(x_dvts)-1, color="blue", linestyle="-", alpha=0.5, label="lim_dvt")

        # Fill margin between lim ± si_stat for both processes
        ax1.fill_between(
            range(len(x_stds)),
            pars[1][0] - pars[2],
            pars[1][0] + pars[2],
            color="green",
            alpha=0.2,
            label="lim_std ± si_stat"
        )
        ax1.fill_between(
            range(len(x_dvts)),
            pars[1][1] - pars[2],
            pars[1][1] + pars[2],
            color="blue",
            alpha=0.2,
            label="lim_dvt ± si_stat"
        )

        # Adjust the title to be below the plot
        tau_str = f"std: {pars[0][0]:.2f}, dvt: {pars[0][1]:.2f}" if self.N_ctx == 2 else f"{pars[0]:.2f}"
        si_q_str = f"std: {pars[3][0]:.2f}, dvt: {pars[3][1]:.2f}" if self.N_ctx == 2 else f"{pars[3]:.2f}"

        title_line1 = f"tau: {tau_str}; si_stat: {pars[2]:.2f}; si_q: {si_q_str}"
        title_line2 = f"(mu_tau: {self.mu_tau:.2f}, mu_si_stat: {self.si_stat:.2f}, mu_si_q: {self.si_stat * ((2 * self.mu_tau - 1) ** 0.5) / self.mu_tau:.2f}, si_r: {self.si_r:.2f})"
        plt.title(f"{title_line1}\n{title_line2}", y=-0.2)
        
        fig.legend(bbox_to_anchor=(1.1, 1))
        plt.tight_layout()
        plt.show()

    def plot_rules_dpos(self, rules, dpos, pars):

        # Visualize hierarchical information: dvt pos and rule

        rules_cmap = {0: "tab:blue", 1: "tab:red", 2: "tab:orange"}

        fig, ax = plt.subplots(figsize=(20, 6))
        for i, y in enumerate(dpos):
            ax.vlines(
                x=i,
                ymin=0,
                ymax=y,
                color="tab:gray",
                linewidth=0.9,
                zorder=1,
                alpha=0.5,
            )
        ax.scatter(
            range(len(dpos)), dpos, c=[rules_cmap[rule] for rule in rules], zorder=2
        )
        ax.set_ylabel("dvt pos")
        ax.set_xlabel("trial")
        ax.set_ylim(1, 8)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xticks(range(len(dpos)))

        handles = [
            plt.Line2D(
                [0], [0], marker="o", color="w", markerfacecolor=color, markersize=10
            )
            for color in rules_cmap.values()
        ]
        labels = rules_cmap.keys()
        ax.legend(handles, labels, title="rule")

        tau_str = f"std: {pars[0][0]:.2f}, dvt: {pars[0][1]:.2f}" if self.N_ctx == 2 else f"{pars[0]:.2f}"
        si_q_str = f"std: {pars[3][0]:.2f}, dvt: {pars[3][1]:.2f}" if self.N_ctx == 2 else f"{pars[3]:.2f}"
        title_line1 = f"tau: {tau_str}; si_stat: {pars[2]:.2f}; si_q: {si_q_str}"
        title_line2 = f"(mu_tau: {self.mu_tau:.2f}, mu_si_stat: {self.si_stat:.2f}, mu_si_q: {self.si_stat * ((2 * self.mu_tau - 1) ** 0.5) / self.mu_tau:.2f}, si_r: {self.si_r:.2f})"
        plt.title(f"{title_line1}\n{title_line2}", y=-0.2)
        fig.tight_layout()
        plt.show()



def example_HGM(config_H):
    gm = HierarchicalAuditGM(config_H)

    rules, rules_long, dpos, timbres, timbres_long, contexts, states, obs, pars = gm.generate_run(return_pars=True)
    # rules_, rules_long_, dpos_, timbres_, timbres_long_, contexts_, states_, obs_ = gm.generate_batch(N_samples=2) # calls generate_run N_samples times and concatenates the return obsjects as (N_samples, object_size) size objects

    # States, current blocks' rules, contexts based on rules and sampled deviant positions, and observations sampled from states based on context
    gm.plot_contexts_rules_states_obs(states[0], states[1], obs, contexts, rules, dpos, pars)

    # Deviant position for each rule
    gm.plot_rules_dpos(rules, dpos, pars)

    # An example of the states and observation sampling for one block
    gm.plot_contexts_states_obs(contexts[0:gm.N_tones], obs[0:gm.N_tones], states[0][0:gm.N_tones], states[1][0:gm.N_tones], gm.N_tones, pars=pars)



def example_NHGM(config_NH):
    gm_NH = NonHierachicalAuditGM(config_NH)

    contexts_NH, states_NH, obs_NH, pars = gm_NH.generate_run(return_pars=True)

    # States and observation sampled based on contexts
    gm_NH.plot_contexts_states_obs(contexts_NH, obs_NH, states_NH[0], states_NH[1], gm_NH.N_tones, pars=pars, figsize=(20, 6))


def example_single(config_single):

    tau_values = [1, 1.5, 2, 4, 8, 16, 32, 50]

    fig, axs = plt.subplots(len(tau_values), 1)

    for i, tau in enumerate(tau_values):
        config_single["mu_tau"]=tau
        gm = NonHierachicalAuditGM(config_single)
        _, states, obs, pars = gm.generate_run(return_pars=True)

        # Plot process states
        axs[i].plot(range(len(states[0])), states[0], label='x_hid', color='orange', linewidth=2)
            
        # Plot observation
        axs[i].plot(range(len(obs)), obs, color='tab:blue', label='y_obs')

        axs[i].set_title(f"mu_tau = {tau}, tau = {pars[0]:.2f}")

    plt.tight_layout()
    plt.show()

    


if __name__ == "__main__":

    # Example hierachical GM (rules, timbres [not implemented], std/dvt)
    config_H = {
        "N_samples": 1,
        "N_blocks": 20,
        "N_tones": 8,
        # "rules_dpos_set": np.array([[3, 4, 5], [4, 5, 6], [5, 6, 7]]),
        "rules_dpos_set": np.array([[3, 4, 5], [5, 6, 7]]),
        "mu_tau": 4,
        "si_tau": 1,
        "si_lim": 5,
        "mu_rho_rules": 0.9,
        "si_rho_rules": 0.05,
        "mu_rho_timbres": 0.8,
        "si_rho_timbres": 0.05,
        # "si_q": 2,  # process noise variance
        "si_stat": 0.5,  # stationary process variance
        "si_r": 0.2,  # measurement noise variance
        "si_d_coef": 0.05,
        "mu_d": 2
    }
    example_HGM(config_H)
    
    # Example non-hierachical GM (no rules, std/dvt)
    config_NH = {
        "N_samples": 1,
        "N_blocks": 1,
        "N_tones": 160,
        "mu_rho_ctx": 0.9,
        "si_rho_ctx": 0.05,
        "mu_tau": 4, # tau = 1 / (1 - a) = x_lim / b
        "si_tau": 1,
        "si_lim": 5,
        # "si_q": 2,  # process noise variance
        "si_stat": 0.2,  # stationary process variance
        "si_r": 0.2,  # measurement noise
        "si_d_coef": 0.05,
        "mu_d": 2
    }
    example_NHGM(config_NH)


    # Example 1 single process (1 context)
    config_single = {
        "N_samples": 1,
        "N_blocks": 1,
        "N_ctx": 1,
        "N_tones": 160,
        "mu_rho_ctx": 0.9,
        "si_rho_ctx": 0.05,
        "mu_tau": 4, # tau = 1 / (1 - a) = x_lim / b
        "si_tau": 1,
        "si_lim": 5,
        "si_stat": 0.2,  # stationary process variance
        "si_r": 0.2,  # measurement noise
    }
    # example_single(config_single)



