Drivethrough of scripts in this folder:

- evaluate_models.py: stores functions computing likelihoods, plotting diverse figures.

- model_analysis.py: Produces visualization evaluation on artificial test dataset (shared across models or one different per model, if different data configuration across models selected)

- model_activations.py: stores tools to extract and analyze activations of hidden states

- model_act_exp_trials.py: computes module activations on the experimental sequences

- model_act_exp_trials_deviant.py: computes module activations on the experimental sequences

- model_prob_exp_trials.py: computes module probabilities on the experimental sequences, on deviant positions and immediate next positions.

- run_exp_trials_pipeline.py: computes activations and probabilities as performed in the previously listed scripts, but all at once, for the models specified at the top of the script.

- exp_trials_selection.py: reads CSV files created by model_prob_exp_trials.py, and produces averages per sequence as well as visualizations

- assess_dpos_and_ctx_detection(_summary).py: produces CSV storing detailing the performance at detecting the deviant and predicting the deviant position for specified models (to match with the .txt legend file)

- plot_exp_trial_activity.py: plots activity per module across entire sequences

- plot_exp_trial_deviant_activity.py: plots activity per module across entire sequences, at deviant positions only

- plot_exp_trial_activity_by_position.py: obsolete


