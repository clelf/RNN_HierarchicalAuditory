import os
import random
from pathlib import Path
import matplotlib.pyplot as plt
import evaluate_models as eval
from pipeline_next import plot_samples
import torch
import numpy as np

# Directory containing models
models_dir = Path(__file__).parent.resolve() / Path('training_results_CORRECT/N_ctx_2/NonHierarchicalGM_selected')

# Load model info
models_info = [eval.ModelInfo.from_path(models_dir / model_dir) for model_dir in os.listdir(models_dir)]

# Use the first model's data config for generating test data
data_info = models_info[0].data_config_dict
N_samples = 5
N_plots = N_samples
test_data_config = eval.TestDataConfig.from_saved_config(data_info, n_samples=N_samples)



# Plot samples for each model
output_dir = Path(__file__).parent.resolve() / Path('evaluation_results/example_samples')
output_dir.mkdir(parents=True, exist_ok=True)

for info in models_info[3:4]: # [3:4]

    test_data = eval.generate_test_data(test_data_config)
    data_config = test_data_config.to_gm_dict()

    model = eval.load_model(info)
    model_name = info.model_dir.name
    print(f"Plotting samples for model: {model_name}")
    # Evaluate model to get predictions
    y = test_data['y']
    y_np = test_data['y_np']
    pars = test_data['pars']
    contexts_np = test_data['contexts_np']
    
    # Forward pass
    with torch.no_grad():
        model_output = model(y[:, :-1, :])
        mu_pred, var_pred, context_output = eval.extract_model_predictions(model, model_output)

    # # Target is y[1:] (predicting next observation)
    # y_target = y_np[:, 1:]

    # Context predictions (aligned with y[1:])
    contexts_logits = context_output.detach().cpu().numpy()
    # Convert logits to probabilities and then to predicted context labels
    contexts_probs = torch.softmax(torch.from_numpy(contexts_logits), dim=-1).numpy()
    contexts_preds = np.argmax(contexts_probs, axis=-1)
    contexts_target = contexts_np[:, 1:]  # Align with predictions
    
    title = f"Samples for {model_name}"
    save_path = output_dir / f"{model_name}_samples"
    plot_samples(
        y_np, mu_pred, np.sqrt(var_pred), save_path, # TODO: check whether alignment between y and mu_pred is correct given mu_pred has one less time step than y
        title=title,
        params=pars,
        N_plots=N_plots,
        data_config=data_config,
        contexts=contexts_np,
        # seq_start=-100,
        seq_end=None,
        shared_ylim=True,
        contexts_probs=contexts_probs,
        contexts_preds=contexts_preds
    )