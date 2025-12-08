import json
import os

import pandas as pd

def load_sparsity_eval_results(base_directory,
                               model_name,
                               dataset_name,
                               task_names,
                               task,
                               prefix="",
                               sparsities=None):
    """
    Load and return a pandas dataframe containing evaluation results for different model sparsity levels.

    Args:
        base_directory: str - directory where results are stored
        model_name: str - name of the model family (as stated on HuggingFace, if using a HF model).
            Example: "facebook/opt-125m"
        dataset_name: str - name of the calibration dataset used for pruning. Example: "wikitext2"
        task_names: list of str - names of tasks on the specified run. Only used to identify the correct file.
            Example: ["squadv2", "coqa", "mmlu", "nq_open"]
        task: str - name of the task whose results are to be loaded. Must be one of task_names.
        prefix: str - file prefix, if any. Example: "debug_"
        sparsities: list of str - model sparsity levels to load. Default: [0.0, 0.1, 0.25, 0.4, 0.6]
    """

    if sparsities is None:
        sparsities = [0.0, 0.1, 0.25, 0.4, 0.6]

    rows = []

    for sparsity in sparsities:
        model_str = model_name.replace("/", "-")
        results_directory = os.path.join(base_directory,
                                         f"{prefix}{model_str}_{dataset_name}")
        file_name = f"results_s{sparsity:.2f}_{"_".join(task_names)}.json"
        file_path = os.path.join(results_directory, file_name)

        with open(file_path, "r") as f:
            data = json.load(f)

        try:
            metrics = data[task]
        except KeyError:
            metrics = data["results"][task]

        row = {"sparsity": sparsity}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                row[key] = value

        rows.append(row)

    df = pd.DataFrame(rows)

    return df