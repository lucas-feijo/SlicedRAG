import json
import logging
import sys
import os

import lm_eval
# import torch
from lm_eval import tasks
from lm_eval import utils as lm_eval_utils
from lm_eval.api.registry import ALL_TASKS
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import initialize_tasks

from slicegpt import gpu_utils, hf_utils, utils
from slicegpt.config import config

logging.basicConfig(
    level=logging.DEBUG,
    stream=sys.stdout,
    format="%(levelname)s: %(message)s"
)

def get_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # remove handlers if already set, to avoid double logging
    for h in list(logger.handlers):
        logger.removeHandler(h)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger


def eval(args):
	"""
	(Based on SliceGPT's run_lm_eval.py)
	Evaluate a sliced model using LM Evaluation Harness
	
	args:
	  - "model": HuggingFace model name, e.g. "facebook/opt-125m"
	  - "sliced_model_path": path to sliced checkpoint dir
      - "sparsity": float, e.g. 0.25
      - "round_interval": int or None (optional)
      - "batch_size": int
      - "tasks": None, string (comma-separated) or list of patterns
      - "num_fewshot": int
      - "limit": int or None
      - "save_dir": output directory for results

	:param args: argument dict
	"""
     
	logger = get_logger()
	logger.info("Running Evaluation")

	logger.info(
		f"Loading sliced {args["model"]} from {args["sliced_model_path"]} "
		f"with sparsity {args["sparsity"]}"
		)
	
	model_adapter, tokenizer = hf_utils.load_sliced_model(
		args["model"],
		args["sliced_model_path"],
		sparsity=args["sparsity"],
		token=None,
		round_interval=args.get(["round_interval"], None),
	)

	if hasattr(model_adapter.model, "tie_weights"):
		model_adapter.model.tie_weights = lambda: None

	model_adapter.model.to(config.device)

	hflm = HFLM(
		pretrained=model_adapter.model,
		tokenizer=tokenizer,
		batch_size=args["batch_size"]
		)

	if args["tasks"] is None:
		task_names = tasks.ALL_TASKS
	else:
		task_names = lm_eval_utils.pattern_match(args["tasks"], ALL_TASKS)

	logger.info(f"Selected Tasks: {task_names}")

	results = lm_eval.simple_evaluate(
		hflm,
		tasks=task_names,
		num_fewshot=args["num_fewshot"],
		batch_size=args["batch_size"],
		limit=args["limit"],
		write_out=True,
		log_samples=True
		)

	logger.info("Results:")
	logger.info(results)

	os.makedirs(args["save_dir"], exist_ok=True)

	sparsity_tag = f"{args["sparsity"]:.2f}"
	result_filename = os.path.join(
		args["save_dir"],
		f"results_s{args["sparsity"]:.2f}_{"_".join(task_names)}.json"
	)
	
	with open(result_filename, "w") as f:
		json.dump(results, f, indent=2)

	logger.info(f"Saved results to {result_filename}")

	return results

