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


def eval_baseline(args):
    """
    Run LM Evaluation Harness on either:
      - a sliced (pruned) model, if 'sliced_model_path' is provided
      - or a dense HF model, if not.
    """
    logger = get_logger()
    logger.info("Running Evaluation")

    use_sliced = args.get("sliced_model_path") is not None and args.get("sparsity", 0.0) > 0.0

    if use_sliced:
        # ---------- SLICED MODEL BRANCH ----------
        logger.info(
            f"Loading SLICED model {args['model']} from {args['sliced_model_path']} "
            f"with sparsity {args['sparsity']}"
        )

        model_adapter, tokenizer = hf_utils.load_sliced_model(
            args["model"],
            args["sliced_model_path"],
            sparsity=args["sparsity"],
            token=None,
            round_interval=args.get("round_interval", None),
        )

        # For sliced models, disable weight tying
        if hasattr(model_adapter.model, "tie_weights"):
            model_adapter.model.tie_weights = lambda *x, **kw: None

        model_adapter.model.to(config.device)

        hflm = HFLM(
            pretrained=model_adapter.model,
            tokenizer=tokenizer,
            batch_size=args["batch_size"],
        )

    else:
        # ---------- DENSE BASELINE BRANCH ----------
        logger.info(
            f"Loading DENSE baseline model {args['model']} directly from HF hub"
        )

        # Here we let LM Eval Harness load the model & tokenizer itself
        hflm = HFLM(
            pretrained=args["model"],
            batch_size=args["batch_size"],
        )

    # ---------- Task selection ----------
    if args["tasks"] is None:
        logger.warning(
            "args['tasks'] is None -> using ALL_TASKS. "
            "This may be very slow / memory-heavy."
        )
        task_names = tasks.ALL_TASKS
    else:
        if isinstance(args["tasks"], str):
            patterns = [t.strip() for t in args["tasks"].split(",") if t.strip()]
        else:
            patterns = args["tasks"]
        task_names = lm_eval_utils.pattern_match(patterns, ALL_TASKS)

    logger.info(f"Selected Tasks: {task_names}")

    # ---------- Run evaluation ----------
    results = lm_eval.simple_evaluate(
        hflm,
        tasks=task_names,
        num_fewshot=args["num_fewshot"],
        batch_size=args["batch_size"],
        limit=args["limit"],
        write_out=True,
        log_samples=False,
    )

    logger.info("Results (metrics only):")
    logger.info(results["results"])


    # ---------- Save results ----------
    os.makedirs(args["save_dir"], exist_ok=True)

    # Use sparsity=0.0 if not present, so filenames still make sense
    sparsity_val = float(args.get("sparsity", 0.0))
    sparsity_tag = f"{sparsity_val:.2f}"

    result_path = os.path.join(
        args["save_dir"],
        f"results_s{sparsity_tag}_{'_'.join(task_names)}_light.json",
    )

    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved results to {result_path}")

    return results


