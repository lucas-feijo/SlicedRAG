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

from slicegpt import gpu_utils, hf_utils, utils
from slicegpt.config import config

import os
import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import evaluate
from tqdm import tqdm


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
        round_interval=args.get("round_interval", None),
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
        limit=args.get("limit", None),
        write_out=True,
        log_samples=args.get("log_samples", False)
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


def run_squad1_evaluation(
    model_name="facebook/opt-125m",
    limit=100,
    max_new_tokens=32,
    save_dir="../../experiments/results/",
    save_name="results_squadv1_manual.json",
    device=None,
):
    """
    Run a manual SQuAD1.1 evaluation using HF transformers.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID, e.g. "facebook/opt-125m".
    limit : int or None
        Number of SQuAD validation examples to evaluate (None = full dataset).
    max_new_tokens : int
        Maximum number of tokens for generation.
    save_dir : str
        Directory where results will be saved.
    save_name : str
        Filename of the output JSON.
    device : str or None
        Override device (e.g. "cpu" or "cuda"). If None, auto-detect.

    Returns
    -------
    dict
        Metrics dictionary with EM and F1.
    """

    # ----------------------------
    # Detect device
    # ----------------------------
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ----------------------------
    # Load tokenizer & model
    # ----------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    # OPT models often have no padding token → use EOS
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ----------------------------
    # Load SQuAD validation set
    # ----------------------------
    dataset = load_dataset("squad", split="validation")

    if limit is not None:
        dataset = dataset.select(range(limit))

    print(f"Evaluating on {len(dataset)} examples")

    predictions = []
    references = []

    # ----------------------------
    # Loop over examples
    # ----------------------------
    for i, doc in enumerate(tqdm(dataset)):
        prompt = (
            "Context: " + doc["context"]
            + "\nQuestion: " + doc["question"]
            + "\nAnswer:"
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Extract only generated tokens
        generated = output[0, inputs["input_ids"].shape[1]:]
        answer_text = tokenizer.decode(generated, skip_special_tokens=True).strip()

        # DEBUG: print first 5 examples
        if i < 5:
            print("\n===== DEBUG EXAMPLE =====")
            print("Q:", doc["question"])
            print("GT:", doc["answers"]["text"])
            print("PRED:", repr(answer_text))

        predictions.append({
            "id": doc["id"],
            "prediction_text": answer_text,
        })

        references.append({
            "id": doc["id"],
            "answers": doc["answers"],
        })

    # ----------------------------
    # Compute SQuAD metrics
    # ----------------------------
    metric = evaluate.load("squad")
    scores = metric.compute(predictions=predictions, references=references)

    print("\n=== FINAL SCORES ===")
    print(scores)

    # ----------------------------
    # Save results
    # ----------------------------
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, save_name)

    with open(out_path, "w") as f:
        json.dump({
            "results": scores,
            "num_examples": len(dataset),
            "model": model_name,
            "max_new_tokens": max_new_tokens,
        }, f, indent=2)

    print(f"\nSaved results to: {out_path}\n")

    return scores

'''
How to call the evalsquad function:

run_squad1_evaluation(
    model_name="facebook/opt-125m",
    limit=100,
    max_new_tokens=16,
    save_dir="../../experiments/results/",
    save_name="opt125m_squadv1.json"
)

'''


def run_squad2_evaluation(
    model_name="facebook/opt-125m",
    limit=100,
    max_new_tokens=32,
    save_dir="../../experiments/results/",
    save_name="results_squadv2_manual.json",
    device=None,
):
    """
    Run a manual SQuAD2.0 evaluation using HF transformers.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID, e.g. "facebook/opt-125m".
    limit : int or None
        Number of SQuAD v2 validation examples to evaluate (None = full dataset).
    max_new_tokens : int
        Maximum number of tokens for generation.
    save_dir : str
        Directory where results will be saved.
    save_name : str
        Filename of the output JSON.
    device : str or None
        Override device (e.g. "cpu" or "cuda"). If None, auto-detect.

    Returns
    -------
    dict
        Metrics dictionary with EM and F1 for SQuAD2.0.
    """

    # ----------------------------
    # Detect device
    # ----------------------------
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ----------------------------
    # Load tokenizer & model
    # ----------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    # OPT models often have no padding token → use EOS
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ----------------------------
    # Load SQuAD2.0 validation set
    # ----------------------------
    dataset = load_dataset("squad_v2", split="validation")

    if limit is not None:
        dataset = dataset.select(range(limit))

    print(f"Evaluating on {len(dataset)} examples")

    predictions = []
    references = []

    # ----------------------------
    # Loop over examples
    # ----------------------------
    for i, doc in enumerate(tqdm(dataset)):
        # For SQuAD v2 we explicitly allow no-answer
        prompt = (
            "Context: " + doc["context"]
            + "\nQuestion: " + doc["question"]
            + "\nIf the question cannot be answered from the context, answer with 'unanswerable'."
            + "\nAnswer:"
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Extract only generated tokens
        generated = output[0, inputs["input_ids"].shape[1]:]
        raw_answer = tokenizer.decode(generated, skip_special_tokens=True).strip()

        # Simple heuristic: interpret "unanswerable" as no-answer (empty string)
        raw_lower = raw_answer.lower()
        if raw_lower.startswith("unanswerable"):
            pred_text = ""
        else:
            pred_text = raw_answer

        # DEBUG: print first 5 examples
        if i < 5:
            print("\n===== DEBUG EXAMPLE =====")
            print("Q:", doc["question"])
            print("GT answers:", doc["answers"]["text"])
            print("RAW PRED:", repr(raw_answer))
            print("PRED USED:", repr(pred_text))

        predictions.append({
            "id": doc["id"],
            "prediction_text": pred_text,
            # required for squad_v2 metric, even if we don't model it
            "no_answer_probability": 0.0,
        })

        references.append({
            "id": doc["id"],
            "answers": doc["answers"],  # may be empty if is_impossible=True
        })

    # ----------------------------
    # Compute SQuAD v2 metrics
    # ----------------------------
    metric = evaluate.load("squad_v2")
    scores = metric.compute(predictions=predictions, references=references)

    print("\n=== FINAL SQuAD v2 SCORES ===")
    print(scores)

    # ----------------------------
    # Save results
    # ----------------------------
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, save_name)

    with open(out_path, "w") as f:
        json.dump({
            "results": scores,
            "num_examples": len(dataset),
            "model": model_name,
            "max_new_tokens": max_new_tokens,
        }, f, indent=2)

    print(f"\nSaved results to: {out_path}\n")

    return scores




