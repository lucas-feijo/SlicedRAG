import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from slicegpt import hf_utils

def build_context(example):
    paragraphs = []

    for paragraph_title, sentences in example["context"]:
        paragraph_text = " ".join(sentences)
        paragraphs.append(f"[{paragraph_title}] {paragraph_text}")

    return "\n".join(paragraphs)


def generate_hotpot_predictions_sliced(model_name, model_path, sparsity, dataset_path, out_path=None, max_new_tokens=64, limit=None):
    """
    Generate predictions for the HotpotQA dataset using a sliced model.

    Args:
        model_name: Name of the model architecture.
        model_path: Path to the sliced model weights.
        sparsity: Sparsity level of the sliced model.
        dataset_path: Path to the HotpotQA dataset in JSON format.
        out_path: Optional path to save the predictions as a JSON file.
        max_new_tokens: Maximum number of new tokens to generate for each answer.
        limit: Optional limit on the number of examples to process.

    Returns:
        A dictionary containing the predictions.
    """
    model_adapter, tokenizer  = hf_utils.load_sliced_model(
        model_name,
        model_path,
        sparsity=sparsity
    )
    model = model_adapter.model.to(device="cuda")
    return generate_hotpot_predictions(model, tokenizer, dataset_path, out_path, max_new_tokens, limit)


def generate_hotpot_predictions_hf(model_name, dataset_path, out_path=None, max_new_tokens=64, limit=None):
    """
    Generate predictions for the HotpotQA dataset using a Hugging Face model.

    Args:
        model_name: Name or path of the Hugging Face model.
        dataset_path: Path to the HotpotQA dataset in JSON format.
        out_path: Optional path to save the predictions as a JSON file.
        max_new_tokens: Maximum number of new tokens to generate for each answer.
        limit: Optional limit on the number of examples to process.

    Returns:
        A dictionary containing the predictions.
    """

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
    return generate_hotpot_predictions(model, tokenizer, dataset_path, out_path, max_new_tokens, limit)


def generate_hotpot_predictions(model, tokenizer, dataset_path, out_path=None, max_new_tokens=64, limit=None):
    """
    Generate predictions for the HotpotQA dataset.

    Args:
        model: The language model to use for generation.
        tokenizer: The tokenizer corresponding to the model.
        dataset_path: Path to the HotpotQA dataset in JSON format.
        out_path: Optional path to save the predictions as a JSON file.
        max_new_tokens: Maximum number of new tokens to generate for each answer.
        limit: Optional limit on the number of examples to process.
    
    Returns:
        A dictionary containing the predictions.
    """

    model.eval()

    with open(dataset_path, "r", encoding="utf-8") as f:
        examples = json.load(f)

    predictions = {
        "answer": {},
        "sp": {}
    }

    if limit is None:
        limit = len(examples)

    for example in tqdm(examples[:limit]):
        question_id  = example["_id"]
        question_text = example["question"]

        context_text = build_context(example)

        prompt = (
            "Answer the question using the context."
            "Reply only with a short answer."
            f"Context: \n{context_text}\n\n"
            f"Question: {question_text}\n\n"
            "Answer: "
        )

        model_inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=tokenizer.model_max_length,
        ).to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )

        generated_text = tokenizer.decode(
            generated_ids[0][model_inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        )

        short_answer = generated_text.splitlines()[0].strip()

        predictions["answer"][question_id] = short_answer
        predictions["sp"][question_id] = []

    if out_path is not None:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(predictions, f)

        print(f"Predictions saved to {out_path}")
    
    return predictions