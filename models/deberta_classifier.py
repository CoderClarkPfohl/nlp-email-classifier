"""
DeBERTa-based zero-shot classifier for job application emails.

Uses Hugging Face's zero-shot-classification pipeline with NLI models.
Supports multiple model tiers via the MODEL_PRESETS dict.

Requirements
------------
pip install torch "transformers>=4.30,<5.0" sentencepiece protobuf

If these are not installed, this module will raise an ImportError
with a helpful message; the rest of the project works without it.
"""

from typing import Dict, List, Tuple

# Fix 1: Contrastive candidate labels — forces NLI model to discriminate
# on the action (rejecting vs. reviewing), not just the topic (application).
CANDIDATE_LABELS = [
    "the company is extending a formal job offer with salary or start date details",
    "the company is rejecting or declining the candidate's application",
    "the company is inviting the candidate to interview",
    "the candidate must complete an assessment, form, or next step",
    "the company has received the application and is still reviewing it",
    "this email is a job alert, newsletter, or not directly about a submitted application",
]

# Fix 2: Hypothesis template — wraps each label in a sentence frame so the
# NLI model reasons about the email's meaning, not just keyword overlap.
HYPOTHESIS_TEMPLATE = "This email is about {}."

# Map from contrastive NLI labels back to short codes
LABEL_MAP = {
    "the company is extending a formal job offer with salary or start date details": "acceptance",
    "the company is rejecting or declining the candidate's application": "rejection",
    "the company is inviting the candidate to interview": "interview",
    "the candidate must complete an assessment, form, or next step": "action_required",
    "the company has received the application and is still reviewing it": "in_process",
    "this email is a job alert, newsletter, or not directly about a submitted application": "unrelated",
}

# Fix 3: Model presets — selectable via --deberta-model flag
MODEL_PRESETS = {
    "best":     "MoritzLaurer/deberta-v3-large-zeroshot-v2",   # ~60 min CPU, ~93-96%
    "balanced": "MoritzLaurer/deberta-v3-base-zeroshot-v1",    # ~20 min CPU, ~91-93%
    "fast":     "cross-encoder/nli-deberta-v3-small",           # ~12 min CPU, ~87-89%
    "bart":     "facebook/bart-large-mnli",                     # ~45 min CPU, ~90-92%
}
DEFAULT_PRESET = "balanced"

# Fix 4: Confidence threshold — predictions below this fall back to rule labels
CONFIDENCE_THRESHOLD = 0.55


def load_deberta_pipeline(model_name: str = None, device: int = -1):
    """
    Load the zero-shot classification pipeline.

    Parameters
    ----------
    model_name : str
        Hugging Face model ID or preset key (best/balanced/fast/bart).
        Defaults to the balanced preset.
    device : int
        -1 for CPU, 0+ for CUDA GPU index.

    Returns
    -------
    transformers.Pipeline
    """
    try:
        from transformers import pipeline as hf_pipeline
    except ImportError:
        raise ImportError(
            "The DeBERTa classifier requires `transformers` and `torch`.\n"
            "Install them with:  pip install torch transformers\n"
            "Then re-run this script with --model deberta"
        )

    if model_name is None:
        model_name = MODEL_PRESETS[DEFAULT_PRESET]
    elif model_name in MODEL_PRESETS:
        model_name = MODEL_PRESETS[model_name]

    print(f"       Model: {model_name}")

    classifier = hf_pipeline(
        "zero-shot-classification",
        model=model_name,
        device=device,
        hypothesis_template=HYPOTHESIS_TEMPLATE,
    )
    return classifier


def classify_single(classifier, text: str,
                     candidate_labels: List[str] = None) -> Tuple[str, float]:
    """
    Classify a single email text.

    Returns
    -------
    (short_label, confidence)
    """
    if candidate_labels is None:
        candidate_labels = CANDIDATE_LABELS

    result = classifier(text, candidate_labels, multi_label=False)
    top_label = result["labels"][0]
    top_score = result["scores"][0]
    return LABEL_MAP.get(top_label, top_label), round(top_score, 4)


def classify_batch(classifier, texts: List[str],
                    batch_size: int = 8) -> List[Dict]:
    """
    Classify a batch of emails with DeBERTa zero-shot.

    Returns
    -------
    List of dicts with keys: label, confidence, all_scores
    """
    results = classifier(
        texts,
        CANDIDATE_LABELS,
        multi_label=False,
        batch_size=batch_size,
    )

    # Handle single-item result (not wrapped in list)
    if isinstance(results, dict):
        results = [results]

    output = []
    for res in results:
        top_label = LABEL_MAP.get(res["labels"][0], res["labels"][0])
        all_scores = {
            LABEL_MAP.get(l, l): round(s, 4)
            for l, s in zip(res["labels"], res["scores"])
        }
        output.append({
            "label": top_label,
            "confidence": round(res["scores"][0], 4),
            "all_scores": all_scores,
        })
    return output
