#CS 7347 Natural Language Processing


#***DeBERTa-based Zero-Shot Classifier***

from typing import Dict, List, Tuple


CANDIDATE_LABELS = [
    "the company is extending a formal job offer with salary or start date details",
    "the company is rejecting or declining the candidate's application",
    "the company is inviting the candidate to interview",
    "the candidate must complete an assessment, form, or next step",
    "the company has received the application and is still reviewing it",
    "this email is a job alert, newsletter, or not directly about a submitted application",
]

#hypothesis template (zero-shot classification)
HYPOTHESIS_TEMPLATE = "This email is about {}."

#mapping from long lbls to short lbls
LABEL_MAP = {
    "the company is extending a formal job offer with salary or start date details": "acceptance",
    "the company is rejecting or declining the candidate's application": "rejection",
    "the company is inviting the candidate to interview": "interview",
    "the candidate must complete an assessment, form, or next step": "action_required",
    "the company has received the application and is still reviewing it": "in_process",
    "this email is a job alert, newsletter, or not directly about a submitted application": "unrelated",
}

#model presets
MODEL_PRESETS = {
    "best":     "MoritzLaurer/deberta-v3-large-zeroshot-v2",   # ~60 min CPU, ~93-96%
    "balanced": "MoritzLaurer/deberta-v3-base-zeroshot-v1",    # ~20 min CPU, ~91-93%
    "fast":     "cross-encoder/nli-deberta-v3-small",           # ~12 min CPU, ~87-89%
    "bart":     "facebook/bart-large-mnli",                     # ~45 min CPU, ~90-92%
}
DEFAULT_PRESET = "balanced"


CONFIDENCE_THRESHOLD = 0.55

#loading deberta
def load_deberta_pipeline(model_name: str = None, device: int = -1):
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

#classify a single email with DeBERTa
def classify_single(classifier, text: str, candidate_labels: List[str] = None) -> Tuple[str, float]:
    
    if candidate_labels is None:
        candidate_labels = CANDIDATE_LABELS

    result = classifier(text, candidate_labels, multi_label=False)
    top_label = result["labels"][0]
    top_score = result["scores"][0]
    return LABEL_MAP.get(top_label, top_label), round(top_score, 4)


def classify_batch(classifier, texts: List[str], batch_size: int = 8) -> List[Dict]:
 
    results = classifier(
        texts,
        CANDIDATE_LABELS,
        multi_label=False,
        batch_size=batch_size,
    )


    if isinstance(results, dict):
        results = [results]

    output = []
    for res in results:
        #map long label to short label, round scores
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
