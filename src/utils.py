import random
from datasets import load_dataset
import os
from transformers import AutoTokenizer

def preprocess_conversations(example, randomize):
    """
    Randomly maps the 'chosen' and 'rejected' columns to 'conversation_a' and 'conversation_b',
    and creates a 'winner' column indicating whether model_a or model_b is the winner.
    
    If the chosen conversation is assigned to conversation_a, then model_a wins;
    otherwise, if it ends up in conversation_b, then model_b wins.
    """
    if (not randomize) or random.random() < 0.5:
        # Map chosen to conversation_a and rejected to conversation_b
        conversation_a = example["chosen"]
        conversation_b = example["rejected"]
        winner = "model_a"
    else:
        # Map rejected to conversation_a and chosen to conversation_b
        conversation_a = example["rejected"]
        conversation_b = example["chosen"]
        winner = "model_b"
    
    # Return a new dictionary with the updated fields
    return {
        "conversation_a": conversation_a,
        "conversation_b": conversation_b,
        "winner": winner
    }

def switch_conversations(example):
    if random.random() < 0.5:
        conversation_a, conversation_b = example["conversation_b"], example["conversation_a"]
        # winner now points to whichever model occupies slot A
        winner = "model_b" if example["winner"] == "model_a" else "model_a"
    else:
        conversation_a, conversation_b = example["conversation_a"], example["conversation_b"]
        winner = example["winner"]

    return {
        "conversation_a": conversation_a,
        "conversation_b": conversation_b,
        "winner": winner
    }

def filter_sample(sample, tokenizer, max_filter_length: int = 512):
    prompt = sample.get("prompt", "")
    try:
        chosen = sample["chosen"][-1]["content"]
        rejected = sample["rejected"][-1]["content"]
    except (KeyError, IndexError):
        return False

    chosen_score = sample.get("score_chosen", 0)
    rejected_score = sample.get("score_rejected", 0)

    return (
        len(tokenizer.encode(prompt)) < max_filter_length
        and len(tokenizer.encode(chosen)) < max_filter_length
        and len(tokenizer.encode(rejected)) < max_filter_length
        and chosen_score > rejected_score
        and "confidence" not in chosen.lower()
        and "confidence" not in rejected.lower()
    )


def apply_template(prompt: str) -> str:
    return f"<|user|>{prompt}<|assistant|>"

def load_ultra_subset(num_examples, randomize = False, ref_model = None, tokenizer = None, train = False):
    split = "train_prefs" if train else "test_prefs"
    print(f"Loading {split} set")
    dataset = load_dataset("tevinw/processed_ultrafeedback_binarized_prefs_sft", split=split)

    if randomize:
        seed = random.randint(0, 2**32 - 1)
        print("Random seed:", seed)
        dataset = dataset.shuffle(seed=seed)

    if train:
        return dataset.select(range(num_examples)) \
            .map(lambda x: preprocess_conversations(x, randomize=False))
    elif tokenizer:
        prompts = []
        for sample in dataset:
            if filter_sample(sample, tokenizer):
                prompts.append(apply_template(sample["prompt"]))
            if len(prompts) >= 1024:
                break
        return prompts
    else:
        raise Exception("tokenizer not found but test split required")

# .filter(lambda x: filter_sample(x, tokenizer)) \
from datasets import Dataset
import numpy as np

def sample_k_per_qid(
    ds: Dataset,
    qid_ranges: list[tuple[int, int]],
    k: int = 8,
    qid_col: str = "question_id",
    seed: int = 42,
    first = False,
) -> Dataset:
    """
    Args
    ----
    ds : Hugging Face `Dataset`
        The original dataset.
    qid_ranges : list of (start, end) tuples
        Inclusive ranges of question-IDs to keep, e.g. [(0, 999), (2000, 2500)].
    k : int, default 8
        Number of samples to keep per question-ID.
    qid_col : str, default "question_id"
        Column that contains the question ID.
    seed : int, default 42
        RNG seed for deterministic sampling.

    Returns
    -------
    Dataset
        A new Dataset containing `k` randomly chosen rows for every
        `question_id` inside the supplied ranges.

    Raises
    ------
    ValueError
        If any question-ID has fewer than `k` rows available.
    """
    # ------------------------------------------------------------------ #
    # 1.  Build a set of all allowed question IDs                        #
    # ------------------------------------------------------------------ #
    allowed_qids = {
        qid
        for start, end in qid_ranges
        for qid in range(start, start + 1)
    }

    print("allowed len:", len(allowed_qids))

    # ------------------------------------------------------------------ #
    # 2.  Filter upfront to those IDs (fast – runs in parallel)          #
    # ------------------------------------------------------------------ #
    filtered = ds.filter(
        lambda ex: ex[qid_col] in allowed_qids,
        num_proc=min(8, ds.num_rows)  # parallel if you like
    )

    # ------------------------------------------------------------------ #
    # 3.  Collect row indices per question-ID                            #
    # ------------------------------------------------------------------ #
    buckets: dict[int, list[int]] = {}
    for idx, qid in enumerate(filtered[qid_col]):
        buckets.setdefault(qid, []).append(idx)

    # ------------------------------------------------------------------ #
    # 4.  Sample exactly `k` indices per bucket with a fixed RNG         #
    # ------------------------------------------------------------------ #
    selected_indices = []
    for qid, idxs in buckets.items():
        if len(idxs) < k:
            selected = idxs
        elif first:
            selected = idxs[:k]
        else:
            rng = np.random.default_rng(seed)
            selected = rng.choice(idxs, size=k, replace=False)
        selected_indices.extend(selected)

    # Keep original order (optional)
    selected_indices.sort()

    return filtered.select(selected_indices)

def load_mt_subset(k: int = 8, train = False):
    if train:
        print("Loading training set")
        return sample_k_per_qid(load_dataset("lmsys/mt_bench_human_judgments", split="human").filter(lambda example: "tie" not in example["winner"].lower()), [(81, 85), (91, 95), (101, 105), (111, 115), (121, 125), (131, 135), (141, 145), (151, 155)], k=k).map(switch_conversations)
    else:
        print("Loading test set")
        return sample_k_per_qid(load_dataset("lmsys/mt_bench_human_judgments", split="human").filter(lambda example: "tie" not in example["winner"].lower()), [(86, 90), (96, 100), (106, 110), (116, 120), (126, 130), (136, 140), (146, 150), (156, 160)], k=k, first=True)

# UltraFeedback rules
# rules = [
#     "The assistant's responses should present explanations in a coherent, step-by-step structure with logical flow, numbered points, and clear sections.",
#     "When addressing user misconceptions, the assistant must clarify misunderstandings before offering solutions.",
#     "Translations must use accurate terminology, preserve original tone and structure, and avoid introducing unrelated content.",
#     "Responses must prioritize technical accuracy, correct formulas, error-free code examples, and validated context alignment.",
#     "Incorporate vivid sensory details, figurative language, and relatable examples when explicitly requested.",
#     "Provide actionable advice, practical steps, and concrete implementation strategies tailored to the user's context.",
#     "Indicate confidence levels while acknowledging uncertainty and limitations when appropriate.",
#     "Maintain a conversational, empathetic, and professional tone while avoiding overly formal or dismissive language.",
#     "Integrate cultural sensitivity, domain-specific terminology, and contextual relevance into explanations.",
#     "Include properly formatted citations, references, and academic conventions when required.",
#     "Address all components of the user's query comprehensively without omission or tangential content.",
#     "Avoid assumptions when ambiguity exists; seek clarification for insufficient context.",
#     "Use illustrative examples of both correct/incorrect approaches to demonstrate concepts.",
#     "Strictly adhere to user-specified formats, structures, and output requirements.",
#     "Address ethical considerations, legal compliance, and recommend professional consultation when relevant.",
#     "Prioritize security measures, error handling, and technical robustness in solutions.",
#     "Ensure conciseness by eliminating redundancy and focusing on core query relevance.",
#     "Explain underlying mechanisms, reasoning processes, and cause-effect relationships explicitly.",
#     "Validate answers against provided context and avoid unsupported extrapolation.",
#     "Maintain narrative coherence with source material when discussing plots or characters.",
#     "Structure comparisons, analyses, and recommendations using clear categorization.",
#     "Anticipate user needs by providing comprehensive details without requiring follow-ups.",
#     "Preserve specific terms, measurements, and formatting conventions during localization.",
#     "Use collaborative language and hierarchical organization for complex information.",
#     "Balance thoroughness with brevity to prevent information overload while ensuring clarity."
# ]



# MT-Bench rules
rules = [
    "The assistant's responses must provide detailed step-by-step explanations and calculations to ensure correctness and clarity.",
    "The assistant's code should avoid unnecessary complexity, handle edge cases, include error handling, and use appropriate data structures.",
    "The assistant's responses must maintain a professional and approachable tone, adapting to the nature of the user's query.",
    "The assistant's responses must strictly adhere to user-specified formats (e.g., JSON/YAML) with correct syntax and structure.",
    "The assistant's explanations should prioritize logical coherence, clarity, and avoidance of redundant or ambiguous content.",
    "The assistant must adhere to ethical guidelines by avoiding medical diagnoses and prioritizing user safety in critical situations.",
    "Creative outputs must maintain structural integrity (e.g., rhyme schemes, metaphors) while retaining key informational elements.",
    "The assistant should proactively address user misunderstandings, anticipate follow-up questions, and provide actionable feedback.",
    "The assistant must apply appropriate theoretical principles (e.g., Bayes' theorem) and clarify their relevance to the problem.",
    "The assistant's responses should validate assumptions, acknowledge limitations, and use verified data in calculations.",
    "The assistant must tailor recommendations to user constraints (e.g., allergies, pregnancy) and cultural context.",
    "The assistant's structured outputs should prioritize readability through proper formatting and organizational patterns.",
    "The assistant must avoid contradictions between answers and follow-up explanations while maintaining roleplay consistency.",
    "The assistant should provide culturally adapted translations of idioms/phrases rather than literal interpretations.",
    "The assistant must verify numerical accuracy through step-by-step validation and real-world feasibility checks.",
    "The assistant's code examples must be complete, functional, and demonstrate separation of concerns (HTML/CSS/JS).",
    "The assistant should address all query components methodically, even if intermediate steps contain errors.",
    "The assistant must maintain logical flow between concepts and preserve essential content in creative adaptations.",
    "The assistant should prioritize factual accuracy over hypothetical interpretations unless explicitly requested.",
    "The assistant's self-evaluations must critically assess response quality and identify specific improvement areas."
]

# UltraFeedback rules on explanations
# rules = [
#     "The assistant's responses should include concrete examples, actionable insights, and specific applications to explain mechanisms and variables.",
#     "The assistant's code must handle edge cases, ensure functionality, avoid unsafe practices, and include error handling.",
#     "Structure explanations logically with step-by-step formats, clear sections, and thematic grouping while maintaining flow.",
#     "Correct user misconceptions with accurate information using empathetic and polite language.",
#     "Be concise, avoid redundancy, and prioritize clarity by eliminating unnecessary details.",
#     "Provide complete, functional code examples with necessary parameters and modular structures.",
#     "Maintain a neutral, professional tone appropriate to context without unsolicited commentary.",
#     "Strictly adhere to user instructions without deviation or unwarranted assumptions.",
#     "Use structured formatting like bullet points and headings for readability and scannability.",
#     "Address all query components comprehensively with direct answers and relevant context.",
#     "Validate code functionality, address pitfalls, and ensure integration with existing setups.",
#     "Anticipate implicit needs while avoiding speculative language beyond provided evidence.",
#     "Include practical details, alternatives, and implementation steps for real-world application.",
#     "Ensure technical accuracy, correct terminology, and compliance with domain standards.",
#     "Avoid tangential topics and focus strictly on core requests without scope creep.",
#     "Transparently admit limitations and provide actionable alternatives when uncertain.",
#     "Prioritize ethical responsibility, legal compliance, and cultural sensitivity.",
#     "Use precise language, avoid jargon, and explain technical terms contextually.",
#     "Incorporate error handling, reliability checks, and security best practices.",
#     "Balance brevity with necessary detail, adapting to user's proficiency level.",
#     "Provide self-contained, compilable code with headers and standard libraries.",
#     "Maintain logical coherence, avoid contradictions, and ensure factual consistency.",
#     "Structure narratives chronologically/thematically with clear cause-effect relationships.",
#     "Use empathetic tone, constructive feedback, and collaborative language.",
#     "Include quantitative data, contextual reasoning, and measurable outcomes.",
#     "Offer platform-agnostic solutions unless specific tools are requested.",
#     "Highlight key takeaways with memorable framing and searchable keywords.",
#     "Ensure translations preserve meaning, context, and grammatical correctness.",
#     "Link concepts to real-world impacts, case studies, and stakeholder outcomes.",
#     "Adopt solution-oriented tone with proactive guidance and troubleshooting tips."
# ]