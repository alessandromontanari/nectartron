"""
Reward functions for fine-tuning NectarCAM Q&A model using reinforcement learning.
Focuses on factual accuracy and information retrieval.
"""

# TODO: this is still a WIP


import logging
import re
from typing import Dict, List
import numpy as np

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not installed. Semantic similarity rewards unavailable.")


# ============================================================================
# ENTITY EXTRACTION
# ============================================================================

def extract_run_number(text: str) -> str:
    """Extract run number from text (e.g., 'Run #635')."""
    match = re.search(r'run\s*#?(\d+)', text, re.IGNORECASE)
    return match.group(1) if match else None


def extract_key_entities(text: str) -> Dict[str, List[str]]:
    """Extract key entities from NectarCAM Q&A answers.
    
    Returns entities like:
    - run_numbers: [635, 632, ...]
    - trigger_modes: [Pedestal, Calibration, ...]
    - setups: [Camera2, CameraQM, ...]
    - conductors: [Patrick Sizun, Eric Delagnes, ...]
    """
    entities = {
        "run_numbers": [],
        "trigger_modes": [],
        "setups": [],
        "conductors": [],
        "modules": [],
        "dates": [],
    }
    
    # Extract run numbers
    run_matches = re.findall(r'run\s*#?(\d+)', text, re.IGNORECASE)
    entities["run_numbers"] = list(set(run_matches))
    
    # Extract trigger modes
    trigger_patterns = [
        "Pedestal", "Calibration", "Internal", "Physics", 
        "Cosmic", "Random", "Periodic", "n-fold", "nan"
    ]
    for pattern in trigger_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            entities["trigger_modes"].append(pattern)
    
    # Extract setups
    setup_patterns = ["Camera2", "CameraQM", "DataRun", "nan"]
    for pattern in setup_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            entities["setups"].append(pattern)
    
    # Extract number of modules
    module_match = re.search(r'(\d+\.?\d*)\s*modules', text, re.IGNORECASE)
    if module_match:
        entities["modules"].append(module_match.group(1))
    
    # Extract names (simple pattern)
    name_patterns = ["Patrick Sizun", "Eric Delagnes", "Vincent Marandon"]
    for name in name_patterns:
        if name in text:
            entities["conductors"].append(name)
    
    return entities


# ============================================================================
# BASIC REWARD FUNCTIONS
# ============================================================================

def exact_match_reward(generated: str, reference: str) -> float:
    """
    Reward 1.0 if answers match exactly (case-insensitive).
    Useful as a binary signal but quite strict.
    """
    return 1.0 if generated.lower().strip() == reference.lower().strip() else 0.0


def normalized_edit_distance_reward(generated: str, reference: str) -> float:
    """
    Reward based on string similarity using normalized edit distance.
    Returns score between 0 and 1, where 1 is identical.
    
    Uses Levenshtein distance normalized by the longest string length.
    """
    try:
        from difflib import SequenceMatcher
    except ImportError:
        logger.warning("difflib not available. Returning 0.0")
        return 0.0
    
    # Normalize text
    gen = generated.lower().strip()
    ref = reference.lower().strip()
    
    # Calculate similarity ratio
    similarity = SequenceMatcher(None, gen, ref).ratio()
    return similarity


def f1_token_overlap_reward(generated: str, reference: str) -> float:
    """
    F1 score based on token-level overlap.
    Compares sets of words between generated and reference answers.
    
    Returns:
        F1 score between 0 and 1
    """
    def tokenize(text):
        # Remove punctuation and split
        text = re.sub(r'[^\w\s]', '', text.lower())
        return set(text.split())
    
    gen_tokens = tokenize(generated)
    ref_tokens = tokenize(reference)
    
    if not gen_tokens and not ref_tokens:
        return 1.0
    if not gen_tokens or not ref_tokens:
        return 0.0
    
    # Calculate precision and recall
    intersection = gen_tokens & ref_tokens
    precision = len(intersection) / len(gen_tokens) if gen_tokens else 0
    recall = len(intersection) / len(ref_tokens) if ref_tokens else 0
    
    # F1 score
    if precision + recall == 0:
        return 0.0
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def semantic_similarity_reward(generated: str, reference: str, model_name: str) -> float:
    """ 
    Reward based on semantic similarity between generated and reference.
    Uses sentence transformers to compute embedding similarity.
    
    Parameters
    ----------
    generated : str 
        Model-generated answer
    reference : str
        Ground truth answer
    model_name: str
        HuggingFace model for embeddings
    
    Returns
    -------
    float
        Cosine similarity between embeddings (0 to 1)
    """
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        logger.warning("sentence-transformers not available. Use pip install sentence-transformers")
        return 0.0
    
    try:
        model = SentenceTransformer(model_name)
        gen_embedding = model.encode(generated, convert_to_tensor=False)
        ref_embedding = model.encode(reference, convert_to_tensor=False)
        
        # Cosine similarity
        similarity = np.dot(gen_embedding, ref_embedding) / (
            np.linalg.norm(gen_embedding) * np.linalg.norm(ref_embedding)
        )
        return float(similarity)
    except Exception as e:
        logger.error(f"Error computing semantic similarity: {e}")
        return 0.0


# ============================================================================
# ENTITY-BASED REWARDS (Domain-Specific)
# ============================================================================

def entity_extraction_reward(generated: str, reference: str) -> float:
    """
    Reward based on correct entity extraction.
    Checks if generated answer contains the same key entities as reference.
    
    Useful for NectarCAM Q&A because answers focus on specific entities
    like run numbers, trigger modes, setups, etc.
    
    Parameters
    ----------
    generated : str 
        Model-generated answer
    reference : str
        Ground truth answer

    Returns
    -------
    float
        Proportion of entity types that match correctly
    """
    gen_entities = extract_key_entities(generated)
    ref_entities = extract_key_entities(reference)
    
    # Count matching entity types
    matches = 0
    total = 0
    
    for entity_type in ref_entities:
        total += 1
        gen_set = set(gen_entities[entity_type])
        ref_set = set(ref_entities[entity_type])
        
        # If reference has no entities of this type, give full score
        if not ref_set:
            matches += 1
        # If reference has entities, check if they match
        elif gen_set == ref_set:
            matches += 1
    
    return matches / total if total > 0 else 0.0


def run_number_accuracy_reward(generated: str, reference: str) -> float:
    """
    Critical check: Does the generated answer mention the correct run number?
    Most critical for NectarCAM Q&A since questions are usually about specific runs.
    
    Parameters
    ----------
    generated : str 
        Model-generated answer
    reference : str
        Ground truth answer

    Returns
    -------
    float
        1.0 if correct run number, 0.0 otherwise
    """
    gen_run = extract_run_number(generated)
    ref_run = extract_run_number(reference)
    
    if ref_run is None:
        return 1.0  # No reference run number to check
    
    return 1.0 if gen_run == ref_run else 0.0


def information_coverage_reward(generated: str, reference: str) -> float:
    """
    Reward based on how much important information from reference is in generated.
    Extracted key phrases and checks coverage.
    
    Parameters
    ----------
    generated : str 
        Model-generated answer
    reference : str
        Ground truth answer

    Returns
    -------
    float
        Proportion of key information preserved (0 to 1)
    """
    # Extract meaningful chunks (words > 4 chars or numbers)
    def extract_key_phrases(text):
        # Keep numbers and words longer than 4 characters
        words = text.split()
        key_phrases = []
        for word in words:
            # Remove punctuation
            clean = re.sub(r'[^\w\d]', '', word.lower())
            if len(clean) > 4 or clean.isdigit():
                key_phrases.append(clean)
        return set(key_phrases)
    
    gen_phrases = extract_key_phrases(generated)
    ref_phrases = extract_key_phrases(reference)
    
    if not ref_phrases:
        return 1.0
    
    coverage = len(gen_phrases & ref_phrases) / len(ref_phrases)
    return coverage


# ============================================================================
# COMBINED REWARD FUNCTION
# ============================================================================

def compute_reward(
    generated: str,
    reference: str,
    weights: Dict[str, float] = None,
    use_semantic: bool = False,
) -> Dict[str, float]:
    """
    Compute combined reward from multiple signals.
    
    Parameters
    ----------
    generated : str 
        Model-generated answer
    reference : str
        Ground truth answer
    weights : Dict[str, float]
        Dictionary of reward function weights (must sum to 1.0)
    use_semantic : bool
        Whether to include semantic similarity (requires sentence-transformers)

    Returns
    -------
    Dict[str, float]
        Dictionary with individual reward scores and combined 'total_reward'
    """
    if weights is None:
        if use_semantic:
            weights = {
                "exact_match": 0.15,
                "run_number_accuracy": 0.25,  # Critical for this task
                "entity_extraction": 0.20,
                "f1_overlap": 0.15,
                "information_coverage": 0.15,
                "semantic_similarity": 0.10,
            }
        else:
            weights = {
                "exact_match": 0.15,
                "run_number_accuracy": 0.30,  # Critical for this task
                "entity_extraction": 0.25,
                "f1_overlap": 0.15,
                "information_coverage": 0.15,
            }
    
    rewards = {}
    
    # Compute individual rewards
    rewards["exact_match"] = exact_match_reward(generated, reference)
    rewards["run_number_accuracy"] = run_number_accuracy_reward(generated, reference)
    rewards["entity_extraction"] = entity_extraction_reward(generated, reference)
    rewards["f1_overlap"] = f1_token_overlap_reward(generated, reference)
    rewards["information_coverage"] = information_coverage_reward(generated, reference)
    
    if use_semantic:
        rewards["semantic_similarity"] = semantic_similarity_reward(generated, reference)
    
    # Compute weighted total
    total_reward = sum(
        rewards.get(key, 0.0) * weight
        for key, weight in weights.items()
    )
    
    rewards["total_reward"] = total_reward
    rewards["weights"] = weights
    
    return rewards


def batch_reward_computation(
    predictions: List[str],
    references: List[str],
    weights: Dict[str, float] = None,
    use_semantic: bool = False,
) -> List[Dict[str, float]]:
    """
    Compute rewards for a batch of predictions.
    
    Parameters
    ----------
    predictions : List[str]
        List of model-generated answers
    references : List[str]
        List of ground truth answers
    weights : Dict[str, float]
        Dictionary of reward function weights (must sum to 1.0)
    use_semantic : bool
        Whether to include semantic similarity (requires sentence-transformers)

    Returns
    -------
    List[Dict[str, float]]
        List of reward dictionaries
    """
    batch_rewards = []
    for pred, ref in zip(predictions, references):
        reward = compute_reward(pred, ref, weights=weights, use_semantic=use_semantic)
        batch_rewards.append(reward)
    
    return batch_rewards


# ============================================================================
# UTILITY & ANALYSIS
# ============================================================================

def print_reward_analysis(rewards: Dict[str, float]) -> None:
    """Print a formatted analysis of computed rewards."""
    print("\n" + "="*60)
    print("REWARD ANALYSIS")
    print("="*60)
    
    for key, value in rewards.items():
        if key not in ["total_reward", "weights"]:
            print(f"  {key:.<40} {value:.4f}")
    
    print("-"*60)
    print(f"  {'TOTAL REWARD':.<40} {rewards['total_reward']:.4f}")
    print("="*60 + "\n")


def average_batch_rewards(batch_rewards: List[Dict[str, float]]) -> Dict[str, float]:
    """Compute average rewards across a batch."""
    if not batch_rewards:
        return {}
    
    avg_rewards = {}
    for key in batch_rewards[0]:
        if key != "weights":
            values = [r.get(key, 0.0) for r in batch_rewards if key in r]
            avg_rewards[key] = np.mean(values) if values else 0.0
    
    return avg_rewards
