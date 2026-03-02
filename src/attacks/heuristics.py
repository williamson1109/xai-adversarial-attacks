# src/attacks/heuristics.py

from typing import List

def significant_word_removal(text: str, shap_vals: List[float],
                             tokenizer, n_words: int = 1) -> str:
    # remove the top-n tokens by absolute SHAP value
    tokens = tokenizer.tokenize(text)
    indices = sorted(range(len(tokens)),
                     key=lambda i: abs(shap_vals[i]),
                     reverse=True)[:n_words]
    for idx in sorted(indices, reverse=True):
        del tokens[idx]
    return tokenizer.convert_tokens_to_string(tokens)

def significant_word_injection(text: str, shap_vals: List[float],
                               tokenizer, candidate_words: List[str]) -> str:
    # try inserting words from candidate_words at positions where
    # the model is most sensitive (highest SHAP); naive example:
    tokens = tokenizer.tokenize(text)
    pos = max(range(len(tokens)), key=lambda i: abs(shap_vals[i]))
    tokens.insert(pos, candidate_words[0])
    return tokenizer.convert_tokens_to_string(tokens)