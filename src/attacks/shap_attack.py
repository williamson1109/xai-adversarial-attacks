# src/attacks/shap_attack.py
import shap
import torch
from transformers import AutoTokenizer
from typing import List, Tuple
from src.models.distilbert_classifier import DistilBERTClassifier

class ShapExplainer:
    def __init__(self, model: DistilBERTClassifier, device: str):
        self.model = model.to(device)
        self.device = device
        self.tokenizer = model.tokenizer
        # wrap model for SHAP
        def f(input_ids, attention_mask):
            inputs = {
                'input_ids': torch.tensor(input_ids).to(device),
                'attention_mask': torch.tensor(attention_mask).to(device)
            }
            with torch.no_grad():
                out = self.model(**inputs)
                probs = torch.softmax(out['logits'], dim=-1).cpu().numpy()
            return probs
        self.explainer = shap.Explainer(f, masker=shap.maskers.Text(self.tokenizer))

    def explain(self, text: str) -> shap.Explanation:
        return self.explainer([text])

def shap_values_for_batch(explainer: ShapExplainer, texts: List[str]):
    return explainer.explain(texts)

# example usage
if __name__ == "__main__":
    # load a trained checkpoint
    clf = DistilBERTClassifier(**model_kwargs)
    clf.load_state_dict(torch.load('path/to/fold_1.pt', map_location='cpu'))
    expl = ShapExplainer(clf, device='cpu')
    ex = expl.explain("We continue to coordinate efforts to prevent and contain COVID-19")
    print(ex.values)   # SHAP values per token