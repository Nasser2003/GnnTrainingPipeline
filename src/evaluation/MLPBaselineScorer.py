import torch
from .BaselineScorer import BaselineScorer


class MLPBaselineScorer(BaselineScorer):
    """
    Wraps a trained MLP (e.g. RawMLPConcatPredictor) to produce probabilities.
    """

    def __init__(self, model):
        self.model = model

    def score(self, data, edge_index):
        self.model.eval()
        batch_size = 100_000
        all_scores = []
        device = next(self.model.parameters()).device
        with torch.no_grad():
            for i in range(0, edge_index.size(1), batch_size):
                batch_ei = edge_index[:, i:i+batch_size].to(device)
                logits = self.model.forward_logits(data, batch_ei).view(-1)
                all_scores.append(torch.sigmoid(logits))
            return torch.cat(all_scores)
