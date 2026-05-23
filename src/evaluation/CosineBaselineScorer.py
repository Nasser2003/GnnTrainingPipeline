import torch
import torch.nn as nn
import torch.nn.functional as F
from .BaselineScorer import BaselineScorer


class CosineBaselineScorer(nn.Module, BaselineScorer):
    """
    Stateless cosine similarity scorer as a PyTorch Module.
    Score = 0.5 * (cosine(src, dst) + 1), mapped to [0, 1].
    """

    def __init__(self):
        super().__init__()
        # Dummy parameter to make it a valid non-empty module state if needed
        self.dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def score(self, data, edge_index):
        x = data.x.float()
        batch_size = 100_000
        all_scores = []
        for i in range(0, edge_index.size(1), batch_size):
            batch_ei = edge_index[:, i:i+batch_size]
            s, d = batch_ei[0], batch_ei[1]
            cos = F.cosine_similarity(x[s], x[d], dim=1)
            all_scores.append(0.5 * (cos + 1.0))
        return torch.cat(all_scores)

    def forward(self, data, edge_index):
        return self.score(data, edge_index)
