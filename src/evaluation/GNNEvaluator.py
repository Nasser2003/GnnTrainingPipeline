import mlflow
from evaluation.Evaluator import Evaluator
from evaluation.GNNScorer import GNNScorer


class GNNEvaluator(Evaluator):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.scorer = GNNScorer(model)

    def evaluate(self, val_data, test_data, neg_ratios=None,
                 neg_ratio_for_threshold=1, full_pos_edges=None):
        if neg_ratios is None:
            neg_ratios = [1, 9, 100]

        results = []
        for r in neg_ratios:
            with mlflow.start_span(name=f"evaluate_neg_ratio_{r}") as span:
                span.set_inputs({"neg_ratio": r})
                bt_r = self.find_best_threshold(
                    val_data, r, self.scorer, full_pos_edges=full_pos_edges)
                metrics = self.evaluate_metrics_master(
                    "GNN", test_data, r, self.scorer,
                    threshold=bt_r, full_pos_edges=full_pos_edges)
                if metrics:
                    span.set_outputs(metrics)
                    results.append(metrics)
                
        return results
