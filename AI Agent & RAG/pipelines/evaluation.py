"""Evaluation module for assessing RAG and agent performance"""

from typing import List, Dict, Any
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np


class Evaluator:
    """Evaluate RAG and agent performance"""

    def __init__(self):
        self.results = []

    def evaluate_retrieval(self, query: str, retrieved_docs: List[str], relevant_docs: List[str]) -> Dict[str, float]:
        """Evaluate retrieval performance"""
        # Simple precision@k and recall@k
        if not relevant_docs:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

        retrieved_set = set(retrieved_docs)
        relevant_set = set(relevant_docs)

        # Calculate precision and recall
        true_positives = len(retrieved_set & relevant_set)

        precision = true_positives / len(retrieved_docs) if retrieved_docs else 0
        recall = true_positives / len(relevant_docs) if relevant_docs else 0

        # F1 score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4)
        }

    def evaluate_answer(self, generated_answer: str, reference_answer: str, query: str = None) -> Dict[str, Any]:
        """Evaluate answer quality"""
        metrics = {
            'exact_match': self._exact_match(generated_answer, reference_answer),
            'contains_answer': self._contains_answer(generated_answer, reference_answer),
            'length_ratio': len(generated_answer) / len(reference_answer) if reference_answer else 0,
            'answer_length': len(generated_answer)
        }

        # BLEU-like score (simplified)
        metrics['bleu_similarity'] = self._compute_bleu_similarity(generated_answer, reference_answer)

        return metrics

    def _exact_match(self, pred: str, ref: str) -> bool:
        """Check for exact match (normalized)"""
        pred_norm = pred.strip().lower()
        ref_norm = ref.strip().lower()
        return pred_norm == ref_norm

    def _contains_answer(self, pred: str, ref: str) -> bool:
        """Check if prediction contains reference key information"""
        pred_lower = pred.lower()
        ref_lower = ref.lower()

        # Simple keyword matching
        ref_keywords = set(ref_lower.split())
        if len(ref_keywords) == 0:
            return False

        matches = sum(1 for kw in ref_keywords if kw in pred_lower)
        return matches / len(ref_keywords) >= 0.5

    def _compute_bleu_similarity(self, pred: str, ref: str) -> float:
        """Compute simplified BLEU score"""
        from collections import Counter

        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()

        if not pred_tokens or not ref_tokens:
            return 0.0

        # Unigram precision
        pred_counter = Counter(pred_tokens)
        ref_counter = Counter(ref_tokens)

        overlap = sum(min(pred_counter[t], ref_counter[t]) for t in pred_counter)

        precision = overlap / len(pred_tokens) if pred_tokens else 0

        # Length penalty
        ratio = len(pred_tokens) / len(ref_tokens) if ref_tokens else 0
        if ratio > 1:
            penalty = np.exp(1 - ratio)
        else:
            penalty = 1

        return float(precision * penalty)

    def evaluate_pipeline(self, pipeline, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate entire pipeline on test cases"""
        results = {
            'total_cases': len(test_cases),
            'successful': 0,
            'failed': 0,
            'retrieval_metrics': [],
            'answer_metrics': []
        }

        for test_case in test_cases:
            query = test_case['question']
            expected = test_case.get('answer', '')
            relevant_docs = test_case.get('relevant_docs', [])

            try:
                # Run pipeline
                response = pipeline.query(query)

                # Evaluate retrieval if relevant docs provided
                if relevant_docs:
                    retrieved_docs = [r['content'] for r in response.get('sources', [])]
                    retrieval_metrics = self.evaluate_retrieval(query, retrieved_docs, relevant_docs)
                    results['retrieval_metrics'].append(retrieval_metrics)

                # Evaluate answer
                answer_metrics = self.evaluate_answer(response['answer'], expected, query)
                results['answer_metrics'].append(answer_metrics)

                results['successful'] += 1
            except Exception as e:
                results['failed'] += 1
                print(f"Error evaluating case '{query[:50]}...': {str(e)}")

        # Aggregate metrics
        if results['retrieval_metrics']:
            results['avg_retrieval'] = self._aggregate_metrics(results['retrieval_metrics'])

        if results['answer_metrics']:
            results['avg_answer'] = self._aggregate_metrics(results['answer_metrics'])

        results['success_rate'] = results['successful'] / results['total_cases'] if results['total_cases'] > 0 else 0

        return results

    def _aggregate_metrics(self, metrics_list: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate average metrics"""
        aggregated = {}
        keys = metrics_list[0].keys() if metrics_list else []

        for key in keys:
            values = [m[key] for m in metrics_list if key in m and isinstance(m[key], (int, float))]
            if values:
                aggregated[f'avg_{key}'] = float(np.mean(values))
                aggregated[f'std_{key}'] = float(np.std(values))

        return aggregated

    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate evaluation report"""
        report = []
        report.append("=" * 50)
        report.append("EVALUATION REPORT")
        report.append("=" * 50)
        report.append(f"Total test cases: {results['total_cases']}")
        report.append(f"Successful: {results['successful']}")
        report.append(f"Failed: {results['failed']}")
        report.append(f"Success rate: {results.get('success_rate', 0):.2%}")
        report.append("")

        if 'avg_retrieval' in results:
            report.append("Retrieval Metrics:")
            for key, value in results['avg_retrieval'].items():
                report.append(f"  {key}: {value:.4f}")
            report.append("")

        if 'avg_answer' in results:
            report.append("Answer Metrics:")
            for key, value in results['avg_answer'].items():
                report.append(f"  {key}: {value:.4f}")
            report.append("")

        report.append("=" * 50)

        return "\n".join(report)
