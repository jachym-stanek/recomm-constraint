import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple, Any

import numpy as np
from scipy.sparse import csr_matrix

from src.algorithms.Preprocessor import ItemPreprocessor
from src.algorithms.algorithm import Algorithm
from src.constraints import Constraint
from src.segmentation import SegmentationExtractor
from src.settings import Settings
from src.util import total_satisfaction


@dataclass
class SolverResults:
    """Keeps track of cumulative time for a single solver"""
    total_time: float = 0.0
    total_time_empty: float = 0.0
    total_constraint_satisfaction: float = 0.0
    total_score: float = 0.0
    calls: int = 0
    calls_empty: int = 0

    def add(self, elapsed: float, satisfaction_score: float, score: float, empty: bool=False) -> None:
        if empty:
            self.total_time_empty += elapsed
            self.calls_empty += 1
        else:
            self.total_time += elapsed
            self.calls += 1
            self.total_constraint_satisfaction += satisfaction_score
            self.total_score += score

    @property
    def avg_time(self) -> float:
        return (self.total_time + self.total_time_empty) / (self.calls + self.calls_empty) if self.calls else 0.0

    @property
    def avg_time_empty(self) -> float:
        return self.total_time_empty / self.calls_empty if self.calls_empty else 0.0

    @property
    def avg_time_non_empty(self) -> float:
        return self.total_time / self.calls if self.calls else 0.0

    @property
    def avg_constraint_satisfaction(self) -> float:
        return self.total_constraint_satisfaction / self.calls if self.calls else 0.0

    @property
    def avg_score(self) -> float:
        return self.total_score / self.calls if self.calls else 0.0


class Evaluator:
    def __init__(self, settings: Settings, segmentation_extractor: SegmentationExtractor):
        self.num_hidden = settings.recommendations['num_hidden']
        self.log_every = settings.log_every
        self.preprocessor = ItemPreprocessor(verbose=False)
        self.segmentation_extractor = segmentation_extractor

    def evaluate(
        self,
        train_dataset,
        test_dataset,
        model,
        *,
        N: int = 10,
        M: int = 100,
        take_random_hidden: bool = False,
        min_relevant_items: int = 3,
        solvers: Dict[str, Algorithm] = None,
        slice_sizes: List[int] = None,
        constraints: List[Constraint] = None,
        precomputed_neighborhoods = None
    ):

        if precomputed_neighborhoods is None:
            precomputed_neighborhoods = model.item_knn.compute_neighborhoods_chunked(model.item_factors)

        total_recall = 0.0
        total_items_recommended = set()
        user_count = 0
        skipped_users = 0
        solver_stats = dict()
        total_candidates = 0

        detailed_rows: List[Dict[str, Any]] = []

        for user_idx in range(len(test_dataset)):
            relevant_items = test_dataset.matrix[user_idx].indices
            if len(relevant_items) < min_relevant_items:
                skipped_users += 1
                continue

            recalls = []
            hidden_pool = list(relevant_items)
            for _ in range(self.num_hidden):
                if not hidden_pool:
                    break

                hidden_item = ( np.random.choice(hidden_pool) if take_random_hidden else relevant_items[_] )
                hidden_pool.remove(hidden_item)

                observed_items = set(relevant_items)
                observed_items.remove(hidden_item)

                user_obs_csr = csr_matrix(test_dataset.matrix[user_idx, list(observed_items)])

                # recommendation
                if not solvers:
                    recomms, _ = model.recommend( # we do not need scores here
                        test_dataset.matrix,
                        user_idx,
                        user_obs_csr,
                        list(observed_items),
                        N=N,
                        precomputed_neighborhoods=precomputed_neighborhoods,
                        test_user=True,
                    )
                else:
                    recomms, num_candidates, solver_metrics = self._recommend_with_solvers(
                        user_idx,
                        test_dataset.matrix,
                        model,
                        self.preprocessor,
                        self.segmentation_extractor,
                        solvers,
                        user_obs_csr,
                        observed_items,
                        precomputed_neighborhoods,
                        N=N,
                        M=M,
                        constraints=constraints,
                        slice_sizes=slice_sizes,
                    )
                    if len(recomms) == 0:
                        continue
                    self._proccess_solver_metrics(user_idx, N, M, constraints, detailed_rows, solver_stats, solver_metrics)
                    total_candidates += num_candidates

                hit = int(hidden_item in recomms)
                recalls.append(hit)
                total_items_recommended.update(recomms)

            if recalls:
                total_recall += float(np.mean(recalls))
                user_count += 1
            else:
                skipped_users += 1

            if user_count and (user_count + skipped_users) % self.log_every == 0:
                print(
                    f"[Evaluator] processed {user_count + skipped_users}/{len(test_dataset)} users "
                    f"({skipped_users} skipped). Average recall@{N} = {total_recall / user_count:.4f}, "
                    f"catalog coverage = {len(total_items_recommended) / train_dataset.num_items:.4f}"
                )
                if solvers:
                    print(f"[Evaluator] Average number of candidates: {total_candidates / user_count:.2f}")
                    for name, stats in solver_stats.items():
                        print(f"[Evaluator] {name}: avg time = {stats.avg_time:.2f} ms, "
                                f"avg time non-empty = {stats.avg_time_non_empty:.2f} ms, "
                                f"avg time empty = {stats.avg_time_empty:.2f} ms, "
                              f"avg score = {stats.avg_score:.4f}, "
                              f"avg constraint satisfaction = {stats.avg_constraint_satisfaction:.4f}")

        # final results
        average_recall = total_recall / user_count if user_count else 0.0
        catalog_coverage = len(total_items_recommended) / train_dataset.num_items

        print(f"[Evaluator] finished. average recall@{N} = {average_recall:.4f}, catalog coverage = {catalog_coverage:.4f}")

        if solvers is None:
            return {'average_recall': average_recall, 'catalog_coverage': len(total_items_recommended) / train_dataset.num_items}
        else:
            aggregated = {name: {"time": solver_stats[name].avg_time, "time_non_empty": solver_stats[name].avg_time_non_empty,
                           "time_empty": solver_stats[name].avg_time_empty, "score": solver_stats[name].avg_score,
                           "constraint_satisfaction": solver_stats[name].avg_constraint_satisfaction}
                    for name in solver_stats} | {"average_num_candidates": total_candidates / user_count}
            return aggregated, detailed_rows

    def _recommend_with_solvers(
        self,
        user_idx: int,
        R: csr_matrix,
        model,
        preprocessor: ItemPreprocessor,
        segmentation_extractor: SegmentationExtractor,
        solvers: Dict[str, Algorithm],
        user_observation: csr_matrix,
        observed_items: Set[int],
        precomputed_neighborhoods,
        N: int,
        M: int,
        constraints: List[Constraint],
        slice_sizes: List[int],
    ) -> Tuple[List[int], int, Dict[str, dict]]:

        solver_metrics: Dict[str, dict] = {}

        inner_recomms, scores = model.recommend(
                        R,
                        user_idx,
                        user_observation,
                        list(observed_items),
                        N=M,
                        precomputed_neighborhoods=precomputed_neighborhoods,
                        test_user=True,
                    )

        # we always keep the top‑N as baseline
        candidates = {item: score for item, score in zip(inner_recomms, scores)}
        num_candidates = len(candidates)
        if num_candidates < N:
            print(f"[Evaluator] WARNING: fewer than {N} candidates found for user {user_idx}.")
            return [], 0, {}

        candidates_segments = segmentation_extractor.get_segments_dict_for_recomms(candidates)
        unconstrained_recomms: List[int] = inner_recomms[:N]

        for name, solver in solvers.items():

            if name == "ilp-slicing":
                for s in slice_sizes:
                    if s >= N: # optimization
                        continue
                    recomms, metrics = self._solve_recomms(solver, preprocessor, candidates, candidates_segments, constraints, N, s)
                    solver_metrics[f"{name}-s={s}"] = metrics
            elif name == "ilp-preprocessing" or name == "cp-preprocessing":
                recomms, metrics = self._solve_recomms(solver, preprocessor, candidates, candidates_segments, constraints, N, preprocess=True)
                solver_metrics[name] = metrics
            else:
                recomms, metrics = self._solve_recomms(solver, preprocessor, candidates, candidates_segments, constraints, N)
                solver_metrics[name] = metrics

        return unconstrained_recomms, num_candidates, solver_metrics

    def _solve_recomms(self, solver, preprocessor, candidates, candidates_segments, constraints, N, s=None, preprocess=False):
        start = time.perf_counter()
        if s is None and not preprocess:
            recomms = solver.solve(candidates, candidates_segments, constraints, N)
        elif s is None and preprocess:
            candidates = preprocessor.preprocess_items(candidates, candidates_segments, constraints, N)
            recomms = solver.solve(candidates, candidates_segments, constraints, N)
        else:
            recomms = solver.solve_by_slicing(preprocessor, candidates, candidates_segments, constraints, N=N, slice_size=s)
        elapsed = (time.perf_counter() - start)*1000 # in ms
        score = sum(candidates[item] for item in recomms.values()) if recomms else 0
        constraint_satisfaction_score = total_satisfaction(recomms, candidates, candidates_segments, constraints)
        metrics = {
            'time': elapsed,
            'score': score,
            'constraint_satisfaction_score': constraint_satisfaction_score,
        }
        return recomms, metrics

    def _proccess_solver_metrics(self, user_idx, N, M, constraints, detailed_rows, solver_stats, solver_metrics) -> None:
        for name, metrics in solver_metrics.items():
            if name not in solver_stats:
                solver_stats[name] = SolverResults()
            solver_stats[name].add(metrics['time'], metrics['constraint_satisfaction_score'], metrics['score'])

        row_base = {
            "useridx": user_idx,  # ← renamed
            "N": N,
            "M": M,
            "constraints": [str(c) for c in (constraints or [])],
        }
        for s_name, m in solver_metrics.items():
            detailed_rows.append({
                **row_base,
                "solver": s_name,
                "time_ms": m["time"],
                "constraint_satisfaction": m["constraint_satisfaction_score"],
                "score": m["score"],
                "empty": m["score"] == 0,
            })
