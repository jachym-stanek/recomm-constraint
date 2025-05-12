import time
import pandas as pd
import os
from typing import List, Dict, Any

from src.algorithms.ItemKnn import ItemKnn
from src.data_split import DataSplitter
from src.evaluator import Evaluator
from src.models import ALSModel, AnnoyALSModel
from src.segmentation import SegmentationExtractor


class ExperimentRunner(object):
    def __init__(self, settings, results_file, train_dataset, test_dataset, constraints=None):
        self.settings = settings
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.results_file = results_file
        self.results_df = results_file.split('.')[0] + '.csv'
        self.constraints = constraints
        self.segmentation_extractor = SegmentationExtractor(settings)

        self._rows_buffer: List[Dict[str, Any]] = []

        self.default_params = {
            'num_factors': 256,
            'regularization': 0.1,
            'num_iterations': 3,
            'alpha': 1.0,
            'nearest_neighbors': 10,
            'num_trees': 50,
            'search_k': -1,
            'bm25_B': 0.8,
        }

    # run experiment for a combination of two parameters value lists
    def run_experiments_on_model_parameters(self, parameter1_values, parameter2_values, parameter1_name, parameter2_name,
                                            use_approximate_model=False, retrain_every_rewrite=True):
        print(f"[ExperimentRunner] Running experiment with parameters: {parameter1_name}, {parameter2_name}...")

        start_time = time.time()
        results = []
        model = None

        for param1 in parameter1_values:
            if not retrain_every_rewrite:
                model = self._get_model_from_params_rewrite({parameter1_name: param1}, use_approximate_model=use_approximate_model)
                model.train(self.train_dataset)

            for param2 in parameter2_values:
                params_rewrite = {
                    parameter1_name: param1,
                    parameter2_name: param2
                }
                metrics = self._run_experiment_for_particular_params(params_rewrite, model=model,
                                                                     use_approximate_model=use_approximate_model)
                results.append(metrics)

        print(f"[ExperimentRunner] Experiments completed in {time.time() - start_time:.2f} seconds.")

        return results

    def run_experiments_on_solver(self, solvers, num_recomms_values, num_candidates_values, constraint_lists, slice_sizes, tested_item_properties, model_params=None):

        if model_params is None:
            model_params = dict()

        segmentation_extractor = SegmentationExtractor(self.settings)
        segmentation_extractor.extract_segments(tested_item_properties)

        evaluator = Evaluator(self.settings, segmentation_extractor)
        model = self._get_model_from_params_rewrite(model_params)
        model.train(self.train_dataset)
        precomputed_neighborhoods = model.item_knn.compute_neighborhoods_chunked(model.item_factors)

        results = []

        for N in num_recomms_values:
            for M in num_candidates_values:
                if N >= M:
                    continue
                for constraints in constraint_lists:
                    print(f"[ExperimentRunner] Running experiment with N={N}, M={M}, constraints={constraints}...")
                    aggregated, detailed_rows = evaluator.evaluate(
                        train_dataset=self.train_dataset,
                        test_dataset=self.test_dataset,
                        model=model,
                        N=N,
                        M=M,
                        min_relevant_items=self.settings.min_relevant_items,
                        take_random_hidden=self.settings.recommendations['take_random_hidden'],
                        solvers=solvers,
                        slice_sizes=slice_sizes,
                        constraints=constraints,
                        precomputed_neighborhoods=precomputed_neighborhoods
                    )
                    rewrites = {"N": N, "M": M, "constraints": [str(c) for c in constraints]}
                    print(f"[ExperimentRunner] Rewrites: {rewrites} Evaluation Metrics: {aggregated}")
                    self._rows_buffer.extend(detailed_rows)
                    self._flush_solver_df()
                    self._save_metrics_to_file(rewrites, aggregated)


    def _run_experiment_for_particular_params(self, params_rewrite, model=None, use_approximate_model=False, solver=None):
        print(f"[ExperimentRunner] Running experiment with special params: {params_rewrite}...")

        # if bm25 normalization is tested, we need to recreate the dataset
        if 'bm25_B' in params_rewrite:
            print(f"[ExperimentRunner] Recreating dataset with bm25_B={params_rewrite['bm25_B']}...")
            data_splitter = DataSplitter(self.settings)
            data_splitter.load_data(self.settings.dataset_name)
            data_splitter.split_data(bmB=params_rewrite['bm25_B'])
            self.train_dataset = data_splitter.get_train_data()
            self.test_dataset = data_splitter.get_test_data()

        start_time = time.time()

        if model is None:
            model = self._get_model_from_params_rewrite(params_rewrite, use_approximate_model=use_approximate_model)
            model.train(self.train_dataset)
            print(f"[ExperimentRunner] Training completed in {time.time() - start_time:.2f} seconds.")
        elif "nearest_neighbors" in params_rewrite: # if testing nearest neighbors and there is a model given, we need to replace ItemKNN
            model.item_knn = ItemKnn(K=params_rewrite['nearest_neighbors'])

        # Evaluate the model
        evaluator = Evaluator(self.settings, self.segmentation_extractor)
        metrics = evaluator.evaluate(
            train_dataset=self.train_dataset,
            test_dataset=self.test_dataset,
            model=model,
            N=self.settings.recommendations['top_n'],
            min_relevant_items=self.settings.min_relevant_items,
            take_random_hidden=self.settings.recommendations['take_random_hidden']
        )

        print(f"[ExperimentRunner] Evaluation Metrics: {metrics}, processing time: {time.time() - start_time:.2f} seconds.")

        self._save_metrics_to_file(params_rewrite, metrics)

        return metrics

    def _save_metrics_to_file(self, params_rewrite, metrics):
        with open(self.results_file, 'a') as f:
            f.write(f'{(params_rewrite, metrics)}\n')

    def _flush_solver_df(self) -> None:
        """Append buffered rows to <results_file.csv> """
        if not self._rows_buffer:
            return

        df_new = pd.DataFrame(self._rows_buffer,
                              columns=["useridx", "N", "M", "constraints",
                                       "solver", "time_ms",
                                       "constraint_satisfaction", "score", "empty"])

        header_needed = (not os.path.exists(self.results_df) # write header we do the first write
                         or os.path.getsize(self.results_df) == 0)
        df_new.to_csv(self.results_df,
                      mode="a",
                      index=False,
                      header=header_needed)

        # Clear buffer so the next flush only writes fresh rows
        self._rows_buffer.clear()

    def _get_model_from_params_rewrite(self, params_rewrite, use_approximate_model=False):
        params = self.default_params.copy()
        params.update(params_rewrite)

        if use_approximate_model:
            return AnnoyALSModel(num_factors=params['num_factors'], regularization=params['regularization'],
                                  num_iterations=params['num_iterations'], alpha=params['alpha'],
                                  num_trees=params['num_trees'], use_gpu=self.settings.use_gpu)
        else:
            return ALSModel(num_factors=params['num_factors'], regularization=params['regularization'],
                             num_iterations=params['num_iterations'], alpha=params['alpha'],
                             use_gpu=self.settings.use_gpu, nearest_neighbors=params['nearest_neighbors'])
