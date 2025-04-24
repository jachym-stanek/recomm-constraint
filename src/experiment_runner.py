import time

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
        self.constraints = constraints
        self.segmentation_extractor = SegmentationExtractor(settings)

        self.default_params = {
            'num_factors': 256,
            'regularization': 0.1,
            'num_iterations': 3,
            'alpha': 1.0,
            'nearest_neighbors': 5,
            'num_trees': 50,
            'search_k': -1,
            'bm25_B': 0.8,
        }

    # run experiment for a combination of two parameters value lists
    def run_experiments(self, parameter1_values, parameter2_values, parameter1_name, parameter2_name,
                       use_approximate_model=False, solver=None):
        print(f"[ExperimentRunner] Running experiment with parameters: {parameter1_name}, {parameter2_name}...")

        start_time = time.time()
        results = []

        for param1 in parameter1_values:
            for param2 in parameter2_values:
                params_rewrite = {
                    parameter1_name: param1,
                    parameter2_name: param2
                }
                metrics = self._run_experiment_for_particular_params(params_rewrite,
                                                                     use_approximate_model=use_approximate_model,
                                                                     solver=solver)
                results.append(metrics)

        print(f"[ExperimentRunner] Experiments completed in {time.time() - start_time:.2f} seconds.")

        return results


    def _run_experiment_for_particular_params(self, params_rewrite, use_approximate_model=False, solver=None):
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

        params = self._get_rewrite_params(params_rewrite)

        if use_approximate_model:
            model = AnnoyALSModel(num_factors=params['num_factors'], regularization=params['regularization'],
                                  num_iterations=params['num_iterations'], alpha=params['alpha'],
                                  num_trees=params['num_trees'], use_gpu=self.settings.use_gpu)
        else:
            model = ALSModel(num_factors=params['num_factors'], regularization=params['regularization'],
                             num_iterations=params['num_iterations'], alpha=params['alpha'],
                             use_gpu=self.settings.use_gpu, nearest_neighbors=params['nearest_neighbors'])

        model.train(self.train_dataset)

        print(f"[ExperimentRunner] Training completed in {time.time() - start_time:.2f} seconds.")

        # Evaluate the model
        evaluator = Evaluator(self.settings)


        if solver is not None:
            metrics = evaluator.evaluate_constrained_model(train_dataset=self.train_dataset, test_dataset=self.test_dataset,
                                                           segmentation_extractor=self.segmentation_extractor,
                                                           constraints=self.constraints,
                                                           model=model, N=self.settings.recommendations['top_n'])
        else:
            metrics = evaluator.evaluate_recall_at_n(
                train_dataset=self.train_dataset,
                test_dataset=self.test_dataset,
                model=model,
                N=self.settings.recommendations['top_n'],
                min_relevant_items=self.settings.min_relevant_items
            )

        print(f"[ExperimentRunner] Evaluation Metrics: {metrics}, processing time: {time.time() - start_time:.2f} seconds.")

        self._save_metrics_to_file(params_rewrite, metrics)

        return metrics

    def _save_metrics_to_file(self, params_rewrite, metrics):
        with open(self.results_file, 'a') as f:
            f.write(f'{(params_rewrite, metrics)}\n')

    def _get_rewrite_params(self, params_rewrite):
        params = self.default_params.copy()
        params.update(params_rewrite)
        return params
