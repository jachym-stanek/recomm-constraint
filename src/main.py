import time, os, re

from src.algorithms.ILP import IlpSolver
from src.constraints import GlobalMaxItemsPerSegmentConstraint
from src.data_split import DataSplitter
from src.evaluator import Evaluator
from src.experiment_runner import ExperimentRunner
from src.models import ALSModel, AnnoyALSModel
from src.settings import Settings
from src.segmentation import SegmentationExtractor
from src.constraints import *
from src.constraint_generator import ConstraintGenerator

# results file will have number one higher than the highest number in the existing results files
RESULTS_FILE = "results" + str(max([int(re.search(r'results(\d+)\.txt', f).group(1)) for f in os.listdir('.') if re.match(r'results\d+\.txt', f)] + [0]) + 1) + ".txt"

def main():
    start_time = time.time()
    settings = Settings()
    settings.set_dataset_in_use('industrial_dataset1')
    # settings.set_dataset_in_use('movielens')
    data_splitter = DataSplitter(settings)
    data_splitter.load_data(settings.dataset_name)
    data_splitter.split_data()
    train_dataset= data_splitter.get_train_data()
    test_dataset = data_splitter.get_test_data()

    print(f"Train rating matrix shape: {train_dataset.matrix.shape}, Number of users: {len(train_dataset)}")
    print(f"Test rating matrix shape: {test_dataset.matrix.shape}, Number of users: {len(test_dataset)}")

    # factors = [1, 2, 5, 7, 10, 20, 50, 100, 200, 500]
    # num_factors = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    # factors = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 32, 64, 128, 256]
    regularizations = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    # nearest_neighbors = [2, 4, 6, 8, 10, 12, 15, 17, 20, 25, 30, 40, 50]
    # nearest_neighbors = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 30, 40, 50]
    # num_trees = [2, 4, 8, 10, 16, 20, 30, 50, 70, 100, 200]
    # num_trees = [2, 10, 50, 70, 100, 200]
    # alphas = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    # num_iterations = [1, 2, 3, 5, 8, 10, 15]
    nearest_neighbors = [1, 2, 4, 6, 8, 10, 15, 20, 30, 50, 60, 70, 100, 150, 200]
    # num_factors = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    num_factors = [256, 512, 1024, 2048]
    bm_bs = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 4.0, 8, 16]
    num_iterations = [1, 2, 3, 5, 8, 10, 15]

    # segmentation_extractor = SegmentationExtractor(settings)
    # segmentation_extractor.extract_segments('genres')
    # segmentation_extractor.extract_segments('category_1')  # segments for industrial_dataset1

    experiment_runner = ExperimentRunner(settings, RESULTS_FILE, train_dataset, test_dataset)

    # results = experiment_runner.run_experiments_on_model_parameters(num_factors, nearest_neighbors, 'num_factors', 'nearest_neighbors',
    #                                                                 use_approximate_model=False, retrain_every_rewrite=False)


    solvers = {'ilp': IlpSolver(verbose=False), 'ilp-preprocessing': IlpSolver(verbose=False), 'ilp-slicing': IlpSolver(verbose=False)}
    num_recomms_values = [10, 15, 20, 30, 50]
    num_candidates_values = [20, 30, 50, 100, 200]
    item_properties = ['category_2', 'category_3', 'key_type'] # properties for industrial_dataset1
    constraint_lists = [
        [
            MinSegmentsConstraint(segmentation_property='category_3', min_segments=2, weight=0.9, window_size=10),
            GlobalMaxItemsPerSegmentConstraint(segmentation_property='category_2', max_items=3, weight=0.9, window_size=10),
        ]
    ]

    slice_sizes = [2, 5, 7, 8, 10, 12, 15, 16, 20]
    experiment_runner.run_experiments_on_solver(solvers, num_recomms_values, num_candidates_values, constraint_lists, slice_sizes, item_properties)

def measure_changes_with_diversity_constraints():
    start_time = time.time()
    settings = Settings()
    settings.split['train_ratio'] = 0.9925 # 1039 user for movielens
    settings.split['random_state'] = 100
    data_splitter = DataSplitter(settings)
    data_splitter.load_data('movielens')
    data_splitter.split_data()
    train_dataset = data_splitter.get_train_data()
    test_dataset = data_splitter.get_test_data()
    segmentation_extractor = SegmentationExtractor(settings)
    segmentation_extractor.extract_segments('genres')  # segmens for movielens
    segments = segmentation_extractor.get_segments()

    print(f"Train rating matrix shape: {train_dataset.matrix.shape}, Number of users: {len(train_dataset)}")
    print(f"Test rating matrix shape: {test_dataset.matrix.shape}, Number of users: {len(test_dataset)}")

    num_factors = 256
    regularization = 10
    num_nearest_neighbors = 10
    num_iterations = 3
    use_gpu = False
    N = 10
    M = 300

    results = []
    max_items = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    model = ALSModel(num_factors=num_factors, regularization=regularization, num_iterations=num_iterations,
                     use_gpu=use_gpu, nearest_neighbors=num_nearest_neighbors)
    model.train(train_dataset)

    print(f"[ExperimentRunner] Training completed in {time.time() - start_time:.2f} seconds.")

    # Evaluate the model
    evaluator = Evaluator(settings)

    for mi in max_items:
        constraints = [GlobalMaxItemsPerSegmentConstraint(segmentation_property='genres', max_items=mi, weight=0.9, window_size=N)]
        metrics = evaluator.evaluate_constrained_model(train_dataset=train_dataset, test_dataset=test_dataset,
                                                       segmentation_extractor=segmentation_extractor,
                                                       constraints=constraints,
                                                       model=model, N=N, M=M)
        print(f"[ExperimentRunner] Max items: {mi}, Evaluation Metrics: {metrics}, processing time: {time.time() - start_time:.2f} seconds.")
        results.append((mi, metrics))

    with open("diversity_experiment_results.txt", 'a') as f:
        f.write(f'{results}\n')


if __name__ == "__main__":
    print(f"Using file '{RESULTS_FILE}' to save results.")
    main()
    # measure_changes_with_diversity_constraints()
