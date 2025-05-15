import time, os, re

from src.algorithms.ILP import IlpSolver
from src.algorithms.CP import CpSolver
from src.algorithms.StateSpaceSearch import StateSpaceSolver
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
    # settings.set_dataset_in_use('industrial_dataset1')
    # settings.set_dataset_in_use('movielens')
    settings.set_dataset_in_use('industrial_dataset2')
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
    nearest_neighbors = [1, 2, 4, 6, 8, 10, 15, 20, 30, 50, 70, 100, 150, 200]
    num_factors = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    # num_factors = [256, 512, 1024, 2048]
    bm_bs = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 4.0, 8, 16]
    num_iterations = [1, 2, 3, 5, 8, 10, 15]

    # segmentation_extractor = SegmentationExtractor(settings)
    # segmentation_extractor.extract_segments('genres')
    # segmentation_extractor.extract_segments('category_1')  # segments for industrial_dataset1

    experiment_runner = ExperimentRunner(settings, RESULTS_FILE, train_dataset, test_dataset)

    # results = experiment_runner.run_experiments_on_model_parameters(regularizations, nearest_neighbors, 'regularization',
    #                                                                 'nearest_neighbors',
    #                                                                 use_approximate_model=False,
    #                                                                 retrain_every_rewrite=False)

    # results = experiment_runner.run_experiments_on_model_parameters(bm_bs, nearest_neighbors,
    #                                                                 'bm25_B',
    #                                                                 'nearest_neighbors',
    #                                                                 use_approximate_model=False,
    #                                                                 retrain_every_rewrite=False)

    results = experiment_runner.run_experiments_on_model_parameters(num_factors, nearest_neighbors,
                                                                    'num_factors',
                                                                    'nearest_neighbors',
                                                                    use_approximate_model=False,
                                                                    retrain_every_rewrite=False)

    # results = experiment_runner.run_experiments_on_model_parameters(num_factors, nearest_neighbors, 'num_factors', 'nearest_neighbors',
    #                                                                 use_approximate_model=False, retrain_every_rewrite=False)


def evaluate_solvers_on_id1():
    start_time = time.time()
    settings = Settings(config_file="../settings/solver_evaluation_config.json")
    settings.set_dataset_in_use('industrial_dataset1')
    data_splitter = DataSplitter(settings)
    data_splitter.load_data(settings.dataset_name)
    data_splitter.split_data()
    train_dataset = data_splitter.get_train_data()
    test_dataset = data_splitter.get_test_data()
    experiment_runner = ExperimentRunner(settings, RESULTS_FILE, train_dataset, test_dataset)
    solvers = {'ilp': IlpSolver(verbose=False), 'ilp-preprocessing': IlpSolver(verbose=False),
               'ilp-slicing': IlpSolver(verbose=False)} # not evaluating CP solver because it cannot handle soft constraints
    num_recomms_values = [10, 15, 20, 30, 50]
    num_candidates_values = [20, 30, 50, 100]
    item_properties = ['category_2', 'category_3', 'key_type']  # properties for industrial_dataset1
    constraint_generator = ConstraintGenerator()
    random_5_constraints = constraint_generator.generate_random_constraints(num_constraints=5, num_recommendations=10,
                                                        segmentation_properties=item_properties, min_window_size=2, weight_type="soft",
                                                        exclude_specific=[ItemAtPositionConstraint, ItemFromSegmentAtPositionConstraint, MinItemsPerSegmentConstraint, MaxItemsPerSegmentConstraint])
    random_10_constraints = constraint_generator.generate_random_constraints(num_constraints=10, num_recommendations=10,
                                                        segmentation_properties=item_properties, min_window_size=2, weight_type="soft",
                                                        exclude_specific=[ItemAtPositionConstraint, ItemFromSegmentAtPositionConstraint])
    constraint_lists = [
        random_5_constraints,
        random_10_constraints,
        [
            MinSegmentsConstraint(segmentation_property='category_3', min_segments=2, weight=0.9, window_size=10),
            GlobalMaxItemsPerSegmentConstraint(segmentation_property='category_2', max_items=3, weight=0.9,
                                               window_size=10),
        ],
        [
            MaxSegmentsConstraint(segmentation_property='category_3', max_segments=3, weight=0.9, window_size=10),
            GlobalMaxItemsPerSegmentConstraint(segmentation_property='category_2', max_items=2, weight=0.9,
                                               window_size=5),
        ]
    ]

    slice_sizes = [2, 5, 7, 8, 10, 12, 15, 16, 20]
    experiment_runner.run_experiments_on_solver(solvers, num_recomms_values, num_candidates_values, constraint_lists,
                                                slice_sizes, item_properties)
    print(f"[ExperimentRunner] Evaluation completed in {time.time() - start_time:.2f} seconds.")

def evaluate_solvers_on_id2():
    start_time = time.time()
    settings = Settings(config_file="../settings/solver_evaluation_config.json")
    settings.set_dataset_in_use('industrial_dataset2')
    data_splitter = DataSplitter(settings)
    data_splitter.load_data(settings.dataset_name)
    data_splitter.split_data()
    train_dataset = data_splitter.get_train_data()
    test_dataset = data_splitter.get_test_data()
    experiment_runner = ExperimentRunner(settings, RESULTS_FILE, train_dataset, test_dataset)
    ILP_time_limit = 5  # seconds
    solvers = {
                'ilp': IlpSolver(verbose=False, time_limit=ILP_time_limit),
                'ilp-preprocessing': IlpSolver(verbose=False, time_limit=ILP_time_limit),
                'ilp-slicing': IlpSolver(verbose=False, time_limit=ILP_time_limit),
                'state_space': StateSpaceSolver(verbose=False),
               } # not evaluating CP solver because it cannot handle soft constraints
    num_recomms_values = [10, 15, 20, 25, 30]
    # num_candidates_values = [20, 30, 50, 100]
    num_candidates_values = [60]
    item_properties = ["custom_label_0","custom_label_2","brand","product_type"]  # properties for industrial_dataset2
    constraint_generator = ConstraintGenerator()
    random_5_constraints = constraint_generator.generate_random_constraints(num_constraints=5, num_recommendations=10,
                                                        segmentation_properties=item_properties, min_window_size=2, weight_type="soft",
                                                        exclude_specific=[ItemAtPositionConstraint, ItemFromSegmentAtPositionConstraint, MinItemsPerSegmentConstraint, MaxItemsPerSegmentConstraint])
    random_10_constraints = constraint_generator.generate_random_constraints(num_constraints=10, num_recommendations=10,
                                                        segmentation_properties=item_properties, min_window_size=2, weight_type="soft",
                                                        exclude_specific=[ItemAtPositionConstraint, ItemFromSegmentAtPositionConstraint])
    constraint_lists = [
        random_5_constraints,
        random_10_constraints,
        [
            MinSegmentsConstraint(segmentation_property='category_3', min_segments=2, weight=0.9, window_size=10),
            GlobalMaxItemsPerSegmentConstraint(segmentation_property='category_2', max_items=3, weight=0.9,
                                               window_size=10),
        ],
        [
            MaxSegmentsConstraint(segmentation_property='category_3', max_segments=3, weight=0.9, window_size=10),
            GlobalMaxItemsPerSegmentConstraint(segmentation_property='category_2', max_items=2, weight=0.9,
                                               window_size=5),
        ]
    ]

    slice_sizes = [1, 2, 3, 4, 5, 7, 8]
    experiment_runner.run_experiments_on_solver(solvers, num_recomms_values, num_candidates_values, constraint_lists,
                                                slice_sizes, item_properties)
    print(f"[ExperimentRunner] Evaluation completed in {time.time() - start_time:.2f} seconds.")

def evaluate_solvers_on_movielens():
    start_time = time.time()
    settings = Settings(config_file="../settings/solver_evaluation_config.json")
    settings.set_dataset_in_use('movielens')
    data_splitter = DataSplitter(settings)
    data_splitter.load_data(settings.dataset_name)
    data_splitter.split_data()
    train_dataset = data_splitter.get_train_data()
    test_dataset = data_splitter.get_test_data()
    experiment_runner = ExperimentRunner(settings, RESULTS_FILE, train_dataset, test_dataset)
    ILP_time_limit = 5 # seconds
    solvers = {'ilp': IlpSolver(verbose=False, time_limit=ILP_time_limit), 'ilp-preprocessing': IlpSolver(verbose=False, time_limit=ILP_time_limit),
               'ilp-slicing': IlpSolver(verbose=False, time_limit=ILP_time_limit)} # not evaluating CP solver because it cannot handle soft constraints
    num_recomms_values = [10, 15, 20, 30, 50]
    num_candidates_values = [20, 30, 50, 100, 200, 300]
    item_properties = ['genres', 'title', 'year']
    constraint_generator = ConstraintGenerator()
    random_5_constraints = constraint_generator.generate_random_constraints(num_constraints=5, num_recommendations=10,
                                                        segmentation_properties=item_properties, min_window_size=2, weight_type="soft",
                                                        exclude_specific=[ItemAtPositionConstraint, ItemFromSegmentAtPositionConstraint, MinItemsPerSegmentConstraint, MaxItemsPerSegmentConstraint])
    random_10_constraints = constraint_generator.generate_random_constraints(num_constraints=10, num_recommendations=10,
                                                        segmentation_properties=item_properties, min_window_size=2, weight_type="soft",
                                                        exclude_specific=[ItemAtPositionConstraint, ItemFromSegmentAtPositionConstraint])
    constraint_lists = [
        random_5_constraints,
        random_10_constraints,
        [
            MinSegmentsConstraint(segmentation_property='year', min_segments=2, weight=0.9, window_size=10),
            GlobalMaxItemsPerSegmentConstraint(segmentation_property='genres', max_items=3, weight=0.9,
                                               window_size=10),
        ],
        [
            MaxSegmentsConstraint(segmentation_property='year', max_segments=3, weight=0.9, window_size=5),
            GlobalMaxItemsPerSegmentConstraint(segmentation_property='genres', max_items=2, weight=0.9,
                                               window_size=5),
        ]
    ]

    slice_sizes = [2, 5, 7, 8, 10, 15, 20]
    experiment_runner.run_experiments_on_solver(solvers, num_recomms_values, num_candidates_values, constraint_lists,
                                                slice_sizes, item_properties)
    print(f"[ExperimentRunner] Evaluation completed in {time.time() - start_time:.2f} seconds.")

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
    # evaluate_solvers_on_id1()
    # evaluate_solvers_on_movielens()
    # evaluate_solvers_on_id2()
