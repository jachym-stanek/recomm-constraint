import time, os, re
import matplotlib.pyplot as plt


from scipy.sparse import csr_matrix, load_npz
import numpy as np

from src.data_split import DataSplitter
from src.evaluator import Evaluator
from src.models import ALSModel
from src.settings import Settings
from segmentation import SegmentationExtractor

# results file will have number one higher than the highest number in the existing results files
RESULTS_FILE_ALS = "results" + str(max([int(re.search(r'results(\d+)\.txt', f).group(1)) for f in os.listdir('.') if re.match(r'results\d+\.txt', f)] + [0]) + 1) + ".txt"

def main():
    start_time = time.time()
    settings = Settings()
    data_splitter = DataSplitter(settings)
    data_splitter.load_data('movielens')
    # data_splitter.load_data('bookcrossing')
    data_splitter.split_data()
    train_dataset= data_splitter.get_train_data()
    test_dataset = data_splitter.get_test_data()

    print(f"Train rating matrix shape: {train_dataset.matrix.shape}, Number of users: {len(train_dataset)}")
    print(f"Test rating matrix shape: {test_dataset.matrix.shape}, Number of users: {len(test_dataset)}")

    # factors = [1, 2, 5, 7, 10, 20, 50, 100, 200, 500]
    factors = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    regularizations = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    # num_nearest_neighbors = [2, 4, 5, 6, 8, 10, 15, 20, 30, 50]
    # alphas = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    # num_iterations = [1, 2, 3, 5, 8, 10, 15]

    segmentation_extractor = SegmentationExtractor(settings)
    segmentation_extractor.extract_segments('genres') # segmens for movielens

    results = []
    for num_factors in factors:
        # for K in num_nearest_neighbors:
        for regularization in regularizations:
        # for num_iters in num_iterations:
        # for alpha in alphas:
        #     metrics = run_experiment_ILP(settings, train_dataset, test_dataset, segmentation_extractor, num_factors, regularization, 3, 1.0)
            # metrics = run_experiment_ALS(settings, train_dataset, test_dataset, num_factors, 0.01, num_iters)
            # metrics = run_experiment_ALS(settings, train_dataset, test_dataset, num_factors, 0.01, 3, alpha)
            # metrics = run_experiment_ILP(settings, train_dataset, test_dataset, segmentation_extractor, num_factors, 0.01, 3, alpha)
            metrics = run_experiment_ALS(settings, train_dataset, test_dataset, num_factors=num_factors, regularization=regularization, num_iters=3)
            # metrics = run_experiment_ALS(settings, train_dataset, test_dataset, 128, regularization, num_iters=3, N=N)
            # metrics = run_experiment_ALS(settings, train_dataset, test_dataset, num_factors, 10., num_iters=3, K=K)
            result = (num_factors, regularization, metrics)
            # result = (N, regularization, metrics)
            # result = (num_factors, K, metrics)
            results.append(result)
            with open(RESULTS_FILE_ALS, 'a') as f:
                    f.write(f'{result}\n')
            # results.append((num_factors, num_iters, metrics))
            # results.append((num_factors, alpha, metrics))

    print(f"Execution time: {time.time() - start_time:.2f} seconds")

def run_experiment_ALS(settings, train_dataset, test_dataset, num_factors=256, regularization=1.0, num_iters=3, K=10, alpha=1.0):
    print("---------------------------------")
    print(f"[ExperimentRunner] Running ALS experiment with num_factors={num_factors}, regularization={regularization}, number of iterations: {num_iters}, alpha: {alpha}, K={K}...")

    start_time = time.time()

    model = ALSModel(num_factors=num_factors, regularization=regularization, num_iterations=num_iters, alpha=alpha,
                     use_gpu=settings.use_gpu, nearest_neighbors=settings.nearest_neighbors)
    model.train(train_dataset)

    print(f"[ExperimentRunner] Training completed in {time.time() - start_time:.2f} seconds.")

    # Evaluate the model
    evaluator = Evaluator(settings)
    metrics = evaluator.evaluate_recall_at_n(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        model=model,
        N=settings.recommendations['top_n'],
        K=K
    )
    # metrics = evaluator.evaluate_recall_at_n_batch(train_dataset, test_dataset, model, N=settings.recommendations['top_n'])
    print(f"[ExperimentRunner] Evaluation Metrics: {metrics}, processing time: {time.time() - start_time:.2f} seconds.")

    return metrics


def run_experiment_ILP(settings, train_dataset, test_dataset, segmentation_extractor, num_factors, regularization, num_iters, alpha=1.0):
    print(
        f"[ExperimentRunner] Running experiment with num_factors={num_factors}, regularization={regularization}, number of iterations: {num_iters}, alpha: {alpha} ...")

    start_time = time.time()

    with open(RESULTS_FILE_ALS, 'a') as f:
        f.write(
            f'Running experiment with num_factors={num_factors}, regularization={regularization}, number of iterations: {num_iters}, alpha: {alpha} ....\n')

    model = ALSModel(num_factors=num_factors, regularization=regularization, num_iterations=num_iters, alpha=alpha,
                     use_gpu=settings.use_gpu)
    model.train(train_dataset)

    print(f"[ExperimentRunner] Training completed in {time.time() - start_time:.2f} seconds.")

    # Evaluate the model
    evaluator = Evaluator(settings)
    metrics = evaluator.evaluate_constrained_model(train_dataset=train_dataset, test_dataset=test_dataset,
                                                   segmentation_extractor=segmentation_extractor, model=model, N=settings.recommendations['top_n'])
    print(f"[ExperimentRunner] Evaluation Metrics: {metrics}, processing time: {time.time() - start_time:.2f} seconds.")
    with open(RESULTS_FILE_ALS, 'a') as f:
        f.write(f'Evaluation Metrics: {metrics}\n')

    return metrics


if __name__ == "__main__":
    print(f"[ExperimentRunner] Using file '{RESULTS_FILE_ALS}' to save results.")
    main()
