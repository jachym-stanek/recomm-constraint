import time
import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix, load_npz
import numpy as np

from src.data_split import DataSplitter
from src.evaluator import Evaluator
from src.models import ALSModel
from src.settings import Settings


def main():
    start_time = time.time()
    settings = Settings()
    data_splitter = DataSplitter(settings)
    data_splitter.load_data('movielens')
    data_splitter.split_data()
    train_dataset= data_splitter.get_train_data()
    test_dataset = data_splitter.get_test_data()

    print(f"Train rating matrix shape: {train_dataset.matrix.shape}, Number of users: {len(train_dataset)}")
    print(f"Test rating matrix shape: {test_dataset.matrix.shape}, Number of users: {len(test_dataset)}")

    factors = [1, 2, 5, 10, 20, 50, 100, 200, 500]
    regularizations = [0.001, 0.005, 0.01, 0.5]

    results = []
    for num_factors in factors:
        for regularization in regularizations:
            metrics = run_experiment_ALS(settings, train_dataset, test_dataset, num_factors, regularization)
            results.append((num_factors, regularization, metrics))

    # save results to a file
    with open('results.txt', 'w') as f:
        for result in results:
            f.write(f'{result}\n')

    # plot results - recall on x-axis, catalog coverage on y-axis, make one plot for fixed num_factors and varying regularization
    # make second plot for fixed regularization and varying num_factors
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    for num_factors in factors:
        recall = []
        catalog_coverage = []
        for result in results:
            if result[0] == num_factors:
                recall.append(result[2]['average_recall'])
                catalog_coverage.append(result[2]['catalog_coverage'])
        ax[0].plot(recall, catalog_coverage, label=f'Num factors: {num_factors}')
    ax[0].set_xlabel('Average Recall@N')
    ax[0].set_ylabel('Catalog Coverage')
    ax[0].legend()

    for regularization in regularizations:
        recall = []
        catalog_coverage = []
        for result in results:
            if result[1] == regularization:
                recall.append(result[2]['average_recall'])
                catalog_coverage.append(result[2]['catalog_coverage'])
        ax[1].plot(recall, catalog_coverage, label=f'Regularization: {regularization}')
    ax[1].set_xlabel('Average Recall@N')
    ax[1].set_ylabel('Catalog Coverage')
    ax[1].legend()

    plt.show()

    print(f"Execution time: {time.time() - start_time:.2f} seconds")

def run_experiment_ALS(settings, train_dataset, test_dataset, num_factors, regularization):
    print(f"[ExperimentRunner] Running ALS experiment with num_factors={num_factors}, regularization={regularization}...")

    model = ALSModel(num_factors=num_factors, regularization=regularization)
    model.train(train_dataset)

    # Evaluate the model
    evaluator = Evaluator(log_every=10, num_hidden=40)
    # metrics = evaluator.evaluate_recall_at_n(
    #     train_dataset=train_dataset,
    #     test_dataset=test_dataset,
    #     model=model,
    #     N=settings.recommendations['top_n']
    # )
    metrics = evaluator.evaluate_recall_at_n_batch(train_dataset, test_dataset, model, N=settings.recommendations['top_n'])
    print("[ExperimentRunner] Evaluation Metrics:", metrics)

    return metrics


if __name__ == "__main__":
    main()
