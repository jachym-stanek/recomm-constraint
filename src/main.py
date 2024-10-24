from scipy.sparse import csr_matrix, load_npz
import numpy as np

from src.data_split import DataSplitter
from src.evaluator import Evaluator
from src.models import ALSModel
from src.settings import Settings


def main():
    settings = Settings()
    data_splitter = DataSplitter(settings)
    data_splitter.load_data('movielens')
    data_splitter.split_data()
    train_dataset= data_splitter.get_train_data()
    test_dataset = data_splitter.get_test_data()

    print(f"Train rating matrix shape: {train_dataset.matrix.shape}, Number of users: {len(train_dataset.user_ids)}")
    print(f"Test rating matrix shape: {test_dataset.matrix.shape}, Number of users: {len(test_dataset.user_ids)}")

    model = ALSModel()
    model.train(train_dataset)

    # Evaluate the model
    evaluator = Evaluator(log_every=2)
    metrics = evaluator.evaluate_recall_at_n(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        model=model,
        N=settings.recommendations['top_n']
    )
    print("[ExperimentRunner] Evaluation Metrics:", metrics)


if __name__ == "__main__":
    main()
