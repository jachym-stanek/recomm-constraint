import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix, load_npz

from src.settings import Settings


SETTINGS = Settings()  # Load default settings


def check_data():
    # load files
    SETTINGS.set_dataset_in_use('industrial_dataset1')
    items_file = SETTINGS.dataset.get('items_file')
    users_file = SETTINGS.dataset.get('users_file')
    bookmarks_file = SETTINGS.dataset.get('bookmarks_file')
    detail_views_file = SETTINGS.dataset.get('detail_views_file')
    purchases_file = SETTINGS.dataset.get('purchases_file')

    # check that all users have at least one detail view
    users = pd.read_csv(users_file)
    detail_views = pd.read_csv(detail_views_file)
    all_users_have_detail_view = detail_views['user_id'].isin(users['user_id']).all()
    print(f"All users have at least one detail view: {all_users_have_detail_view}")


def analyse_detail_views():
    # load files
    SETTINGS.set_dataset_in_use('industrial_dataset1')
    detail_views_file = SETTINGS.dataset.get('detail_views_file')
    users_file = SETTINGS.dataset.get('users_file')

    # check that all users have at least one detail view
    detail_views = pd.read_csv(detail_views_file)
    num_detail_views = len(detail_views)
    print(f"Number of detail views: {num_detail_views}")

    # check that all users have at least one detail view
    num_users = detail_views['user_id'].nunique()
    print(f"Number of users with detail views: {num_users}")
    # compare with users file
    users = pd.read_csv(users_file)
    num_users_in_file = len(users)
    print(f"Number of users in users file: {num_users_in_file}")

    num_items = detail_views.groupby('user_id')['item_id'].nunique()

    labels = ['0', '1', '2', '3', '4', '5', '6', '7', '≥8']
    # cut off everything ≥8 into the last bin
    bins = [-0.5, 0.5, 1.5, 2.5, 3.5,
            4.5, 5.5, 6.5, 7.5, float('inf')]

    # bucket and count
    buckets = pd.cut(num_items, bins=bins, labels=labels)
    counts = buckets.value_counts().loc[labels]  # ensure correct order

    # plot discrete bars
    fig, ax = plt.subplots(figsize=(8, 4))
    counts.plot(kind='bar', ax=ax, rot=0)
    ax.set_xlabel("Number of unique items viewed")
    ax.set_ylabel("Number of users")
    ax.set_title("Distribution of unique items viewed per user")
    plt.tight_layout()
    plt.show()

def analyze_rating_matrix():
    SETTINGS.set_dataset_in_use('industrial_dataset1')
    rating_matrix_file = SETTINGS.dataset.get('rating_matrix_file')
    rating_matrix = load_npz(rating_matrix_file)
    num_ratings = rating_matrix.getnnz(axis=1)  # array of length n_users

    labels = [str(i) for i in range(0, 8)] + ['≥8']
    bins = [-0.5, 0.5, 1.5, 2.5, 3.5,
            4.5, 5.5, 6.5, 7.5, np.inf]

    buckets = pd.cut(num_ratings, bins=bins, labels=labels)
    counts = buckets.value_counts().loc[labels]

    # plot
    fig, ax = plt.subplots(figsize=(8, 4))
    counts.plot(kind='bar', ax=ax, rot=0, width=0.8)

    ax.set_xlabel('Number of ratings per user')
    ax.set_ylabel('Number of users')
    ax.set_title('Distribution of ratings per user')
    plt.tight_layout()
    plt.show()


def analyze_number_of_interactions():
    # industrial dataset1
    SETTINGS.set_dataset_in_use('industrial_dataset1')
    users_file = SETTINGS.dataset.get('users_file')
    items_file = SETTINGS.dataset.get('items_file')
    bookmarks_file = SETTINGS.dataset.get('bookmarks_file')
    detail_views_file = SETTINGS.dataset.get('detail_views_file')
    purchases_file = SETTINGS.dataset.get('purchases_file')
    users = pd.read_csv(users_file)
    items = pd.read_csv(items_file)
    bookmarks = pd.read_csv(bookmarks_file)
    detail_views = pd.read_csv(detail_views_file)
    purchases = pd.read_csv(purchases_file)

    num_users = users['user_id'].nunique()
    num_items = items['item_id'].nunique()
    num_bookmarks = len(bookmarks)
    num_detail_views = len(detail_views)
    num_purchases = len(purchases)
    print("=== Industrial dataset1 ===")
    print(f"Number of users: {num_users}")
    print(f"Number of items: {num_items}")
    print(f"Number of bookmarks: {num_bookmarks}")
    print(f"Number of detail views: {num_detail_views}")
    print(f"Number of purchases: {num_purchases}")
    print(f"Total number of interactions: {num_bookmarks + num_detail_views + num_purchases}")

    # count number of interactions (bookmarks + detail views + purchases) per user
    interactions = pd.concat([bookmarks, detail_views, purchases], ignore_index=True)
    num_interactions_per_user = interactions.groupby('user_id').size() # multiple interactions with the same item are counted

    # count 1st percentile, 25th percentile, 50th percentile, 75th percentile, 99th percentile
    percentiles = np.percentile(num_interactions_per_user, [1, 25, 50, 75, 99])
    print("Percentiles of number of interactions per user:")
    print(f"1st percentile: {percentiles[0]}")
    print(f"25th percentile: {percentiles[1]}")
    print(f"50th percentile: {percentiles[2]}")
    print(f"75th percentile: {percentiles[3]}")
    print(f"99th percentile: {percentiles[4]}")

    # count number of interactions per item
    num_interactions_per_item = interactions.groupby('item_id')['user_id'].size()
    # count 1st percentile, 25th percentile, 50th percentile, 75th percentile, 99th percentile
    percentiles = np.percentile(num_interactions_per_item, [1, 25, 50, 75, 99])
    print("Percentiles of number of interactions per item:")
    print(f"1st percentile: {percentiles[0]}")
    print(f"25th percentile: {percentiles[1]}")
    print(f"50th percentile: {percentiles[2]}")
    print(f"75th percentile: {percentiles[3]}")
    print(f"99th percentile: {percentiles[4]}")


    # count movielens
    SETTINGS.set_dataset_in_use('movielens')
    users_file = SETTINGS.dataset.get('users_file')
    items_file = SETTINGS.dataset.get('items_file')
    ratings_file = SETTINGS.dataset.get('ratings_file')
    users = pd.read_csv(users_file)
    items = pd.read_csv(items_file)
    ratings = pd.read_csv(ratings_file)

    num_users = users['user_id'].nunique()
    num_items = items['item_id'].nunique()
    num_ratings = len(ratings)
    print("\n=== Movielens ===")
    print(f"Number of users: {num_users}")
    print(f"Number of items: {num_items}")
    print(f"Number of ratings: {num_ratings}")

    # count number of interactions per user
    num_interactions_per_user = ratings.groupby('userId')['movieId'].size()
    # count 1st percentile, 25th percentile, 50th percentile, 75th percentile, 99th percentile
    percentiles = np.percentile(num_interactions_per_user, [1, 25, 50, 75, 99])
    print("Percentiles of number of interactions per user:")
    print(f"1st percentile: {percentiles[0]}")
    print(f"25th percentile: {percentiles[1]}")
    print(f"50th percentile: {percentiles[2]}")
    print(f"75th percentile: {percentiles[3]}")
    print(f"99th percentile: {percentiles[4]}")

    # count number of interactions per item
    num_interactions_per_item = ratings.groupby('movieId')['userId'].size()
    # count 1st percentile, 25th percentile, 50th percentile, 75th percentile, 99th percentile
    percentiles = np.percentile(num_interactions_per_item, [1, 25, 50, 75, 99])
    print("Percentiles of number of interactions per item:")
    print(f"1st percentile: {percentiles[0]}")
    print(f"25th percentile: {percentiles[1]}")
    print(f"50th percentile: {percentiles[2]}")
    print(f"75th percentile: {percentiles[3]}")
    print(f"99th percentile: {percentiles[4]}")


if __name__ == '__main__':
    # check_data()
    # analyse_detail_views()
    # analyze_rating_matrix()
    analyze_number_of_interactions()
