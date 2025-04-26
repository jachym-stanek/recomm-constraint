import os


# file = 'results_id1_factors_vs_num_neighbors.txt'
file = 'results_movielens_factors_vs_num_neighbors.txt'
results = []


with open(file, 'r') as f:
    for line in f:
        results.append(eval(line))

print(results)

# new_file = 'results_id1_factors_vs_num_neighbors_fixed.txt'
new_file = 'results_movielens_factors_vs_num_neighbors_fixed.txt'
new_results = []

nearest_neighbors = [2, 4, 6, 8, 10, 15, 20, 30, 50]
num_factors = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
for i in range(len(nearest_neighbors)):
    for j in range(len(num_factors)):
        index = i * len(num_factors) + j
        fixed_result = ({"nearest_neighbors": nearest_neighbors[i], "num_factors": num_factors[j]}, results[index])

        new_results.append(fixed_result)
with open(new_file, 'w') as f:
    for result in new_results:
        f.write(str(result) + '\n')