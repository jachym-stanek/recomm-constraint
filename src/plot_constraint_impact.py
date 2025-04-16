import matplotlib.pyplot as plt

file_name = "diversity_experiment_results.txt"
results = []
with open(file_name, 'r') as f:
    for line in f:
        results = eval(line)
        break

print(f"results: {results}")

# remove the value for max_item=1 as its misleading
results = [result for result in results if result[0] not in [1, 10]]

nums_max_items = [result[0] for result in results]
constrained_recalls = [result[1]["average_recall_constrained"] for result in results]
constrained_catalog_coverages = [result[1]["catalog_coverage_constrained"] for result in results]

print(f"nums_max_items: {nums_max_items}")
print(f"constrained_recalls: {constrained_recalls}")
print(f"constrained_catalog_coverages: {constrained_catalog_coverages}")

# Plot results
# x-axis recall@N, y-axis catalog coverage, plot entry for each max_items
fig, ax = plt.subplots(1, 1, figsize=(10, 7))
line, = ax.plot(constrained_recalls, constrained_catalog_coverages, marker='o', linewidth=2, label=f'Max items per segment: {nums_max_items}')
# Annotate the points with max_items values
for i, max_items in enumerate(nums_max_items):
    if max_items == 8:
        ax.annotate(f'{max_items}', (constrained_recalls[i], constrained_catalog_coverages[i]), xytext=(-13, -5),
                    textcoords='offset points', fontsize=13)
    elif max_items == 9:
        ax.annotate(f'{max_items}', (constrained_recalls[i], constrained_catalog_coverages[i]), xytext=(+5, -8),
                    textcoords='offset points', fontsize=13)
    else:
        ax.annotate(f'{max_items}', (constrained_recalls[i], constrained_catalog_coverages[i]), xytext=(3, 3),
                    textcoords='offset points', fontsize=13)

ax.grid(True)
ax.set_xlabel(f'Average Recall@N', fontsize=18)
ax.set_ylabel('Catalog Coverage', fontsize=18)
# ax.set_title('Impact of Diversity Constraints on Catalog Coverage and Recall')

plt.show()
