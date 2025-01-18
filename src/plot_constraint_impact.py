import matplotlib.pyplot as plt

file_name = "diversity_experiment_results.txt"
results = []
with open(file_name, 'r') as f:
    for line in f:
        results = eval(line)
        break

print(f"results: {results}")

results = results

nums_max_items = [result[0] for result in results]
constrained_recalls = [result[1]["average_recall_constrained"] for result in results]
constrained_catalog_coverages = [result[1]["catalog_coverage_constrained"] for result in results]

# Plot results
# x-axis recall@N, y-axis catalog coverage, plot entry for each max_items
fig, ax = plt.subplots(1, 1, figsize=(10, 7))
line, = ax.plot(constrained_recalls, constrained_catalog_coverages, marker='o', label=f'Max items per segment: {nums_max_items}')
# Annotate the points with max_items values
for i, max_items in enumerate(nums_max_items):
    if i in [0,1,2,3,5,6]:
        ax.annotate(f'{max_items}', (constrained_recalls[i], constrained_catalog_coverages[i]), xytext=(3, 3), textcoords='offset points', fontsize=13)
    elif i in [4, 8]:
        ax.annotate(f'{max_items}', (constrained_recalls[i], constrained_catalog_coverages[i]), xytext=(-12, -12), textcoords='offset points', fontsize=13)
    elif i == 7:
        ax.annotate(f'{max_items}', (constrained_recalls[i], constrained_catalog_coverages[i]), xytext=(8, -10), textcoords='offset points', fontsize=13)

ax.grid(True)
ax.set_xlabel(f'Average Recall@N')
ax.set_ylabel('Catalog Coverage')
ax.set_title('Impact of Diversity Constraints on Catalog Coverage and Recall')

plt.show()
