import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# factors = [50, 100, 200, 500]
# regularizations = [0.001, 0.005, 0.01, 0.5]
# factors = [1, 2, 5, 10, 20, 50, 100, 200, 500]
# num_iterations = [1, 2, 3, 5, 8, 10, 15]
# factors = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]
# regularizations = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
# alphas = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
factors = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25]
regularizations = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]

PLOT_CONSTRAINED = True

results = []
with open('results7.txt', 'r') as f:
    for line in f:
        # num_factors, num_iters, metrics = eval(line)
        # num_factors, alpha , metrics = eval(line)
        num_factors, regularization, metrics = eval(line)
        # results.append((num_factors, num_iters, metrics))
        results.append((num_factors, regularization, metrics))
        # results.append((num_factors, alpha, metrics))
print(results)


# Plot results - recall on x-axis, catalog coverage on y-axis
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# First plot: fixed num_factors, varying alpha
for num_factors in factors:
    recall = []
    catalog_coverage = []
    recall_constrained = []
    catalog_coverage_constrained = []
    # alpha_values = []
    regularization_values = []
    for result in results:
        if result[0] == num_factors:
            recall.append(result[2]['average_recall'])
            catalog_coverage.append(result[2]['catalog_coverage'])
            recall_constrained.append(result[2]['average_recall_constrained'])
            catalog_coverage_constrained.append(result[2]['catalog_coverage_constrained'])
            # alpha_values.append(result[1])
            regularization_values.append(result[1])
    # Plot unconstrained metrics
    line, = ax[0].plot(recall, catalog_coverage, marker='o', label=f'Num factors: {num_factors}')
    # Plot constrained metrics with same color, dotted line
    if PLOT_CONSTRAINED:
        ax[0].plot(recall_constrained, catalog_coverage_constrained, linestyle='dotted', color=line.get_color())
    # Annotate the points with alpha values for unconstrained
    # for i, alpha_value in enumerate(alpha_values):
    #     ax[0].annotate(f'{alpha_value}', (recall[i], catalog_coverage[i]))
    for i, regularization_value in enumerate(regularization_values):
        ax[0].annotate(f'{regularization_value}', (recall[i], catalog_coverage[i]))
    # Annotate the points with alpha values for constrained
    if PLOT_CONSTRAINED:
        # for i, alpha_value in enumerate(alpha_values):
        #     ax[0].annotate(f'{alpha_value}', (recall_constrained[i], catalog_coverage_constrained[i]))
        for i, regularization_value in enumerate(regularization_values):
            ax[0].annotate(f'{regularization_value}', (recall_constrained[i], catalog_coverage_constrained[i]))
ax[0].set_xlabel(f'Average Recall@N')
ax[0].set_ylabel('Catalog Coverage')
# ax[0].set_title('Fixed Num Factors, Varying Alpha')
ax[0].set_title('Fixed Num Factors, Varying Regularization')

# Create custom legend entries for unconstrained and constrained
handles, labels = ax[0].get_legend_handles_labels()
line_unconstrained = Line2D([0], [0], color='black', linestyle='-')
if PLOT_CONSTRAINED:
    line_constrained = Line2D([0], [0], color='black', linestyle='dotted')
    ax[0].legend(handles + [line_unconstrained, line_constrained],
                 labels + ['Unconstrained', 'Constrained'])
else:
    ax[0].legend(handles + [line_unconstrained], labels + ['Unconstrained'])

# Second plot: fixed alpha, varying num_factors
# for alpha in alphas:
for regularization in regularizations:
    recall = []
    catalog_coverage = []
    recall_constrained = []
    catalog_coverage_constrained = []
    num_factors_values = []
    for result in results:
        # if result[1] == alpha:
        if result[1] == regularization:
            recall.append(result[2]['average_recall'] )
            catalog_coverage.append(result[2]['catalog_coverage'])
            recall_constrained.append(result[2]['average_recall_constrained'])
            catalog_coverage_constrained.append(result[2]['catalog_coverage_constrained'])
            num_factors_values.append(result[0])
    # Plot unconstrained metrics
    # line, = ax[1].plot(recall, catalog_coverage, marker='o', label=f'Alpha: {alpha}')
    line, = ax[1].plot(recall, catalog_coverage, marker='o', label=f'Regularization: {regularization}')
    # Plot constrained metrics with same color, dotted line
    if PLOT_CONSTRAINED:
        ax[1].plot(recall_constrained, catalog_coverage_constrained, linestyle='dotted', color=line.get_color())
    # Annotate the points with num_factors values
    for i, num_factors_value in enumerate(num_factors_values):
        ax[1].annotate(f'{num_factors_value}', (recall[i], catalog_coverage[i]))
    # Annotate the points with num_factors values for constrained
    if PLOT_CONSTRAINED:
        for i, num_factors_value in enumerate(num_factors_values):
            ax[1].annotate(f'{num_factors_value}', (recall_constrained[i], catalog_coverage_constrained[i]))
ax[1].set_xlabel(f'Average Recall@N')
ax[1].set_ylabel('Catalog Coverage')
# ax[1].set_title('Fixed Alpha, Varying Num Factors')
ax[1].set_title('Fixed Regularization, Varying Num Factors')

# Create custom legend entries for unconstrained and constrained
handles, labels = ax[1].get_legend_handles_labels()
line_unconstrained = Line2D([0], [0], color='black', linestyle='-')
if PLOT_CONSTRAINED:
    line_constrained = Line2D([0], [0], color='black', linestyle='dotted')
    ax[1].legend(handles + [line_unconstrained, line_constrained],
                 labels + ['Unconstrained', 'Constrained'])
else:
    ax[1].legend(handles + [line_unconstrained], labels + ['Unconstrained'])

plt.tight_layout()
plt.show()
