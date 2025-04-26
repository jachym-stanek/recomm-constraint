import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# factors = [50, 100, 200, 500]
# regularizations = [0.001, 0.005, 0.01, 0.5]
# factors = [1, 2, 5, 10, 20, 50, 100, 200, 500]
# num_iterations = [1, 2, 3, 5, 8, 10, 15]
# factors = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]
# regularizations = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
# alphas = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
# factors = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25]
# regularizations = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
# factors = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
# regularizations = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
# regularizations = [0.01, 0.1, 1.0, 10.0, 100.0]
# num_nearest_neighbors = [2, 4, 5, 6, 8, 10, 15, 20, 30, 50]
# regularizations = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
# num_nearest_neighbors = [2, 4, 6, 8, 10, 15, 20, 30, 50]
# factors = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
# num_nearest_neighbors = [2, 4, 6, 8, 10, 15, 20, 30, 50]
# factors = [8, 16, 32, 64, 128, 256, 512, 1024]
# num_nearest_neighbors = [2, 4, 8, 15, 30, 50]
# factors = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 32, 64, 128, 256]
# factors = [3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 32, 64, 128, 256, 512, 1024]
# num_trees = [2, 10, 50, 70, 100, 200]

# nearest_neighbors = [2, 4, 6, 8, 10, 12, 15, 17, 20, 25, 30, 40, 50]
# skipped_neighbors = [20, 25, 40]
# factors = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
# skipped_factors = [2, 512]

nearest_neighbors = [2, 4, 6, 8, 10, 15, 20, 30, 50]
factors = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
skipped_neighbors = []
skipped_factors = []

nearest_neighbors = [nn for nn in nearest_neighbors if nn not in skipped_neighbors]
factors = [nf for nf in factors if nf not in skipped_factors]


PLOT_CONSTRAINED = False

results = []
# with open('results_id1_factors_vs_num_neighbors_fixed.txt', 'r') as f:
with open('results_movielens_nearest_neighbors_vs_factors.txt', 'r') as f:
    for line in f:
        # num_factors, num_iters, metrics = eval(line)
        # num_factors, alpha , metrics = eval(line)
        # num_factors, regularization, metrics = eval(line)
        # num_factors, num_nn, metrics = eval(line)
        # if num_factors not in (4, 2048) and num_nn not in (6, 10, 20):
        #     results.append((num_factors, num_nn, metrics))
        # num_nn, regularization, metrics = eval(line)
        # results.append((num_factors, num_iters, metrics))
        # results.append((num_factors, regularization, metrics))
        # results.append((num_factors, alpha, metrics))
        # if regularization != 1000:
        #     results.append((num_nn, regularization, metrics))
        # f, nt, metrics = eval(line)
        params_rewrite, metrics = eval(line)
        facs = params_rewrite['num_factors']
        K = params_rewrite['nearest_neighbors']
        if facs not in factors or K not in nearest_neighbors:
            continue
        results.append((facs, K, metrics))
print(results)


# Plot results - recall on x-axis, catalog coverage on y-axis
fig, ax = plt.subplots(1, 2, figsize=(18, 7))

# First plot: fixed num_factors, varying alpha
for nf in factors:
# for nn in num_nearest_neighbors:
    recall = []
    catalog_coverage = []
    recall_constrained = []
    catalog_coverage_constrained = []
    # alpha_values = []
    K_values = []
    regularization_values = []
    for result in results:
        if result[0] == nf and result[2]['average_recall'] > 0.0:
        # if result[0] == nn and result[2]['average_recall'] > 0.0:
            recall.append(result[2]['average_recall'])
            catalog_coverage.append(result[2]['catalog_coverage'])
            if PLOT_CONSTRAINED:
                recall_constrained.append(result[2]['average_recall_constrained'])
                catalog_coverage_constrained.append(result[2]['catalog_coverage_constrained'])
            # alpha_values.append(result[1])
            # regularization_values.append(result[1])
            K_values.append(result[1])
    # Plot unconstrained metrics
    line, = ax[0].plot(recall, catalog_coverage, marker='o', label=f'Num factors: {nf}')
    # line, = ax[0].plot(recall, catalog_coverage, marker='o', label=f'K: {nn}')
    # Plot constrained metrics with same color, dotted line
    if PLOT_CONSTRAINED:
        ax[0].plot(recall_constrained, catalog_coverage_constrained, linestyle='dotted', color=line.get_color())
    # Annotate the points with alpha values for unconstrained
    # for i, alpha_value in enumerate(alpha_values):
    #     ax[0].annotate(f'{alpha_value}', (recall[i], catalog_coverage[i]))
    # for i, regularization_value in enumerate(regularization_values):
    #     ax[0].annotate(f'{regularization_value}', (recall[i], catalog_coverage[i]))
    for i, K_value in enumerate(K_values):
        ax[0].annotate(f'{K_value}', (recall[i], catalog_coverage[i]))
    # Annotate the points with alpha values for constrained
    if PLOT_CONSTRAINED:
        # for i, alpha_value in enumerate(alpha_values):
        #     ax[0].annotate(f'{alpha_value}', (recall_constrained[i], catalog_coverage_constrained[i]))
        for i, regularization_value in enumerate(regularization_values):
            ax[0].annotate(f'{regularization_value}', (recall_constrained[i], catalog_coverage_constrained[i]))
ax[0].set_xlabel(f'Average Recall@N')
ax[0].set_ylabel('Catalog Coverage')
# ax[0].set_title('Fixed Num Factors, Varying Alpha')
# ax[0].set_title('Fixed Num Factors, Varying Regularization')
# ax[0].set_title('Fixed Number of Nearest Neighbors, Varying Regularization')
ax[0].set_title('Fixed Number of Factors, Varying Number of Nearest Neighbors')
# ax[0].set_title('Fixed Number of Factors, Varying Number of Trees')

# Create custom legend entries for unconstrained and constrained
handles, labels = ax[0].get_legend_handles_labels()
line_unconstrained = Line2D([0], [0], color='black', linestyle='-')
if PLOT_CONSTRAINED:
    line_constrained = Line2D([0], [0], color='black', linestyle='dotted')
    ax[0].legend(handles + [line_unconstrained, line_constrained],
                 labels + ['Unconstrained', 'Constrained'])
else:
    ax[0].legend(handles, labels)

# Second plot: fixed alpha, varying num_factors
# for alpha in alphas:
# for regularization in regularizations:
for K in nearest_neighbors:
# for nt in num_trees:
    recall = []
    catalog_coverage = []
    recall_constrained = []
    catalog_coverage_constrained = []
    num_factors_values = []
    for result in results:
        # if result[1] == alpha:
        # if result[1] == regularization and result[2]['average_recall'] > 0.0:
        if result[1] == K and result[2]['average_recall'] > 0.0:
            recall.append(result[2]['average_recall'] )
            catalog_coverage.append(result[2]['catalog_coverage'])
            if PLOT_CONSTRAINED:
                recall_constrained.append(result[2]['average_recall_constrained'])
                catalog_coverage_constrained.append(result[2]['catalog_coverage_constrained'])
            num_factors_values.append(result[0])
    # Plot unconstrained metrics
    # line, = ax[1].plot(recall, catalog_coverage, marker='o', label=f'Alpha: {alpha}')
    # line, = ax[1].plot(recall, catalog_coverage, marker='o', label=f'Regularization: {regularization}')
    line, = ax[1].plot(recall, catalog_coverage, marker='o', label=f'K: {K}')
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
# ax[1].set_title('Fixed Regularization, Varying Num Factors')
# ax[1].set_title('Fixed Regularization, Varying Number of Nearest Neighbors')
ax[1].set_title('Fixed Number of Nearest Neighbors, Varying Number of Factors')
# ax[1].set_title('Fixed Number of Trees, Varying Number of Factors')

# Create custom legend entries for unconstrained and constrained
handles, labels = ax[1].get_legend_handles_labels()
line_unconstrained = Line2D([0], [0], color='black', linestyle='-')
if PLOT_CONSTRAINED:
    line_constrained = Line2D([0], [0], color='black', linestyle='dotted')
    ax[1].legend(handles + [line_unconstrained, line_constrained],
                 labels + ['Unconstrained', 'Constrained'])
else:
    ax[1].legend(handles, labels)

plt.tight_layout()
plt.show()
