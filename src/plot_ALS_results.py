import matplotlib.pyplot as plt

# factors = [50, 100, 200, 500]
# regularizations = [0.001, 0.005, 0.01, 0.5]
# factors = [1, 2, 5, 10, 20, 50, 100, 200, 500]
# regularizations = [0.001, 0.005, 0.01, 0.5]
# num_iterations = [1, 2, 3, 5, 8, 10, 15]
factors = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]
# regularizations = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
alphas = [0.1, 0.5, 1.0, 2.0, 5.0]

# results = [
#     (50, 0.001, {'average_recall': 0.006570397111913353, 'catalog_coverage': 0.039775643375614046}),
#     (50, 0.005, {'average_recall': 0.006534296028880859, 'catalog_coverage': 0.03988562211305814}),
#     (50, 0.01, {'average_recall': 0.0063176895306859115, 'catalog_coverage': 0.04028887748368649}),
#     (50, 0.5, {'average_recall': 0.006389891696750897, 'catalog_coverage': 0.03959234547987389}),
#     (100, 0.001, {'average_recall': 0.0051263537906137075, 'catalog_coverage': 0.046960920888628195}),
#     (100, 0.005, {'average_recall': 0.005776173285198542, 'catalog_coverage': 0.04703424004692426}),
#     (100, 0.01, {'average_recall': 0.005523465703971106, 'catalog_coverage': 0.046887601730332136}),
#     (100, 0.5, {'average_recall': 0.005992779783393491, 'catalog_coverage': 0.04758413373414473}),
#     (200, 0.001, {'average_recall': 0.0056317689530685795, 'catalog_coverage': 0.05557592198841557}),
#     (200, 0.005, {'average_recall': 0.0056317689530685795, 'catalog_coverage': 0.05616247525478407}),
#     (200, 0.01, {'average_recall': 0.005234657039711179, 'catalog_coverage': 0.05550260283011951}),
#     (200, 0.5, {'average_recall': 0.005379061371841142, 'catalog_coverage': 0.05524598577608329}),
#     (500, 0.001, {'average_recall': 0.004981949458483744, 'catalog_coverage': 0.07643522252364543}),
#     (500, 0.005, {'average_recall': 0.0048014440433212895, 'catalog_coverage': 0.07610528631131315}),
#     (500, 0.01, {'average_recall': 0.005884476534296016, 'catalog_coverage': 0.07661852041938559}),
#     (500, 0.5, {'average_recall': 0.005487364620938616, 'catalog_coverage': 0.0763985629444974})
# ]
results = [
]
with open('results19.txt', 'r') as f:
    for line in f:
        # num_factors, num_iters, metrics = eval(line)
        num_factors, alpha , metrics = eval(line)
        # num_factors, regularization, metrics = eval(line)
        # results.append((num_factors, num_iters, metrics))
        # results.append((num_factors, regularization, metrics))
        results.append((num_factors, alpha, metrics))
print(results)


# Multiply recall values to spread them out
recall_multiplier = 1

# Plot results - recall on x-axis, catalog coverage on y-axis
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# First plot: fixed num_factors, varying regularization
for num_factors in factors:
    recall = []
    catalog_coverage = []
    # regularization_values = []
    # num_iters_values = []
    alpha_values = []
    for result in results:
        if result[0] == num_factors:
            recall.append(result[2]['average_recall'] * recall_multiplier)
            catalog_coverage.append(result[2]['catalog_coverage'])
            # regularization_values.append(result[1])
            # num_iters_values.append(result[1])
            alpha_values.append(result[1])
    ax[0].plot(recall, catalog_coverage, marker='o', label=f'Num factors: {num_factors}')
    # Annotate the points with regularization values
    # for i, reg_value in enumerate(regularization_values):
    # for i, num_iters_value in enumerate(num_iters_values):
    for i, alpha_value in enumerate(alpha_values):
        # ax[0].annotate(f'{reg_value}', (recall[i], catalog_coverage[i]))
        # ax[0].annotate(f'{num_iters_value}', (recall[i], catalog_coverage[i]))
        ax[0].annotate(f'{alpha_value}', (recall[i], catalog_coverage[i]))
ax[0].set_xlabel(f'Average Recall@N x {recall_multiplier}')
ax[0].set_ylabel('Catalog Coverage')
# ax[0].set_title('Fixed Num Factors, Varying Regularization')
ax[0].set_title('Fixed Num Factors, Varying Alpha')
# ax[0].set_title('Fixed Num Factors, Varying Num Iterations')
ax[0].legend()

# Second plot: fixed regularization, varying num_factors
# for regularization in regularizations:
# for num_iters in num_iterations:
for alpha in alphas:
    recall = []
    catalog_coverage = []
    num_factors_values = []
    for result in results:
        # if result[1] == regularization:
        # if result[1] == num_iters:
        if result[1] == alpha:
            recall.append(result[2]['average_recall'] * recall_multiplier)
            catalog_coverage.append(result[2]['catalog_coverage'])
            num_factors_values.append(result[0])
    # ax[1].plot(recall, catalog_coverage, marker='o', label=f'Regularization: {regularization}')
    # ax[1].plot(recall, catalog_coverage, marker='o', label=f'Num Iterations: {num_iters}')
    ax[1].plot(recall, catalog_coverage, marker='o', label=f'Alpha: {alpha}')
    # Annotate the points with num_factors values
    for i, num_factors_value in enumerate(num_factors_values):
        ax[1].annotate(f'{num_factors_value}', (recall[i], catalog_coverage[i]))
ax[1].set_xlabel(f'Average Recall@N x {recall_multiplier}')
ax[1].set_ylabel('Catalog Coverage')
# ax[1].set_title('Fixed Regularization, Varying Num Factors')
# ax[1].set_title('Fixed Num Iterations, Varying Num Factors')
ax[1].set_title('Fixed Alpha, Varying Num Factors')
ax[1].legend()

plt.tight_layout()
plt.show()
