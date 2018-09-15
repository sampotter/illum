import matplotlib.pyplot as plt
plt.style.use('ggplot')

import numpy as np

# sizes = np.array([49152, 98304, 196608, 393216, 786432, 1572864, 3145728])
sizes = np.array([49152, 98304, 196608, 393216, 786432])

vesta_stats = dict()
vesta_stats['horizon_time'] = np.array([11.157, 25.2314, 57.7525, 133.304, 305.158])
vesta_stats['F_time'] = np.array([4.95167, 20.77, 85.3855, 403.645, 1851.22])
vesta_stats['F_nnz'] = np.array([1773548, 6498013, 24993548, 97525169, 382757235])
vesta_stats['gs_avg_time'] = np.array([0.0315, 0.129, 0.593, 2.61, 10.53])
vesta_stats['gs_res'] = np.array([
    [0.19, 5e-4, 1.4e-6, 3.2e-9, 7.66e-12, 2.88e-12],
    [0.206, 6.9e-4, 2.05e-6, 5.4686e-9, 1.42e-11, 5.45e-12],
    [0.263, 9.2e-4, 2.81e-6, 7.75e-9, 2.24e-11, 1.09e-11],
    [0.359, 1.3e-3, 4.14e-6, 1.2e-8, 3.89e-11, 2.15e-11],
    [0.516, 1.9e-3, 6.44e-6, 1.97e-8, 7.02e-11, 4.23e-11]])
    
S = vesta_stats

n_gs_steps = 6
assert(S['gs_res'].shape[1] == n_gs_steps)
K = np.arange(1, n_gs_steps + 1)

fig, axes = plt.subplots(4, 1, figsize=(8, 12))

ax = axes[0]
ax.loglog(sizes, S['horizon_time'], '*-', markersize=4, label='Horizons (initialization)')
ax.loglog(sizes, S['F_time'], '*-', markersize=4, label='Construction of $K$ (initialization)')
ax.loglog(sizes, S['gs_avg_time'], '*-', markersize=4, label='G-S Iterations (one iterate)')
ax.loglog(sizes, 6*S['gs_avg_time'], '*-', markersize=4, label='G-S Iterations (total)')
ax.set_ylabel('CPU time for one step [s]')
ax.set_xlabel('# Faces')
ax.set_xticks(sizes)
ax.set_xticklabels(['%.2E' % size for size in sizes])
ax.minorticks_off()

ax.legend()

ax = axes[1]
for i, row in enumerate(S['gs_res']):
    ax.semilogy(K, row, '*-', markersize=4, label='N = %d' % sizes[i])
ax.set_ylabel('Residual [W/m$^2$]')
ax.set_xlabel('Iteration')
ax.minorticks_off()
ax.legend()

ax = axes[2]
ax.loglog(sizes, S['F_nnz'], '*-', markersize=4)
ax.set_ylabel('Number of nonzeros in $K$')
ax.set_xlabel('# Faces')
ax.set_xticks(sizes)
ax.set_xticklabels(['%.2E' % size for size in sizes])
ax.minorticks_off()

ax = axes[3]
ax.semilogx(sizes, S['F_nnz']/(sizes**2), '*-', markersize=4)
ax.set_ylabel('Sparsity of $K$ [%]')
ax.set_xlabel('# Faces')
ax.set_yticklabels(['{:,.3%}'.format(perc) for perc in ax.get_yticks()])
ax.set_xticks(sizes)
ax.set_xticklabels(['%.2E' % size for size in sizes])
ax.minorticks_off()

fig.subplots_adjust(0.13, 0.05, 0.99, 1.0)

fig.show()

fig.savefig('stats.pdf', dpi=300, transparent=True)
