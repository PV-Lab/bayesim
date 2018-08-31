m1 = bym.Model(load_state=True, state_file='after_run_1.h5')
m2 = bym.Model(load_state=True, state_file='after_run_2.h5')
m3 = bym.Model(load_state=True, state_file='after_run_3.h5')

m1_n_bins, m1_n_probs = m1.probs.project_1D(m1.params.fit_params[1])
m2_n_bins, m2_n_probs = m2.probs.project_1D(m2.params.fit_params[1])
m3_n_bins, m3_n_probs = m3.probs.project_1D(m3.params.fit_params[1])

m1_Bp_bins, m1_Bp_probs = m1.probs.project_1D(m1.params.fit_params[0])
m2_Bp_bins, m2_Bp_probs = m2.probs.project_1D(m2.params.fit_params[0])
m3_Bp_bins, m3_Bp_probs = m3.probs.project_1D(m3.params.fit_params[0])

m1_patches = m1.probs.project_2D(m1.params.fit_params[0], m1.params.fit_params[1])
m2_patches = m2.probs.project_2D(m2.params.fit_params[0], m2.params.fit_params[1])
m3_patches = m3.probs.project_2D(m3.params.fit_params[0], m3.params.fit_params[1])

import matplotlib.patches as mplp

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,10))

fs=20

for rownum in [0,1]:
    for colnum in [0,1]:
        for item in ([axes[rownum][colnum].xaxis.label, axes[rownum][colnum].yaxis.label] +axes[rownum][colnum].get_xticklabels() + axes[rownum][colnum].get_yticklabels()):
                    item.set_fontsize(fs)

axes[0,0].hist(m1.params.fit_params[0].vals, weights=m1_Bp_probs, bins=m1_Bp_bins)
axes[0,0].hist(m2.params.fit_params[0].vals, weights=m2_Bp_probs, bins=m2_Bp_bins)
axes[0,0].hist(m3.params.fit_params[0].vals, weights=m3_Bp_probs, bins=m3_Bp_bins)
axes[0,0].set_xscale('log')
axes[0,0].scatter(258, 0.05, 200, 'r', marker='*', zorder=20)

axes[1,1].hist(m1.params.fit_params[1].vals, weights=m1_n_probs, bins=m1_n_bins)
axes[1,1].hist(m2.params.fit_params[1].vals, weights=m2_n_probs, bins=m2_n_bins)
axes[1,1].hist(m3.params.fit_params[1].vals, weights=m3_n_probs, bins=m3_n_bins)
axes[1,1].scatter(1.36, 0.05, 200, 'r', marker='*', zorder=20)

i=0
for patches in [m1_patches, m2_patches, m3_patches]:
    for patch in patches:
        new_patch=mplp.Rectangle((patch._x, patch._y), patch._width, patch._height, facecolor=colors[i], alpha=patch._alpha, fill=True)
        axes[1,0].add_patch(new_patch)
    i = i+1
axes[1,0].set_xscale('log')
axes[1,0].set_xlim(axes[0,0].get_xlim())
axes[1,0].set_ylim(axes[1,1].get_xlim())
axes[1,0].set_xlabel("B'", fontsize=fs+4)
axes[1,0].set_ylabel('n', fontsize=fs+4)
axes[1,0].scatter(258, 1.36, 200, c="None", marker='o', linewidths=3, edgecolors='r', zorder=20)

fig.delaxes(axes[0,1])

axes[0,0].yaxis.set_label_position("right")
axes[0,0].set_ylim([0,1])
axes[1,1].set_ylim([0,1])
axes[1,1].yaxis.set_label_position("right")
axes[0][0].set_ylabel("P(B')", rotation=270, labelpad=24, fontsize=fs+4)
axes[1][1].set_ylabel("P(n)", rotation=270, labelpad=24, fontsize=fs+4)
axes[1,1].set_xlabel("n", fontsize=fs+4)

plt.savefig('../../paper/all_PMFs.png')
