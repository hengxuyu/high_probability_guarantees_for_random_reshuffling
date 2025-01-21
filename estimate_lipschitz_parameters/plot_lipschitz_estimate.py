import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import matplotlib.patches as patches

lipschitz_grad_estimate_trajectory = np.load("lipschitz_grad_estimate_trajectory.npy")
hessian_lip_estimate_trajectory = np.load("hessian_lip_estimate_trajectory.npy")

# change as you see fit
text_font_size = 28
number_font_size = 28


plt.gcf().set_size_inches(12, 9)
plt.plot(lipschitz_grad_estimate_trajectory, alpha=0.5)
plt.xlabel('Inner iteration', fontfamily='serif', fontsize=text_font_size)
plt.ylabel('Lipschitz gradient parameter', fontfamily='serif', fontsize=text_font_size)
plt.xticks(fontsize=number_font_size)
plt.yticks(fontsize=number_font_size)

plt.savefig(f"grad_lip_estimate_trajectory.pdf")

fig, ax = plt.subplots()
plt.gcf().set_size_inches(12, 9)
ax.plot(hessian_lip_estimate_trajectory)

ax_inset = inset_axes(ax, width="60%", height="30%", loc='center')
ax_inset.plot(range(5000, 5051), hessian_lip_estimate_trajectory[5000:5051], alpha=0.5)
ax_inset.set_xlim(5000, 5050)
ax_inset.set_ylim(min(hessian_lip_estimate_trajectory[5000:5051]), max(hessian_lip_estimate_trajectory[5000:5051]))

ax_inset.set_xticks([5000, 5050])
ax_inset.set_yticks([min(hessian_lip_estimate_trajectory[5000:5051]), max(hessian_lip_estimate_trajectory[5000:5051])])



ax_inset.tick_params(axis='both', which='major', labelsize=number_font_size - 10)

rect = patches.Rectangle((5000, min(hessian_lip_estimate_trajectory[5000:5050])), 50, max(hessian_lip_estimate_trajectory[5000:5050]) - min(hessian_lip_estimate_trajectory[5000:5050]), edgecolor='r', facecolor='none')

ax.add_patch(rect)
mark_inset(ax, ax_inset, loc1=3, loc2=4, fc="none", ec="0.5")

ax.set_xlabel('Inner iteration', fontfamily='serif', fontsize=text_font_size)
ax.set_ylabel('Lipschitz Hessian parameter', fontfamily='serif', fontsize=text_font_size)
ax.set_xticks(range(0, len(hessian_lip_estimate_trajectory), 1000))
ax.tick_params(axis='both', which='major', labelsize=number_font_size)

plt.savefig(f"hessian_lip_estimate_trajectory.pdf")

