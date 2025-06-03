import numpy as np
import matplotlib.pyplot as plt

# Roskam figures 4.9

CL0 = 0.17
CLalpha = 0.1
CLdeltaE = 0.25 / 30
CLiH = 3 * CLdeltaE
CM0 = 0.025
CMalpha = -0.015
CMdeltaE = -0.0062
CMiH = 3 * CMdeltaE
iH = 0
mom_ref_pt = 0.25
forward_cg = 0.15
aft_cg = 0.30
alpha_stall = 10
Cl_PlotMax = 1.2
alpha_PlotMax = 10

deltaE_values = np.array([-30, -20, -10, 0, 10])
color_specs_str = "-bo-gx-r+-c*-md-yv-k^"

legend_strings = [f"De={k_val} deg." for k_val in deltaE_values]

alpha_vector = np.arange(0, alpha_PlotMax + 1, 1)
dCMdCL = CMalpha / CLalpha
CM0bar = CM0 - dCMdCL * CL0
CMiHbar = CMiH - dCMdCL * CLiH
CMdeltaEbar = CMdeltaE - dCMdCL * CLdeltaE

fig = plt.figure(num=2, figsize=(12, 6))
fig.clf()

CL_plot_vector = np.arange(0, Cl_PlotMax + 0.01, 0.01)

ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

for i in range(len(deltaE_values)):
    dE = deltaE_values[i]
    current_style = color_specs_str[3 * i : 3 * i + 3]

    CL_alpha_data = CL0 + CLalpha * alpha_vector + CLiH * iH + CLdeltaE * dE
    ax1.plot(alpha_vector, CL_alpha_data, current_style, label=legend_strings[i])

    CM_CL_data = CM0bar + dCMdCL * CL_plot_vector + CMiHbar * iH + CMdeltaEbar * dE
    ax2.plot(CM_CL_data, CL_plot_vector, current_style, label=legend_strings[i])

ax1.legend(loc="lower right")
ax1.axis([min(alpha_vector), max(alpha_vector), 0, Cl_PlotMax])
ax1.set_ylabel("CL")
ax1.set_xlabel("alpha (deg)")
ax1.set_title("Lift Curve Slopes\nRef: Roskam Fig 4.9 (left), 4.10 (left)")
ax1.grid(True)
ax1.text(0.1, 0.95, f"iH= {iH} deg.", transform=ax1.transAxes)

ax2.plot([0, 0], [0, Cl_PlotMax], "k")

delta_cg_fwd = mom_ref_pt - forward_cg
Cm_fwd_cg = Cl_PlotMax * delta_cg_fwd
ax2.plot([0, Cm_fwd_cg], [0, Cl_PlotMax], "k")

delta_cg_aft = mom_ref_pt - aft_cg
Cm_aft_cg = Cl_PlotMax * delta_cg_aft
ax2.plot([0, Cm_aft_cg], [0, Cl_PlotMax], "k")

CL_at_stall = CL0 + CLalpha * alpha_stall + CLiH * iH + CLdeltaE * deltaE_values
CM_at_stall = CM0 + CMalpha * alpha_stall + CMiH * iH + CMdeltaE * deltaE_values
ax2.plot(CM_at_stall, CL_at_stall, "k--", linewidth=1.0)

ax2.legend(loc="lower right")
ax2.set_xlabel(f"CM about {mom_ref_pt*100:.0f}% c")  # Use .0f for integer percentage
ax2.set_ylabel("CL")
ax2.set_title(
    "A/C Trim Diagram\nRef: Roskam Fig 4.9 (right), 4.10 (right), 4.11b (right)"
)
ax2.grid(True)

ax2.text(0.18, 0.75, f"iH= {iH} deg.", transform=ax2.transAxes)
ax2.text(0.8, 0.95, f"fwd cg xbar={forward_cg}", transform=ax2.transAxes, ha="right")
ax2.text(0.1, 0.95, f"aft cg xbar={aft_cg}", transform=ax2.transAxes, ha="left")
ax2.text(
    0.5, 0.85, f"alpha stall={alpha_stall} deg.", transform=ax2.transAxes, ha="center"
)

ax2.axis([-0.2, 0.2, 0, Cl_PlotMax])
ax2.invert_xaxis()

plt.tight_layout()
plt.show()
