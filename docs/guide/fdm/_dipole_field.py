import matplotlib.pyplot as plt
import numpy as np

width = 544
height = 336
px_in_inches = 0.01024

upsample_factor = 4
x = np.linspace(-width / 2, width / 2, (upsample_factor * width) + 1)
y = np.linspace(-height / 2, height / 2, (upsample_factor * height) + 1)
XY = np.meshgrid(x, y)

L_tip = 100
semi_min_axis = 50
r_tip = 25
d_Q0 = 1.31 * L_tip / (L_tip + 2 * r_tip)

Q_loc = np.array([0, r_tip * d_Q0 - L_tip])[:, np.newaxis, np.newaxis]

E_0 = (XY - Q_loc) / np.linalg.norm(XY - Q_loc, axis=0) ** 3
E_dipole = E_0 + E_0[:, ::-1]

ellipse = (XY[0] / semi_min_axis) ** 2 + (XY[1] / L_tip) ** 2 <= 0.95

fig, ax = plt.subplots(figsize=(width * px_in_inches, height * px_in_inches))

E_z = (E_dipole * ~ellipse)[1]
E_lim = np.abs(E_z).max()
ax.pcolormesh(
    XY[0],
    XY[1],
    (E_dipole * ~ellipse)[1],
    cmap=plt.cm.Spectral,
    vmin=-E_lim,
    vmax=E_lim,
)

ax.set_axis_off()
fig.subplots_adjust(top=1, right=1, bottom=0, left=0)
fig.savefig("_dipole_field.png", dpi=144)
plt.show()
