import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_hex

filename = r"palette.txt"
cmap = plt.cm.Spectral
palette = np.array(
    [cmap._segmentdata[component][:, 1] for component in ("red", "green", "blue")]
).T

with open(filename, "w") as file:
    file.writelines([to_hex(c) + "\n" for c in palette])
