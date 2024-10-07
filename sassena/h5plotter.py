import h5py
import numpy as np
import matplotlib
from matplotlib.pyplot import *
from scipy import *
import pandas as pd
import sys

filename = sys.argv[1]

matplotlib.rc("font", size=28)

coh_f = h5py.File("./sassena/signal.h5", "r")
coh_fq = coh_f["fq"][:]
coh_fq0 = coh_f["fq0"][:]
# coh_fqt = coh_f['fqt'][:]
coh_q = coh_f["qvectors"][:]
coh_f.close()

qmin = 0.3
qdel = 0.1
qmax = 2.0
q_avg = np.arange(qmin, qmax + qdel, qdel).round(2)


x, y = coh_q, (coh_fq / 3000.0 - (5.803**2 / 3.0 + 3.739**2 / 3.0 * 2.0)) / 100.0

x = x[:, 0]
y = y[:, 0]

df = pd.DataFrame({"Q": x, "I_Q": y})
df.to_csv(f"./data/diffraction_patterns/{filename}", index=False)


plot(x, y, "r-")

ylim(-0.6, 0.6)


xlabel("Q/A^-1", fontsize="x-large")
ylabel("I(Q)", fontsize="x-large")
legend(title=r"$Q / \AA^{-1}$", fontsize="small")

# a = axes([0.38, 0.6, 0.52, 0.3])
# # errorbar(d.xa, d.ya, d.dya)
# plot(coh_q, (coh_fq / 3000.0 - (5.803**2 / 3.0 + 3.739**2 / 3.0 * 2.0)) / 100.0, "r-")
# xlim(10, 30)
# ylim(-0.03, 0.03)

show()
