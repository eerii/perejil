def inicio(c, s=[5.0, 3.5]):
    import matplotlib
    from cycler import cycler

    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
        'figure.figsize': s,
        'scatter.edgecolors': "black"
    })

    ccycler = 0
    if c <= 2:
        ccycler = (cycler(color=["royalblue", "indianred"]))
    if c == 3:
        ccycler = (cycler(color=["royalblue", "mediumseagreen", "tomato"]))
    if c >= 4:
        ccycler = (cycler(color=["royalblue", "mediumseagreen", "sandybrown", "tomato", "orchid"]))
    matplotlib.rcParams['axes.prop_cycle'] = ccycler

def guardar(n, xl, yl, leg=True, lab=True):
    import matplotlib.pyplot as plt
    if lab:
        plt.xlabel(xl)
        plt.ylabel(yl, rotation=0, labelpad=20)
    if leg: plt.legend()
    plt.savefig(n + ".pgf", bbox_inches = "tight")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

inicio(1)

d = pd.read_csv("pasaaltospython.csv", sep=',', decimal='.')

f = d["f"].to_numpy()
o = d["desfase"].to_numpy()

def arctanf(x,b,c):
    return b * np.arctan(c / x)
popt, pcov = curve_fit(arctanf, f, o)

print(popt)

xs = np.logspace(0., 4., 500)
ys = (arctanf(xs, *popt))

plt.xscale("log")
plt.xscale("log")
plt.plot(xs, ys, color="royalblue")
plt.scatter(f, o, linewidth=0.5, s=10)

guardar("pasaaltos", "$\\log(f(Hz))$", "$\\theta(rad)$", leg=False)

v2 = d["V2"].to_numpy()

plt.clf()
plt.xscale("log")
plt.yscale("log")
plt.scatter(f, v2, linewidth=0.5, s=10)

x1 = np.logspace(0., 4., 500)
y1 = np.ones(x1.shape[0])
plt.plot(x1, y1,'--', color="sandybrown")

x2 = np.logspace(0., 2.5, 500)
y2 = x2 * 0.005076
plt.plot(x2, y2,'--', color="sandybrown")

plt.scatter(197, 1, linewidth=0.5, s=20, color="tomato")

guardar("pasaaltos2", "$\\log(f(Hz))$", "$V_2/V_1$", leg=False)