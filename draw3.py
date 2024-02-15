import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns

plt.rcParams["font.family"] = "Times New Roman"
sns.set_theme()
# color1='#8da0cb'
# color2='#fc8d62'
# color3='#66c2a5'
# color4='brown'

color1 = '#66c2a5'
color2 = '#fc8d62'
color3 = '#8da0cb'
color4 = '#e78ac3'
_, ax1 = plt.subplots()
device = ["L40", "A100"]
speedup_32 = [2.54, 2.69]
speedup_64 = [2.75, 2.69]
speedup_128 = [2.61, 2.83]
x = np.arange(len(device)) + 0.2
width = 0.1
x_32 = x - width
x_64 = x
x_128 = x + width
# ax1.plot(budget, tree4specinfer, label="SpecInfer (4$\\times$ Tree)", color=color2, linewidth=3)
# ax1.plot(budget[1:], tree8specinfer, label="SpecInfer (8$\\times$ Tree)", color=color3, linewidth=3)
# ax1.plot(budget[2:], tree16specinfer, label="SpecInfer (16$\\times$ Tree)", color=color4, linewidth=3)
plt.title("Speed Up v.s. Budget on Different Hardware", fontsize=15)
plt.bar(x_32,speedup_32,width=width,color=color3, label="Budget 32")
plt.bar(x_64,speedup_64,width=width,color=color1, label="Budget 64")
plt.bar(x_128,speedup_128,width=width,color=color2, label="Budget 128")
plt.legend(fontsize=12, loc='upper left')
plt.ylabel('Speed Up', fontsize=15)
plt.ylim(2.0, 3.0)
plt.xlim(0.0, 1.5)
plt.xticks(x, labels=device, fontsize=12)
plt.savefig("hardware.pdf", bbox_inches='tight')
