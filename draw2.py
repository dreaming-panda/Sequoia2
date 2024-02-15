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
budget = [4, 8, 16, 32, 64, 128, 256, 512]
sequoia = [2.00, 2.31, 2.63, 2.95, 3.22, 3.49, 3.81, 3.97]
tree4sequoia = [1.74, 2.15, 2.50, 2.63, 2.65, 2.68, 2.70, 2.70]
tree8sequoia = [1.82, 2.24, 2.64, 2.76, 2.80, 2.80, 2.83]
tree16sequoia = [1.87, 2.34, 2.74, 2.89, 2.98, 2.98]
ax1.plot(budget, sequoia, label="Sequoia", color=color1, linewidth=3)
ax1.plot(budget, tree4sequoia, label="Sequoia (4$\\times$ Tree)", color=color2, linewidth=3)
ax1.plot(budget[1:], tree8sequoia, label="Sequoia (8$\\times$ Tree)", color=color3, linewidth=3)
ax1.plot(budget[2:], tree16sequoia, label="Sequoia (16$\\times$ Tree)", color=color4, linewidth=3)
plt.title("Sequoia Tree Structure Ablation", fontsize=15)
ax1.set_xscale('log', base=2)
ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
ax1.legend(fontsize=12, loc='lower right')
ax1.set_xlabel('Budget', fontsize=15)
ax1.set_ylabel('#Accepted Tokens', fontsize=15)
ax1.set_ylim(1.5, 4.0)
plt.xticks(fontsize=12)
plt.savefig("Structure.pdf", bbox_inches='tight')
def plot_acceptance_rates(df, draft, metrics, temp, top_p=1.0, target_only=True, save_fig=True, yticks=None, ylims=None, max_k=512):
  pretty_draft_names = {
      'EleutherAI/pythia-410m': 'Pythia-410m',
      'EleutherAI/pythia-2.8b': 'Pythia-2.8b',
      'llama-7b': 'Llama2-7b',
      'TheBloke/Llama-2-7B-Chat-GPTQ': 'Llama2-7B-Chat',
  }
  target_names = {
      'EleutherAI/pythia-410m': 'Pythia-12b',
      'EleutherAI/pythia-2.8b': 'Pythia-12b',
      'llama-7b': 'Llama2-70b',
      'TheBloke/Llama-2-7B-Chat-GPTQ': 'Llama2-70B-Chat',
  }
  pretty_metric_names = {
      'cover_acceptance_k': 'Top-$k$ Sampling',
      'spectr_acceptance_k': 'SpecTr',
      'specinfer_acceptance_replace_k': 'SpecInfer',
      'specinfer_acceptance_no_replace_k': 'Sequoia (ours)',
  }
  colors = {
      'specinfer_acceptance_no_replace_k': color1,
      'specinfer_acceptance_replace_k': color2,
      'spectr_acceptance_k': color3,
      'cover_acceptance_k': color4,
  }
  

  for metric in metrics:
    # alpha = alphas[f'{draft}_{metric}_{temp}'][1:]
    df_1 = df[(df['Metric name'] == metric) & (df['k'] <= max_k) & (df['Draft'] == draft) & (df['Top-p'] == top_p) & (df['Target-only top-p'] == target_only) & (df['Temp'] == temp)]
    pretty_metric = pretty_metric_names[metric]
    ax1.plot(df_1['k'], 1 - df_1['Metric value'], label=pretty_metric, color=colors[metric], linewidth=4)
    # ax1.plot(np.arange(1, len(sequoia) - 1), sequoia[2:], label='Sequoia (ours)', color=color3)
    pretty_draft = pretty_draft_names[draft]
    target = target_names[draft]
    plt.title(f'Draft: {pretty_draft}, Target: {target} (Temp: {temp})', fontsize=15)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend(fontsize=15, loc='lower left')
    ax1.set_xlabel('# of Speculated Tokens', fontsize=20)
    ax1.set_ylabel('Rejection Rate', fontsize=20)
    plt.xticks(fontsize=15)
    if yticks is not None:
      ytick_labels = [f'{t:.2f}' for t in yticks]
      # ax1.yticks(ytick_positions, ytick_labels, fontsize=15)
      ax1.set_yticks(yticks, ytick_labels)
      # ax1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
      ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
      if not ylims:
        ax1.set_ylim(yticks[0]-0.2, yticks[-1] + 0.8)
      else:
        ax1.set_ylim(ylims[0], ylims[1])

  if save_fig:
    plt.savefig(f'/var/cr06_data/zhuoming/robustness_{pretty_draft}_{target}_{temp}.pdf', bbox_inches='tight')
    