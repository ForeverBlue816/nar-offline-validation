"""Exact group-range ECDFs: all observations, without quantile thinning."""
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import ticker
from .plot import style, save, METHODS, LABELS, LAYERS, METHOD_COLORS, number


def run(source, destination):
    source, destination = Path(source), Path(destination)
    torch.set_num_threads(1); style()
    plt.rcParams['path.simplify'] = False
    records = []
    for mode in ('paired_local', 'end_to_end'):
        methods = METHODS if mode == 'paired_local' else METHODS[1:]
        for site in ('down_proj', 'q_proj'):
            arrays = {}; counts = {}
            for layer in LAYERS:
                for method in methods:
                    chunks = []
                    for sample in range(8):
                        y = torch.load(source/'activations'/mode/method/f's{sample:02d}_l{layer:02d}_{site}.pt', map_location='cpu', weights_only=True)
                        group = y.reshape(2048, -1, 128)
                        chunks.append((group.amax(-1)-group.amin(-1)).numpy().ravel())
                    values, multiplicity = np.unique(np.concatenate(chunks), return_counts=True)
                    key = f'{method}__{layer}'
                    arrays[key+'__values'] = values
                    arrays[key+'__counts'] = multiplicity.astype(np.uint32)
                    counts[key] = int(multiplicity.sum())
                    assert counts[key] == 8*2048*(96 if site == 'down_proj' else 32)
            target = destination/'exact_ecdf'; target.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target/f'{mode}__{site}.npz', **arrays)
            fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.3), sharey=True)
            fig.subplots_adjust(left=.07, right=.98, bottom=.23, top=.73, wspace=.25)
            for ax, layer in zip(axes, LAYERS):
                maximum = 0
                for method in methods:
                    key = f'{method}__{layer}'; values = arrays[key+'__values']
                    probability = arrays[key+'__counts'].astype(np.float64).cumsum()/counts[key]
                    ax.step(np.r_[values[0], values], np.r_[0., probability], where='post',
                            color=METHOD_COLORS[METHODS.index(method)], linewidth=1,
                            label=LABELS[method], rasterized=True)
                    maximum = max(maximum, float(values[-1]))
                ax.set_title(f'Block {layer+1}'); ax.set_xlabel('Group range'); ax.set_ylim(0, 1)
                ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
                ax.set_xscale('symlog', linthresh=.01); ax.set_xlim(0, maximum*1.05 if maximum else 1)
                powers = 10.**np.arange(-2, int(np.floor(np.log10(maximum)))+1) if maximum >= .01 else np.array([])
                chosen = powers[np.linspace(0, len(powers)-1, min(3, len(powers)), dtype=int)] if len(powers) else np.array([maximum/2, maximum])
                ax.set_xticks(np.unique(np.r_[0, chosen])); ax.xaxis.set_major_formatter(ticker.FuncFormatter(number))
                ax.xaxis.set_minor_locator(ticker.NullLocator()); ax.grid(axis='y', alpha=.5)
                ax.spines[['top', 'right']].set_visible(False); ax.tick_params(length=2)
            axes[0].set_ylabel('Cumulative probability')
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .935), ncol=len(methods), frameon=False, fontsize=6)
            label = 'Paired local' if mode == 'paired_local' else 'End to end'
            fig.text(.52, .99, f'{label} | {site} | Exact group-range ECDF', ha='center', va='top', fontsize=7)
            save(fig, destination/'figures'/mode/site/'group_range_ecdf')
            records.append({'mode': mode, 'site': site, 'observations_per_curve': counts,
                            'quantile_thinning': False, 'path_simplification': False,
                            'duplicate_values': 'exact multiplicities; no binning',
                            'curve_marks': 'rasterized at 600 dpi from all distinct values and exact cumulative counts'})
            print('EXACT ECDF', mode, site, flush=True)
    (destination/'exact_ecdf'/'manifest.json').write_text(json.dumps(records, indent=2)+'\n')
