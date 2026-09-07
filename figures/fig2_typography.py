"""Figure 2's explicit Times New Roman Bold typography; no font fallback."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from fontTools.ttLib import TTFont

from figure_style import configure_style, save_panel


def configure_fig2_style():
    configure_style()
    font_dir = Path(os.environ.get(
        'FIGURE2_FONT_DIR', Path.home() / '.local/share/figure-fonts/times-new-roman'))
    if font_dir.is_dir():
        for path in sorted(font_dir.iterdir()):
            if path.suffix.lower() == '.ttf':
                font_manager.fontManager.addfont(str(path))
    prop = font_manager.FontProperties(family='Times New Roman', weight='bold', style='normal')
    try:
        path = Path(font_manager.findfont(prop, fallback_to_default=False))
    except ValueError as exc:
        raise RuntimeError('Install Times New Roman Bold or set FIGURE2_FONT_DIR; Figure 2 forbids font substitution.') from exc
    with TTFont(path) as font:
        family = font['name'].getDebugName(1)
        weight = font['OS/2'].usWeightClass
        if family != 'Times New Roman' or weight != 700 or font['post'].italicAngle != 0:
            raise RuntimeError(f'Expected upright Times New Roman Bold, got {path}')
        provenance = {'family': family, 'weight': 'bold', 'weight_class': weight,
                      'postscript_name': font['name'].getDebugName(6),
                      'version': font['name'].getDebugName(5), 'file': path.name,
                      'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    mpl.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman'],
                         'font.weight': 'bold', 'axes.labelweight': 'bold',
                         'axes.titleweight': 'bold'})
    return provenance


def export_caption(here):
    """Typeset the unchanged caption separately from the bare scientific panels."""
    caption = (here / 'fig2_caption.txt').read_text().strip()
    width, margin, size, spacing = 6.4, .16, 8, 1.3
    prop = font_manager.FontProperties(family='Times New Roman', weight='bold', size=size)
    fig = plt.figure(figsize=(width, 2), dpi=600)
    renderer = fig.canvas.get_renderer()
    max_width = (width - 2 * margin) * fig.dpi
    lines = []
    for paragraph in caption.splitlines():
        line = ''
        for word in paragraph.split():
            trial = f'{line} {word}'.strip()
            if line and renderer.get_text_width_height_descent(trial, prop, False)[0] > max_width:
                lines.append(line)
                line = word
            else:
                line = trial
        lines.append(line)
    height = 2 * margin + len(lines) * size * spacing / 72
    fig.set_size_inches(width, height)
    ax = fig.add_axes([margin / width, margin / height,
                       1 - 2 * margin / width, 1 - 2 * margin / height])
    ax.set_axis_off()
    ax.text(0, 1, '\n'.join(lines), transform=ax.transAxes, va='top', ha='left',
            fontproperties=prop, linespacing=spacing)
    save_panel(fig, here / 'fig2_caption')
