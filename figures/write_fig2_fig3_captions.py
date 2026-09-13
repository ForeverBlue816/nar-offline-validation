#!/usr/bin/env python3
"""Scientific captions derived from the frozen plotting and audit tables."""
import json
from pathlib import Path
import textwrap
from deployment_figure_style import *


def caption_export(text,number):
    from matplotlib.font_manager import FontProperties
    prop=FontProperties(family='Times New Roman',weight='bold',size=8)
    fig=plt.figure(figsize=(5.5,2),dpi=150);renderer=fig.canvas.get_renderer()
    lines=[];line=''
    for word in text.split():
        trial=f'{line} {word}'.strip()
        if line and renderer.get_text_width_height_descent(trial,prop,False)[0]>(5.5-.32)*fig.dpi:
            lines.append(line);line=word
        else:line=trial
    lines.append(line)
    height=.32+len(lines)*8*1.35/72;fig.set_size_inches(5.5,height)
    ax=fig.add_axes([.16/5.5,.16/height,1-.32/5.5,1-.32/height]);ax.axis('off')
    ax.text(0,1,'\n'.join(lines),transform=ax.transAxes,ha='left',va='top',fontsize=8,linespacing=1.35)
    export(fig,HERE/f'fig{number}_caption',qa_directory=HERE/'qa/fig2_fig3')


def main():
    style();source=json.loads((HERE/'fig2_fig3_source_metadata.json').read_text())
    values=json.loads((HERE/'fig2_revised_metadata.json').read_text())['mean_reduction_percent']
    s=source['moe_diagnostics']
    fig2=f'''Null-space energy placement accompanies lower activation range and quantization error. All 28 Llama-3.2-3B down-input layers are shown for Hadamard, DuQuant and PrismQuant at maximal alignment rank. (a) Whole-activation energy fraction in the groupwise quantizer null space; the dashed reference is 1/128. (b) Paired mean group range. (c) Dynamic asymmetric group-128 INT4 activation NMSE. The arithmetic means of per-layer PrismQuant-versus-Hadamard percentage reductions are {values['b']:.2f}% and {values['c']:.2f}%, respectively. These are descriptive layer measurements, without inferred seed confidence intervals. DuQuant denotes the existing DuQuant-style diagnostic, not the complete official implementation. The shared legend applies to all panels.'''
    fig3=f'''Quantizer-aware geometry, energy concentration and range-law behavior. (a) All 8,064 non-BOS layer-27 down-input token projections in the frozen (v1,v2) basis, standardized separately, with unit free-direction projections shown at the same multiplier of 1.0. Their in-plane lengths are 0.015 for Hadamard and 1.000 for PrismQuant; direction cosines are distinct from token standard deviations. (b) Seven depth-spanning layers: 1, 5, 9, 13, 18, 22 and 27. Solid curves preserve the original BOS-excluded rank-256 spectra at layers 1/13/27. Dashed curves use existing all-token rank-64 spectra at layers 5/9/18/22 and stop at the measured endpoint; they are not continuations of the solid-curve protocol. The common all-token protocol for all seven layers is shown in the appendix. (c) All 2,520 activation, 280 V-cache and 112 multi-slot observations, with the unchanged pooled reference y=0.059802+0.866515x (R²=0.86); x=√(1−f) and y is range relative to paired Hadamard. The gray dashed line is identity. (d) All 5,342 available Qwen3-30B-A3B-Base expert observations, excluded from the reference fit. Marker area is fixed; teal denotes <2,048 routed calibration tokens and blue ≥2,048. Brown points connect medians in ten equal-count f bins; vertical intervals show Q1–Q3, not confidence intervals. The blue line is the unchanged reference from c; no MoE line is fitted for display. Within-MoE OLS association R² is {s['full_rows_256']['within_subset_ols_r_squared']:.2f} for the 4,955 experts reaching the original 256-row cap and {s['all']['within_subset_ols_r_squared']:.2f} for all experts; transferred-reference predictive R² is {s['full_rows_256']['transferred_reference_predictive_r_squared']:.2f}/{s['all']['transferred_reference_predictive_r_squared']:.2f}, respectively. The full-row median y/x is {s['full_rows_256']['median_y_over_sqrt']:.2f}. Removing the {s['f_gt_0_5_count']} experts with f>0.5 reduces x variance by {100*(1-s['f_le_0_5']['x_variance_ddof0']/s['all']['x_variance_ddof0']):.2f}% and within-subset R² to {s['f_le_0_5']['within_subset_ols_r_squared']:.2f}, illustrating range sensitivity. Cold-expert reference residuals have {s['cold_over_hot_reference_residual_sd']:.2f}× the hot-expert standard deviation. Their covariance uses the original layer-pooled shrinkage prior; this association does not isolate its causal effect. Expert evaluation rows come from their calibration capture, so cross-model holdout applies to the dense reference fit, not to expert calibration.'''
    appendix='''All-token energy context. Cumulative uncentered energy at layers 1, 5, 9, 13, 18, 22 and 27 of Llama-3.2-3B down inputs, from the same existing rank-64 randomized-eigenspectrum table. BOS is included for every curve. Layer 1 is consequently dominated by BOS energy; this panel is not substituted for the BOS-excluded original curve in Figure 3b. All measured ranks 1–64 are retained without extrapolation or smoothing.'''
    texts={2:fig2,3:fig3}
    for number,text in texts.items():
        (HERE/f'fig{number}_caption.txt').write_text(text+'\n');caption_export(text,number)
    (HERE/'captions_fig2_fig3.txt').write_text('Figure 2\n'+fig2+'\n\nFigure 3\n'+fig3+'\n\nAppendix energy context\n'+appendix+'\n')
    def tex(s):
        for old,new in [('×',r'\ensuremath{\times}'),('√(1−f)',r'\ensuremath{\sqrt{1-f}}'),('R²',r'\ensuremath{R^2}'),('≥',r'\ensuremath{\geq}'),('−','-'),('–','--'),('%',r'\%'),('<',r'\ensuremath{<}')]:s=s.replace(old,new)
        return s
    (HERE/'captions_fig2_fig3.tex').write_text('\n'.join(r'\expandafter\def\csname fig'+str(n)+r'RevisedCaption\endcsname{'+tex(t)+'}' for n,t in texts.items())+'\n')
    (HERE/'include_fig2_fig3.tex').write_text(r'''% Main figures are designed at 5.5 inches, matching the deployment figures.
\input{figures/captions_fig2_fig3.tex}
\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figures/fig2_revised.pdf}
  \caption{\csname fig2RevisedCaption\endcsname}
  \label{fig:nullspace-range-error}
\end{figure}
\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figures/fig3_revised.pdf}
  \caption{\csname fig3RevisedCaption\endcsname}
  \label{fig:geometry-reference-experts}
\end{figure}
''')

if __name__=='__main__':main()
