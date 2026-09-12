#!/usr/bin/env python3
"""Generate source-derived captions and a LaTeX inclusion snippet; CPU only."""
import json
from pathlib import Path
import pandas as pd
from make_deployment_efficiency import select
HERE=Path(__file__).resolve().parent


def main():
    d=pd.read_csv(HERE/'deployment_efficiency_data.csv')
    a=pd.read_csv(HERE/'fig4_revised_data.csv')
    overhead=[select(d,'decode_overhead',m,'nar',mode='cuda_graph_sequence')[0] for m in ['3b','8b']]
    recover=[float(a[a.kind.eq('recovery')&a.model.eq(m)&a.k_category.eq('8')].recovery_percent.iloc[0]) for m in ['llama32_3b','llama31_8b','qwen3_8b_base']]
    savings=[select(d,'memory_saving',m,'nar',mode='cuda_graph_sequence')[0] for m in ['3b','8b']]
    speed=select(d,'decode_speedup','8b','nar',mode='cuda_graph_sequence')[0]
    fig4=f'''Design choices and measured incremental deployment cost. (a) Activation representation budget versus WikiText-2 perplexity for Llama-3.2-3B; all 11 existing (g, m) configurations and the bf16 reference are retained. The budget is 4 + 16(m+1)/g activation bits/value, not E28 model-file size or whole-model effective bits. Point positions retain full precision; labels use two decimals. Filled circles/squares connect m=1 measurements; open circles/triangles retain additional directions. The horizontal bracket compares scale resolution at g=256 versus 128; the vertical bracket identifies the group-constant/null-space contribution at the shared 4.25-bit budget (Hadamard versus PrismQuant, g=128, m=1). PPL is averaged over three seeds and 64 chunks per seed; bf16 uses one reference. The 8B counterpart is in the appendix. (b) Activation-only PPL recovery is 100(PPL_Hadamard−PPL_k)/(PPL_Hadamard−PPL_bf16). All 15 measurements, including nonmonotone changes, remain visible. The k=8 operating point recovers {recover[0]:.2f}%, {recover[1]:.2f}%, and {recover[2]:.2f}% for Llama 3B, Llama 8B, and Qwen3-8B-Base. Rank labels are categories: qkv/down caps are respectively 24/64, 32/112, and 32/96, applied per site. Llama uses E11 (three seeds, 64 chunks); Qwen uses E18-v2 (one seed, 146 chunks), with bf16 weights/KV and group-128 activation quantization. The retained summary PPLs do not supply a common uncertainty estimate across these protocols; no interval or monotonic benefit is claimed. (c) On A40, k=8 adds {overhead[0]:.2f}% and {overhead[1]:.2f}% decode latency versus matched Hadamard. Centers are 100(t_PrismQuant/t_Hadamard−1) from pooled medians of 150 runs; small points and whiskers show all three paired-session median ratios and their min–max, not a confidence interval. Accuracy panels and the random-weight E28 kernel-swap panel use different protocols and do not form a checkpoint-level joint accuracy–speed Pareto point.'''
    efficiency=f'''Deployment efficiency of PrismQuant. The implementation strip shows compact R4, U = X A and Z = X H_block − U B, with offline factors A/B and a direct X-to-B branch; box widths are not timing proportions. Kernel B outputs FP16 to the shared per-token symmetric INT4 quantizer and CUTLASS INT4 GEMM. (a) Eager prefill input throughput, normalized to the matched FP16 model/batch in the original E28-v2 uniform0..255 cohort, with 2048 input tokens. The dashed line is FP16 = 1.00; source tables include absolute input tokens/s. (b) Matched private current-stream CUDA Graph decode latency for FP16, Hadamard, and PrismQuant. PrismQuant is {speed:.2f}× faster than FP16 on 8B; 3B remains slower than FP16. (c) Peak allocated inference bytes divided by 1e9, from the same Graph mode, with {savings[0]:.2f}%/{savings[1]:.2f}% PrismQuant savings relative to FP16 for 3B/8B. This is inference memory, not reserved/NVML memory or weight-file size; centers are the maximum of three independent-session peaks. (d) Candidate/reference wall-time ratios for prebound versus generic launch at T=1 (dispatch-sensitive) and the complete R4 path with TC-B versus shuffle-B at T=2048 (bulk execution). Each comparison has its own gray reference at 1.00; this is not FP16 or Hadamard. Centers are medians of three paired-session ratios, not pooled timing ratios. These are local implementation ablations, not B-only or complete-model speedups. All T=1/2048/32768 cases, including unfavorable TC-B cases, are retained in the appendix. Deployment timing centers use pooled medians over 150 formal runs (three sessions, 50 runs each after 10 warmups); prefill ratios use the corresponding pooled throughput medians. Whiskers show min–max of the three session medians, paired within each session for ratios; memory whiskers show session-peak ranges. They are not 95% confidence intervals, and tiny ranges are not enlarged. All performance models have random weights. Graph decode uses batch 1, prefix 2048, 128 causal steps with the first 8 discarded, and one preallocated 2176-token page. Independent page-64 correctness tests verify page crossing; the formal timing does not measure dynamic page allocation. FP16 is the performance reference, distinct from the accuracy panels' bf16 reference. The E28 per-token symmetric quantizer and KV interfaces are not the paper-native group-128 asymmetric full deployment. The frozen report discloses large-accumulator FP16-conversion stress-test failures in the existing backend; 6/6 private Graph correctness passes do not imply every numerical test passes.'''
    appendix='''Appendix evidence. The original E17 v3 rank-cost figure retains four PrismQuant points (k=8 and k=32 for both models) and both k=8 Hadamard references, on NVIDIA RTX PRO 6000 Blackwell Server Edition, T=2048. The source statistic is 100 times the separately benchmarked transform duration divided by decoder-layer duration plus transform duration; it is not measured end-to-end decode overhead. The original Triton do_bench protocol and selected configurations are preserved in E17V3_DONE.json and the original timing CSVs; no session uncertainty is fabricated. The matched eager/Graph figure uses only the private current-stream panel, with 150 samples per method/mode, and shows each method's corresponding latency. The all-shapes figure retains all 30 kernel comparisons: five reference families, two models, and three token counts. Centers are medians of three paired-session wall-time ratios and whiskers show their min–max; gray marks denote each family's own reference at 1.00. The B comparison measures full R4 paths with different B implementations. E17-native rows here are separate local A40 microbenchmarks of asymmetric packed-output transforms and must not be confused with the original RTX PRO 6000 rank-cost figure or the E28 FP16-output interface. The 8B metadata plot uses the same E20 definition as Figure 4a and retains all 11 observations plus bf16.'''
    common='''Frozen source: commit 6747960b96b0b03e626e98eaeeccf0d1abc34e79, results/e28_v2/20260912_a40_v2_full_int4. Every plotted value and paired-session value is traceable in fig4_revised_data.csv and deployment_efficiency_data.csv. Original accuracy CSVs and old exports remain archived. No training, calibration, GPU benchmarking, or kernel changes were performed to create these figures.'''
    captions={'fig4_revised':fig4,'fig_deployment_efficiency':efficiency,'appendix':appendix}
    (HERE/'captions.txt').write_text('\n\n'.join([f'{k}\n{v}' for k,v in captions.items()])+f'\n\n{common}\n')
    def tex(s):
        s=s.replace('%',r'\%').replace('_',r'\_').replace('−',r'\ensuremath{-}').replace('×',r'\ensuremath{\times}').replace('–','--')
        return s
    (HERE/'captions.tex').write_text('% Generated from frozen plotting data; no fixed figure number for deployment.\n'+
        '\n\n'.join(r'\expandafter\def\csname '+k+r'Caption\endcsname{'+tex(v)+'}' for k,v in captions.items())+'\n')
    (HERE/'include_deployment_figures.tex').write_text(r'''% Include in the manuscript preamble/body after loading graphicx.
% Physical width is 5.5 in; do not shrink to an unreadable single-column panel.
\input{figures/captions.tex}
\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figures/fig4_revised.pdf}
  \caption{\csname fig4_revisedCaption\endcsname}
  \label{fig:design-cost}
\end{figure}
\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figures/fig_deployment_efficiency.pdf}
  \caption{\csname fig_deployment_efficiencyCaption\endcsname}
  \label{fig:deployment-efficiency}
\end{figure}
% Numbering is supplied by the full manuscript, not forced to Figure 5.
% Appendix: figures/appendix/fig4_metadata_8b.pdf, fig_e17_rank_cost.pdf,
% fig_matched_eager_graph.pdf, and fig_kernel_all_shapes.pdf.
''')
    # A compact table retains the raw throughput/latency/memory and every comparison.
    central=d[d.session.eq(0)]
    lines=['# Frozen E28 plotting values','','Displayed values use two decimals; the CSV retains unrounded values.','',
           '| Metric | Model | Method | Baseline | Mode / scope | B / T | Value | Unit |','|---|---|---|---|---|---|---:|---|']
    for r in central.itertuples():
        lines.append(f'| {r.metric} | {r.model} | {r.method} | {"" if pd.isna(r.baseline) else r.baseline} | {r.mode} / {r.phase} | {r.batch} / {r.tokens} | {r.value:.2f} | {r.unit} |')
    (HERE/'deployment_efficiency_table.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
