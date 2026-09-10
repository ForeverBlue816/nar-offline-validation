# PrismQuant 精修交付

保留原图的 Transformer 主干、Construct / Represent / Deploy 三卡片、Attention / FFN 展开区和底部图例。画布收为 2:1，统一浅底、细边框、Times New Roman 文字及颜色语义。

- `PrismQuant_method_refined.pptx`：完整可编辑源文件。7 个模块组；文字、算子、缓存、圆点及连接线均为 PowerPoint 原生对象，取消组合后可逐个编辑。
- `PrismQuant_method_refined.pdf`：183 × 91.5 mm 的矢量版本，嵌入 Times New Roman。
- `PrismQuant_method_refined.png`：600 dpi 预览。
- `refine_prismquant.py`：生成及导出脚本。
- `qa/`：对齐、可编辑性、字体及实际字形碰撞检查记录。

科学内容依据指定的四个实现文件：`e14_w4a4kv4.py`、`activation_experiments.py`、`fold_signed_permutation.py`、`e17_v3.py`，对应 [nar-offline-validation](https://github.com/ForeverBlue816/nar-offline-validation/tree/9040075cee056f127300b638993ef676d51f1d37/nar)。没有引入实验数据或速度声明。散点与小柱是机制示意，不是实测样本。

已核对两条 residual bypass、Q/K scores 与加权 V 两步、gate-only SiLU、gate 的 Π / up 的 DΠ 折叠、R₂ 的层内跨 head 共享、K/V 不同分组轴，以及 G₄′ → H₁₂₈ → A₄ 的数学顺序。FFN 标注两阶段 Triton 实现。中间小柱使用相同纵轴尺度，平移前后的 range 保持不变。

## 复现

安装 Python 依赖并准备 Times New Roman 常规、粗体、斜体字体；设置 `PRISMQUANT_FONT_DIR` 指向字体目录。支持 Microsoft core fonts 的 Times.TTF / Timesbd.TTF / Timesi.TTF / Timesbi.TTF 文件名，大小写不限。

```bash
python -m pip install -r requirements.txt
python refine_prismquant.py --out .
```

安装 LibreOffice 后，可直接渲染 PPT 并导出 PDF / PNG：

```bash
python refine_prismquant.py --out . --render --soffice soffice
python qa/verify_delivery.py
python qa/audit_panel_alignment.py qa/alignment_measured.json --strict
python qa/audit_native_ppt.py PrismQuant_method_refined.pdf --out qa/native_collisions.json
```

导出由实际 PPT 渲染得到。PPT 无内嵌图片；PDF 无栅格图片。数学上下标采用独立定位的可编辑文字，避免渲染器二次缩字。按 183 mm 图宽检查：主要算子约 8–9 pt，辅助说明约 7–8 pt；最小上下标约 5.4 pt。插入论文时继续缩窄图幅会同比缩小文字。

碰撞检查沿用原几何检测阈值，并从 PDF 内嵌 TrueType 字形读取实际边界，避免把数学上下标和图例的空白字体区域错误合并成重叠段落。最终字形检查为 0 FAIL / 0 WARN，面板对齐检查通过。
