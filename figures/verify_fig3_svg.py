#!/usr/bin/env python3
"""Verify actual Figure 3 SVG structure, plotted observations, and rendered halves."""
from pathlib import Path
import json
import re
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pymupdf

HERE = Path(__file__).resolve().parent
NS = {'s': 'http://www.w3.org/2000/svg'}


def render(path):
    with pymupdf.open(path) as doc:
        pix = doc[0].get_pixmap(matrix=pymupdf.Matrix(3, 3), alpha=False)
        return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]


def ink(image):
    return int(np.any(image < 230, axis=2).sum())


def main():
    law = pd.read_csv(HERE / 'fig3_range_law.csv')
    counts = law.source_family.value_counts().to_dict()
    report = {'panels': {}, 'compositions': {}}
    for stem in ['fig3c1', 'fig3c2', 'fig3c', 'fig3']:
        root = ET.parse(HERE / f'{stem}.svg').getroot()
        ids = [e.get('id') for e in root.iter() if e.get('id')]
        assert len(ids) == len(set(ids)), 'Duplicate SVG IDs'
        refs = []
        for element in root.iter():
            for key, value in element.attrib.items():
                refs.extend(re.findall(r'url\(#([^)]*)\)', value))
                if key.endswith('href') and value.startswith('#'):
                    refs.append(value[1:])
        assert not set(refs) - set(ids), 'Unresolved marker or clipping reference'
        assert not root.findall('.//s:svg', NS), 'Nested viewport may lose right-hand panels on import'
        if stem in ['fig3c1', 'fig3c2']:
            assert not root.findall('.//s:image', NS), 'Range-law marks must remain native vector'
            collections = [e for e in root.findall('.//s:g', NS)
                           if (e.get('id') or '').startswith('PathCollection_')]
            n = 1 if stem == 'fig3c1' else 2
            plotted = [len(g.findall('.//s:use', NS)) + len(g.findall('s:path', NS))
                       for g in collections[:n]]
            expected = ([counts['E1c activations']] if n == 1
                        else [counts['E7 V cache'], counts['E20 multi-slot']])
            assert plotted == expected, (stem, plotted, expected)
            formula = next(e for e in root.iter() if e.get('id') == 'sqrt_one_minus_f')
            assert ''.join(formula.itertext()).strip() == '1 − f'
            radical = next(e for e in formula.iter() if e.get('id') == 'radical_with_full_overbar')
            path = radical.find('s:path', NS)
            coords = [float(v) for v in re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', path.get('d'))]
            points = np.array(coords).reshape(-1, 2)
            assert len(points) == 5 and np.isclose(points[-1, 1], points[-2, 1])
            assert points[-1, 0] - points[-2, 0] > 10, 'Missing full radicand overbar'
            report['panels'][stem] = {'plotted_points': plotted, 'expected_points': expected,
                                      'embedded_images': 0, 'complete_vector_radical': True}
    for composite, stems in [('fig3c', ['fig3c1', 'fig3c2']),
                              ('fig3', ['fig3a', 'fig3b', 'fig3c1', 'fig3c2'])]:
        combined = render(HERE / f'{composite}.svg')
        rows = len(stems) // 2
        ratios = {}
        for i, stem in enumerate(stems):
            y0, y1 = [round(v * combined.shape[0] / rows) for v in [i // 2, i // 2 + 1]]
            x0, x1 = [round(v * combined.shape[1] / 2) for v in [i % 2, i % 2 + 1]]
            ratio = ink(combined[y0:y1, x0:x1]) / ink(render(HERE / f'{stem}.svg'))
            assert .90 < ratio < 1.10, (composite, stem, 'Missing or clipped content', ratio)
            ratios[stem] = ratio
        report['compositions'][composite] = {'svg_renderer': 'PyMuPDF', 'ink_ratio_vs_standalone': ratios}
    report['verdict'] = 'PASS'
    (HERE / 'qa/fig3.svg-integrity.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
