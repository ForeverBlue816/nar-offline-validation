# Palette accessibility report

Variant B is derived numerically from Variant A in CAM02-UCS: supporting hues use 60% of the original chroma, PrismQuant uses 70%, and J′ is spread to satisfy the hard grayscale gate. Values are converted back to sRGB, clipped to gamut, and rounded to 8-bit hex. The accessibility constraint necessarily requires a larger lightness separation than the aesthetic target alone.

| Method | Variant A | Variant B | J′ A → B | Chroma reduction |
|---|---:|---:|---:|---:|
| identity / raw | `#7F7F7F` | `#858484` | 55.8 → 57.9 | 29.7% |
| Hadamard | `#E69F00` | `#FFE39C` | 74.5 → 92.8 | 42.6% |
| DuQuant-style | `#009E73` | `#8DBEA6` | 58.8 → 74.9 | 39.7% |
| PrismQuant | `#0072B2` | `#001638` | 46.8 → 11.1 | 36.4% |

Distances are Euclidean CAM02-UCS ΔE after colorspacious simulation at severity 100. A row passes only when the minimum over all six method pairs is at least 15.

| Variant | Viewing condition | Minimum ΔE | Limiting pair | Hadamard–PrismQuant ΔE | Gate |
|---|---|---:|---|---:|---:|
| A | normal | 24.27 | identity / raw vs DuQuant-style | 64.68 | PASS |
| A | deuteranopia | 6.96 | identity / raw vs DuQuant-style | 64.44 | FAIL |
| A | protanopia | 13.43 | identity / raw vs DuQuant-style | 56.88 | FAIL |
| A | tritanopia | 11.73 | DuQuant-style vs PrismQuant | 53.45 | FAIL |
| A | grayscale | 4.66 | identity / raw vs DuQuant-style | 24.99 | FAIL |
| B | normal | 22.36 | identity / raw vs DuQuant-style | 88.98 | PASS |
| B | deuteranopia | 17.34 | identity / raw vs DuQuant-style | 90.01 | PASS |
| B | protanopia | 18.72 | Hadamard vs DuQuant-style | 86.13 | PASS |
| B | tritanopia | 22.98 | identity / raw vs DuQuant-style | 81.53 | PASS |
| B | grayscale | 16.63 | Hadamard vs DuQuant-style | 80.72 | PASS |

Variant A is retained for visual comparison but fails the hard all-pairs accessibility gate. Variant B passes under normal vision, deuteranopia, protanopia, tritanopia, and grayscale; Hadamard and PrismQuant are also well above threshold in every condition.

Full-figure simulations written: 24 PNGs under `figures/accessibility/` (three figures × two variants × four simulated conditions).
