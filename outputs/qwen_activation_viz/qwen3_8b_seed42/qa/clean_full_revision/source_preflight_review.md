# Source preflight interpretation

The complete plotting source closure passes (19 PASS, 2 WARN, no FAIL).
`source_closure_preflight.json` identifies every included source by SHA-256.
Standalone module reports are also retained: export and font declarations live
in imported helpers, so rasterizer/audit-only modules are not complete figure
programs. Their missing-export/font findings are resolved by auditing the full
plotting closure and the actual final PDFs, not by weakening the detector.
The shared legacy style declares Times New Roman, but both selected matrix
entry points explicitly override it with DejaVu Sans before drawing.

Two remaining source warnings are expected. PNG is the requested raster preview;
PDF and SVG are the publication assets, so no TIFF is requested. The export
width is calculated in Python; final PDF text is separately checked at the
contracted 7.2-inch insertion width, with a 5 pt minimum (detail >=6.99 pt).
The 100/120-dpi PDF review images are QA previews only; all figure PNGs and
surface marks use 600 dpi. Review-preview DPI is not a publication setting.
