# Figure audit scope

Current local revision: detail_render_config.json, detail_integrity.json,
detail_pdf_render_index.json, detail_main_panel_review.json and the current
rendered_audit_index.json define the delivered local surfaces. The final PDF
review includes typography at the intended 183 mm matrix insertion width.

The original full_resolution_integrity, full_surface_checks, delivery_review
and visual_review records describe the preceding full-domain rendering. They
are preserved for history. initial_local_layout_review and detail_previews
record real-data preflights before the final paper-size typography adjustment.
They do not override the current render configuration or final PDF reviews.

The validation_report and supplementary numerical failures are experimental
records, preserved unchanged. A PASS in a figure integrity or layout check
does not change those experimental outcomes.

The current final visual decision is [detail_delivery_review.json](detail_delivery_review.json).
Initial zoom spacing failures and the passing repair with unchanged data are
recorded in [zoom_spacing_repair.json](zoom_spacing_repair.json). PDF audit
WARN records remain visible; their final-size visual resolution is documented.
