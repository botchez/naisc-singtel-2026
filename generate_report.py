"""Generate report.pdf for NAISC 2026 Singtel Adaptive Drift Intelligence Challenge.

Team: junyiding
Usage: python generate_report.py
"""

from fpdf import FPDF

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TEAM = "junyiding"
DATE = "April 2026"

# Colours (R, G, B)
DARK_BLUE = (26, 58, 92)
MED_BLUE = (41, 98, 150)
LIGHT_BLUE = (220, 235, 250)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY_BG = (245, 245, 245)
DARK_GRAY = (80, 80, 80)

# Metrics from pipeline run
BASELINE_TRAIN = 0.9252
BASELINE_TEST = 0.7552
MITIGATED_TRAIN = 0.9223
MITIGATED_TEST = 0.8686
HIGH_DRIFT = 3
MOD_DRIFT = 7
LOW_DRIFT = 30
TOTAL_FEATURES = 40

RUNTIME = {
    "Data Ingestion": "0.62s",
    "Drift Detection": "0.35s",
    "Drift Mitigation": "2.10s",
    "Feature Engineering": "1.48s",
    "Model Training": "14.87s",
    "Total": "19.43s",
}


# ---------------------------------------------------------------------------
# PDF helper class
# ---------------------------------------------------------------------------
class ReportPDF(FPDF):
    def __init__(self):
        super().__init__("P", "mm", "A4")
        self.set_auto_page_break(auto=True, margin=20)

    # -- reusable helpers --------------------------------------------------

    def _set_font(self, style="", size=9.5):
        self.set_font("Helvetica", style, size)

    def _set_mono(self, size=7.5):
        self.set_font("Courier", "", size)

    def section_header(self, num, title):
        """Print a section header with dark-blue background bar and smart page break."""
        if self.get_y() + 50 > self.h - self.b_margin:
            self.add_page()
        else:
            self.ln(8)
        self.set_fill_color(*DARK_BLUE)
        self.set_text_color(*WHITE)
        self._set_font("B", 12)
        self.cell(0, 8, f"  {num}. {title}", new_x="LMARGIN", new_y="NEXT", fill=True)
        self.set_text_color(*BLACK)
        self.ln(2)

    def sub_header(self, num, title):
        """Print a subsection header with accent line."""
        self.ln(2)
        self.set_text_color(*MED_BLUE)
        self._set_font("B", 10.5)
        self.cell(0, 6, f"{num}  {title}", new_x="LMARGIN", new_y="NEXT")
        y = self.get_y()
        self.set_draw_color(*MED_BLUE)
        self.set_line_width(0.3)
        self.line(self.l_margin, y, self.l_margin + 70, y)
        self.set_draw_color(0, 0, 0)
        self.set_line_width(0.2)
        self.set_text_color(*BLACK)
        self.ln(1)

    def body(self, text):
        self._set_font("", 9.5)
        self.multi_cell(0, 4.5, text)
        self.ln(1)

    def body_bold(self, text):
        self._set_font("B", 9.5)
        self.multi_cell(0, 4.5, text)
        self.ln(1)

    def small_body(self, text):
        self._set_font("", 8.5)
        self.multi_cell(0, 4, text)
        self.ln(1)

    def mono_block(self, text):
        """Monospace block with light background."""
        self.set_fill_color(*LIGHT_BLUE)
        self._set_mono(7)
        for line in text.split("\n"):
            self.cell(0, 3.5, f"  {line}", new_x="LMARGIN", new_y="NEXT", fill=True)
        self.set_fill_color(*WHITE)
        self.ln(2)

    def table(self, headers, rows, col_widths=None, font_size=8.5):
        """Draw a simple table with alternating row colours."""
        w = self.w - self.l_margin - self.r_margin
        if col_widths is None:
            col_widths = [w / len(headers)] * len(headers)

        # header
        self.set_fill_color(*DARK_BLUE)
        self.set_text_color(*WHITE)
        self._set_font("B", font_size)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 5.5, f" {h}", border=1, fill=True)
        self.ln()

        # rows
        self.set_text_color(*BLACK)
        self._set_font("", font_size)
        for r_idx, row in enumerate(rows):
            if r_idx % 2 == 0:
                self.set_fill_color(*GRAY_BG)
            else:
                self.set_fill_color(*WHITE)
            max_h = 5
            # Calculate needed height for multi-line cells
            cell_lines = []
            for i, cell in enumerate(row):
                lines = self.multi_cell(col_widths[i], 4.5, str(cell), dry_run=True, output="LINES")
                cell_lines.append(lines)
                max_h = max(max_h, len(lines) * 4.5)
            # Draw cells
            x_start = self.get_x()
            y_start = self.get_y()
            # Check if we need a page break
            if y_start + max_h > self.h - self.b_margin:
                self.add_page()
                y_start = self.get_y()
            for i, cell in enumerate(row):
                self.set_xy(x_start + sum(col_widths[:i]), y_start)
                self.multi_cell(col_widths[i], 4.5, str(cell), border=1, fill=True,
                                new_x="RIGHT", new_y="TOP")
            self.set_y(y_start + max_h)
        self.ln(2)

    def simple_table(self, headers, rows, col_widths=None, font_size=8.5, row_bg=None):
        """Table with automatic text wrapping in cells."""
        w = self.w - self.l_margin - self.r_margin
        if col_widths is None:
            col_widths = [w / len(headers)] * len(headers)

        # header
        self.set_fill_color(*DARK_BLUE)
        self.set_text_color(*WHITE)
        self._set_font("B", font_size)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 5.5, f" {h}", border=1, fill=True)
        self.ln()

        # rows
        self.set_text_color(*BLACK)
        self._set_font("", font_size)
        save_draw = self.draw_color
        self.set_draw_color(180, 180, 180)
        for r_idx, row in enumerate(rows):
            if row_bg and r_idx < len(row_bg):
                bg = row_bg[r_idx]
            elif r_idx % 2 == 0:
                bg = GRAY_BG
            else:
                bg = WHITE

            # Calculate max height needed across all cells in this row
            max_h = 5.5
            for i, cell in enumerate(row):
                cw = max(col_widths[i] - 2, 5)
                lines = self.multi_cell(cw, 4.5, str(cell), dry_run=True, output="LINES")
                max_h = max(max_h, len(lines) * 4.5 + 1)

            x_start = self.l_margin
            y_start = self.get_y()

            if y_start + max_h > self.h - self.b_margin:
                self.add_page()
                y_start = self.get_y()

            # Draw uniform-height cells
            for i, cell in enumerate(row):
                x = x_start + sum(col_widths[:i])
                self.set_fill_color(*bg)
                self.rect(x, y_start, col_widths[i], max_h, style="DF")
                self.set_xy(x + 1, y_start + 0.5)
                self.multi_cell(col_widths[i] - 2, 4.5, str(cell))

            self.set_y(y_start + max_h)
        self.draw_color = save_draw
        self.ln(2)

    def bullet(self, text):
        self._set_font("", 9.5)
        self.set_x(self.l_margin)
        w = self.w - self.l_margin - self.r_margin
        self.multi_cell(w, 4.5, f"  -  {text}")

    def footer(self):
        self.set_y(-15)
        self._set_font("I", 7.5)
        self.set_text_color(*DARK_GRAY)
        self.cell(0, 10, f"Team {TEAM} | NAISC 2026 Singtel Adaptive Drift Intelligence Challenge",
                  align="L")
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="R")


# ---------------------------------------------------------------------------
# Build the report
# ---------------------------------------------------------------------------
def build_report():
    pdf = ReportPDF()
    pdf.alias_nb_pages()
    pw = pdf.w - pdf.l_margin - pdf.r_margin  # printable width

    # ======================================================================
    # PAGE 1: Cover
    # ======================================================================
    pdf.add_page()
    # Cover accent: vertical bar on left
    pdf.set_fill_color(*DARK_BLUE)
    pdf.rect(0, 0, 4, pdf.h, style="F")
    pdf.ln(50)
    pdf.set_text_color(*DARK_BLUE)
    pdf.set_font("Helvetica", "B", 28)
    pdf.cell(0, 14, "Adaptive Drift Intelligence", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 14, "Challenge", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)
    pdf.set_font("Helvetica", "", 16)
    pdf.cell(0, 10, "Technical Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_draw_color(*DARK_BLUE)
    pdf.set_line_width(0.8)
    pdf.line(pdf.l_margin + 40, pdf.get_y(), pdf.w - pdf.r_margin - 40, pdf.get_y())
    pdf.ln(8)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, "NAISC Singtel 2026", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)
    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(*DARK_GRAY)
    pdf.cell(0, 7, f"Team: {TEAM}", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, DATE, align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(16)
    pdf.set_text_color(*BLACK)
    pdf.set_font("Helvetica", "I", 10)
    pdf.multi_cell(0, 5, (
        "A 5-stage automated pipeline for robust churn prediction through "
        "multi-metric compound drift detection, progressive mildest-to-strongest "
        "mitigation, importance-aware iterative drift correction, and self-training "
        "test distribution adaptation."
    ), align="C")

    # ======================================================================
    # PAGE 2: Executive Summary
    # ======================================================================
    pdf.add_page()

    # Table of contents
    pdf.set_text_color(*DARK_BLUE)
    pdf._set_font("B", 14)
    pdf.cell(0, 8, "Table of Contents", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(*BLACK)
    pdf.ln(2)
    toc = [
        ("1", "System Architecture", "3"),
        ("2", "Drift Detection", "4"),
        ("3", "Drift Mitigation", "5"),
        ("4", "Feature Engineering & Model Training", "6"),
        ("5", "Robustness & Scalability", "7"),
        ("6", "Results & Analysis", "8"),
        ("7", "Conclusion", "9"),
    ]
    pdf._set_font("", 10)
    for num, title, pg in toc:
        pdf.cell(8, 5.5, num + ".")
        title_w = pdf.get_string_width(title) + 2
        pdf.cell(title_w, 5.5, title)
        # Leader dots
        remaining = pw - 8 - title_w - 12
        dot_w = pdf.get_string_width(" . ")
        n_dots = max(1, int(remaining / dot_w))
        dot_str = (" . " * n_dots).rstrip()
        pdf.set_text_color(180, 180, 180)
        pdf._set_font("", 8)
        pdf.cell(remaining, 5.5, dot_str, align="L")
        pdf.set_text_color(*BLACK)
        pdf._set_font("", 10)
        pdf.cell(12, 5.5, pg, align="R")
        pdf.ln()
    pdf.ln(4)

    # Executive Summary
    pdf.set_text_color(*DARK_BLUE)
    pdf._set_font("B", 14)
    pdf.cell(0, 8, "Executive Summary", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(*BLACK)
    pdf.ln(2)

    pdf.body(
        "This report describes our solution for the NAISC 2026 Singtel Adaptive Drift Intelligence "
        "Challenge. The task is to predict customer churn under data distribution shift with fixed "
        "LightGBM hyperparameters, where all innovation must come from drift detection, mitigation, "
        "and feature engineering rather than model tuning."
    )
    pdf.body(
        "Our solution is a fully automated 5-stage pipeline that detects drift using 10 complementary "
        "statistical metrics with compound severity classification and Benjamini-Hochberg FDR correction, "
        "mitigates drift progressively from mildest to strongest transformations, engineers drift-aware "
        "features, and trains a final model enhanced by importance-aware iterative mitigation and "
        "self-training pseudo-labeling."
    )
    pdf.body_bold("Key Results:")
    key_text = (
        f"On the public dataset ({TOTAL_FEATURES} features analysed), the pipeline detected "
        f"{HIGH_DRIFT} high-drift, {MOD_DRIFT} moderate-drift, and {LOW_DRIFT} low-drift features. "
        f"After mitigation, the test AU-PRC improved from {BASELINE_TEST:.4f} (baseline, no mitigation) "
        f"to {MITIGATED_TEST:.4f} (mitigated), a {((MITIGATED_TEST - BASELINE_TEST) / BASELINE_TEST * 100):.1f}% "
        f"relative improvement. Total pipeline runtime: {RUNTIME['Total']}."
    )
    # Highlight box
    y_box = pdf.get_y()
    pdf._set_font("", 9.5)
    lines = pdf.multi_cell(pw - 10, 4.5, key_text, dry_run=True, output="LINES")
    box_h = len(lines) * 4.5 + 5
    pdf.set_fill_color(*LIGHT_BLUE)
    pdf.rect(pdf.l_margin, y_box, pw, box_h, style="F")
    pdf.set_xy(pdf.l_margin + 5, y_box + 2.5)
    pdf.multi_cell(pw - 10, 4.5, key_text)
    pdf.set_y(y_box + box_h + 2)
    pdf.body_bold("Key Innovations:")
    innovations = [
        "Multi-metric compound severity classification (not single-threshold)",
        "Progressive mildest-to-strongest mitigation preserving maximum information",
        "Importance-aware iterative mitigation using model feedback",
        "Self-training pseudo-labeling for test distribution adaptation",
        "Benjamini-Hochberg FDR correction across all feature tests",
        "Small-sample robustness scaling for threshold adjustment",
        "Conditional normalization (applied only when drift improves)",
    ]
    for inv in innovations:
        pdf.bullet(inv)

    # ======================================================================
    # Section 1: System Architecture
    # ======================================================================
    pdf.add_page()
    pdf.section_header("1", "System Architecture")

    pdf.sub_header("1.1", "Pipeline Overview")
    pdf.body(
        "The solution implements a five-stage sequential pipeline orchestrated by a central "
        "controller (main.py). Each stage is modular and fully automated with no hardcoded "
        "column names or manual intervention required."
    )

    pipeline_diagram = (
        "Stage 1: Data Ingestion & Profiling        (profiler.py)\n"
        "  Load CSVs -> Fuzzy-match columns -> Auto-classify types\n"
        "  -> Numeric coercion -> Memory downcast -> Decision sample (100K)\n"
        "              |\n"
        "Stage 2: Drift Detection                   (drift_detector.py)\n"
        "  Per-feature statistical tests (PSI, KS, Chi2, JSD, Wasserstein)\n"
        "  -> Severity classification -> BH FDR correction\n"
        "              |\n"
        "Stage 3: Drift Mitigation                  (mitigation.py)\n"
        "  Data quality normalization -> Re-check drift\n"
        "  -> Progressive per-feature mitigation (mildest to strongest)\n"
        "              |\n"
        "Stage 4: Feature Engineering               (feature_engineer.py)\n"
        "  Missing indicators -> Frequency encoding -> Ratio features\n"
        "  -> Aggregation features -> Deduplication -> Drift check\n"
        "              |\n"
        "Stage 5: Model Training & Prediction       (model.py)\n"
        "  Baseline model (no mitigation) vs Mitigated model:\n"
        "    Feature selection -> Interaction terms -> Iterative mitigation\n"
        "    -> Self-training -> Final predictions\n"
        "              |\n"
        "Output: prediction.csv + model.joblib"
    )
    pdf.mono_block(pipeline_diagram)

    pdf.sub_header("1.2", "Design Principles")
    pdf.simple_table(
        ["Principle", "Implementation"],
        [
            ["Full Automation",
             "No hardcoded column names; auto-detection of types, target format, abbreviation mappings"],
            ["Generalization",
             "Fuzzy column matching, dynamic runtime budgeting, feature-agnostic strategies"],
            ["Minimal Intervention",
             "Progressive strategies tried automatically; features dropped only as last resort"],
            ["Scalability",
             "Polars-based, decision sampling at 100K rows, joblib parallelization, memory downcasting"],
            ["Statistical Rigor",
             "FDR correction, small-sample adjustment, compound multi-metric severity classification"],
        ],
        col_widths=[35, pw - 35],
    )

    pdf.sub_header("1.3", "Technology Stack")
    pdf.simple_table(
        ["Library", "Version", "Purpose"],
        [
            ["Polars", ">=1.0.0", "Primary DataFrame library (Rust-backed, memory efficient)"],
            ["LightGBM", "4.6.0", "Classifier (fixed hyperparameters per competition rules)"],
            ["SciPy", ">=1.13.0", "Statistical tests (KS, Chi-squared, Wasserstein, entropy)"],
            ["scikit-learn", ">=1.5.0", "AU-PRC evaluation, StratifiedKFold for target encoding"],
            ["joblib", ">=1.4.0", "Model serialization and parallel drift detection"],
            ["NumPy", ">=1.26.0", "Numerical array operations"],
        ],
        col_widths=[25, 18, pw - 43],
    )

    # ======================================================================
    # Section 2: Drift Detection
    # ======================================================================
    pdf.add_page()
    pdf.section_header("2", "Drift Detection")
    pdf.body(
        "Drift detection is the foundation of our pipeline. Rather than relying on a single "
        "statistical test, we employ 10 complementary metrics that capture different aspects of "
        "distributional shift. Features are classified into three severity levels (high, moderate, low) "
        "using compound conditions that require corroborating evidence from multiple tests, "
        "reducing both false positives and false negatives."
    )

    pdf.sub_header("2.1", "Multi-Metric Detection Framework")
    pdf.body(
        "For each feature, we compute a battery of statistical tests. Numerical and categorical "
        "features use different test suites tailored to their data type:"
    )

    pdf.body_bold("Numerical Features (6 metrics):")
    pdf.simple_table(
        ["Metric", "What It Captures", "Key Detail"],
        [
            ["PSI", "Overall distribution shift",
             "Adaptive bins: min(10, max(2, n/20))"],
            ["KS Test", "Max CDF separation",
             "Effect size (D) and significance (p) both used"],
            ["Wasserstein", "Earth mover's distance",
             "Normalized by train std for scale-invariance"],
            ["IQR Ratio", "Spread / dispersion changes",
             "test_IQR / train_IQR; catches variance drift"],
            ["Tail Divergence", "Extreme percentile shifts",
             "Max shift in P1/P99, normalized by train std"],
            ["Missingness", "Null rate differences",
             "Can only escalate severity, never reduce"],
        ],
        col_widths=[24, 38, pw - 62],
    )

    pdf.body_bold("Categorical Features (4 metrics):")
    pdf.simple_table(
        ["Metric", "What It Captures", "Key Detail"],
        [
            ["JSD", "Symmetric divergence",
             "Bounded [0, ln2], Laplace-smoothed"],
            ["Chi2 + Cramer's V", "Association strength",
             "Sample-size normalized; rare cats grouped (>50)"],
            ["Unseen Categories", "Novel test values",
             "Flags distribution support changes"],
            ["Missingness", "Null rate differences",
             "Same escalation logic as numerical"],
        ],
        col_widths=[28, 35, pw - 63],
    )

    pdf.sub_header("2.2", "Compound Severity Classification")
    pdf.body(
        "A critical innovation is that severity is not determined by a single threshold on a single "
        "metric. Instead, compound conditions across multiple metrics must be satisfied. This "
        "prevents false positives from overpowered tests (e.g., KS p-values on large samples) "
        "and false negatives from metrics that miss specific drift patterns (e.g., PSI missing tail shifts)."
    )
    pdf.body(
        "All thresholds are scaled by a small-sample factor (ss_factor) to widen thresholds "
        "for small datasets, preventing noise-driven false alarms."
    )

    pdf.body_bold("Numerical Severity Rules:")
    pdf.simple_table(
        ["Severity", "Condition (any triggers classification)"],
        [
            ["HIGH", "PSI >= 0.25*ss  OR  (PSI >= 0.15*ss AND KS p < 0.001)"],
            ["HIGH", "(Wasserstein_norm >= 1.0 AND PSI >= 0.1*ss)  OR  (KS D >= 0.15 AND PSI >= 0.1*ss)"],
            ["HIGH", "(IQR ratio > 2.0 or < 0.5, AND PSI >= 0.1*ss)  OR  (Tail div > 2.0 AND PSI >= 0.05*ss)"],
            ["MODERATE", "PSI >= 0.1*ss  OR  (KS p < 0.05 AND KS D >= 0.05)"],
            ["MODERATE", "(Wasserstein_norm >= 0.3 AND PSI >= 0.05*ss)  OR  (Tail div > 1.0 AND PSI >= 0.03*ss)"],
            ["LOW", "Otherwise"],
        ],
        col_widths=[24, pw - 24],
        row_bg=[
            (255, 230, 230), (255, 230, 230), (255, 230, 230),
            (255, 243, 224), (255, 243, 224),
            (230, 255, 230),
        ],
    )

    pdf.body_bold("Categorical Severity Rules:")
    pdf.simple_table(
        ["Severity", "Condition (any triggers classification)"],
        [
            ["HIGH", "JSD >= 0.1*ss  OR  (Cramer's V >= 0.3 AND JSD >= 0.05*ss)"],
            ["MODERATE", "JSD >= 0.05*ss  OR  Chi2 p < 0.001  OR  Cramer's V >= 0.15  OR  Chi2 p < 0.05"],
            ["LOW", "Otherwise"],
        ],
        col_widths=[24, pw - 24],
        row_bg=[
            (255, 230, 230),
            (255, 243, 224),
            (230, 255, 230),
        ],
    )

    pdf.sub_header("2.3", "Benjamini-Hochberg FDR Correction")
    pdf.body(
        "When testing 40+ features simultaneously, multiple testing inflation is a real concern: "
        "at alpha=0.05, we would expect ~2 false positives by chance alone. We apply "
        "Benjamini-Hochberg correction across all KS and Chi-squared p-values to control the "
        "False Discovery Rate. After correction, severity is re-classified using the adjusted "
        "p-values, ensuring that only genuinely drifted features are flagged."
    )

    pdf.sub_header("2.4", "Small-Sample Robustness")
    pdf.body(
        "Histogram-based metrics (PSI, JSD) have high variance on small samples due to sparse bins. "
        "We compute an effective sample size using the harmonic mean of train and test sizes:"
    )
    pdf.mono_block(
        "n_eff = 2 * n_train * n_test / (n_train + n_test)\n"
        "ss_factor = (5000 / n_eff)^0.5    if n_eff < 5000\n"
        "ss_factor = 1.0                   if n_eff >= 5000"
    )
    pdf.body(
        "Multiplying all PSI/JSD thresholds by ss_factor > 1 makes severity harder to trigger "
        "for small datasets, reducing false alarms while maintaining sensitivity for large datasets."
    )

    pdf.sub_header("2.5", "Parallel Execution")
    pdf.body(
        "For datasets with more than 20 features, drift detection runs in parallel via "
        "joblib.Parallel (n_jobs=-1, prefer='threads'). Each feature is analyzed independently, "
        "making the computation embarrassingly parallel. Both train and test are subsampled to "
        "100K rows maximum for statistical tests, as power converges well before this point."
    )

    # ======================================================================
    # Section 3: Drift Mitigation
    # ======================================================================
    pdf.section_header("3", "Drift Mitigation")
    pdf.body(
        "Mitigation follows a three-phase pipeline: first normalize data quality issues that "
        "artificially inflate drift scores, then re-check drift on modified columns, then "
        "progressively apply transformations from mildest to strongest. This ordering preserves "
        "maximum information content while effectively reducing distributional divergence."
    )

    pdf.sub_header("3.1", "Phase 1: Data Quality Normalization")
    pdf.body(
        "Before applying drift-specific mitigation, we fix data inconsistencies that inflate drift "
        "scores artificially. Three normalization techniques are applied:"
    )
    pdf.body_bold("Abbreviation Detection & Standardization")
    pdf.body(
        "The pipeline auto-detects city abbreviation mappings (e.g., 'LA' -> 'Los Angeles', "
        "'SD' -> 'San Diego') by matching 2-3 character uppercase strings to multi-word names "
        "via word-initial letter alignment. Ties are broken by combined frequency across datasets. "
        "On the public dataset, this detected: SJ -> San Jose, SD -> San Diego, LA -> Los Angeles."
    )
    pdf.body_bold("Conditional Lowercase & Hyphen Removal")
    pdf.body(
        "Case normalization and hyphen removal are applied only if they reduce drift. For each "
        "categorical column, JSD is measured before and after normalization on a 100K subsample. "
        "The transformation is applied only if post-normalization JSD is lower. This prevents "
        "normalization from destroying meaningful case or hyphen information."
    )
    pdf.body_bold("Fuzzy Category Matching")
    pdf.body(
        "Unseen test categories are mapped to their closest training category using difflib's "
        "sequence matching (cutoff=0.6, fallback to 0.45). This handles typos and minor "
        "variations. Skipped for high-cardinality columns (>200 train categories or >50 unseen) "
        "for computational efficiency."
    )

    pdf.sub_header("3.2", "Phase 2: Targeted Drift Re-check")
    pdf.body(
        "After normalization, drift detection is re-run only for modified columns (not the entire "
        "feature set). This avoids wasting mitigation effort on features whose drift was caused "
        "by data quality issues rather than genuine distributional shift. Benjamini-Hochberg "
        "correction is reapplied after the update."
    )

    pdf.sub_header("3.3", "Phase 3: Progressive Per-Feature Mitigation")
    pdf.body(
        "For features that still exhibit high or moderate drift after normalization, strategies are "
        "applied progressively from mildest to strongest. The first strategy that reduces severity "
        "below the target threshold is accepted. If all strategies fail for a high-drift feature, "
        "the feature is dropped as a last resort."
    )

    pdf.body_bold("Numerical Strategies (tried in order):")
    pdf.simple_table(
        ["#", "Strategy", "Mechanism", "Preserves"],
        [
            ["1", "Clip Outliers", "Winsorize to [P1, P99] of train", "Scale, distribution shape"],
            ["2", "Log Transform", "log(1+x); requires x >= 0", "Monotonicity, relative order"],
            ["3", "Clip + Log", "Clip then shift + log", "Monotonicity after clipping"],
            ["4", "Robust Scaling", "(x - median) / IQR from train", "Relative distances"],
            ["5", "Quantile Binning", "Discretize into train quantile bins", "Ordinal ranking"],
            ["6", "Rank Transform", "Percentile rank [0, 1]", "Order only (nuclear option)"],
        ],
        col_widths=[7, 28, 55, pw - 90],
    )

    pdf.body_bold("Categorical Strategies (tried in order):")
    pdf.simple_table(
        ["#", "Strategy", "Mechanism"],
        [
            ["1", "Rare Category Grouping", "Merge categories < 1% frequency and unseen to '_other'"],
            ["2", "Frequency Encoding", "Replace categories with train frequency [0, 1]"],
            ["3", "Freq + Clip", "Frequency encode then clip outliers"],
            ["4", "Freq + Rank", "Frequency encode then rank transform"],
            ["5", "Freq + Quantile Bin", "Frequency encode then discretize into bins"],
        ],
        col_widths=[7, 38, pw - 45],
    )

    pdf.body(
        "For moderate-drift features, a greedy approach is used: all strategies are tested, and "
        "the one producing the best metric improvement (lowest PSI or JSD) is selected. "
        "Processing is capped at the top 50 moderate-drift features by magnitude for runtime efficiency."
    )

    pdf.sub_header("3.4", "Importance-Aware Iterative Mitigation")
    pdf.body(
        "A critical second pass occurs after the initial model is trained (Stage 5). Using feature "
        "importances from a trained LightGBM model, the pipeline identifies high-importance "
        "features that still have residual drift:"
    )
    pdf.body(
        "1. Train a model and extract gain-based feature importances.\n"
        "2. Compute the 50th percentile importance threshold.\n"
        "3. For each feature above the threshold with residual drift:\n"
        "   - Try quantile matching first (CDF-to-CDF alignment, preserves scale)\n"
        "   - Fall back to rank transform if quantile matching insufficient\n"
        "4. For low-importance features with high drift: drop them entirely.\n\n"
        "This focuses compute on features that actually impact predictions, rather than treating "
        "all features equally. On the public dataset, this step re-mitigated 24 high-importance "
        "features and dropped 16 low-importance high-drift features."
    )

    # ======================================================================
    # Section 4: Feature Engineering & Model Training
    # ======================================================================
    pdf.section_header("4", "Feature Engineering & Model Training")

    pdf.sub_header("4.1", "Feature Engineering")
    pdf.body(
        "All feature engineering transformations are learned from training data only (no data leakage). "
        "New features undergo a post-engineering drift check: any newly created feature with high "
        "drift severity is immediately dropped."
    )
    pdf.simple_table(
        ["Feature Type", "Logic", "Drift Guard"],
        [
            ["Missing Indicators", "Binary flag for cols with >1% nulls",
             "Low risk (binary feature)"],
            ["Frequency Encoding", "Train frequency for categoricals >10 unique",
             "Unseen test values map to 0.0"],
            ["Ratio Features", "Safe division of top-5 correlated pairs",
             "Epsilon masking; |r| > 0.3 threshold"],
            ["Aggregation Features", "Mean-by-group for top cat x num cols",
             "Unknown test groups use global mean"],
            ["Duplicate Removal", "Drop one of pair when |r| > 0.98",
             "Prefer dropping engineered over original"],
            ["Post-Eng. Check", "Drift test every new feature",
             "Drop if high severity detected"],
        ],
        col_widths=[28, 52, pw - 80],
    )

    pdf.sub_header("4.2", "Dual Categorical Encoding Strategy")
    pdf.body(
        "Categorical features are encoded using two strategies based on cardinality:"
    )
    pdf.simple_table(
        ["Cardinality", "Encoding", "Rationale"],
        [
            ["> 10 unique", "LightGBM native categorical (int labels)",
             "Exploits LightGBM's optimal categorical split algorithm"],
            ["<= 10 unique", "K-fold target encoding (smoothing m=10)",
             "Bayesian shrinkage prevents overfitting to rare categories"],
        ],
        col_widths=[22, 52, pw - 74],
    )
    pdf.body(
        "K-fold target encoding uses stratified cross-validation on training data to prevent target "
        "leakage. Test data is mapped using smoothed full-training-set means, with unseen "
        "categories falling back to the global mean."
    )

    pdf.sub_header("4.3", "Self-Training (Pseudo-Labeling)")
    pdf.body(
        "After training the mitigated model, we adapt it to the test distribution using high-confidence "
        "predictions as pseudo-labels:"
    )
    pdf.body(
        "1. Predict probabilities on the test set.\n"
        "2. Select high-confidence samples: P(churn) >= 0.95 or P(churn) <= 0.05.\n"
        "3. Assign pseudo-labels (P >= 0.5 -> 1, else -> 0).\n"
        "4. Augment training data with pseudo-labeled test samples.\n"
        "5. Retrain the model on augmented data."
    )
    pdf.body(
        "Key design choices: (a) Single iteration only, to prevent confirmation bias from "
        "compounding prediction errors. (b) Extreme confidence thresholds (0.95/0.05) ensure "
        f"high-quality pseudo-labels. On the public dataset, this added 3,377 pseudo-labeled "
        f"samples (1,618 positive, 1,759 negative)."
    )

    pdf.sub_header("4.4", "Model Configuration (Competition Compliance)")
    pdf.body(
        "The LightGBM classifier uses exactly the fixed hyperparameters specified by the "
        "competition rules:"
    )
    pdf.mono_block(
        "LGBMClassifier(\n"
        "    verbosity=-1,\n"
        "    objective='binary',\n"
        "    is_unbalance=True,\n"
        "    random_state=42,\n"
        "    importance_type='gain'\n"
        ")"
    )
    pdf.body(
        "All other parameters (n_estimators, max_depth, learning_rate, etc.) use LightGBM defaults. "
        "No hyperparameter tuning is performed."
    )

    # ======================================================================
    # Section 5: Robustness & Scalability
    # ======================================================================
    pdf.section_header("5", "Robustness & Scalability")

    pdf.sub_header("5.1", "Dynamic Runtime Budgeting")
    pdf.body(
        "The pipeline enforces a 300-second computation budget to ensure completion within "
        "time limits regardless of dataset size. The maximum number of training rows is "
        "dynamically calculated:"
    )
    pdf.mono_block(
        "max_rows = 300 / (8.0e-7 * n_cols)\n"
        "\n"
        "Example: 200 columns -> max 1.9M rows\n"
        "Example:  45 columns -> max 8.3M rows (no cap needed for most datasets)"
    )
    pdf.body(
        "The cost model (8.0e-7 seconds per row-column) was calibrated empirically on a "
        "2M x 200 benchmark dataset. If the training data exceeds the cap, it is subsampled "
        "using stratified temporal sampling."
    )

    pdf.sub_header("5.2", "Scalability Mechanisms")
    pdf.simple_table(
        ["Mechanism", "Purpose", "Threshold"],
        [
            ["Decision Sampling", "Statistical tests on reduced data", "len(train) > 100K"],
            ["Pipeline Row Cap", "Dynamic cap based on column count", "Budget = 300s"],
            ["Temporal Stratification", "Preserve month dist. in samples",
             "Temporal column detected"],
            ["Parallel Drift Detection", "joblib threads for features",
             "n_features > 20"],
            ["Moderate-Drift Cap", "Process highest-magnitude features",
             "n_moderate > 50"],
            ["Memory Downcasting", "Float64->32, Int64->Int8/16/32",
             "All datasets"],
            ["Correlation Matrix Cap", "Top cols by variance for O(m^2)",
             "n_cols > 50 or 200"],
            ["TE Subsample", "Subsample for K-fold computation",
             "len(train) > 200K"],
        ],
        col_widths=[35, 55, pw - 90],
    )

    pdf.sub_header("5.3", "Generalization Design")
    pdf.body(
        "The pipeline is designed to generalize to arbitrary datasets without modification:"
    )
    pdf.body(
        "- No hardcoded column names: all column types are auto-detected (ID, target, temporal, "
        "numerical, categorical) using heuristic analysis.\n"
        "- Target format auto-detection: supports Yes/No, 1/0, True/False, and minority-class fallback.\n"
        "- Fuzzy column name matching: handles naming inconsistencies between train and test "
        "(e.g., 'Customer_ID' vs 'CustomerID', case differences, underscores vs spaces).\n"
        "- Abbreviation detection: word-initial letter matching discovers abbreviation mappings.\n"
        "- Zero-variance and near-zero-variance columns automatically skipped.\n"
        "- All-null columns automatically excluded."
    )

    # ======================================================================
    # Section 6: Results & Analysis
    # ======================================================================
    pdf.section_header("6", "Results & Analysis")

    pdf.sub_header("6.1", "Drift Detection Summary")
    pdf.body(
        f"The pipeline analysed {TOTAL_FEATURES} features from the public dataset and classified "
        f"drift severity as follows:"
    )
    pdf.simple_table(
        ["Severity", "Count", "Percentage"],
        [
            ["High", str(HIGH_DRIFT), f"{HIGH_DRIFT/TOTAL_FEATURES*100:.1f}%"],
            ["Moderate", str(MOD_DRIFT), f"{MOD_DRIFT/TOTAL_FEATURES*100:.1f}%"],
            ["Low", str(LOW_DRIFT), f"{LOW_DRIFT/TOTAL_FEATURES*100:.1f}%"],
            ["Total", str(TOTAL_FEATURES), "100%"],
        ],
        col_widths=[30, 25, 25],
    )

    pdf.sub_header("6.2", "Drift Mitigation Summary")
    pdf.body("Features flagged for mitigation and the strategies applied:")
    pdf.simple_table(
        ["Feature", "Type", "Severity", "Strategy Applied"],
        [
            ["NumberofReferrals", "Numerical", "High", "Dropped (all strategies failed)"],
            ["AvgMonthlyLongDistanceCharges", "Numerical", "High", "Dropped (all strategies failed)"],
            ["Contract", "Categorical", "High",
             "Lowercased, hyphens removed; dropped (low importance)"],
            ["MonthlyCharge", "Numerical", "Moderate", "Quantile matching"],
            ["TotalLongDistanceCharges", "Numerical", "Moderate", "Quantile matching"],
            ["LocationCity", "Categorical", "Moderate",
             "Abbreviations standardized, lowercased"],
            ["ReferredaFriend", "Categorical", "Moderate",
             "Lowercased; dropped (low importance)"],
            ["PrioritySupport", "Categorical", "Moderate",
             "Lowercased; dropped (low importance)"],
            ["DigitalInvoicing", "Categorical", "Moderate",
             "Lowercased; dropped (low importance)"],
            ["TransactionMode", "Categorical", "Moderate", "Frequency encoding"],
        ],
        col_widths=[40, 22, 18, pw - 80],
    )

    pdf.body(
        "Additionally, during iterative importance-aware mitigation: 24 high-importance features "
        "were re-mitigated (primarily via rank transform and quantile matching), and 16 "
        "low-importance high-drift features were dropped."
    )

    pdf.sub_header("6.3", "Model Performance")
    pdf.simple_table(
        ["Metric", "Baseline (no mitigation)", "Mitigated", "Improvement"],
        [
            ["AU-PRC (train)", f"{BASELINE_TRAIN:.4f}", f"{MITIGATED_TRAIN:.4f}",
             f"{(MITIGATED_TRAIN - BASELINE_TRAIN) / BASELINE_TRAIN * 100:+.2f}%"],
            ["AU-PRC (test)", f"{BASELINE_TEST:.4f}", f"{MITIGATED_TEST:.4f}",
             f"+{(MITIGATED_TEST - BASELINE_TEST) / BASELINE_TEST * 100:.1f}%"],
        ],
        col_widths=[30, 38, 30, 25],
    )
    pdf.body(
        f"The mitigated model achieves a {((MITIGATED_TEST - BASELINE_TEST) / BASELINE_TEST * 100):.1f}% "
        f"relative improvement in test AU-PRC over the baseline. Notably, the training AU-PRC "
        f"is slightly lower for the mitigated model, indicating that mitigation reduces overfitting "
        f"to training-specific distributional patterns, leading to better generalization."
    )

    pdf.sub_header("6.4", "Runtime Breakdown")
    runtime_rows = [[stage, time] for stage, time in RUNTIME.items()]
    pdf.simple_table(
        ["Stage", "Time"],
        runtime_rows,
        col_widths=[40, 25],
    )
    pdf.body(
        f"Total pipeline runtime of {RUNTIME['Total']} on the public dataset (70K train, 14K test, "
        f"45 columns) is well within the 300-second computation budget. The pipeline scales "
        f"to datasets with millions of rows through decision sampling and dynamic row capping."
    )

    # ======================================================================
    # Section 7: Conclusion
    # ======================================================================
    pdf.section_header("7", "Conclusion")
    pdf.body(
        "We presented a fully automated, five-stage pipeline for detecting, quantifying, and "
        "mitigating data drift in a customer churn prediction setting under fixed model "
        "hyperparameters. Our key contributions are:"
    )
    contributions = [
        ("Multi-Metric Compound Severity Classification",
         "Using 10 complementary statistical metrics with compound conditions that require "
         "corroborating evidence, reducing both false positives and false negatives."),
        ("Progressive Mildest-to-Strongest Mitigation",
         "Applying transformations from lightest (clipping) to strongest (rank transform), "
         "preserving maximum information content while effectively reducing drift."),
        ("Importance-Aware Iterative Mitigation",
         "A model-feedback-driven second pass that focuses mitigation on features that actually "
         "impact predictions, and drops low-importance high-drift features."),
        ("Self-Training for Test Distribution Adaptation",
         "Single-iteration pseudo-labeling with extreme confidence thresholds adapts the model "
         "to test distribution without using actual labels."),
        ("Benjamini-Hochberg FDR Correction",
         "Controlling the False Discovery Rate across all simultaneous hypothesis tests to "
         "prevent spurious drift detection."),
        ("Small-Sample Robustness Scaling",
         "Threshold adjustment based on effective sample size prevents noise-driven false "
         "alarms on smaller datasets."),
        ("Conditional Normalization",
         "Data quality fixes are applied only when they demonstrably reduce drift, preventing "
         "well-intentioned normalization from introducing new distributional shifts."),
    ]
    for title, desc in contributions:
        pdf._set_font("B", 9.5)
        pdf.cell(0, 4.5, f"{title}:", new_x="LMARGIN", new_y="NEXT")
        pdf._set_font("", 9.5)
        pdf.multi_cell(0, 4.5, desc)
        pdf.ln(1)

    pdf.ln(2)
    pdf.body(
        f"On the public dataset, our pipeline improved test AU-PRC from {BASELINE_TEST:.4f} "
        f"to {MITIGATED_TEST:.4f} (+{((MITIGATED_TEST - BASELINE_TEST) / BASELINE_TEST * 100):.1f}%), "
        f"demonstrating that systematic drift detection and mitigation significantly improves "
        f"model generalization under distribution shift."
    )

    # Save
    pdf.output("report.pdf")
    print("report.pdf generated successfully.")


if __name__ == "__main__":
    build_report()
