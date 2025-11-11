<h1 align="center">AdaCHEMICA</h1>
<p align="center">
  <strong><em>Ada</em></strong>ptive 
  <strong><em>CHE</em></strong>mical-based 
  <strong><em>MI</em></strong>neral 
  <strong><em>C</em></strong>lassification
  <strong><em>A</em></strong>pproach
</p>

<p align="center">
  <em>Hybrid framework for open-set mineral classification from compositional data with adaptive class-specific thresholds</em>
</p>

<p align="center">
  <img src="logo.png" alt="AdaCHEMICA Logo" width="300"><br>
  <em>Ada Lovelace, the first programmer, holding a rock — the union of computational innovation and mineral science</em>
</p>

<p align="center">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-MIT-green.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.9%2B-blue.svg">
  <img alt="Status" src="https://img.shields.io/badge/status-publication--ready-6aa84f.svg">
</p>

<hr>

<h2>📚 Table of Contents</h2>
<ul>
  <li><a href="#overview">Overview</a></li>
  <li><a href="#features">Features</a></li>
  <li><a href="#installation">Installation</a></li>
  <li><a href="#quickstart">Quickstart</a></li>
  <li><a href="#workflows">Workflows</a>
    <ul>
      <li><a href="#workflow-1-spiral-plots">Workflow 1 — Spiral Plots</a></li>
      <li><a href="#workflow-2-preprocessing">Workflow 2 — Dataset Preprocessing</a></li>
      <li><a href="#workflow-3-all-mode">Workflow 3 — All-in-One</a></li>
      <li><a href="#workflow-4-hybrid-learning">Workflow 4 — Hybrid Learning</a></li>
    </ul>
  </li>
  <li><a href="#defaults">Threshold Defaults</a></li>
  <li><a href="#cli">Key CLI Flags</a></li>
  <li><a href="#outputs">Outputs</a></li>
  <li><a href="#reproducibility">Reproducibility</a></li>
  <li><a href="#citation">Citation & License</a></li>
</ul>

<hr>

<h2 id="overview">📄 Overview</h2>
<p>
  <strong>AdaCHEMICA</strong> is a supervised learning framework for mineral classification from compositional data, with explicit <em>open-set</em> handling. It integrates class-specific confidence thresholds with robust class-specific Mahalanobis limits (via MCD) to decide among <strong>confident</strong>, <strong>uncertain</strong>, and <strong>unknown</strong>.
</p>
<p>Included components:</p>
<ul>
  <li>Publication-ready spiral plots of class sample counts.</li>
  <li>Comprehensive preprocessing: iterative robust Mahalanobis outlier filtering, class balancing with KMeans undersampling and SMOTE, and automatic engineered geochemical features.</li>
  <li>Hybrid model: Random Forest + class-specific confidence thresholds + class-specific Mahalanobis limits.</li>
  <li>Open-set evaluation with LOCO, coverage-aware reports, and rich diagnostics.</li>
</ul>

<hr>

<h2 id="features">⚙️ Features</h2>
<ul>
  <li>
    <strong>Spiral plots with DOCX-driven highlighting</strong> — Optionally bold series/groups/polymorphs in labels when a DOCX with bolded names is provided.
  </li>
  <li>
    <strong>Iterative, gap-aware robust outlier filtering</strong> — Per-class Mahalanobis distances (MCD) over multiple rounds, with tail gap analysis and conservative safeguards.
  </li>
  <li>
    <strong>KMeans undersampling + SMOTE oversampling</strong> — Preserve diversity while reducing large classes and synthetically enlarging minority classes to a common target size where feasible.
  </li>
  <li>
    <strong>Automatic engineered features</strong> — e.g., highest and second-highest constituents, their sum, totals, total of major constituents &gt; 1 wt%, and number of major constituents.

  </li>
  <li>
    <strong>Hybrid open-set decision</strong> — Combine class-specific confidence and Mahalanobis distance, with a controlled relaxation for the “uncertain” band.
  </li>
  <li>
    <strong>LOCO open-set evaluation</strong> — Simulate unseen classes; quantify <em>unknown</em>/<em>uncertain</em>/<em>confident</em>, mappings, and confusions.
  </li>
  <li>
    <strong>Detailed plots</strong> — Confidence-threshold diagnostics, coverage–precision/recall/F1 curves, confidence distributions, class-wise mosaic, Mahalanobis profiles, feature importance, and LOCO stacked bars.<br>
    <em>Note:</em> some plots (confidence distributions & “most confused” heatmaps) require <code>seaborn</code>; if not installed, they are skipped automatically.
  </li>
  <li>
    <strong>Reproducibility</strong> — Fixed seeds, controlled threading, environment snapshots, structured logs and diagnostics.
  </li>
</ul>

<hr>

<h2 id="installation">🚀 Installation</h2>
<pre><code>git clone https://github.com/math-rs/AdaCHEMICA.git
cd AdaCHEMICA
pip install -r requirements.txt
</code></pre>
<p><em>Main libs:</em> numpy, pandas, scikit-learn, matplotlib, scipy, joblib, openpyxl, imbalanced-learn, threadpoolctl, Pillow. <code>seaborn</code> is optional (for certain plots).</p>

<hr>

<h2 id="quickstart">⚡ Quickstart</h2>

<h3 id="workflow-1-spiral-plots"># 1) Spiral plots</h3>
<pre><code>python preprocessing.py --mode spiral \
  --input data/mineral_data.xlsx --sheet data --label label \
  --docx docs/classes.docx \
  --outdir outputs/run_spiral
</code></pre>

<h3 id="workflow-2-preprocessing"># 2) Dataset preprocessing</h3>
<pre><code>python preprocessing.py --mode preprocess \
  --input data/mineral_data.xlsx --sheet data --label label \
  --outdir outputs/run_prep \
  --output outputs/run_prep/datasets/mineral_data_balanced.xlsx
</code></pre>

<h3 id="workflow-3-all-mode"># 3) All-in-one (spiral → preprocess)</h3>
<pre><code>python preprocessing.py --mode all \
  --input data/mineral_data.xlsx --sheet data --label label \
  --docx docs/classes.docx \
  --outdir outputs/run_all \
  --output outputs/run_all/datasets/mineral_data_balanced.xlsx
</code></pre>

<h3 id="workflow-4-hybrid-learning"># 4) Hybrid learning model (classification + evaluation)</h3>
<pre><code>python hybrid_learning.py \
  --input_xlsx outputs/run_prep/datasets/mineral_data_balanced.xlsx \
  --sheet data \
  --out_dir outputs/run_model \
  --plot
</code></pre>

<hr>

<h2 id="workflows">🧭 Workflows</h2>

<h3 id="workflow-1-spiral-plots">1) Spiral Plots (<code>preprocessing.py --mode spiral</code>)</h3>
<p>
  Generates a multi-panel spiral visualization of per-class sample counts (color scaled logarithmically).
  Optionally parses a DOCX file to bold broader categories (series, groups, and polymorphs) in labels.
  Exports high-resolution PNG, TIFF, and PDF figures.
</p>

<h3 id="workflow-2-preprocessing">2) Dataset Preprocessing (<code>preprocessing.py --mode preprocess</code>)</h3>
<ul>
  <li>Iterative, per-class robust Mahalanobis filtering (MCD) with adaptive tail-gap thresholds; diagnostics saved as PNG and CSV.</li>
  <li>KMeans undersampling for large classes and SMOTE oversampling for minorities to achieve balanced class sizes.</li>
  <li>The balanced dataset includes a <code>synthetic_data</code> column flagging samples generated by SMOTE.</li>
  <li>Automatic engineered features (e.g., highest and second-highest constituents, totals, and number of major constituents &gt;1 wt%).</li>
  <li>Outputs a balanced, traceable Excel dataset with full diagnostics and environment snapshots.</li>
</ul>

<h3 id="workflow-3-hybrid-learning">3) Hybrid Learning (<code>hybrid_learning.py</code>)</h3>
<ul>
  <li>Random Forest classifier with stratified out-of-fold predictions used to derive class-specific confidence thresholds, safeguarded by fallback limits.</li>
  <li>Hybrid open-set decision combining confidence and Mahalanobis distance (MCD-based), with controlled relaxation for the <strong>uncertain</strong> band.</li>
  <li>Leave-One-Class-Out (LOCO) evaluation producing coverage-aware reports, class-wise coverage tables, and publication-ready plots.</li>
  <li>Confusion matrices (<em>strict</em> and <em>lenient</em> modes) visualized as top-N heatmaps of the most confused classes.</li>
  <li>Feature-importance analysis quantifying each variable’s contribution to model performance.</li>
</ul>

<hr>

<h3 id="defaults">⚙️ Threshold Defaults</h3>
<p>The hybrid decision combines confidence and Mahalanobis distance with safety guardrails:</p>
<ul>
  <li><strong>Confident fallback threshold</strong>: <code>0.8</code> — used if the auto-derived confident threshold is missing or exceeds 0.8; lower bound of 0.3 is enforced.</li>
  <li><strong>Minimum uncertainty gap</strong>: <code>0.2</code> — ensures the <em>uncertain</em> threshold is at least 0.2 lower than <em>confident</em>; <em>uncertain</em> also has a hard floor of <strong>0.1</strong>.</li>
  <li><strong>Mahalanobis (per class)</strong>: default mode <code>inflated_quantile</code> with <code>q=0.95</code> and <code>inflate=1.05</code> (tunable via <code>--mahal_quantile</code> and <code>--mahal_inflate</code>).</li>
  <li><strong>Loose factor</strong>: <code>--loose_factor=2.0</code> — relaxes the Mahalanobis gate only for the <em>uncertain</em> band (helps retain near-boundary true samples for review).</li>
  <li><strong>Covariance shrink</strong>: <code>--cov_shrink_alpha=0.10</code> — diagonal shrinkage for numerical stability (0 disables).</li>
</ul>

<hr>

<h2 id="cli">🧰 Key CLI Flags</h2>

<h4>preprocessing.py</h4>
<ul>
  <li><code>--mode</code>: <code>spiral</code> | <code>preprocess</code> | <code>all</code></li>
  <li><code>--input</code>, <code>--sheet</code>, <code>--label</code></li>
  <li><code>--outdir</code> and <code>--output</code> (balanced XLSX)</li>
  <li><code>--docx</code> (optional, bold-driven spiral highlighting)</li>
  <li><code>--elements</code> (JSON list of chemistry columns, e.g. <code>'["SiO2","Al2O3","Fe2O3t"]'</code>) and <code>--metadata</code> (JSON list of metadata columns to preserve)</li>
</ul>

<h4>hybrid_learning.py</h4>
<ul>
  <li><code>--input_xlsx</code>, <code>--sheet</code>, <code>--out_dir</code>, <code>--plot</code></li>
  <li><code>--features_json</code> (JSON list of features; by default includes 15 element columns <em>plus</em> engineered features produced in preprocessing)</li>
  <li><code>--n_estimators</code> (RF), <code>--cv_folds</code></li>
  <li><code>--fallback_conf</code>, <code>--min_uncertainty_gap</code></li>
  <li><code>--threshold_mode</code>: <code>max</code> | <code>quantile</code> | <code>inflated_quantile</code></li>
  <li><code>--mahal_quantile</code> (default 0.95), <code>--mahal_inflate</code> (default 1.05), <code>--cov_shrink_alpha</code> (default 0.10)</li>
  <li><code>--loose_factor</code> (default 2.0)</li>
</ul>

<hr>

<p><em>Note:</em> Plots marked “requires seaborn” are generated only if <code>seaborn</code> is installed; otherwise the pipeline runs normally.</p>

<h2 id="outputs">🗂 Outputs</h2>
<p>All results are written under the directory provided via <code>--outdir</code> (for <em>preprocessing</em>) or <code>--out_dir</code> (for <em>hybrid learning</em>).</p>

<pre><code># Workflow 1 — Spiral Plots (preprocessing.py --mode spiral)
outdir/ (= outputs/run_spiral)
  plots/
    spiral/                         # Spiral plots of class size distributions (PNG/TIFF/PDF)
  diagnostics/
    run_params.json                 # CLI and resolved-parameters snapshot
  logs/
    run.log                         # Execution log (INFO)
  ENVIRONMENT.txt                   # Python/OS/package versions snapshot

# Workflow 2 — Dataset Preprocessing (preprocessing.py --mode preprocess)
outdir/ (= outputs/run_prep)
  datasets/
    mineral_data_balanced.xlsx      # Balanced dataset with engineered features (XLSX)
  plots/
    mahalanobis/                    # Mahalanobis diagnostics per class and round (PNG)
  diagnostics/
    run_params.json                 # CLI and parameters snapshot
    class_counts_initial.csv        # Sample counts before filtering
    class_counts_after_filtering.csv# Sample counts after Mahalanobis filtering
    mahalanobis_diagnostic_table_iterative.csv  # Iterative Mahalanobis diagnostics table
    mahalanobis_summary_lightweight.csv         # Summary of retained/excluded samples
    class_counts_balanced.csv       # Final class counts after KMeans/SMOTE balancing
  logs/
    run.log                         # Execution log (INFO)
  ENVIRONMENT.txt                   # Python/OS/package versions snapshot

# Workflow 3 — Hybrid Learning (hybrid_learning.py)
out_dir/ (= outputs/run_model)
  models/
    learning_model.pkl              # Final Random Forest model (joblib)
    hybrid_thresholds.json          # Class-specific confidence + Mahalanobis thresholds
  diagnostics/
    thresholds_per_class.csv        # Derived confidence thresholds per class
    classwise_coverage.csv          # Coverage table by class
    feature_importance_full.csv     # Feature-importance values for full model
    loco_openset_full_rates.csv     # LOCO open-set coverage/precision/recall per class
    loco_detailed_mappings.csv      # Detailed LOCO mappings (true vs predicted)
    open_set_metrics.txt            # Coverage-aware metrics and summary report
  plots/
    confidence_thresholds/          # Confidence-threshold diagnostics per class (PNG)
    coverage_precision_recall/      # Coverage–Precision/Recall/F1 curves (PNG/PDF)
    confidence_distributions/       # Confidence distributions (requires seaborn)
    most_confused/                  # Strict/lenient confusion heatmaps (requires seaborn)
    classwise_coverage/             # Per-class coverage mosaics (PNG/PDF)
    mahalanobis_profiles/           # Mahalanobis distance profiles per class (PNG)
    feature_importance/             # Feature-importance bar plots (PNG/PDF)
    loco/                           # LOCO stacked-bar summaries (PNG/PDF)
  logs/
    run.log                         # Execution log (INFO)
  ENVIRONMENT.txt                   # Python/OS/package versions snapshot
</code></pre>

<hr>

<h2 id="reproducibility">🧪 Reproducibility</h2>
<ul>
  <li><strong>Deterministic seeds</strong> and limited threading (e.g., <code>OMP_NUM_THREADS=1</code>).</li>
  <li><strong>Environment snapshots</strong> (<code>ENVIRONMENT.txt</code>) including package and OS versions.</li>
  <li>Headless, high-DPI figures suitable for CI environments.</li>
</ul>

<hr>

<h2 id="citation">📜 Citation & License</h2>
<p>
  Cite once the associated paper is published. Licensed under the <a href="LICENSE">MIT License</a>.
</p>

<p align="center">
  &copy; 2025 Matheus Rossi Santos — ORCID:
  <a href="https://orcid.org/0000-0002-1604-381X">0000-0002-1604-381X</a>
</p>
