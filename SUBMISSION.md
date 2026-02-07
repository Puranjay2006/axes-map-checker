# HACKATHON SOLUTION SUBMISSION DOCUMENT

---

## TEAM DETAILS

| Field | Details |
|-------|---------|
| **Team Name** | FutureFormers |
| **Problem Statement Chosen** | Problem 2 |
| **Team Members** | Puranjay (Developer, UI/UX) • Akshobhya (System Design, QA Logic) |
| **GitHub Repository Link** | https://github.com/Puranjay2006/axes-map-checker |
| **Demo Link** | https://axes-map-checker.streamlit.app |

---

## 1. PROBLEM UNDERSTANDING & SCOPE

### 1.1 Explain the problem you are solving in your own words.

We are solving the problem of **automated quality assurance for ride-hailing routing infrastructure**. When map vendors supply road network data (as WKT LINESTRING geometries), the data often contains connectivity and continuity errors — broken routes, isolated road segments, unrealistically short links, and near-miss endpoint gaps where roads should connect but don't.

- **Inputs:** A `.wkt` file containing LINESTRING geometries representing a road/street network.
- **Core Challenge:** Automatically detect route continuity and connectivity errors without any manual labeling or hardcoded thresholds, since every dataset has different coordinate systems, scales, and density.
- **Expected Output:** A comprehensive QA report identifying each problematic geometry with its error type, severity, confidence score, and coordinates — exportable as `error_report.json`.

### 1.2 What assumptions or simplifications did you make to stay within the hackathon scope?

- **Error scope is strictly limited** to route continuity and connectivity errors only (broken continuity, isolated segments, unrealistically short links, missing connections). We intentionally excluded geometric quality issues like self-intersections, duplicate geometries, or attribute validation.
- **Coordinate system agnostic:** We assume all coordinates in a single file share the same projected CRS. We do not perform CRS transformations.
- **2D only:** We work with 2D coordinates (X, Y). Z-values are ignored if present.
- **No manual labeling required:** All thresholds are derived from the dataset's own statistical distribution (percentile-based), making the system work on any dataset without tuning.
- **Single-file analysis:** Each analysis run operates on one `.wkt` file at a time. Cross-file or temporal comparisons are out of scope.

---

## 2. SOLUTION APPROACH & DESIGN

### 2.1 Describe your overall approach to solving the problem.

We built a **four-stage detection pipeline** that combines rule-based validation with unsupervised machine learning:

```
┌─────────────────────────────────────────────────────────┐
│                   INPUT: .wkt file                      │
│              (LINESTRING geometries)                    │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────┐
│  STAGE A: Feature Extraction (FeatureExtractor)          │
│  • Line length, vertex count, vertex density             │
│  • Start/end coordinates                                 │
│  • Endpoint degree (how many segments share each node)   │
│  • Connectivity score (distance to nearest other road)   │
└──────────────────────┬───────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────┐
│  STAGE B: Rule-Based Validation (RuleBasedValidator)     │
│  Rule 1: Isolated segments — both endpoints unconnected  │
│          + high connectivity score (> 75th percentile)   │
│  Rule 2: Short segments — length below 5th percentile    │
│          or < 2% of median length                        │
│  Rule 3: Endpoint gaps — dangling endpoints near but     │
│          not touching another road (adaptive threshold)  │
└──────────────────────┬───────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────┐
│  STAGE C: ML Anomaly Detection (AnomalyDetector)         │
│  • Isolation Forest (unsupervised) trained on the        │
│    uploaded dataset's feature matrix                     │
│  • Flags geometric outliers that rules may miss          │
│  • Adjustable contamination parameter (sensitivity)      │
└──────────────────────┬───────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────┐
│  STAGE D: Decision Logic (DecisionEngine)                │
│  • Merges rule-based + ML flags                          │
│  • Boosts confidence by 30% for double-flagged segments  │
│  • Assigns severity: HIGH (≥70%), MEDIUM (≥40%), LOW     │
│  • Deduplicates and sorts by confidence                  │
└──────────────────────┬───────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────┐
│  OUTPUT: error_report.json + Interactive Map + Dashboard  │
└──────────────────────────────────────────────────────────┘
```

### 2.2 Why did you choose this approach?

**Alternatives considered:**

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Pure rule-based (hardcoded thresholds) | Simple, fast | Breaks on different datasets/CRS | ❌ Rejected |
| Supervised ML (Random Forest / SVM) | High accuracy | Requires labeled training data we don't have | ❌ Rejected |
| Pure unsupervised ML | No labels needed | Hard to interpret, misses domain-specific errors | ❌ Rejected |
| **Hybrid: Data-driven rules + Isolation Forest** | **No labels, adaptive thresholds, interpretable** | Slightly more complex | ✅ **Chosen** |

**Key trade-offs:**
- We chose **percentile-based thresholds** over hardcoded values so the system adapts to any coordinate system and data density.
- We chose **Isolation Forest** over other anomaly detectors (LOF, DBSCAN) because it handles mixed feature scales well, works with small datasets, and has a single interpretable parameter (contamination).
- We combine both approaches because rules catch domain-specific errors (orphaned segments, gaps) while ML catches unexpected geometric anomalies that rules might miss.

---

## 3. TECHNICAL IMPLEMENTATION

### 3.1 Describe the technical implementation of your solution.

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Web Framework** | Streamlit | Interactive single-page application |
| **Geometry Engine** | Shapely 2.0+ | WKT parsing, spatial computations, distance calculations |
| **ML Engine** | scikit-learn (Isolation Forest) | Unsupervised anomaly detection |
| **Data Processing** | Pandas, NumPy | Feature matrix, statistical analysis |
| **Map Visualization** | Folium + streamlit-folium | Interactive map with layers, popups, markers |
| **Deployment** | Streamlit Cloud + GitHub | Auto-deploy on push to main branch |
| **UI Design** | Custom CSS (glassmorphism) | Professional, modern interface with DM Sans typography |

**Architecture (OOP):**
- `FeatureExtractor` — builds endpoint map, computes per-segment features
- `RuleBasedValidator` — three adaptive rules with percentile-based thresholds
- `AnomalyDetector` — StandardScaler + IsolationForest pipeline
- `DecisionEngine` — static combiner with confidence boosting
- `build_error_report()` — generates the JSON export structure

### 3.2 What were the main technical challenges and how did you overcome them?

1. **No hardcoded thresholds across different CRS:** Road networks come in different coordinate systems (meters, feet, degrees, local projected). We solved this by deriving ALL thresholds from the dataset itself using percentiles (5th percentile for short segments, 75th percentile for connectivity outliers, median-based gap thresholds).

2. **No labeled training data:** We cannot train a supervised classifier without labeled errors. We solved this by using Isolation Forest (unsupervised) — it learns the "normal" distribution of features from the uploaded data and flags statistical outliers automatically.

3. **Streamlit Cloud + Python 3.13 compatibility:** Shapely 2.0.2 had no wheels for Python 3.13. We fixed this by using flexible version requirements (`>=2.0.4`) to pick up compatible builds.

4. **Map coordinate normalization:** The WKT data uses local projected coordinates (not lat/lon), so Folium maps can't display them directly. We solved this by normalizing coordinates relative to the dataset centroid with a small scale factor.

5. **Double-counting issues:** A segment flagged by both rules and ML would appear twice. The `DecisionEngine` deduplicates by (geometry_id, error_type) and boosts confidence instead of duplicating.

---

## 4. RESULTS & EFFECTIVENESS

### 4.1 What does your solution successfully achieve?

| Requirement | Status | Details |
|-------------|--------|---------|
| Detect isolated/orphaned road segments | ✅ Fully met | Rule 1 flags segments where both endpoints are unconnected |
| Detect unrealistically short segments | ✅ Fully met | Rule 2 uses 5th-percentile adaptive threshold |
| Detect broken continuity (endpoint gaps) | ✅ Fully met | Rule 3 finds near-miss gaps with adaptive distance |
| ML-based anomaly detection | ✅ Fully met | Isolation Forest catches geometric outliers |
| No hardcoded thresholds | ✅ Fully met | All thresholds derived from data distribution |
| No manual labeling | ✅ Fully met | Fully unsupervised pipeline |
| Exportable error report | ✅ Fully met | `error_report.json` with geometry ID, error type, coordinates, confidence |
| Interactive visualization | ✅ Fully met | Color-coded map with severity layers, popups, layer controls |
| One-click analysis | ✅ Fully met | Upload file → automatic full pipeline execution |
| Works with any .wkt file | ✅ Fully met | API-agnostic, tested with multiple demo datasets |
| Self-intersection detection | ✅ Fully met | Flags non-simple geometries automatically |
| Duplicate segment detection | ✅ Fully met | Exact and near-duplicate detection via buffered containment |
| Sharp angle detection | ✅ Fully met | Flags vertices with angles below 10° |
| Auto-fix suggestions | ✅ Fully met | Snap dangling endpoints + download corrected WKT |
| Multi-file comparison | ✅ Fully met | Upload old version to see added/removed/unchanged segments |
| False positive feedback | ✅ Fully met | Mark issues as false positives, excluded from reports |
| Report generation | ✅ Fully met | Downloadable QA report (.txt) with full details |

### 4.2 How did you validate or test your solution?

We created **5 purpose-built test datasets** to validate each detection capability:

| Test File | Segments | Purpose | Expected Result | Actual Result |
|-----------|----------|---------|-----------------|---------------|
| `clean_network.wkt` | 12 | Well-connected grid | Few/no issues | ✅ Minimal flags |
| `errors_mixed.wkt` | 20 | All error types combined | Multiple error types | ✅ Detects short, isolated, and gap errors |
| `isolated_segments.wkt` | 14 | Spatially isolated segments far from network | Flag isolated segments | ✅ Correctly identifies orphaned segments |
| `short_segments.wkt` | 16 | Tiny segments (~0.3–0.5 units) in normal network | Flag short segments | ✅ Short segments flagged below percentile threshold |
| `endpoint_gaps.wkt` | 14 | Small coordinate gaps (0.3–0.8 units) between roads | Flag broken continuity | ✅ Gap detection with adaptive threshold |

Additionally, we tested with the **provided hackathon dataset** (`Problem 2 - streets_xgen.wkt`, 56 segments) — the system identifies connectivity issues and geometric anomalies consistent with manual inspection.

---

## 5. INNOVATION & PRACTICAL VALUE

### 5.1 What is innovative or unique about your solution?

1. **Hybrid detection pipeline:** Combining data-driven rules with unsupervised ML is uncommon in map QA tools. Rules catch domain-specific errors with interpretable explanations, while ML catches unexpected patterns humans might miss.

2. **Zero-configuration adaptive thresholds:** Every threshold is derived from the dataset's own statistical distribution. The same system works on a city block in meters and a highway network in degrees without any parameter tuning.

3. **Confidence scoring with cross-validation:** When both the rule engine and ML independently flag the same segment, confidence is boosted by 30% — providing a built-in cross-validation mechanism.

4. **Full transparency:** The Feature Matrix tab exposes every extracted feature and ML score, making the system auditable. The Pipeline tab explains exactly how each detection stage works.

5. **Production-grade UI:** Glassmorphism design, interactive Folium maps with severity-coded layers, downloadable JSON/CSV reports — built for professional use, not just a hackathon demo.

6. **Auto-fix with corrected file export:** Not just detection — the system suggests exact coordinate corrections for endpoint gaps and lets users download a fully corrected `.wkt` file in one click.

7. **Network version diffing:** Upload an old and new version of a road network to instantly see what changed — segments added, removed, or unchanged — enabling regression testing.

8. **In-session feedback loop:** Users can mark false positives, which are immediately excluded from all reports, building toward a human-in-the-loop QA workflow.

### 5.2 How can this solution be useful in a real-world or production scenario?

- **Map vendor QA:** Ride-hailing companies (Ola, Uber) receiving road network updates from vendors can run this tool as an automated quality gate before ingesting data into routing engines.
- **Municipal GIS teams:** City planners can validate road network datasets for connectivity completeness before using them for infrastructure planning.
- **Batch processing:** The core engine classes (`FeatureExtractor`, `RuleBasedValidator`, `AnomalyDetector`, `DecisionEngine`) are decoupled from the UI and can be imported as a Python library for batch pipeline integration.
- **CI/CD integration:** The `error_report.json` output is machine-readable and can feed into automated acceptance/rejection pipelines.

---

## 6. LIMITATIONS & FUTURE IMPROVEMENTS

### 6.1 What are the current limitations of your solution?

- **No CRS awareness:** Cannot transform between coordinate systems; assumes all coordinates in a single file share the same CRS.
- **Small dataset sensitivity:** Isolation Forest may be less reliable with very small datasets (< 10 segments) due to limited training samples.
- **2D only:** Z-coordinates are ignored.
- **Auto-fix is suggestion-only:** Corrections are proposed but the user must download the corrected file — there is no in-place editing of the original data source.

### 6.2 Previously listed as "future improvements" — now implemented in v5.0:

| Feature | Status | Details |
|---------|--------|---------|
| Multi-file comparison | ✅ Implemented | Upload old + new versions, see added/removed/unchanged segments |
| Auto-fix suggestions | ✅ Implemented | Snap dangling endpoints to nearest road, download corrected WKT |
| Additional geometry checks | ✅ Implemented | Self-intersections, duplicate/near-duplicate segments, sharp angle detection |
| Feedback loop | ✅ Implemented | Mark false positives in-session, excluded from reports |
| Report generation | ✅ Implemented | Downloadable QA report (.txt) with full issue details and fix suggestions |

### 6.3 If you had more time, what further improvements would you make?

1. **CRS detection and transformation:** Auto-detect EPSG codes and normalize coordinates using pyproj.
2. **Persistent feedback database:** Store false-positive labels across sessions for supervised ML refinement over time.
3. **REST API / batch mode:** Expose the core engine as a FastAPI endpoint for CI/CD pipeline integration.
4. **PDF with embedded maps:** Generate rich PDF reports with map screenshots (requires headless browser rendering).
5. **3D geometry support:** Handle Z-coordinates for elevated road networks and bridges.

---

## FINAL DECLARATION

We confirm that this submission is our own work and was developed during the hackathon period.

| Field | Value |
|-------|-------|
| **Team Representative Name** | Puranjay |
| **Confirmation** | Yes |
