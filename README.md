# E-Nose & Graph Neural Networks for Molecular Property Prediction
**Internship Progress Log**

---

## 1. Purpose of the Internship

The core goal of this internship is to build a machine learning pipeline capable of predicting how an electronic nose (e-nose) would respond to a given molecule — using only the molecule's identity (its name/structure) as input.

More precisely: **given the name of a molecule, predict the 8 continuous sensor activation values** that the Aryballe NeOse Pro device would produce when exposed to it.

This is fundamentally a **representation problem**: how do you encode a molecule in a way that a model can learn from? The internship explores three approaches — handcrafted molecular fingerprints, physico-chemical descriptors, and graph-based representations — with the final aim of building and evaluating a **Graph Neural Network (GNN)** model. Vinícius's work is focused primarily on the graph representation and GNN modelling side of the project.

---

## 2. Understanding the Electronic Nose

### What Is It?

The Aryballe **NeOse Pro** is a bio-based opto-electronic nose. Unlike traditional chemical sensors, it mimics the human olfactory system using a surface of **64 peptide sensors** — each a short protein chain with a different 3D shape — deposited on a gold film.

The detection principle is **Surface Plasmon Resonance imaging (SPRi)**:
- A polarised laser shines through a prism onto the gold-coated sensor surface.
- A camera measures the intensity of reflected light in each sensor zone.
- When volatile molecules pass through the device and **bind to the peptides**, the local refractive index changes, causing a measurable shift in reflectivity.
- Each of the 64 sensors responds differently depending on its peptide's affinity for the molecule — producing a unique multi-dimensional signal per compound.

### The Measurement Process

A measurement cycle has two phases:
1. **Baseline phase**: clean air is flowed through; each sensor's resting signal is recorded.
2. **Analyte phase**: the molecule's vapour is introduced; the signal rises as binding occurs.

The result is a **sensogram** — a time-series of sensor readings across the full cycle. The raw data file (`7Q27_sensograms.csv`) contains ~62,000 rows of timestamped readings across 8 selected sensors (IDs: 1, 105, 106, 24, 25, 34, 36, 55) for all runs and molecules in the dataset.

### From Sensogram to Fingerprint (Normalised Signature)

The sensogram is a rich but noisy signal. To extract a compact, stable representation of the molecule's "smell", the signal is processed into a **normalised signature**:
- The **delta** (Δ) between the analyte plateau and the baseline is computed per sensor.
- This delta is **normalised** (e.g., L2 or by baseline value) to remove run-to-run drift.
- The result is an **8-dimensional vector** — one value per sensor — called the **smell fingerprint**.

This vector is what the model is trained to predict. The normalised signatures file (`7Q27_normalized_signatures.csv`) contains 21 rows (one per measurement cycle), each with 8 sensor values for each molecule tested.

**Example — Normalised Sensor Values:**

| Molecule     | S1    | S105  | S106  | S24   | S25   | S34   | S36   | S55   |
|--------------|-------|-------|-------|-------|-------|-------|-------|-------|
| Ocimene      | 0.537 | 0.440 | 0.333 | 0.286 | 0.291 | 0.314 | 0.261 | 0.269 |
| Δ3-Carene    | 0.418 | 0.452 | 0.330 | 0.327 | 0.325 | 0.363 | 0.274 | 0.302 |
| Linalool     | 0.265 | 0.359 | 0.299 | 0.381 | 0.415 | 0.439 | 0.297 | 0.327 |
| α-Pinene     | 0.170 | 0.236 | 0.249 | 0.562 | 0.380 | 0.430 | 0.308 | 0.331 |
| (S)-Limonene | 0.363 | 0.247 | 0.292 | 0.477 | 0.291 | 0.398 | 0.349 | 0.344 |
| (R)-Limonene | 0.400 | 0.250 | 0.308 | 0.259 | 0.291 | 0.432 | 0.440 | 0.383 |

> A key observation: (S)-Limonene and (R)-Limonene are enantiomers (mirror images) — identical in composition but different in 3D shape. The e-nose distinguishes them, which has direct implications for which molecular features the model must be able to encode (chirality matters here).

---

## 3. Data Processing & Feature Engineering

### Dataset

The working dataset (device `7Q27`) covers a small set of volatile organic compounds (VOCs), primarily terpenes. After cleaning, 18 usable molecules remain with valid SMILES strings and sensor values.

### Step 1 — Name Cleaning & SMILES Retrieval

Molecule names from the lab often contain abbreviations, prefixes, or non-standard spellings. The pipeline:
1. **Cleans the molecule name** (e.g., "Delta 3 Carene" → "3-Carene").
2. **Queries the PubChem API** by name to retrieve the canonical `IsomericSMILES` string.
3. When PubChem fails to find a match, the name is corrected manually.

The SMILES string is the universal machine-readable representation of a molecule's structure, and all downstream features (fingerprints, descriptors, graph) are computed from it via RDKit.

### Step 2 — Feature Extraction (Three Complementary Families)

Three families of molecular features were computed and explored:

#### MACCS Keys (166 bits)
A fixed vocabulary of 166 binary structural fragments (e.g., "does this molecule contain a hydroxyl group?"). Generated with RDKit's `MACCSkeys`. Captures high-level structural blocks. Simple and interpretable, but lossy.

#### Morgan Fingerprints (512 bits)
Circular fingerprints that encode the local chemical environment around each atom out to a given radius. Each bit represents the presence of a particular substructure pattern. Generated with `rdFingerprintGenerator`. Sensitive to local topology, but two different structures can hash to the same bit (collision).

**Jaccard similarity analysis** using Morgan fingerprints revealed that (S)-Limonene and (R)-Limonene have a similarity of 0.96 — nearly identical fingerprints — yet the e-nose distinguishes them clearly. This demonstrates a key limitation of bit-vector fingerprints for this task.

#### Mordred Descriptors (48 clean descriptors from 1700+)
Mordred computes 1700+ physico-chemical descriptors from a SMILES string. These capture properties relevant to e-nose binding: volatility, lipophilicity (LogP), polarity, size, hydrogen bonding capacity, functional groups, etc.

A three-step filtering pipeline was applied:
1. Drop descriptors with >20% missing values (NaN).
2. Drop zero-variance and near-constant columns (>95% same value).
3. Drop one of any pair with Pearson correlation >0.95.

This reduced the feature space from 1700+ to **48 clean, informative descriptors**. These cover four property clusters: *Volatility & logP* (MW, SLogP, VPCvt), *Functional Groups* (nAcid, nBase, nThiol), *Steric/Size* (VSA_EState, MV, Diameter), and *Binding Polarity* (TPSA, nHBDon, nHBArc).

A Daylight-style path-based fingerprint (512 bits) and standalone LogP were also computed and added to the enriched dataset, resulting in a final **750-column CSV**.

### Step 3 — Exploratory Visualisation

A notebook (`plots_dataset.ipynb`) was written to visualise every run side-by-side:
- **Left panel**: the raw sensogram (time-series of all 8 sensors during the run).
- **Right panel**: the normalised fingerprint as a bar chart (the 8-value vector the model targets).

This helped build intuition for what the data looks like and confirmed that the normalised signatures cleanly summarise the sensogram response.

### Results from the Feature Exploration Phase

Three initial analyses were run on the small dataset:

**Result 1 — MACCS Active Bits:** Visualised which of the 166 MACCS structural fragments are actually "on" across the molecule set. Given the dominance of terpenes (mostly carbon/hydrogen with limited functional groups), the active bits are sparse and clustered.

**Result 2 — Jaccard Similarity (Morgan):** Pairwise structural similarity matrix revealed the (S)/(R)-Limonene near-identity problem. It also showed that Ocimene, Linalool, and other molecules are structurally quite distinct from the rest — the dataset is chemically diverse relative to its small size.

**Result 3 — Mordred (z-score PCA):** PCA on z-scored Mordred descriptors provided a 2D view of the chemical space. The 48 retained descriptors spread the molecules reasonably well, with the two Limonene enantiomers overlapping (as expected — they have identical 2D descriptors but differ in 3D chirality).

---

## 4. Graph Representation & GNN Modelling

*(This is the primary focus of Vinícius's work in the internship.)*

### Why Represent Molecules as Graphs?

Fingerprints and descriptors are fixed-length vectors computed by predetermined algorithms — they compress molecular structure through a lens the algorithm designer chose, not one the model can learn. Key limitations:

- **Morgan FP**: circular hashing can produce collisions; cannot distinguish chirality without explicit encoding.
- **MACCS keys**: fixed vocabulary misses novel substructures.
- **Mordred**: captures physico-chemical properties well, but is an aggregate — it discards topology.

A **graph representation** preserves the full molecular topology without any predefined compression. Each atom becomes a node, each bond becomes an edge, and the model itself learns which structural patterns are predictive for the task.

### Molecule → Graph (PyTorch Geometric)

Using PyTorch Geometric's `from_smiles` utility, each molecule is converted into a `Data` object with the following attributes:

| Attribute | Shape | Content |
|---|---|---|
| `data.x` | [atoms × 9] | Atom (node) feature matrix |
| `data.edge_index` | [2 × E] | Graph connectivity (source/target pairs) |
| `data.edge_attr` | [E × 5] | Bond (edge) feature matrix |
| `data.y` | [8] | Target: normalised e-nose sensor values |

**Node features (9 per atom):**

| # | Feature |
|---|---|
| 0 | Atomic number (C=6, O=8, …) |
| 1 | Chirality ★ |
| 2 | Degree (bond count) |
| 3 | Formal charge |
| 4 | Implicit H count |
| 5 | Radical electrons |
| 6 | Hybridisation (SP/SP2/SP3) |
| 7 | Is aromatic? |
| 8 | Is in ring? ★ |

★ Chirality and ring membership are particularly relevant here: they allow the graph to distinguish (S)- and (R)-Limonene — something MACCS and Morgan fingerprints largely fail to do.

**Edge features (5 per bond):**

| # | Feature |
|---|---|
| 0 | Bond type (single/double/triple/aromatic) |
| 1 | Stereo (Z/E, cis/trans) |
| 2 | Is aromatic? |
| 3 | Is conjugated? |
| 4 | Is in ring? |

### The GNN Model — `SensorGCN`

The GNN model (`SensorGCN`) was adapted from an existing multi-label classification architecture (`SmellGCN_Paper`). The adaptation converted it from a classification to a **multi-output regression** task, targeting continuous sensor values.

Key architectural choices (preserved from the original design):
- **GCNConv layers**: graph convolutional layers that aggregate features from neighbouring atoms.
- **Global add pooling with skip connections**: all convolutional outputs are concatenated before the MLP head, giving the model access to representations at multiple levels of abstraction.
- **BatchNorm + Dropout MLP head**: regularised fully-connected layers mapping the graph embedding to the 8 output values.

Key adaptations made for regression:
- **Output layer**: changed from sigmoid-activated classification head to a linear regression head (no activation).
- **Loss function**: `BCEWithLogitsLoss` (with `pos_weight`) replaced by **HuberLoss** — robust to outliers, appropriate for continuous targets.
- **Metrics**: classification metrics (AUROC, AUCPR, F1) replaced by regression metrics: R², MAE, RMSE, Pearson correlation, and a tolerance-based within-δ accuracy (a regression analogue of threshold sweeping). These are computed globally and per-sensor.
- **`find_optimal_tolerance`**: sweeps the validation set (range 0.01–0.5, calibrated for normalised targets) to find the best δ.
- **Plotting**: updated to show loss curves, a dual-axis R²/MAE chart over epochs, and per-sensor R² evolution.

The `calculate_all_metrics` function now returns a **dictionary** (rather than a tuple) to cleanly handle the expanded set of regression metrics with global and per-sensor breakdowns.

---

## 5. Open Research Questions

The following questions frame the work going forward:

1. **How does the Aryballe e-nose work, and what does its sensing mechanism tell us about which molecular properties to use for prediction?**
   — Understanding SPRi-based binding physics motivates the choice of physico-chemical features (logP, polarity, size, hydrogen bonding capacity).

2. **Can we build a pipeline that automatically fetches SMILES from PubChem and computes a reproducible, filtered descriptor set?**
   — Answered: yes. The name-cleaning → PubChem query → RDKit extraction → Mordred filtering pipeline is operational.

3. **Which descriptor family best captures the variation across these VOCs?**
   — Still under investigation. Early results suggest Mordred descriptors spread the chemical space better than bit-vector fingerprints, but the dataset is too small for definitive conclusions.

4. **How should descriptor selection be adapted when working with a very small molecule set, anticipating a larger dataset in future iterations?**
   — The correlation-based filtering (threshold 0.95) is calibrated to be conservative at small N. As the dataset grows, the retained descriptor set may change.

5. **Does representing molecules as graphs preserve enough structural information to outperform handcrafted fingerprints on this regression task?**
   — This is the central question Vinícius is addressing. The GNN pipeline is now built; evaluation and comparison against the fingerprint baselines is the next step.
