# E-Nose & Graph Neural Networks for Molecular Property Prediction

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

---

## 6. ISIPCA Meeting — Questions & Answers

At the end of Week 2, a meeting was held with the ISIPCA team to clarify several open questions that had been compiled and formalised into a structured document (`meeting_questions.tex`). The questions covered four domains: signal processing, sensor architecture, dataset size, and the final scientific objective of the project. A critical molecule validation question was also raised. Below are the questions as they were asked, followed by the answers received.

---

### Q1 — What Signal Should We Be Capturing?

**Background:** Looking at the raw sensograms, sensor signals rise continuously toward the end of the analyte phase — meaning the slowly-rising tail dominates the L2-normalised value. The sharp peak that appears mid-run (around index 1450) may carry more chemically specific information but is underweighted in the current normalisation. The sensogram also shows a smaller earlier peak (around index 500). The questions raised were: should we define a time window around the peak instead of using the full signal? Which peak is analytically relevant? Is a binary diode-like signal available to precisely locate the peak? And if not, how should the window be defined?

**Answer:** The relevant region to capture is the interval around the **highest peak** only. If the sensogram shows two peaks close together, the one with the greater amplitude is the analytically significant one — the other is likely an artefact or a secondary event and should be ignored. The correct approach is therefore to define a **time window centred around the maximum peak** and compute the normalised signature from that interval alone, rather than using all time points. Regarding the binary diode sensor that fires at peak onset — that signal is **not available** to us. The recommended method for defining the peak window is to do it **manually**, using proprietary Aryballe software that localises the peak. This software has not yet been provided to us; its delivery is a pending action item.

<div align="center">
   <img src="https://github.com/Baiaopires/Internship_project/blob/main/images/sensogram_peak_zoom_mark.png">
</div>

> **Implication:** The normalised signatures in the current CSV (`7Q27_normalized_signatures.csv`) may not be computed from the optimal signal window. Once the Aryballe software is available, the signatures should be recomputed from the peak-window region. Until then, the existing values are used as targets.

---

### Q2 — 64 Sensors vs. 8

**Background:** The device has 64 sensors in total, but the dataset only contains readings for 8 of them. It was unclear whether this was intentional, what the selection criterion was, and whether the full 64-sensor data could be accessed.

**Answer:** The 64 sensors on the gold film are not 64 distinct peptide types — they are **8 peptide variants, each replicated 8 times** across the surface (8 × 8 = 64 spots). The 8 sensor values in the dataset are therefore already the **averages over each group of 8 replicate spots** for each peptide type. There is no additional information being withheld; the 8 values in the CSV represent the complete, non-redundant signal produced by the device for a given molecule.

> **Implication:** The dataset is complete as-is. There is no richer 64-sensor signal to retrieve. The 8 values are the device's full output, after spatial averaging across replicates.

---

### Q3 — Expected Size of the Full Dataset

**Background:** The current dataset covers only 6 distinct molecules (3 experimental replicates each, giving 18 usable rows after cleaning). It was unclear how many molecules would be available by the end of the internship, whether replicates would be added, and what chemical families would be represented.

**Answer:** ISIPCA currently has only the **18 experiments for the 6 molecules** already in the dataset, and they do not have significant time to run additional experiments in the near future. It is therefore unlikely that the dataset will grow substantially through their experimental pipeline alone by the end of the internship. To address this constraint, ISIPCA suggested that we **go to their laboratory at ISIPCA and run the experiments ourselves**. Each molecule requires 3 experimental runs. The chemical family of the molecules to be tested was not specified at this stage. Additionally, ISIPCA will provide a list of **approximately 800 molecules** they have physically available for experimentation, which will inform which new compounds can be tested.

> **Implication:** Dataset growth is our responsibility to drive. The path forward is to visit ISIPCA, select molecules from their available list, and conduct experiments directly. This also means the final model will need to be designed to handle a potentially still-small dataset, favouring architectures and regularisation strategies that generalise under low-data conditions.

---

### Q4 — Sensor Prediction or Smell Prediction?

**Background:** ISIPCA is both a chemistry research institute and a perfumery school. The project could plausibly aim at predicting raw sensor values (a physico-chemical objective), or at predicting human olfactory perception descriptors such as "floral", "woody", or "citrus" (a perceptual objective), or at a sequential combination of both.

**Answer:** The final objective was not formally specified, but the direction strongly suggested is toward **predicting olfactory/perceptual descriptors** — i.e., predicting what a molecule smells like to a human — rather than stopping at sensor values. The sensor prediction step is understood as an intermediate representation. The broader goal is likely to build a pipeline that goes from molecular structure to perceived smell, which would have direct applications for fragrance design and ingredient evaluation.

> **Implication:** This shapes the long-term architecture of the project. A two-stage pipeline is possible: (1) predict e-nose sensor values from molecular structure, then (2) map those sensor values to olfactory descriptors. Alternatively, the model could be trained end-to-end on odour labels if sufficient labelled data becomes available. This motivates the effort described in Section 7 below to collect odour descriptor data.

---

### Molecule Validation

**Background:** Since the SMILES strings used throughout the pipeline were retrieved automatically from PubChem based on molecule names, their correctness could not be independently verified without chemistry expertise. Errors in SMILES would propagate through the entire pipeline. A particular concern was raised for Ocimene, which has multiple isomers (α, β-cis, β-trans) — PubChem may not return the one actually used in the experiment.

**Answer:** ISIPCA will provide a list of the approximately **800 molecules they have physically available** for experiments. This list will serve as the ground truth for molecule identity validation — the SMILES, CAS numbers, and other identifiers for the molecules we already have can be cross-referenced against this list. In the meantime, the current working task is to search public libraries for **odour descriptor data** for the molecules on that list, to begin building the labelled dataset needed for the perceptual prediction objective.

---

## 7. Building an Odour Descriptor Dataset

Following the ISIPCA meeting, the next concrete step was established: gather odour descriptor labels for the molecules we already have, and prepare infrastructure to extend this to the ~800 molecules ISIPCA will provide. This section documents that process in order.

### Step 1 — Searching for Odour Descriptors for the 6 Existing Molecules

The first task was to find published odour descriptors for the 6 molecules already in the e-nose dataset (Ocimene, Δ3-Carene, Linalool, α-Pinene, (S)-Limonene, (R)-Limonene). Rather than scraping from scratch at this stage, a pre-assembled Kaggle dataset was used as a first quick-lookup source: [**Multi-labelled SMILES Odors Dataset**](https://www.kaggle.com/datasets/aryanamitbarsainyan/multi-labelled-smiles-odors-dataset), which is itself a combination of Good Scents and Leffingwell data. The dataset encodes odour descriptors as a multi-hot binary matrix — one column per descriptor category (floral, fruity, sweet, woody, green, spicy, animal_musk, earthy, citrus, chemical, gourmand, powdery_amber, etc.) — with each molecule represented as a row.

Searching for the 6 molecules against this dataset yielded only **2 matches**: Δ3-Carene and α-Pinene. Their descriptor profiles were:

| Molecule | floral | fruity | sweet | woody | green | spicy | earthy | citrus | chemical |
|---|---|---|---|---|---|---|---|---|---|
| α-Pinene | 0 | 0 | 1 | 1 | 1 | 1 | 1 | 0 | 0 |
| Δ3-Carene | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 1 | 1 |

The remaining 4 molecules (Ocimene, Linalool, (S)-Limonene, (R)-Limonene) were not present, likely due to limited coverage and SMILES string mismatch issues. This initial result made clear that a more exhaustive search across multiple data sources would be necessary, motivating the scraping and dataset synthesis process described in Steps 2 and 3.

---

### Step 2 — Scraping the Good Scents Company Website

Rather than relying on the version of the Good Scents data available in the `pyrfume` GitHub repository (which was last updated approximately 4 years ago and may be significantly outdated), a decision was made to **scrape the Good Scents website directly** to obtain the most current dataset.

This was implemented in `Dataset_Creation.ipynb`. The scraper works as follows:

1. A list of all known odour descriptor terms (e.g., "sweet", "floral", "woody", "citrus", …) is compiled from the Good Scents vocabulary. This list was loaded from `smell_descriptors.csv`.

2. For each descriptor term, a **POST request** is sent to the Good Scents search endpoint (`/search.php?qOdor=<descriptor>`), which returns a paginated list of molecules matching that odour profile.

3. The HTML response is parsed with `BeautifulSoup`. For each molecule found, the parser extracts:
   - Molecule **name**
   - **CAS number** (a unique chemical registry identifier)
   - Full list of **odour descriptors** associated with the molecule

4. Pagination is followed automatically: if a "Next" link is present, the scraper continues to subsequent pages until all results for that descriptor are collected.

5. Results are **deduplicated** by CAS number across all descriptor searches, so a molecule appearing under multiple descriptor queries is only stored once, with its complete descriptor list.

The scraper was tested first on the descriptor "sweet", which returned 1526 molecules on the first page alone, confirming the scale and structure of the data. The full run collected the complete up-to-date Good Scents molecule inventory.

---

### Step 3 — Building the Final Odour Descriptor Dataset (`Preprocessing_Datasets.ipynb`)

Before assembling any dataset, all datasets in the `pyrfume` repository tagged with `organism: human` and `data: odorCharacteristic` were systematically evaluated. Not all of them were usable — several were discarded for reasons such as missing odour descriptors, non-unique labels (multiple people smelling the same compound with no consensus), concentration-dependent responses, or data that only captured inter-molecule similarity rather than per-molecule descriptors. The full evaluation is documented in `Datasets_to_analyse_-_STAGE.pdf`.

The datasets that were identified as usable and incorporated into the final merge are:

- **Arctander** (`arctander_1960`): a curated dataset of ~2,772 molecules with SMILES strings and odour descriptors using the Chastrette classification scheme. Already contains SMILES directly; required only minimal preprocessing (dropping rows with missing values).

- **Good Scents** (`goodscents`, scraped in Step 2): the freshly scraped full inventory of the Good Scents Company website. Provides molecule names and CAS numbers with associated odour descriptors. A **SMILES enrichment step** was applied: for each molecule, the CAS number (or name as fallback) was queried against the **PubChem API** to retrieve the corresponding `IsomericSMILES`. The process was rate-limited, cached to disk (`smiles_cache.json`) for resumability, and molecules with no resolved SMILES were dropped.

- **FlavorDB** (`flavordb`): a database of flavour molecules and their associated odour/flavour descriptors. Identified as usable with minimal preprocessing.

- **FlavorNet** (`flavornet`): another flavour-focused reference dataset with odour descriptors per compound. Identified as directly usable.

- **Leffingwell** (`leffingwell`): a complete dataset with SMILES and odour descriptors, available in its entirety without additional processing.

- **Sharma 2021a** (`sharma_2021a`): contains SMILES and odour labels presumably acquired via human evaluation. Used as-is.

- **Sharma 2021b** (`sharma_2021b`): a larger dataset requiring more preprocessing to extract (SMILES, odour descriptors) pairs into a uniform shape. Incorporated after cleaning.

- **Sigma 2014** (`sigma_2014`): contains SMILES and odour descriptors; origin not fully documented but included given the availability of the key fields.

The following datasets from the pyrfume repository were evaluated and **excluded**: `aromadb` (poor data quality), `bushdid_2014` (no odour descriptors), `dravnieks_1985` (only 161 samples, descriptors not unambiguously linked to individual molecules), `foodb` / `freesolve` (no smell data), `ifra_2019` (only 3 descriptors per molecule, deemed too sparse), `keller_2012` / `keller_2016` (concentration-dependent labels, no expert consensus), `nat_geo_1986` / `nhanes_2014` / `weiss_2012` (not relevant to the task), `ravia_2020` (no clear per-molecule odour descriptors), and `snitz_2013` / `snitz_2019` (similarity comparisons rather than individual odour labels).

The final merged dataset (`final_odor_dataset.csv`) contains columns `smiles` and `odor_descriptors`, spanning thousands of molecules across all included sources.

**Fuzzy matching for descriptor normalisation:** The library `fuzzywuzzy` (with the `python-Levenshtein` backend for speed) was installed to handle approximate string matching between descriptor vocabularies across sources, since the same odour concept may appear under slightly different spellings (e.g., "citrus" vs. "citric", "woody" vs. "wood").

---

### Step 4 — Validating Coverage for Our 6 Molecules

After constructing the final dataset, the 6 molecules from the e-nose experiment were checked for presence using exact SMILES string matching. A key fix was required for α-Pinene: the molecule had been stored in the e-nose dataset under the abbreviated name "a-pinene", which PubChem could not resolve correctly, returning a non-stereospecific SMILES. By changing the query name to **"alpha-pinene"** when retrieving the SMILES from PubChem, the correct canonical SMILES (`CC1=CCC2CC1C2(C)C`) was obtained — and that form was found in the final dataset. Results:

| Molecule | SMILES used | In Final Dataset? |
|---|---|---|
| Ocimene | `CC(C)/C=C/C=C(\C)/C=C` | ✗ Not found |
| Δ3-Carene | `CC1=CCC2C(C1)C2(C)C` | ✓ Found |
| Linalool | `CC(=CCCC(C)(C=C)O)C` | ✓ Found |
| α-Pinene | `CC1=CCC2CC1C2(C)C` | ✓ Found (after name fix) |
| (S)-Limonene | `CC1=CC[C@H](CC1)C(=C)C` | ✓ Found |
| (R)-Limonene | `CC1=CC[C@@H](CC1)C(=C)C` | ✓ Found |

**5 out of 6** molecules from the e-nose dataset are now covered by the final odour descriptor dataset. Only Ocimene remains absent — its multiple structural isomers (α, β-cis, β-trans) make it inherently ambiguous: the SMILES string returned by PubChem for the generic name "ocimene" may not correspond to the specific isomer used in the ISIPCA experiment, and no matching entry could be confirmed. Resolving this will require the CAS number or explicit SMILES to be provided by ISIPCA.

## 8. Odour Descriptor Prediction — GNN with Hierarchical Multi-Label Classification

### Overview

Following the dataset construction work described in Section 7, the primary modelling objective of the internship shifted toward **predicting odour descriptors from molecular structure** — i.e., given a molecule's SMILES string, predict which of 138 perceptual odour categories (floral, woody, fruity, etc.) it belongs to. This is a **multi-label classification problem**: a molecule can simultaneously belong to many descriptors, and those descriptors are not independent — they are organised into a natural hierarchy.

The dataset used throughout this section is the [Multi-labelled SMILES Odors Dataset](https://www.kaggle.com/datasets/aryanamitbarsainyan/multi-labelled-smiles-odors-dataset), a combination of Good Scents and Leffingwell data containing **4,983 molecules** encoded as a binary label matrix of **138 odour descriptors** alongside their non-stereo SMILES strings.

---

### Foundational Papers

Two papers directly inform the architecture and methodology used in this section.

**Sanchez-Lengeling et al. (2019) — "Machine Learning for Scent"**
This paper introduced the `SmellGCN` architecture — a graph convolutional network trained on SMILES representations to predict multi-label odour descriptors. The core design choices are a four-layer GCNConv encoder with skip-connected global add pooling, followed by a two-layer BatchNorm + Dropout MLP head. This architecture forms the backbone of the model used throughout this work. The paper also established the dataset (Good Scents + Leffingwell) and the multi-label BCE training paradigm that we extend here.

**Wehrmann et al. (2018) — "Hierarchical Multi-Label Classification Networks" (HMCN)**
This paper proposed a neural network architecture specifically designed for hierarchical multi-label classification, where labels are organised in a tree or DAG structure. The key insight is that rather than choosing between a global classifier (which captures full label space relationships) or local classifiers (one per hierarchical level, which captures fine-grained within-level structure), HMCN simultaneously optimises both. It introduces dual information flows — a global flow and a local flow — combined with a hierarchical violation penalty that discourages logically inconsistent predictions (e.g., predicting `rose` without predicting `floral`). This paper motivates the architectural extension from the flat `SmellGCN` baseline to the `SmellGCN_HMCNF` model described below.

---

### Data Preprocessing

#### Graph Construction

Each molecule is converted from a SMILES string into a PyTorch Geometric `Data` object using `from_smiles`. Hydrogen atoms are excluded (`with_hydrogen=False`). Node features (`data.x`) encode 9 per-atom properties: atomic number, chirality, degree, formal charge, implicit hydrogen count, radical electrons, hybridisation type, aromaticity, and ring membership. Edge features encode 5 bond properties: bond type, stereo configuration, aromaticity, conjugation, and ring membership. The target vector `data.y` is a binary tensor of shape `(138,)` encoding the molecule's active odour descriptor labels.

#### Stratified Train/Validation/Test Split

A naive random split is inappropriate for multi-label datasets because it can leave rare labels unrepresented in one of the splits, making evaluation unreliable. Instead, **second-order iterative stratification** is applied using `skmultilearn.model_selection.IterativeStratification` with `order=2`.

Order 2 means the algorithm attempts to preserve not only the marginal frequency of each individual label across splits (order 1), but also the co-occurrence frequencies of all label pairs. This is the appropriate choice here because the hierarchy constraint relies on label co-occurrences — if `rose` and `floral` never co-occur in the validation set, the hierarchy penalty cannot be evaluated properly. Higher orders (3+) were considered and rejected: with 4,983 molecules and 138 labels, the number of possible label triplets (~430,000) vastly exceeds the number of samples, making order-3 stratification statistically infeasible and computationally prohibitive.

The split is performed in two sequential steps to avoid leakage: first an 80/20 train/holdout split, then the holdout is further split 50/50 into validation and test. The final proportions are **80% train / 10% validation / 10% test**.

#### Class Imbalance Handling

The label matrix is severely imbalanced — most of the 138 descriptors are positive for only a small fraction of molecules. To address this during training, per-label positive weights are computed as:

```
pos_weight[j] = num_negatives[j] / (num_positives[j] + ε)
```

where ε = 1e-5 prevents division by zero for labels with zero positive examples. These weights are passed to `BCEWithLogitsLoss` to upweight the gradient contribution of rare positive examples.

---

### Hierarchy Construction

#### Meta-Category Grouping

The 138 fine-grained labels are organised into 12 macro-categories (`META_CATEGORIES`), defined manually based on fragrance domain knowledge. Each macro-category acts as a parent group for a set of semantically related fine labels. The 12 macro-categories are named with the `macro_` prefix to distinguish them unambiguously from the fine-grained labels — both levels of the hierarchy are treated as distinct objects:

| Macro-Category | Example Fine Labels |
|---|---|
| `macro_floral` | floral, rose, jasmin, lily, muguet, violet, lavender, geranium |
| `macro_fruity` | fruity, apple, apricot, banana, berry, cherry, peach, strawberry |
| `macro_sweet` | sweet, vanilla, caramellic, honey, chocolate, coconut, creamy |
| `macro_woody` | woody, cedar, sandalwood, pine, vetiver, terpenic, balsamic |
| `macro_green` | green, grassy, herbal, leafy, hay, fresh, cucumber, vegetable |
| `macro_spicy` | spicy, cinnamon, clove, warm, pungent, mint, camphoreous |
| `macro_animal_musk` | animal, musk, leathery, fishy, sweaty, meaty, musty |
| `macro_earthy` | earthy, mushroom, nutty, roasted, coffee, tobacco, smoky |
| `macro_citrus` | citrus, bergamot, ozone, clean, soapy |
| `macro_chemical` | solvent, ethereal, metallic, medicinal, phenolic, sulfurous |
| `macro_gourmand` | almond, malty, rummy, brandy, winey, cooked, savory, garlic |
| `macro_powdery_amber` | amber, powdery, anisic, coumarinic, orris, waxy, aldehydic |

#### Validated Child-Parent Pairs

Not all manually assigned child-parent pairs are consistent with the data. For example, `chamomile` was assigned to `macro_floral`, but in the dataset `chamomile` frequently appears without `floral` co-occurring — the hierarchy assumption is wrong for that pair. Including invalid pairs in the hierarchy penalty actively hurts the model by penalising correct predictions.

To address this, a **data-driven validation step** filters the child-parent pairs using conditional probability:

```
P(parent = 1 | child = 1) ≥ 0.6
```

For each candidate pair, this probability is computed empirically from the full dataset. Only pairs where the child reliably implies the parent are retained. This filter is critical: experiments confirmed that running the hierarchy penalty with unvalidated pairs significantly degraded all metrics, while validated pairs preserved or improved them.

The final `validated_pairs` list is a set of `(child_column_index, parent_group_index)` tuples used exclusively inside the loss function. Labels that lose their parent assignment through filtering are not removed from the dataset — they continue to contribute to the BCE loss as flat labels.

---

### Model Architecture — `SmellGCN_HMCNF`

The model is an adaptation of `SmellGCN_Paper` (from Sanchez-Lengeling et al.) to the HMCN-F paradigm (from Wehrmann et al.). The GCN encoder is preserved entirely; the MLP head is restructured to produce three simultaneous outputs.

#### GCN Encoder (unchanged from baseline)

Four `GCNConv` layers with progressively increasing hidden dimensions:

```
GCNConv(node_features → 15) → SELU
GCNConv(15 → 20)            → SELU
GCNConv(20 → 27)            → SELU
GCNConv(27 → 36)            → SELU
```

After each layer, `global_add_pool` produces a graph-level representation by summing all node features. The five pooled representations (including the raw node features before any convolution) are concatenated into a single graph embedding of dimension `node_features + 15 + 20 + 27 + 36`. This skip-connected pooling strategy gives the model access to structural information at multiple levels of abstraction simultaneously.

#### Global Flow

A two-layer MLP processes the graph embedding into a sequence of hidden representations:

```
Linear(mlp_input_dim → 96) → BatchNorm → ReLU → Dropout(0.47)   [h1, level-1 repr.]
Linear(96 → 63)             → BatchNorm → ReLU → Dropout(0.47)   [h2, level-2 repr.]
Linear(63 → 138)                                                   [logits_global]
```

`logits_global` is the model's global prediction over all 138 fine-grained labels.

#### Local Flow — Level 1 (12 macro-categories)

A transition branch off `h1` (the intermediate 96-dimensional representation) produces predictions for the 12 macro-categories:

```
Linear(96 → 48) → BatchNorm → ReLU   [local hidden]
Linear(48 → 12)                       [logits_local1]
```

This head is trained with its own BCE loss term against the ground-truth 12-group labels (derived on the fly from `y_138` via OR-aggregation over validated child indices). It forces `h1` to explicitly encode coarse odour family information.

#### Local Flow — Level 2 (138 fine-grained labels)

A transition branch off `h2` (the deeper 63-dimensional representation) produces a second independent prediction over the 138 fine-grained labels:

```
Linear(63 → 63) → BatchNorm → ReLU   [local hidden]
Linear(63 → 138)                      [logits_local2]
```

This head is trained with its own BCE loss term against `y_138`, reinforcing that `h2` encodes fine-grained odour descriptor information.

#### Final Combined Prediction

At inference time, `logits_local2` and `logits_global` are combined as a weighted average:

```
logits_final = β × logits_local2 + (1 − β) × logits_global
```

with `β = 0.5` (equal weight to local and global information, as in Wehrmann et al.). The combined logit is used for threshold optimisation and all final evaluation metrics.

---

### Loss Function — `hmcnf_loss`

The total loss has four terms:

```
L_total = L_global + L_local1 + L_local2 + λ × L_hierarchy
```

**`L_global`** — `BCEWithLogitsLoss` with `pos_weight` applied to `logits_global` against `y_138`. This is the primary fine-label training signal.

**`L_local1`** — unweighted `BCEWithLogitsLoss` applied to `logits_local1` against the derived 12-group ground truth `y_12`. This trains the level-1 local head.

**`L_local2`** — `BCEWithLogitsLoss` with `pos_weight` applied to `logits_local2` against `y_138`. This trains the level-2 local head with the same class-imbalance correction as the global head.

**`L_hierarchy`** — the hierarchical violation penalty from Wehrmann et al. (Eq. 16), applied exclusively over `validated_pairs`:

```
L_hierarchy = (1 / |P|) × Σ_{(child, parent) ∈ P} mean[ relu(p_child − p_parent)² ]
```

where `p_child` and `p_parent` are sigmoid probabilities from `logits_global`, and `p_parent` is derived as the elementwise max over all children of that group. The squared form penalises large violations more aggressively than the linear form. Normalisation by `|P|` (number of validated pairs) keeps the penalty magnitude stable regardless of how many pairs survive the validation filter.

**λ (lambda)** controls the weight of the hierarchy penalty relative to the BCE terms. Setting λ too large forces the model to prioritise consistency over correct prediction, degrading precision. Setting λ too small makes the penalty ineffective.

---

### Training Configuration & Hyperparameters

| Hyperparameter | Value | Rationale |
|---|---|---|
| Optimiser | Adam | Standard choice for GNNs; adaptive learning rates handle the sparse gradient signal from rare labels |
| Learning rate | 0.001 | Default Adam LR; warm restarts reduce sensitivity to this choice |
| Scheduler | `CosineAnnealingWarmRestarts` | Periodically resets the learning rate to escape local minima in a non-convex 138-label loss landscape |
| T_0 | 50 | Restart period in epochs; each cycle allows the model to converge before the next reset |
| T_mult | 1 | Equal-length cycles; appropriate for long runs where consistent exploration is preferred over progressive refinement |
| Batch size | 32 | Standard for graph datasets of this scale; consistent with established practice |
| Dropout | 0.47 | Inherited from the SmellGCN_Paper architecture; prevents overfitting on the ~4,000 training molecule set |
| β | 0.5 | Equal weighting of local and global predictions; fixed as in Wehrmann et al. |
| λ (lambda) | 0.3 | Determined empirically; validated pairs allow a higher lambda than would be safe with unvalidated pairs |
| P threshold | 0.6 | Conditional probability threshold for validated pairs; balances hierarchy coverage against signal quality |
| Epochs | 1000 | Model has not fully converged at 500 epochs; AUROC and Macro F1 continue improving past this point |

---

### Threshold Optimisation

Since the model outputs continuous probabilities for each label, a binary prediction requires a threshold. Rather than using a fixed threshold of 0.5 (which performs poorly on imbalanced labels), **per-label threshold optimisation** is performed on the validation set after training.

For each of the 138 fine labels and each of the 12 macro-categories separately, the threshold is swept from 0.01 to 0.99 in steps of 0.01. At each threshold, the per-label macro-F1 is computed. The threshold that maximises macro-F1 for each label is retained. The resulting `thresholds_138` and `thresholds_12` tensors are applied at test time to produce the final binary predictions.

Threshold optimisation is always performed after training, never during it. The fixed-threshold metrics monitored during the training loop serve only as a training signal proxy and do not reflect final model performance.

---

### Evaluation Metrics

Metrics are computed separately for the two prediction levels: the 138 fine-grained labels (HMCN Fine) and the 12 macro-category predictions from `logits_local1` (HMCN Meta). All metrics use the `macro` averaging strategy unless otherwise stated, which treats all labels equally regardless of their frequency — appropriate for this dataset where rare labels are of equal scientific interest to common ones.

Labels for which only one class is present in the test set are excluded from AUC computations to avoid undefined metric errors; the number of included labels is reported explicitly.

#### Metrics computed for both levels

**ROC AUC (macro)** — area under the receiver operating characteristic curve, averaged across labels. Measures ranking quality: how well the model separates positive from negative examples regardless of threshold. Threshold-independent, making it a reliable primary metric even before threshold optimisation. Macro averaging ensures rare labels are not dominated by frequent ones.

**PR AUC (macro)** — area under the precision-recall curve, averaged across labels. More informative than ROC AUC for severely imbalanced label distributions: a model that trivially predicts all negatives achieves high ROC AUC but near-zero PR AUC. For this dataset, where most labels are positive for fewer than 10% of molecules, PR AUC is a more honest measure of predictive quality.

**Hierarchical Violation Rate** — the fraction of predicted child-label activations that are inconsistent with their parent: `child=1` in `y_pred_138` while the corresponding `parent=0` in `y_pred_12`. Computed jointly across both prediction levels. Directly quantifies whether the HMCN architecture achieves its primary design goal of hierarchical consistency. A lower rate indicates more structurally coherent predictions.

#### Metrics computed for HMCN Fine (138 labels) only

**F1 (macro, top-20)** — macro-averaged F1 score computed over the 20 labels with the highest representation in the test set. Reported on a subset because the majority of the 138 labels are too rare in the test split to produce reliable per-label estimates; including highly sparse labels in macro averaging inflates variance without adding information. The top-20 subset provides a stable and interpretable view of fine-grained prediction quality.

#### Metrics computed for HMCN Meta (12 labels) only

**F1 (macro)** — macro-averaged F1 over all 12 macro-categories. The 12-group prediction task is far more tractable than the 138-label task: all groups are reasonably represented in the test set, so full macro-F1 is meaningful and stable here.

**Balanced Accuracy (macro)** — the average of sensitivity and specificity per label, then averaged across labels. Unlike standard accuracy, balanced accuracy gives equal weight to correct positive and correct negative predictions. This matters for the 12-group evaluation because some macro-categories (e.g., `macro_chemical`) are less frequent than others (e.g., `macro_fruity`), and standard accuracy would be dominated by the majority groups.

**Sensitivity (macro)** — true positive rate per label, averaged across the 12 groups. Measures how well the model identifies molecules that belong to each odour family. Low sensitivity on a group means the model systematically misses molecules of that family.

**Specificity (macro)** — true negative rate per label, averaged across the 12 groups. Measures how well the model rejects molecules that do not belong to each family. Together with sensitivity, it provides a complete view of per-group discrimination ability and directly informs the balanced accuracy score.

**Label Co-occurrence Consistency** — measures how well the model's predicted label co-occurrence structure matches the ground-truth co-occurrence structure at the 12-group level. The ground-truth co-occurrence matrix `C_true` and the predicted co-occurrence matrix `C_pred` are both normalised to [0, 1], and consistency is computed as `1 − mean(|C_true − C_pred|)`. A value near 1 means the model predicts label combinations that reflect real-world odour family co-occurrence patterns; a value near 0 means predictions are structurally incoherent. This metric is reported only at the 12-group level because at 138 labels the co-occurrence space is too large and sparse to be meaningful given the dataset size.
