# HIPPIE Web

HIPPIE Web is a coding-free interface for embedding and clustering extracellular
electrophysiology recordings. It wraps the HIPPIE deep-learning encoder behind a
Streamlit application so that users can upload a recording, obtain a learned
embedding, explore the resulting clusters, and inspect per-cluster physiology
without writing any code.

## Input modes

The app accepts five kinds of input:

1. **CSV** — three header-less CSV files containing the autocorrelogram (ACG),
   inter-spike-interval (ISI) distribution, and waveform data, one file per modality.
2. **ACQM zip** — a single ACQM archive that is parsed by the bundled neurocurator
   to extract the waveform and spiking-dynamics features automatically.
3. **NWB** — a Neurodata Without Borders file.
4. **PHY / Kilosort output zip** — a zipped Kilosort/PHY sorting output.
5. **Google Drive download link** — a shareable link to any of the file types above,
   downloaded directly by the app.

## Recording-technology selector

Before processing, select the recording technology used to acquire the data:

- Neuropixels (1.0 / 2.0)
- Silicon probe (non-Neuropixels)
- Juxtacellular (glass micropipette)

This selection conditions the HIPPIE encoder so the embedding accounts for the
acquisition technology.

## What the app produces

- A UMAP embedding of the recorded units, clustered with HDBSCAN.
- Interactive cluster inspection: select a cluster to highlight it and view the
  cluster-averaged waveform, ISI distribution, and ACG.
- If you already have your own cell-type labels, you can upload them as a CSV.
- All computed data and figures can be downloaded for further analysis.

## Running locally

```bash
pip install -r requirements.txt
streamlit run web_code.py
```

A hosted version of the app is also available, so no local installation is
required to try it out.

## Repository layout

- `web_code.py` — the Streamlit application and the main processing flow.
- `neurocurator.py` — parses ACQM recordings into the features HIPPIE expects.
- `utils.py` — shared helpers for loading data, running the model, and clustering.
