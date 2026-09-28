# RetinaProgress AI

[![CI](https://github.com/Anurag-YadavIIH/retina-progress-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/Anurag-YadavIIH/retina-progress-ai/actions/workflows/ci.yml)

Longitudinal retinal disease progression analysis. Given two fundus photographs
from two patient visits, the system segments retinal vessels, the optic
disc and cup, and diabetic retinopathy lesions, extracts clinical biomarkers,
quantifies the change between visits, and generates a clinical report.

> Status: early build. The scaffolding, configuration, device auto-detection
> and continuous integration are in place. The model, biomarker, longitudinal
> and reporting layers are being implemented phase by phase (see the roadmap
> below). This is a research and educational project, not a medical device.

## Clinical problem

Diseases such as glaucoma and diabetic retinopathy progress slowly and are
monitored across repeated visits. Clinicians track the cup-to-disc ratio (CDR),
vessel density and tortuosity, and lesion burden over time. Automating this
comparison gives an objective, reproducible second read of the question that
matters at follow-up: did this eye get worse, and by how much?

## Pipeline

```
   Visit 1 fundus                              Visit 2 fundus
        |                                            |
        v                                            v
  +-----------------------------------------------------------+
  |                     Preprocessing                         |
  |     resize, FOV mask, green-channel CLAHE, normalise      |
  +-----------------------------------------------------------+
        |                                            |
        v                                            v
  +-------------------+  +-------------------+  +-------------------+
  |   Vessel U-Net    |  | Attention U-Net   |  |    U-Net++        |
  |   (DRIVE)         |  | disc/cup (REFUGE) |  |  lesions (IDRiD)  |
  +-------------------+  +-------------------+  +-------------------+
        |                        |                        |
        +------------------------+------------------------+
                                 v
                    +-------------------------+
                    |   Biomarkers per visit  |
                    | density, tortuosity,    |
                    | CDR, lesion areas       |
                    +-------------------------+
                                 v
                    +-------------------------+
                    |  Longitudinal engine    |
                    |  deltas + risk score    |
                    +-------------------------+
                                 v
              +----------------------------------------+
              |  Difference maps, charts, PDF report,  |
              |  FastAPI backend + HTML frontend       |
              +----------------------------------------+
```

## Phase roadmap

| Phase | Area                                   | Status      |
|-------|----------------------------------------|-------------|
| 1     | Config, requirements, gitignore        | Done        |
| 1b    | README, device auto-detect, CI + tests | Done        |
| 2     | Preprocessing (transforms)             | Planned     |
| 3     | Models (U-Net, Attention U-Net, U-Net++, losses) | Planned |
| 4     | Dataset classes (DRIVE, REFUGE, IDRiD) | Planned     |
| 5     | Metrics                                | Planned     |
| 6     | Training scripts                       | Planned     |
| 7     | Inference scripts                      | Planned     |
| 8     | Biomarker extraction                   | Planned     |
| 9     | Longitudinal analysis engine           | Planned     |
| 10    | Explainability and visualisation       | Planned     |
| 11    | Clinical PDF report                    | Planned     |
| 12    | FastAPI backend                        | Planned     |
| 13    | Frontend                               | Planned     |
| 14    | Docker                                 | Planned     |
| 15    | Tests (per module)                     | In progress |
| 16    | Utility scripts                        | In progress |

## Datasets

Datasets are not committed to the repository (see `.gitignore`). Download them
into `data/raw/` to match the paths in `configs/config.yaml`.

| Dataset    | Use                     | Download |
|------------|-------------------------|----------|
| DRIVE      | vessel segmentation     | https://www.kaggle.com/datasets/andrewmvd/drive-digital-retinal-images-for-vessel-extraction |
| REFUGE     | optic disc / cup        | https://refuge.grand-challenge.org/ |
| IDRiD      | lesion segmentation     | https://idrid.grand-challenge.org/Data_Download/ |

RIM-ONE DL is a good alternative source of expert disc and cup segmentations if
REFUGE access is slow: https://github.com/miag-ull/rim-one-dl

## Installation

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

For an NVIDIA GPU, install the matching CUDA build of PyTorch from
https://pytorch.org first, then the rest of `requirements.txt`.

## Device selection

There is no hard-coded `cuda`. The compute device is auto-detected:

```python
from src.utils import get_device, load_config

cfg = load_config()
device = get_device(cfg["runtime"]["device"])  # "auto" -> cuda, else mps, else cpu
```

Override by setting `runtime.device` in `configs/config.yaml` to `cpu`, `cuda`
or `mps`.

## Verify the environment

```bash
python scripts/verify_setup.py
```

Prints each dependency version, CUDA availability, and the resolved device.

## Tests

```bash
python -m pytest tests/ -v
```

Tests run on CPU and need no datasets. They are also run automatically by GitHub
Actions on every push and pull request (see the CI badge above).

## Project structure

```
retina-progress-ai/
├── configs/
│   └── config.yaml            # paths, hyperparameters, runtime.device
├── src/
│   ├── utils.py               # load_config, get_device
│   ├── preprocessing/         # (planned) transforms
│   ├── models/                # (planned) unet, attention_unet, unet_plusplus, losses
│   ├── training/              # (planned) datasets, metrics, train scripts
│   ├── inference/             # (planned) predict scripts
│   ├── biomarkers/            # (planned) vessel, cdr, lesion metrics
│   ├── longitudinal/          # (planned) progression engine
│   ├── explainability/        # (planned) visualiser
│   └── report/                # (planned) PDF generator
├── scripts/
│   └── verify_setup.py        # environment check
├── tests/                     # pytest suite (config + device so far)
├── .github/workflows/ci.yml   # GitHub Actions: pytest on push/PR
├── requirements.txt
└── README.md
```

## Roadmap for running real training

Cloud CPU environments are ideal for writing and testing the code on small
synthetic images. Actual training of the three models is best done on a GPU
(for example Google Colab or Kaggle). A training notebook that clones the repo
and pulls a dataset is planned as part of a later phase.

## Acknowledgements

DRIVE (Staal et al., 2004), REFUGE (Orlando et al., 2020), RIM-ONE DL (Fumero
Batista et al., 2020) and IDRiD (Porwal et al., 2018). Please cite the original
dataset papers in any publication. Research and educational use only. This
project is not a medical device and must not be used for clinical decisions.
