# Multimodal Process Explorer

A what-if analysis tool for multimodal predictive process monitoring. Load an
event log whose events can carry images. The explorer runs up to three models
as each case unfolds, one reading the event history, one the images and one
both, and lets you change what they know to see how their predictions move.

It ships with two examples:

- **Pediatric respiratory admissions** (`examples/xray-pathways`): synthetic
  pathways with real, openly licensed chest X-rays, that anyone can use. Start
  here if you don't have PhysioNet credentials: install Docker (below), then
  follow [its README](examples/xray-pathways/README.md#run-it).
- **COVID-19 pathways from CDSL**: the setting of the MEDUSA paper. The data
  needs credentialed PhysioNet access; with it, the models can be retrained
  following the MEDUSA setup.

You can also load your own XES log with images: the setup reads the event types,
the image events and the outcomes from the log, so no configuration file is
needed, but you need at least one model trained on your data (see
[Your own dataset](#your-own-dataset) and [Models](#models)).

## Install Docker

The explorer runs in Docker, so you need nothing else: no Python, no CUDA.

**Windows 10 or 11**

1. Download Docker Desktop from
   [docs.docker.com/desktop/setup/install/windows-install](https://docs.docker.com/desktop/setup/install/windows-install/)
   and run the installer with the default options (it uses WSL 2).
2. Restart when asked. If Docker Desktop says WSL is missing, open PowerShell
   as administrator, run `wsl --install`, and restart again.
3. Start Docker Desktop and wait until the bottom left says *Engine running*.
4. With an NVIDIA graphics card, the usual Windows driver is all the GPU needs.

**macOS**

1. Download Docker Desktop for your chip (Apple silicon or Intel) from
   [docs.docker.com/desktop/setup/install/mac-install](https://docs.docker.com/desktop/setup/install/mac-install/).
2. Open the `.dmg`, drag Docker to Applications, start it and wait until it says
   *Engine running*.
3. Macs have no NVIDIA GPU, so the models run on the CPU, which is slower (see
   Quick start).

**Linux (Ubuntu, Debian, Fedora, ...)**

1. Install Docker Engine and the Compose plugin:

   ```bash
   curl -fsSL https://get.docker.com | sh
   sudo usermod -aG docker $USER      # then log out and back in
   ```

2. For an NVIDIA GPU, also install the driver and the
   [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

Check the installation with `docker compose version`.

## Quick start

You need about 20 GB of free disk (15 GB for the image, the rest for the
checkpoints and the uploaded files) and 8 GB of RAM; on Windows and macOS,
check that Docker Desktop may use at least 8 GB (Settings, Resources). Downloading the image takes
from a few minutes to half an hour, depending on the connection. Without a GPU,
everything runs on the CPU: predictions are slower and explanations of the
whole history can take minutes. The image is built for x86-64; on Macs with
Apple silicon, Docker runs it through emulation, which is slower still.

1. Download this repository (green **Code** button, then **Download ZIP**, or
   `git clone`) and open a terminal in its folder.
2. Download the three checkpoints of the open example (`text.pth`,
   `image.pth`, `multimodal.pth`, 1.8 GB in total) from the
   [v1.0 release](https://github.com/Havel29/multimodal-process-explorer/releases/tag/v1.0)
   into a folder, for example `models/xray-pathways/`.
3. Start the application. The first time, this downloads the prebuilt image
   ([ghcr.io/havel29/multimodal-process-explorer](https://github.com/users/Havel29/packages/container/package/multimodal-process-explorer),
   about 15 GB once unpacked):

   ```bash
   docker compose up -d
   ```

   With an NVIDIA GPU, add the GPU file instead:

   ```bash
   docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
   ```

   To use the GPU every time without typing both files, add these lines to
   `.env`:

   ```text
   COMPOSE_PATH_SEPARATOR=,
   COMPOSE_FILE=docker-compose.yml,docker-compose.gpu.yml
   ```

   If the inference menu shows "GPU (unavailable)", the container was started
   without the GPU file.

   To build the image from this folder instead (15 minutes to an hour), add
   `--build`.

4. Open [http://localhost:8000](http://localhost:8000).
5. Go through the two-step setup:
   - **Step 1:** select the XES file `examples/xray-pathways/pathways.xes`,
     the image folder `examples/xray-pathways/images` and the folder with the
     three checkpoints, then click **Parse inputs**. The browser asks to
     confirm the folder uploads, and the 1.8 GB of checkpoints take a minute
     or two to upload.
   - **Step 2:** the fields are already filled in; click **Apply setup**.
     The setup then lists what it read from the log; click **Open the
     explorer** to start. The case list should show 20 cases.

The explorer starts with the configuration of the open example. To use the
CDSL data or your own, see the sections below.

After a restart, **Continue with the data already loaded** reopens the last
setup without uploading anything again. To stop the application, run
`docker compose down`; to also remove the loaded data and the image (about
17 GB), run `docker compose down -v` and
`docker image rm ghcr.io/havel29/multimodal-process-explorer:1.0`.

## Inputs

| Input | What it is |
| --- | --- |
| Event log | An `.xes` or `.xes.gz` file, with any number of cases (the statistics page takes longer to fill with more). |
| Images | A folder. The image field of the log holds paths inside it, e.g. `p12/study1/image.jpg`. |
| Checkpoints | A folder with one to three checkpoints (`.pth`, `.pt`, `.bin`, `.ckpt` or `.safetensors`): a text, an image and/or a multimodal model. In Step 2, a model left on *Not used* is left out. |

The files are uploaded from the browser and copied into a private Docker volume.

In Step 2, the form fills in the field names it recognises, such as
`case:concept:name`, `concept:name` and `time:timestamp`. Pick any field it
leaves blank. The label field is optional. Without it, the observed outcomes
and the Statistics page are hidden.

## Using the explorer

- **Case:** choose one at the top. Its events appear at once, and the
  predictions are computed in the background.
- **Timeline:** one lane per event type. Images are the `IMG` markers. Below the
  lanes, a chart shows how the risk changes over time for each model.
- **What the models know:**
  - drag the black line, click the timeline or pick a day to set the cutoff;
  - click an image marker to predict right after that image;
  - click a lane name to remove its events, and click it again to add them back;
  - drag a box over events to use only those, or to exclude them;
  - tick or untick single events in **Events given to the models**.
- **Predictions:** each model's predicted outcome and probability, whether it
  matches the observed outcome, and the inference time.
- **Why?:** one **Explain** button per explained model (by default the
  multimodal model, with Integrated Gradients) runs that model's explanation
  method and shades the image regions that drove its prediction. For a model that reads text, the **model context** (the
  last 512 tokens it reads) shows the text of those events with every word
  coloured by its attribution, and the **whole history** ranks all events by
  their summed attribution.
- **Statistics:** how the models behave over time across the loaded cases:
  discrimination, calibration, earliness, stability, and which event types move
  the predictions.
- **Inference:** switch between Auto, CPU and GPU.
- **Download CSV:** what the explorer shows for the open case, as one table:
  each model's prediction at the cutoff (class probabilities, device and
  inference time), the points of the prediction lines, and the event scores of
  the last explanation.

## Your own dataset

You don't need to write a configuration. Create a file named `.env` in the
repository folder (or copy `.env.example`) with this line, and restart with
`docker compose up -d`:

```text
EXPLORER_CONFIG=config/generic.toml
```

During setup, the explorer reads the following from the event log:

| What | How |
| --- | --- |
| Timeline lanes | One per activity. |
| Image events | The activities whose events have a value in the image field. |
| Outcome classes | The values of the label field, two or more. The rarest one is the risk class, unless `risk_class` in `[outcome]` names another. |
| Event texts | The activity and all event attributes, except the label. |

A model's prediction is its most probable class; accuracy, settling, flips and
agreement are based on it. The prediction lines, ROC AUC, Brier score, event
importance and explanation colours follow the risk class against all the
others, so they read the same with two or more classes.

Before the explorer opens, the setup lists what it read. Check one thing: the
log can't tell the order of the models' outputs, so the classes are assumed to
be in alphabetical order. If your models use another order, add an `[outcome]`
section (see the commented example in `config/generic.toml`).

The setup saves the completed configuration. To rename lanes, group activities
or write better event texts, copy it out and edit it:

```bash
docker compose cp medusa-demo:/app/data/runtime/config.toml config/mine.toml
```

Then put `EXPLORER_CONFIG=config/mine.toml` in `.env` and restart with
`docker compose up -d`: the `config/` folder is mounted into the container, so
no rebuild is needed. You can also draft one before any setup, with the Python
environment of [Development](#development):

```bash
python scripts/draft_config.py log.xes --image photo_path --label case:outcome -o config/mine.toml
```

Every section of a configuration is optional. `config/cdsl_covid.toml` is the
complete example: event types, outcome classes, models and text templates.

The defaults are RoBERTa (text), ViT (image) and FLAVA (multimodal), fine-tuned
on your data; [Models](#models) explains how to use others.

Explanation and adapter files are copied into the image, so restart with
`docker compose up -d --build` after editing one; configurations are read
from the mounted `config/` folder.

### Wording

Everything the interface says can be adapted to the scenario. The nouns go in
`[demo]`:

```toml
[demo]
case_noun = "claim"        # default: case
pathway_noun = "claim"     # default: trace
image_noun = "photo"       # default: image
stage_noun = "photo"       # the events that carry an image
day_noun = "day"
```

Whole sentences go in `[texts]`, one key each. The configuration saved by the
setup lists every key with its current text; `app/texts.py` has the defaults.
Only the keys you change need to stay in your file:

```toml
[texts]
why_title = "What drives this decision?"
timeline_help = "Pick a {day} or click a {stage} marker to see what the models predicted then."
```

`{case}`, `{pathway}`, `{image}`, `{stage}` and `{day}` are replaced by the
nouns (add `s` for the plural); other fields, such as `{count}`, are filled in
by the page. Event names, lane names, outcome labels and model names are set
in their own sections (`[[event_types]]`, `[[outcome.classes]]`, `[[models]]`).

### Explanation method

Each model can have its own explanation method, and the page shows one
**Explain** button per explained model. Without an `[[explanation]]` section,
the multimodal model is explained with the built-in Integrated Gradients, as in
MEDUSA. To explain other models too, add one `[[explanation]]` per model (a model
left out gets no button):

```toml
[[explanation]]
model = "multimodal"
method = "integrated_gradients"   # built in, for the multimodal model

[[explanation]]
model = "text"
method = "explainers/example_occlusion.py"
```

`method` can also be a Python file in the repository with an `explain()`
function. It receives the model, the history text and the image, and returns
scores for spans of the text and a grid of scores for the image; the explorer
colours the words, sums them per event and draws the grid over the image.
`explainers/example_occlusion.py` is a complete example that only calls the
model's `predict`, and `app/explainers.py` documents the interface:

```toml
[[explanation]]
model = "multimodal"
method = "explainers/example_occlusion.py"
label = "Sentence occlusion"
options = { max_sentences = 40 }      # passed to explain()
```

### Models

The explorer runs one to three models, at most one per role, set by `kind` in
`[[models]]`: `text` reads the event history, `image` the latest image, and
`flava` (multimodal) both. A configuration lists the roles it offers, and the
setup uses those you give a checkpoint, so a single text model is enough to
explore a classic, text-based predictive model. A checkpoint is a PyTorch `state_dict` file (`.pth`,
`.pt`, `.bin`, `.ckpt` or `.safetensors`) whose model has one output per
outcome class, in the order of `[outcome]` in the configuration. There are
three ways to get one:

1. **The default models, trained on your log.** `scripts/train_models.py`
   fine-tunes RoBERTa, ViT and FLAVA exactly as the explorer feeds them; see
   `examples/xray-pathways/README.md` for a complete run.
2. **Another Hugging Face model of the same type**, e.g. BERT instead of
   RoBERTa or Swin instead of ViT: set `base` to its name. The image downloads
   the base models of every file in `config/` when it is built, so add the
   model to your configuration before `docker compose up -d --build`.

   ```toml
   [[models]]
   kind = "text"
   base = "google-bert/bert-base-uncased"
   checkpoint = "models/my-data/text.pth"
   ```

3. **Any other PyTorch model**: write an adapter, a Python file with a class
   `Model` that builds the network (`_build`) and turns the history and image
   into class probabilities (`predict`); the explorer loads the checkpoint
   into the network. `adapters/example_bag_of_words.py` is a complete text
   model in 60 lines, and `app/models/__init__.py` documents the interface.
   Train it with your own code and save its `state_dict`.

   ```toml
   [[models]]
   kind = "text"
   name = "Bag of words"
   adapter = "adapters/example_bag_of_words.py"
   checkpoint = "models/my-data/bag_of_words.pth"
   ```

The built-in Integrated Gradients explains the multimodal model. Any other
model, including one from an adapter file, gets an Explain button once it has
an `[[explanation]]` with a method file (see
[Explanation method](#explanation-method)).

## The CDSL example

CDSL is distributed through
[PhysioNet](https://physionet.org/content/covid-data-shared-learning/1.0.0/)
and requires credentialed access. Its data-use agreement forbids sharing the
event data, the images and the checkpoints trained on them; Git ignores `data/`
and `models/`, so files kept there are never committed by accident.

1. Get credentialed access on PhysioNet, download CDSL, and copy the files
   listed in `MEDUSA/dataset/dataset.txt` into `MEDUSA/dataset/`.
2. Build the event log with the MEDUSA preprocessing (it writes
   `MEDUSA/preprocessing/covid_log.csv`). Steps 2 to 4 run outside Docker, in
   a Python 3.11 environment with the packages of `requirements-preprocess.txt`:

   ```bash
   pip install -r requirements-preprocess.txt
   bash scripts/generate_event_log.sh
   ```

3. Build the log of the chronological test partition: its 509 cases with a
   radiograph (use `--cases 20` for a quick sample of the longest pathways):

   ```bash
   python scripts/create_demo_xes.py --config config/cdsl_covid.toml \
     --log MEDUSA/preprocessing/covid_log.csv \
     --images /path/to/CDSL/IMAGES \
     --cases 509 --output data/runtime/cdsl_test.xes
   ```

4. Extract only the radiographs that log needs from the PhysioNet archive. Run
   it once as a dry run to see the count and size, then add `--extract`:

   ```bash
   python scripts/extract_demo_images.py \
     --xes data/runtime/cdsl_test.xes \
     --archive /path/to/the/CDSL/archive.zip \
     --output data/runtime/test-images
   ```

5. Train the three models on the CDSL data following the MEDUSA setup, with the
   authors' code in `MEDUSA/` (see its README; use step 2 above instead of its
   `generation.sh`, and your own image path instead of
   `/home/vincenzo/Scaricati/covid_log/IMAGES/` in
   `mm_log_generation/create_mm_examples.py`),
   and put the checkpoints in a folder, e.g. `models/`. They fall under the same
   data-use agreement as the data.
6. Put `EXPLORER_CONFIG=config/cdsl_covid.toml` in `.env`, restart with
   `docker compose up -d`, and select in the setup:

   ```text
   XES:          data/runtime/cdsl_test.xes
   Images:       data/runtime/test-images
   Models:       the folder with your three checkpoints
   Image field:  path
   ```

   The configuration expects the checkpoint names `covid_roberta4.pth`,
   `covid_vit2.pth` and `covid_flava_7_full.pth`; rename your files, or pick
   them in the setup's model fields. The hospital discharge, which states the
   outcome and the discharge diagnoses, stays on the timeline but is never
   given to the models (`reveals_outcome` in the configuration).
7. Optionally, check the checkpoints against the results of MEDUSA, outside
   Docker (put the three files in the repository folder, or write their paths
   in the `checkpoint` fields of the configuration):

   ```bash
   python scripts/evaluate_models.py --config config/cdsl_covid.toml \
     --log data/runtime/cdsl_test.xes --images data/runtime/test-images \
     --train-log data/runtime/cdsl_test.xes
   ```

   `--train-log` only feeds the majority baseline: without a CDSL training
   log, the test log gives the majority of the test set instead. With the
   original checkpoints of the study, the block "every image" (one prediction
   per radiograph, as in MEDUSA) gives, for FLAVA, 78.3% accuracy, 0.834 ROC
   AUC and 0.744 macro F1, the values reported in the MEDUSA paper; the
   Statistics page shows the same block as "At each radiograph". Retrained
   checkpoints give close but not identical values.

## Repository layout

| Path | What it holds |
| --- | --- |
| `app/` | The explorer: web server and API (`main.py`, `api.py`), setup (`setup.py`, `inference.py`), predictions, explanations and statistics (`service.py`), configuration (`config.py`, `texts.py`), built-in models (`models/`) and the page (`templates/`, `static/`). |
| `config/` | Configurations: `xray_pathways.toml` (open example, the default), `cdsl_covid.toml` (MEDUSA/CDSL) and `generic.toml` (starting point for your data). |
| `examples/xray-pathways/` | The open example: 20 cases with their radiographs, ready to load. |
| `adapters/` | Your own models: `example_bag_of_words.py` is a template (see [Models](#models)). |
| `explainers/` | Your own explanation methods: `example_occlusion.py` is a template (see [Explanation method](#explanation-method)). |
| `scripts/` | Tools around the explorer, listed below. |
| `tests/` | The test suite (`pytest -q`, see [Development](#development)). |
| `MEDUSA/` | The code of the MEDUSA study; `mm_log_generation/` builds the CDSL event log (see [The CDSL example](#the-cdsl-example)). |
| `data/`, `models/` | Your local data and checkpoints; ignored by Git. |
| `Dockerfile`, `docker-compose*.yml` | The image and how to run it, with the GPU add-on. |
| `requirements*.txt` | Python packages of the explorer and of the CDSL preprocessing. |

## Scripts

| Script | Purpose |
| --- | --- |
| `draft_config.py` | Draft a configuration from an XES file. |
| `create_demo_xes.py` | Build the CDSL log of the test partition (all 509 cases, or a sample). |
| `extract_demo_images.py` | Extract the images the log needs from the PhysioNet archive. |
| `prepare_demo_data.py` | Build the runtime bundle offline (the setup does this for you). |
| `generate_event_log.sh` | Regenerate the CDSL event log with the MEDUSA preprocessing. |
| `download_model_assets.py` | Download the base models (run when building the image). |
| `train_models.py` | Fine-tune the text, image and multimodal models of a configuration on your own log. |
| `make_xray_pathways.py` | Generate the open X-ray example and its training and test logs. |
| `evaluate_models.py` | Accuracy, ROC AUC and macro F1 of the three models on a test log, against the majority baseline. |
| `benchmark_scalability.py` | Time every model on every prefix of a log, by prefix length; `plot_scalability.py` plots it. |

## Development

With Python 3.11 or 3.12 (some pinned packages have no wheels for newer
versions). For a GPU, install the CUDA build of PyTorch first, e.g.
`pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128`;
a plain `pip install` gives the CPU build on Windows.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt pytest httpx
uvicorn app.main:app --port 8000 --workers 1
pytest -q
```

`EXPLORER_CONFIG` selects the configuration (set it in the shell, or add
`--env-file .env` to `uvicorn` if you have a `.env`). Run this way, the default
is `config/xray_pathways.toml`, as with Docker.

## Citation

The CDSL configuration and checkpoints reproduce the setting of:

```bibtex
@article{PASQUADIBISCEGLIE2026102698,
title = {Multimodal predictive process monitoring and its application to explainable clinical pathways},
journal = {Information Systems},
volume = {139},
pages = {102698},
year = {2026},
issn = {0306-4379},
doi = {https://doi.org/10.1016/j.is.2026.102698},
author = {Vincenzo Pasquadibisceglie and Ivan Donadello and Annalisa Appice and Oswald Lanz and Fabrizio Maria Maggi and Giuseppe Fiameni and Donato Malerba}
}
```

## Licence

The explorer builds on the MEDUSA code and, like it, is released under
[CC BY-NC-SA 4.0](LICENSE.md). Third-party material is listed in
[NOTICE.md](NOTICE.md).
