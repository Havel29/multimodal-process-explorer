# Open example: pediatric respiratory admissions

An event log that anyone can use, with no credentials. The pathways are
synthetic; the chest X-rays are real radiographs of children aged one to five
from Kermany et al. (CC BY 4.0, see below). Every case is an admission:

```text
Emergency admission -> Laboratory tests -> Chest X-ray -> (Oxygen therapy started)
  -> (Antibiotics started) -> Vital signs, once a day -> (Respiratory deterioration)
  -> (Follow-up chest X-ray)
```

The child is then discharged home or transferred to intensive care. The
transfer follows a pneumonia on a radiograph, which only the image shows,
together with an urgent triage at admission (oxygen saturation below 93%) or a
respiratory deterioration during the stay, which only the history says; 5% of
the courses do not follow the rule. Some pneumonias show only on the follow-up
X-ray, and a deterioration is recorded on the day it happens, so the outcome
becomes predictable over the stay. Nothing else in the pathway depends on the
finding or on the outcome, so neither the text model nor the image model can
predict the transfer alone. The example is meant for trying the explorer, not
as a benchmark of the models.

| File | What it is |
| --- | --- |
| `pathways.xes` | The 20 cases to load in the explorer, 10 per outcome, with radiographs from the dataset's test split. |
| `images/` | Their radiographs; the `image` attribute of the log holds paths inside it. |
| `../../config/xray_pathways.toml` | The configuration: wording, outcome order and checkpoint names. Everything else is read from the log. |

## Run it

1. Get the three checkpoints (`text.pth`, `image.pth`, `multimodal.pth`) into
   `models/xray-pathways/`: download them from the
   [v1.0 release](https://github.com/Havel29/multimodal-process-explorer/releases/tag/v1.0),
   or train them yourself (below).
2. Start the explorer as described in the main README; with Docker, this
   example's configuration is the default.
3. In the setup, select `examples/xray-pathways/pathways.xes`, the folder
   `examples/xray-pathways/images` and the folder `models/xray-pathways`. The
   label field is `outcome`.

## Reproducing the example (optional)

You don't need this to try the explorer. The repository contains only the 20
cases above, and the trained checkpoints are in the release. This section
rebuilds everything else from scratch: the 2000 training and 500 test cases,
the three models and their evaluation on the test cases. It is also a
complete example of training models for your own log. Training takes about an
hour on a recent NVIDIA GPU, and needs the Python environment of the main
README (Development).

Download `ChestXRay2017.zip` (1.2 GB) from
[Mendeley Data](https://doi.org/10.17632/rscbjbr9sj.2) into
`data/xray-pathways/source/`, then:

```bash
python scripts/make_xray_pathways.py      # 2000 training, 500 test and the 20 example cases
python scripts/train_models.py --config config/xray_pathways.toml \
    --log data/xray-pathways/pathways_full.xes --images data/xray-pathways/images --epochs 4
python scripts/evaluate_models.py --config config/xray_pathways.toml \
    --log data/xray-pathways/pathways_test.xes --images data/xray-pathways/images \
    --train-log data/xray-pathways/pathways_full.xes
```

The training cases use the dataset's training images; the test and example
cases use its test images. Training keeps the epoch with the best accuracy on
10% of the training cases; the test cases are used only by
`evaluate_models.py`. `scripts/train_models.py` works for any log and configuration, so
it is also the way to get checkpoints for your own data.

## Credits

Kermany, D., Zhang, K., Goldbaum, M. (2018). Labeled Optical Coherence
Tomography (OCT) and Chest X-Ray Images for Classification. Mendeley Data, v2.
https://doi.org/10.17632/rscbjbr9sj.2 (CC BY 4.0). Described in Kermany et al.,
*Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep
Learning*, Cell 172(5), 2018.
