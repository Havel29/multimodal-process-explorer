# Third-party material

- **MEDUSA** (`MEDUSA/`): code of Pasquadibisceglie et al., *Multimodal
  predictive process monitoring and its application to explainable clinical
  pathways*, Information Systems 139 (2026). CC BY-NC-SA 4.0, see
  `MEDUSA/LICENSE.md`. The explorer adapts its models and its Integrated
  Gradients configuration.
- **Chest X-ray images** (`examples/xray-pathways/images/`): Kermany, D.,
  Zhang, K., Goldbaum, M. (2018). Labeled Optical Coherence Tomography (OCT)
  and Chest X-Ray Images for Classification. Mendeley Data, v2.
  https://doi.org/10.17632/rscbjbr9sj.2. CC BY 4.0. The images were resized;
  the pathways around them are synthetic.
- **Pretrained models** downloaded when the image is built: RoBERTa
  (`FacebookAI/roberta-base`, MIT), ViT (`google/vit-base-patch16-224`,
  Apache 2.0) and FLAVA (`facebook/flava-full`, BSD 3-Clause), under their own licences.
- **Python packages** installed when the image is built (`requirements.txt`),
  under their own licences; among them PM4Py, used to read XES files, is
  AGPL v3.
- **Inter** font, loaded by the page from Google Fonts (SIL Open Font
  License 1.1); without internet access the page uses the system font.
- **CDSL** data and the checkpoints trained on it are not included; they need
  credentialed access through PhysioNet.
