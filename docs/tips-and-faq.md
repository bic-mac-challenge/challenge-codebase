
# Tips

We strongly suggest you start by reading [data-background.md](data-background.md) and [reconstruction.md](reconstruction.md).


### Suggested public datasets
We recommend the following public datasets if you wish to perform pretraining etc. 
### PET/CT
> Note that all PET images in the following datasets are attenuation corrected (usually by the accompanying CT), which means that the PET may encode some of the CT information.
- [Vienna QUADRA_HC](https://zenodo.org/records/16588733) 96 whole-body 18F-FDG PET/CT studies from 48 participants. Like BIC-MAC, the PET/CT is acquired on a Siemens Biograph Vision Quadra and the participants are healthy controls. [citation](https://www.nature.com/articles/s41597-025-05997-4)

- [AutoPET V](https://fdat.uni-tuebingen.de/records/0zs4c-89f12) 1014 whole-body 18F-FDG PET/CT studies, 597 PSMA PET/CT studies [citation](https://www.nature.com/articles/s41597-022-01718-3)

- [PETWB-REP](https://zenodo.org/records/18670487) 565 whole-body 18F-FDG PET/CT studies [citation](https://arxiv.org/pdf/2508.04062)

- [ENHANCE.PET](https://pubmed.ncbi.nlm.nih.gov/40799763/) 1,597 whole-body 18F-FDG PET/CT studies. Downloaded by running `moosez -dtd -dd path/to/download/` (install [moosez](https://github.com/ENHANCE-PET/MOOSE)) [citation](https://pmc.ncbi.nlm.nih.gov/articles/PMC12340901/#S1).

- [ViMED-PET](https://huggingface.co/datasets/dacthai2807/ViMed-PET)  2,757 whole-body 18F-FDG PET/CT studies. [citation](https://arxiv.org/abs/2509.24739v1)

- [Lung-PET-CT-Dx](https://www.cancerimagingarchive.net/collection/lung-pet-ct-dx/) 436 whole-body (no head) 18F-FDG PET/CT studies [citation](https://doi.org/10.7937/TCIA.2020.NNC2-0461)

- [Deep-PSMA](https://zenodo.org/records/15281784) 100 whole-body PSMA and 18F-FDG PET/CT studies [citation](https://doi.org/10.5281/zenodo.15281783)

### MRI/CT

- [SynthRAD2025](https://zenodo.org/records/14918089) 890 paired MRI–CT and 1,472 CBCT–CT sets covering head-and-neck, thorax, and abdomen from 5 European university medical centers. [citation](https://arxiv.org/abs/2502.17609)

- [CHAOS](https://zenodo.org/records/3431873) 40 abdominal CT and 40 abdominal MRI studies (T1-DUAL, T2-SPIR) from healthy subjects. CT and MRI are from **different** patients (unpaired). [citation](https://doi.org/10.1016/j.media.2020.101950)

- [Paired CT–MRI (T1+T2)](https://doi.org/10.1016/j.dib.2025.111768) Small co-registered CT and MRI (T1- and T2-weighted) dataset from the same patients. [citation](https://doi.org/10.1016/j.dib.2025.111768)

- [Learn2Reg Abdomen MR-CT](https://learn2reg.grand-challenge.org/Datasets/) 16 paired and 90 unpaired abdominal CT and MRI scans [citation](https://doi.org/10.1109/TMI.2022.3213983)

- [RIRE](https://rire.insight-journal.org/) ~20 brain patients with paired CT, T1, T2, and PD MRI with gold-standard marker-based registration transforms. [citation](https://rire.insight-journal.org/)


### CT

- [NLST](https://www.cancerimagingarchive.net/collection/nlst/) ~26,000 low-dose chest CT studies. [citation](https://doi.org/10.7937/TCIA.HMQ8-J677)

- [CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE) 47,149 chest CT volumes with paired radiology reports. [citation](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE)

- [TotalSegmentator CT](https://zenodo.org/records/10047292) 1,228 whole-body CT studies with segmentations of 117 anatomical structures. [citation](https://doi.org/10.1148/ryai.230024)

- [AbdomenAtlas 1.0 Mini](https://huggingface.co/datasets/AbdomenAtlas/AbdomenAtlas1.0Mini) 5,195 abdominal CT studies with 9-organ segmentations. [citation](https://arxiv.org/abs/2305.09666)

- [Merlin Abdominal CT Dataset](https://stanfordaimi.azurewebsites.net/datasets/60b9c7ff-877b-48ce-96c3-0194c8205c40) 25,494 abdominal CT studies with radiology reports. [citation](https://doi.org/10.1038/s41586-026-10181-8)

### MRI
- [TotalSegmentator MRI](https://zenodo.org/records/14710732) 616 whole-body MRI studies. [citation](https://doi.org/10.1148/ryai.230024)

- [FOMO-300K](https://huggingface.co/datasets/FOMO-MRI/FOMO300K) 81,282  brain MRI studies with a total of 306,303 scans. [citation](https://arxiv.org/abs/2506.14432)

### Chest X-ray (Topogram-like)
- [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/) 224,316 chest radiographs of 65,240 patients with 14 pathology labels. [citation](https://arxiv.org/abs/1901.07031)

---


# FAQ


**Do I need to understand PET reconstruction to participate?**

No. You only need to predict a pseudo-CT from the input features. The reconstruction pipeline is provided and run for you. See [reconstruction.md](reconstruction.md) if you want to understand what the reconstruction does and why CT quality matters for PET accuracy.


**What data can my pseudo-CT model use as input?**

Any and all files under the `features/` folder. The baseline uses just the `nacpet.nii.gz`, but you are free to combine modalities and demographic features in any way you see fit. 

**Do I need to resample or register any of the images?**

All images under `features/` have been resampled to the dimensions of `ct.nii.gz (512x512x531)`. The topogram is 2D and therefore resampled to `(512x1x531)`. Prior to resampling, the MR have been rigidly translated to PET/CT space. The MRI aligns crudely with the PET/CT/Topogram, and it is up to you to decide whether your model should incoorporate registration as a preprocessing step. 

**Can I use other data than the BIC-MAC dataset?**
Additional training data is allowed, as long as it was released, publicly available and accessible to all participants without restrictions prior the start of the challenge (April 1st). Private datasets are NOT allowed. 

**Can I use pretrained models?**

Yes, you are allowed to use and finetune pretrained models, as longs as they were publicly available and accessible to all participants without restrictions (e.g. on GitHub, Huggingface, Zenodo, or a comparable platform) *prior to the start of the challenge* (April 1st).

**What subjects have sinogram data for local reconstruction?**

Sinogram data is provided for the following subjects:
- `train/`: `sub-000`, `sub-001`, `sub-002`, `sub-005`, `sub-006`, `sub-008`, `sub-013`, `sub-014`
- `val/`: `sub-004`, `sub-009` ,`sub-010` ,`sub-018` (all of them)

The remaining 67 training subjects have `features/` and `ct-label/` only — you can train and evaluate CT metrics on them, but cannot run closed-loop PET reconstruction locally. We chose to provide sinogram data for only 8 of the 67 training subjects to keep the dataset size managable. 

**Why is `prediction_mask.nii.gz` in `ct-label/`?**

It marks the voxels your model is responsible for predicting (body minus face and scanner bed). During training you may want to restrict your loss to this mask so the model is not penalised for face/bed regions that are overwritten anyway during reconstruction.

**The MRI comes in chunks — do I need to stitch them?**

Pre-stitched versions (`mri_combined_in_phase.nii.gz`, `mri_combined_out_phase.nii.gz`) are provided if you want a single whole-body volume. The individual chunks (`mri_chunk_{0-3}_{in/out}_phase.nii.gz`) are available if you prefer to work per bed position. 


**What format does my pseudo-CT need to be in?**

A NIfTI file (`.nii.gz`) in Hounsfield units, with the same shape and affine as `features/nacpet.nii.gz`. Copying the header directly from the NAC-PET when saving is the safest approach:

```python
ref = nib.load("features/nacpet.nii.gz")
nib.save(nib.Nifti1Image(pred_hu, ref.affine, ref.header), "ct.nii.gz")
```

**Do I need to install STIR locally?**

No, you do not even need to run reconstruction locally - unless you want to validate using the PET-based challenge metrics. If we you do wish to do reconstruction, we recommend using the Docker image, which includes STIR and all dependencies. The image wraps the python code in (`src/recon`) (see ['reconstruction.md](reconstruction.md) for details). Alternatively, you can run the reconstruction locally if you have a local STIR build. Please see [STIR User Guide](https://stir.sourceforge.net/documentation/STIR-UsersGuide.pdf) for installation instructions. IMPORTANT: Make sure to install STIR from source and not a prepackaged version, since the critical reconstruction bugs related to Quadra Sinograms remain present in version 6.3. 

**Reconstruction is slow — how long should I expect it to take?**

Roughly 20–120 minutes per subject on a modern CPU, dominated by the OSEM reconstruction step (step 9). Intermediate outputs are cached, so re-runs resume from where they left off unless `OVERWRITE=1` is set.

**How do I debug a failed reconstruction?**

Check `output_dir/intermediates/recon.log` for the full STIR log. Rerun with `VERBOSE=1` (Docker) or `-v` (Python) to stream STIR output to the terminal in real time.

**Which metrics are reported on the validation leaderboard?**

Four metrics in total: Whole-body SUV MAE, Brain Outlier Score, Organ Bias, and CT μ-MAE. See the evaluation section of the main README for descriptions. The Brain Outlier Score is a dataset-level metric — it cannot be computed for a single subject. The fifth and final metric "TAC Bias" is only computed for the final test set. The metric calculation requires reconstruction using dynamic sinograms, which are unfortunately too large to share. 

**Can I evaluate without running reconstruction?**

Yes — CT μ-MAE only requires your pseudo-CT and the ground-truth CT. Pass `--pred_ct` without `--pred_pet` to `eval_subject.py`:

```bash
python src/evaluation/eval_subject.py \
  --subject_dir data/sub-000 \
  --pred_ct outputs/sub-000/ct.nii.gz
```

**Do I need to submit both a pseudo-CT and a PET?**
For the validation phase, you can submit CT-only, PET-only, or both to CodaBench in a zip file. Please see [Submission Guide](submission-guide.md) for instructions. Submitting both PET and CT unlocks all four metrics. Note that to submit a PET image, you have to run reconstruction locally. For the Final Test phase, the organizers will run reconstruction so you only submit the the Docker image with your pseudo-CT model. 

**How can I make sure that my submitted Docker image will work?**
Once the validation phase starts, you can submit your pseudo-CT container for "Dry-Run". The organizers will the run your container on the hardware used for final evaluation and report back the CT-based metrics for the validation set. This way you can check that the container runs successfully and within the 5-minute time limit. 

**Does my final container need to be the same as the dry-run container?**
No. But we recommend doing a dry-run for the container you intend to submit for the final validation to ensure that it will not crash or run out of memory. 