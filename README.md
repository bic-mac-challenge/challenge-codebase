# BIC-MAC Challenge: Big Cross-Modal Attenuation Correction

Synthesizing pseudo-CT (mu-map) for PET attenuation correction from multimodal inputs (PET, MRI, Topogram).

## Tasks

**Task 1**: Generate mu-map in CT-space (evaluated vs reference CT-mu-map)

**Task 2**: Generate mu-map for PET reconstruction in PET-space (evaluated vs reference CT-AC PET)

## Docker Interface

### Task 1 and 2: Mu-map Prediction

```bash
docker run --rm \
  -v /path/to/subject:/input:ro \
  -v /path/to/output/dir:/output \
  your-solution:latest \
  /input /output/predicted_ct.nii.gz
```

**Arguments:**
1. Input directory: PET, MRI, Topogram, metadata.json
2. Output file path: Predicted CT (NIfTI)

**Input files:**
- `*nacstat_pet.nii.gz`
- `*DIXONbodyIN_T1w.nii.gz`, `*DIXONbodyOUT_T1w.nii.gz`
- `*DIXONbodyIN_chunk-{1-4}_T1w.nii.gz`, `*DIXONbodyOUT_chunk-{1-4}_T1w.nii.gz`
- `*DIXONheadIN_T1w.nii.gz`, `*DIXONheadOUT_T1w.nii.gz`
- `*acq-TOPOGRAM_rec-tr20f_Xray.nii.gz`
- `metadata.json`

### Task 2: PET Reconstruction

```bash
docker run --rm \
  -v /path/to/predicted_ct.nii.gz:/input_ct:ro \
  -v /path/to/recon_dir:/recon:ro \
  -v /path/to/output/dir:/output \
  reconstruction-software:latest \
  /input_ct /recon /output/reconstructed_pet.nii.gz
```

**Arguments:**
1. Input CT: Predicted mu-map from Task 1
2. Reconstruction directory: Sinograms, scatter maps, normalization data
3. Output file path: Reconstructed PET (NIfTI)

## Baseline

```bash
cd src/baseline
docker build -t baseline-solution:latest .

docker run --rm \
  -v /path/to/subject:/input:ro \
  -v /path/to/output:/output \
  baseline-solution:latest \
  /input /output/predicted_ct.nii.gz
```

## Evaluation

### Task 1

```bash
python src/evaluation/run_docker_model.py your-solution:latest /path/to/subject /path/to/output.nii.gz
python src/evaluation/evaluate_task1.py /path/to/predictions /path/to/ground_truth
```

### Task 2

```bash
python src/evaluation/run_docker_model.py your-solution:latest /path/to/subject /path/to/predicted_ct.nii.gz
python src/evaluation/run_docker_reconstruction.py reconstruction-software:latest /path/to/predicted_ct.nii.gz /path/to/recon_dir /path/to/output_pet.nii.gz
python src/evaluation/evaluate_task2.py /path/to/reconstructed_pets /path/to/reference_ct_ac_pets
```
