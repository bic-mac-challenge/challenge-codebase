from pathlib import Path
import json

def find_file(input_dir, pattern):
    """Find a single file matching the given pattern in the input directory."""
    input_dir = Path(input_dir)
    # assert that exactly one file matches the pattern
    files = list(input_dir.glob(pattern))
    assert len(files) == 1, f"Expected exactly one file matching {pattern}, found {len(files)}"
    return files[0]

def get_input_images(input_dir):
    """Find and return paths to all required input images in the given directory."""
    input_dir = Path(input_dir)
    files = {}
    
    files["nacstat_pet"] = find_file(input_dir, "*nacstat_pet.nii.gz")
    files["dixon_body_full_inphase"] = find_file(input_dir, "*DIXONbodyIN_T1w.nii.gz")
    files["dixon_body_full_outphase"] = find_file(input_dir, "*DIXONbodyOUT_T1w.nii.gz")

    for i in range(4):
        files[f"dixon_body_chunk{i+1}_inhase"] = find_file(input_dir, f"*DIXONbodyIN_chunk-{i+1}_T1w.nii.gz")
        files[f"dixon_body_chunk{i+1}_outphase"] = find_file(input_dir, f"*DIXONbodyOUT_chunk-{i+1}_T1w.nii.gz")
    files["dixon_head_inphase"] = find_file(input_dir, "*DIXONheadIN_T1w.nii.gz")
    files["dixon_head_outphase"] = find_file(input_dir, "*DIXONheadOUT_T1w.nii.gz")

    files["topogram"] = find_file(input_dir, "*acq-TOPOGRAM_rec-tr20f_Xray.nii.gz")

    return files

def get_input_metadata(input_dir):
    """Load and return metadata from the input directory."""
    input_dir = Path(input_dir)
    metadata_file = input_dir / "metadata.json"
    with open(metadata_file, "r") as f:
        metadata = json.load(f)
    return metadata
