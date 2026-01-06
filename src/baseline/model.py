from utils import get_input_images, get_input_metadata
import nibabel as nib
import sys


def main(input_dir, output_ct_path):
    """Baseline method that fills a PET-derived body region with the HU value of water (0 HU)."""
    input_images = get_input_images(input_dir)
    metadata = get_input_metadata(input_dir)
    suv = metadata["suv"]
    nacstat_pet = nib.load(input_images["nacstat_pet"])
    arr = nacstat_pet.get_fdata() / suv
    body_mask = arr > 0.1
    HU_water = 0
    ct_pred_arr = body_mask * HU_water
    ct_pred = nib.Nifti1Image(ct_pred_arr, nacstat_pet.affine, nacstat_pet.header)
    nib.save(ct_pred, output_ct_path)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python model.py <input_dir> <output_ct_path>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_ct_path = sys.argv[2]
    main(input_dir, output_ct_path)

