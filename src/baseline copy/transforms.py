from monai.transforms import *


#NOTE: The baseline uses only the NAC-PET as input
# however, your model may use all the images and metadata available
# under the /features folder

def get_transforms(patch_size, num_samples=2):

    transforms = Compose(
        [

            LoadImaged(keys=["nacpet", "ct", "prediction_mask"]),

            EnsureChannelFirstd(keys=["nacpet", "ct", "prediction_mask"]),

            NormalizeIntensityd(
                keys=["nacpet"],
                nonzero=False,
                channel_wise=True,
                subtrahend=[0]
                
            ),

            ScaleIntensityRanged(
                keys=["ct"],
                a_min=-1000,
                a_max=2000,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),

            # now input is just nacpet
            ConcatItemsd(
                keys=["nacpet"],
                name="input"
            ),

            # Crop first so all random augmentations run on small patches
            RandSpatialCropSamplesd(
                keys=["input", "ct", "prediction_mask"],
                roi_size=patch_size,
                random_size=False,
                num_samples=num_samples
            ),
            EnsureTyped(keys=["input", "ct", "prediction_mask"]),

        ]
    )

    return transforms