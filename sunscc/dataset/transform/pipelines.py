import albumentations as A
from functools import partial

from albumentations.augmentations import transforms

Compose = partial(
    A.Compose, additional_targets={"mip-image": "image", "segmentation": "mask"},
    #is_check_shapes=False # May be needed for more recent versions of albumentations
)

crop_and_rotate = Compose(
    transforms=[A.RandomResizedCrop(512, 512, p=1), A.Flip(p=0.5),],
)

crop_and_transform = Compose(
    transforms=[A.RandomResizedCrop(512, 512, p=1), A.ElasticTransform()]
)

crop = Compose(transforms=[A.RandomResizedCrop(512, 512, p=1)])

