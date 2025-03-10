from ultralytics.data.augment import Compose, RandomHSV, RandomFlip
transforms = [RandomHSV(), RandomFlip()]
compose = Compose(transforms)

transforms = [RandomFlip(), RandomPerspective(10), RandomHSV(0.5, 0.5, 0.5)]
compose = Compose(transforms)
single_transform = compose[1]  # Returns a Compose object with only RandomPerspective
multiple_transforms = compose[0:2]  # Returns a Compose object with RandomFlip and RandomPerspective