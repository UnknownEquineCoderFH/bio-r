from torchvision.models import vgg19, VGG16_Weights
from algorithms.pipeline import normal_ds

model = vgg19(
    normal_ds,
    weights=VGG16_Weights([0.2, 0.2, 0.1, 0.2, 0, 0, 0, 1.0]),
)

model.compile()

print(model.classifier)
