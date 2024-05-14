from torchvision.models import vgg16, VGG16_Weights
from algorithms.pipeline import normal_ds

model = vgg16(
    normal_ds,
    weights=VGG16_Weights([0.2, 0.2, 0.1, 0.2, 0, 0, 0, 1.0]),
)

model.compile()

print(model.classifier)
