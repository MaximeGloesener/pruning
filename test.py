import torch
from benchmark import benchmark
from models.vgg import VGG

device = "cuda" if torch.cuda.is_available() else "cpu"

example_input = torch.randn(1, 3, 32, 32).to(device)

# load le modèle non pruné
checkpoint = torch.load('models/vgg.cifar.pretrained.pth')
model = VGG().to(device)
model.load_state_dict(checkpoint["state_dict"])
model.eval()
print('résultats du modèle non pruné:')
benchmark(model, example_input)

print('')

# load le modèle pruné 
model = torch.load('results/test.pth')
model.eval()
model.to(device)
print('résultats du modèle pruné:')
benchmark(model, example_input)