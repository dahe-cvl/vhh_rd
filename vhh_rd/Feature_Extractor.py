import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

class FeatureExtractor(nn.Module):
    def __init__(self, model_name, evaluate = True):
        super(FeatureExtractor, self).__init__()
        self.model_name = model_name

        print(model_name)
        if model_name == "resnet152":
            resnet152 = models.resnet152(pretrained=True)
            modules = list(resnet152.children())[:-1]                
            resnet152 = nn.Sequential(*modules)
            self.model = resnet152
        elif model_name == "resnet50":
            resnet50 = models.resnet50(pretrained=True)
            modules = list(resnet50.children())[:-1]
            resnet50 = nn.Sequential(*modules)
            self.model = resnet50
        else:
            print("Unknown model")
            quit()

        
        for p in self.model.parameters():
            p.requires_grad = False
        if evaluate:
            self.model.eval()

        print(self.model)

        if model_name == "resnet152":
            for name, child in self.model.named_children():
                # Unfreeze last trainable layer 
                # (layer 8 is just average pooling)
                if name == '7':
                    for param in child.parameters():
                        param.requires_grad = True

        elif model_name == "resnet50":
            print("IMPLEMENT LAYER UNFREEZING FOR RESNET50!")
            quit()

        

    def load_weights(self, modelPath):
        self.load_state_dict(torch.load(modelPath))

    def get_preprocessing(self, siamese = False):
        """
        If siamese is true then the image will not be turned to a tensor to allow augmentations
        """
        if self.model_name == "resnet152" or self.model_name == "resnet50":
            return transforms.Compose([
                transforms.ToTensor() if not siamese else torch.nn.Identity(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

    def forward(self, x):
        y = self.model(x)
        y = torch.squeeze(y, 2)
        y = torch.squeeze(y, 2)
        return y