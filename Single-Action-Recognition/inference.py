from PIL import Image
from torchvision import transforms
import torch
import cv2
test_transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
                    ])
class WarpModule(torch.nn.Module):
        def __init__(self, model) -> None:
            super().__init__()
            self.model = model

        def forward(self,  x):
            x = self.model(x)
            x = self.model.fc(x)
            return x

def load_model(path, num_classes):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=False)
    model.fc = torch.nn.Linear(1000, 46)
    model = WarpModule(model)

    model.load_state_dict(torch.load('best_mobile.pt'))
    model.eval()
    return model.to('cuda')

def predict_label(img = [], model= None):
        img_copy = img.copy()
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
        img_copy = Image.fromarray(img_copy)
        img_copy = test_transform(img_copy).to('cuda')
        model = model.to('cuda')
        img_copy = torch.unsqueeze(img_copy,dim = 0)
        predict = model(img_copy)
        predict = torch.softmax(predict, dim=1)
        predict = torch.argmax(predict,dim =1)

        return predict[0]
