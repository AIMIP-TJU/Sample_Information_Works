import torch
from torchvision import transforms

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# DEVICE = 'cpu'
batch_size =  50   # VGG16:150; ResNet18:50; WRN-22-8 :25 
mission = 'classification' #extraction,classification
EPOCH = 100
num_workers = 8
lr_list = [0.01]  
lrf = 0.0001

dataset_names = ['NICO_Animal_test']

classifier_names = ['ResNet18']    #VGG16、ResNet18、WRN-22-8
method_names = ['random']
proportions = [1]

transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    "test": transforms.Compose([transforms.Resize(224),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
