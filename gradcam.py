import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
import sys

class FeatureExtractor():

    def __init__(self, model, target_layers):
        self.model = model
        self.model_features = model.features
        self.target_layers = target_layers
        self.gradients = list()
    def save_gradient(self, grad):
        self.gradients.append(grad)
    def get_gradients(self):
        return self.gradients
    def __call__(self, x):
        target_activations = list()
        self.gradients = list()
        for name, module in self.model_features._modules.items(): 
            x = module(x) #input 
            if name in self.target_layers: 
                x.register_hook(self.save_gradient)
                target_activations += [x] #features
        x = x.view(x.size(0), -1) #reshape
        x = self.model.classifier(x)
        return target_activations, x,

class GradCam():

    def __init__(self, model, target_layer_names):
        self.model = model


        self.extractor = FeatureExtractor(self.model, target_layer_names)
    def forward(self, input):
        return self.model(input)
    def __call__(self, input):
        features, output = self.extractor(input) 
        output.data
        one_hot = output.max() 
            
        self.model.features.zero_grad() 
        self.model.classifier.zero_grad() 
        one_hot.backward(retain_graph=True) 
        
        grad_val = self.extractor.get_gradients()[-1].data.numpy()
        
        
        target = features[-1] 
        
        target = target.data.numpy()[0, :] #(1, 512, 14, 14) > (512, 14, 14) 
        
        
        weights = np.mean(grad_val, axis = (2, 3))[0, :] 
        
        cam = np.zeros(target.shape[1:]) 
        
        
       
        for i, w in enumerate(weights): 
            cam += w * target[i, :, :] 

            
        cam = cv2.resize(cam, (224, 224)) 
        cam = cam - np.min(cam)
        cam = cam  / np.max(cam)
        return cam