import numpy as np
import torch
from torch import optim
from torchvision import models
from PIL import Image
import json
import random
import os
import argparse
from model_definition import Classifier

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = getattr(models, checkpoint['model'])(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer = optim.Adam(model.classifier.parameters())
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return model, optimizer

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Find the shorter side and resize it to 256 keeping aspect ration
    # find the shorter side
    if image.width > image.height:        
        # Constrain the height to be 256
        image.thumbnail((1000, 256))
    else:
        # Constrain the width to be 256
        image.thumbnail((256, 1000))
    
    #  crop out the center 224x224 portion
    width, height = image.size
    new_width, new_height = (224,224)
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2
    image = image.crop((left, top, right, bottom))
    
    # Convert values to range of 0 to 1
    np_image = np.array(image)
    np_image = np_image / 255
    
    # normalize the image
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - means) / stds
    
    # Move the color values to be accepted by pyrotch
    np_image = np_image.transpose(2, 0, 1)
    
    return np_image

def rand_sub_img(): 
    classes = np.arange(1, 102)
    selection = random.choice(classes)
    image_path = 'flowers/valid/'+selection.astype(str)+'/'
    dir_list = os.listdir(image_path)
    filename = random.choice(dir_list)
    path = os.path.join(image_path, filename)
    return selection, path

def img_class_search(json_path, search):
    with open(json_path, 'r') as f:
        cat_to_name = json.load(f)

    # Check the contents of cat_to_name
    for key in (cat_to_name.keys()):
        if int(key) == search:
            return cat_to_name[key]

def predict(image_path, model, topk, gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    #set mode to eval
    model.eval()

    device = "cpu"
    if gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    #load and process image to be fed into network
    image = Image.open(image_path)
    image = process_image(image)
    image = torch.from_numpy(image).type(torch.FloatTensor).cpu()
    image = image.unsqueeze(0) #need to add a dimension to the tensor for a 'batch'

    #get probabilities from network
    probs = torch.exp(model.forward(image.to(device)))
    probs, labels = probs.topk(topk)
    if gpu:
        probs = probs.cpu()
        labels = labels.cpu()
    probs = probs.detach().numpy().tolist()[0]
    indexes = labels.detach().numpy().tolist()[0]
    
    #now to convert the top predicted indicies to their propper classes
    labels = []
    idx_to_class_map = {}
    #first need to flip the class_to_idx so we can look up by indicie
    for key in model.class_to_idx:
        idx_to_class_map[model.class_to_idx[key]] = key
    for index in indexes:
        labels.append(idx_to_class_map[index])
    return probs, labels



def main():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--image_path', help='path to image', required=False)
    parser.add_argument('--checkpoint_path', help='dir where checkpoint should be saved', required=False)
    parser.add_argument('--top_k', help='top # of results', required=False)
    parser.add_argument('--json_path', help='path to json dict', required=False)
    parser.add_argument('--gpu', help='use gpu for prediction', action='store_true')
    parser.add_argument('--random', help='get a rondom image from default directory' ,action='store_true')
    args = vars(parser.parse_args())
    
    #defaults 
    image_path = './flowers/valid/1/image_06755.jpg'
    checkpoint_path = './checkpoint.pth'
    top_k = 5
    json_path = './cat_to_name.json'
    gpu = False
    random = False
    classification = 1 #deafault for selection. will be overwritten by --random
    
    #get args
    if args['image_path']:
        image_path = args['image_path']
        image_path = image_path.replace('\\','/') #for windows paths
    
    if args['checkpoint_path']:
        checkpoint_path = args['checkpoint_path']
        checkpoint_path = checkpoint_path.replace('\\','/') #for windows paths
        
    if args['top_k']:
        top_k = int(args['top_k'])
    if args['json_path']:
        json_path = args['json_path']
    if args['gpu']:
        gpu = True
    if args['random']:
        random = True
        
    model, optimizer = load_checkpoint(checkpoint_path)
    #only if we are using a random image
    if random:
        classification, image_path = rand_sub_img()
    #get predicitons
    probs, classes = predict(image_path, model, top_k, gpu)
    #translate correct classifications to their proper names
    translated_classes = []
    adjusted_probs = []
    for folder in classes:
        translated_classes.append(img_class_search(json_path, int(folder)))
    for prob in probs:
        adjusted_probs.append(round(prob*100,2))
    #print out predictions
    for i in range(len(probs)):
        print("{0:20} -----> {1}".format(translated_classes[i], adjusted_probs[i]))
    #only if we are using a random image
    if random:
        title = img_class_search(json_path, classification)
        print("path of image selected: ", image_path)
        print("true class: ", title)
        
    
    
    
    
main()