# Python Libraries
import pickle
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import keras
from keras import backend as K
# from keras.preprocessing.image import load_img, img_to_array
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
from keras.preprocessing.image import load_img, img_to_array, save_img,array_to_img
from keras.applications import inception_v3
import glob as glob
# Helper functions
import helper
from attack import PixelAttacker
import os, sys

matplotlib.style.use('ggplot')
np.random.seed(100)

import argparse

import numpy as np
import pandas as pd
from keras.datasets import cifar10
import pickle

# Custom Networks
from networks.lenet import LeNet
from networks.pure_cnn import PureCnn
from networks.network_in_network import NetworkInNetwork
from networks.resnet import ResNet
from networks.densenet import DenseNet
from networks.wide_resnet import WideResNet
from networks.capsnet import CapsNet

# Helper functions
from differential_evolution import differential_evolution
import helper

class PixelAttacker_JLCHEN:
    def __init__(self, models, data, class_names, dimensions=(32, 32)):
        # Load data and model
        self.models = models
        self.x_test, self.y_test = data
        self.class_names = class_names
        self.dimensions = dimensions

        network_stats, correct_imgs = helper.evaluate_models(self.models, self.x_test, self.y_test)
        self.correct_imgs = pd.DataFrame(correct_imgs, columns=['name', 'img', 'label', 'confidence', 'pred'])
        self.network_stats = pd.DataFrame(network_stats, columns=['name', 'accuracy', 'param_count'])

    def deprocess_image(self,x):
        # Util function to convert a tensor into a valid image.
        if K.image_data_format() == 'channels_first':
            x = x.reshape((3, x.shape[2], x.shape[3]))
            x = x.transpose((1, 2, 0))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 3))
        x /= 2.
        x += 0.5
        x *= 255.
        x = np.clip(x, 0, 255).astype('uint8')
        return x

    def preprocess_image(self, image_path):
        # Util function to open, resize and format pictures
        # into appropriate tensors.
        img = load_img(image_path)
        img = img_to_array(img)
        print('111', img.shape)
        img = np.expand_dims(img, axis=0)
        print('222', img.shape)
        img = inception_v3.preprocess_input(img)
        return img
    
    def preprocess_image2(self, img):
        img = np.expand_dims(img, axis=0)
        print('222', img.shape)
        img = inception_v3.preprocess_input(img)
        return img
    
    def predict_classes(self, xs, img, target_class, model, minimize=True):
        # Perturb the image with the given pixel(s) x and get the prediction of the model
        imgs_perturbed = helper.perturb_image(xs, img)
        predictions = model.predict(imgs_perturbed)[:,target_class]
        # This function should always be minimized, so return its complement if needed
        return predictions if minimize else 1 - predictions

    def attack_success(self, x, img, target_class, model, targeted_attack=False, verbose=False):
        # Perturb the image with the given pixel(s) and get the prediction of the model
        attack_image = helper.perturb_image(x, img)

        confidence = model.predict( inception_v3.preprocess_input(attack_image))[0]        
        predicted_class = np.argmax(confidence)
        
        # If the prediction is what we want (misclassification or 
        # targeted classification), return True
        if (verbose):
            print('Confidence:', confidence[target_class])
        if ((targeted_attack and predicted_class == target_class) or
            (not targeted_attack and predicted_class != target_class)):
            return True

    def attack(self, img, in_name, model, target=None, pixel_count=1, 
            maxiter=75, popsize=400, verbose=False, plot=False):
        # Change the target class based on whether this is a targeted attack or not
        targeted_attack = target is not None
        target_class = target if targeted_attack else self.y_test[img,0]
        
        # Define bounds for a flat vector of x,y,r,g,b values
        # For more pixels, repeat this layout
        dim_x, dim_y = self.dimensions
        bounds = [(0,dim_x), (0,dim_y), (0,256), (0,256), (0,256)] * pixel_count
        
        # Population multiplier, in terms of the size of the perturbation vector x
        popmul = max(1, popsize // len(bounds))
        
        my_deprocess = self.deprocess_image(self.x_test[img].copy())
        ori_test = self.x_test[img]
        # Format the predict/callback functions for the differential evolution algorithm
        predict_fn = lambda xs: self.predict_classes(
            xs, my_deprocess, target_class, model, target is None)
        callback_fn = lambda x, convergence: self.attack_success(
            x, my_deprocess, target_class, model, targeted_attack, verbose)
        
        # Call Scipy's Implementation of Differential Evolution
        attack_result = differential_evolution(
            predict_fn, bounds, maxiter=maxiter, popsize=popmul,
            recombination=1, atol=-1, callback=callback_fn, polish=False)

        # Calculate some useful statistics to return from this function
        attack_image = helper.perturb_image(attack_result.x, my_deprocess)[0]
        prior_probs = model.predict(np.array([self.x_test[img]]))[0]        
        prior_class = np.argmax(prior_probs)
        print('prior class:', prior_class)
                
        predicted_probs = model.predict(self.preprocess_image2(attack_image.copy()))[0]
        predicted_class = np.argmax(predicted_probs)
        actual_class = self.y_test[img,0]
        success = predicted_class != actual_class
        cdiff = prior_probs[actual_class] - predicted_probs[actual_class]
        
        if(success):
            out_path = '/home/imagenet_attacked/'            
            save_img(out_path+in_name,attack_image)                    
        # Show the best attempt at a solution (successful or not)
        #if plot:
            #attack_image = self.deprocess_image(attack_image)
            #helper.plot_image(attack_image, actual_class, self.class_names, predicted_class)

        return [model.name, pixel_count, img, actual_class, predicted_class, success, cdiff, prior_probs, predicted_probs, attack_result.x]

#model = keras.applications.MobileNet()
model = inception_v3.InceptionV3(weights='imagenet',include_top=True)
with open('data/imagenet_classes.pkl', 'rb') as f:
    class_names = pickle.load(f)
word_to_class = {i:w for i,w in enumerate(class_names)}
#print(word_to_class[605])
# print(class_names)
#class_names['Carassius auratus']
def preprocess_image(image_path):
    # Util function to open, resize and format pictures
    # into appropriate tensors.
    img = load_img(image_path)
    img = img_to_array(img)
    print('111', img.shape)
    img = np.expand_dims(img, axis=0)
    print('222', img.shape)
    img = inception_v3.preprocess_input(img)
    return img

in_path = '/home/pred_scuss/'
in_list = sorted(glob.glob(in_path + '*.png'))
# Should output /device:GPU:0
K.tensorflow_backend._get_available_gpus()

#for eps in range(0, len(filenames), 20):    
models = [model]
count = 0

for in_name in in_list:
    print(in_name)
    print('-'*10)
    processed_images = preprocess_image(in_name)
    head, t_name = os.path.split(in_name)
    labels = int(t_name.split('_')[1])
    test = processed_images, np.array([[labels]])
    attacker = PixelAttacker_JLCHEN(models, test, class_names, dimensions=(299, 299))
    #result = attacker.attack(0, t_name,model,maxiter=100, verbose=True, plot=True, pixel_count=10)
    result = attacker.attack(0, t_name,model,maxiter=70, verbose=True, plot=True, pixel_count=100)
 