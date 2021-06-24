
# Python Libraries
#%matplotlib inline
import pickle
import numpy as np
import pandas as pd
import scipy.misc
#from PIL import Image
import cv2 as cv
import os,sys
#import matplotlib
from keras.datasets import cifar10
from keras import backend as K

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
#from scipy.misc import imsave

#matplotlib.style.use('ggplot')
np.random.seed(100)


def perturb_image(xs, img):
    # If this function is passed just one perturbation vector,
    # pack it in a list to keep the computation the same
    if xs.ndim < 2:
        xs = np.array([xs])
    
    # Copy the image n == len(xs) times so that we can 
    # create n new perturbed images
    tile = [len(xs)] + [1]*(xs.ndim+1)
    imgs = np.tile(img, tile)
    
    # Make sure to floor the members of xs as int types
    xs = xs.astype(int)
    
    for x,img in zip(xs, imgs):
        # Split x into an array of 5-tuples (perturbation pixels)
        # i.e., [[x,y,r,g,b], ...]
        pixels = np.split(x, len(x) // 5)
        for pixel in pixels:
            # At each pixel's x,y position, assign its rgb value
            x_pos, y_pos, *rgb = pixel
            img[x_pos, y_pos] = rgb
    
    return imgs

def predict_classes(xs, img, target_class, model, minimize=True):
    # Perturb the image with the given pixel(s) x and get the prediction of the model
    imgs_perturbed = perturb_image(xs, img)
    predictions = model.predict(imgs_perturbed)[:,target_class]
    # This function should always be minimized, so return its complement if needed
    return predictions if minimize else 1 - predictions

def attack_success(x, img, target_class, model, targeted_attack=False, verbose=False):
    # Perturb the image with the given pixel(s) and get the prediction of the model
    attack_image = perturb_image(x, x_test[img])

    confidence = model.predict(attack_image)[0]
    predicted_class = np.argmax(confidence)
    
    # If the prediction is what we want (misclassification or 
    # targeted classification), return True
    if (verbose):
        print('Confidence:', confidence[target_class])
    if ((targeted_attack and predicted_class == target_class) or
        (not targeted_attack and predicted_class != target_class)):
        return True
    # NOTE: return None otherwise (not False), due to how Scipy handles its callback function

count = 0
import os
def attack(img, model,cls_id, case_path, target=None, pixel_count=1, 
           maxiter=75, popsize=400,verbose=False):
    # Change the target class based on whether this is a targeted attack or not
    targeted_attack = target is not None
    target_class = target if targeted_attack else y_test[img,0]
    
    # Define bounds for a flat vector of x,y,r,g,b values
    # For more pixels, repeat this layout
    bounds = [(0,32), (0,32), (0,256), (0,256), (0,256)] * pixel_count
    
    # Population multiplier, in terms of the size of the perturbation vector x
    popmul = max(1, popsize // len(bounds))
    
    # Format the predict/callback functions for the differential evolution algorithm
    predict_fn = lambda xs: predict_classes(
        xs, x_test[img], target_class, model, target is None)
    callback_fn = lambda x, convergence: attack_success(
        x, img, target_class, model, targeted_attack, verbose)
    
    # Call Scipy's Implementation of Differential Evolution
    attack_result = differential_evolution(
        predict_fn, bounds, maxiter=maxiter, popsize=popmul,
        recombination=1, atol=-1, callback=callback_fn, polish=False)

    # Calculate some useful statistics to return from this function
    attack_image = perturb_image(attack_result.x, x_test[img])[0]
    prior_probs = model.predict_one(x_test[img])
    predicted_probs = model.predict_one(attack_image)
    predicted_class = np.argmax(predicted_probs)
    actual_class = y_test[img,0]
    success = predicted_class != actual_class
    
#     if(success):
#         #count += 1
#         name = 'horrse_attacked_'+str(img)+'_'+str(actual_class) +'_'+str(predicted_class)+'.png'
#         save_success(attack_image,name)
    
    cdiff = prior_probs[actual_class] - predicted_probs[actual_class]
    if(predicted_probs[actual_class] < 0.5):
    # Show the best attempt at a solution (successful or not)
        #helper.plot_image(attack_image, actual_class, class_names, predicted_class)
        #saved
        
        cls_name = case_path + str(cls_id)+'_'+class_names[cls_id]        
        ori_name = cls_name + '/original/'+str(img) + '_' + str(actual_class) + '.png'
        ori_path = cls_name + '/original/'
        if not os.path.exists(ori_path):
            #os.makedirs(Annotations_path)
            os.system('mkdir -p %s' % (ori_path))
        #scipy.misc.imsave(ori_name, x_test[img])
        #result = Image.fromarray((x_test[img] * 255).astype(np.uint8))        
        #result.save(ori_name)
        cv.imwrite(ori_name, x_test[img])
        at_name = cls_name + '/attacked/'+str(img) +'_'+str(actual_class) +'_'+str(predicted_class)+'.png'
        at_path = cls_name + '/attacked/'
        if not os.path.exists(at_path):
            #os.makedirs(Annotations_path)
            os.system('mkdir -p %s' %(at_path))
        #scipy.misc.imsave(at_name, attack_image)
        #result = Image.fromarray((attack_image * 255).astype(np.uint8))
        #result.save(at_name)
        cv.imwrite(at_name, attack_image)
        #np.savetxt('horse_cor_'+str(img)+'.txt', attack_result.x,delimiter=',')
        #np.savetxt('test.out', x, delimiter=',')
        print("success:", prior_probs[actual_class], predicted_probs[actual_class])
    else:
        ok_cls_name =  case_path+str(cls_id)+'_'+class_names[cls_id]
        ok_name = ok_cls_name + '/OK/'+str(img) +'_' + str(actual_class)+ '.png'
        ok_path = ok_cls_name + '/OK/'
        if not os.path.exists(ok_path):
            #os.makedirs(Annotations_path)
            os.system('mkdir -p %s' %(ok_path))
        #scipy.misc.imsave(ok_name, x_test[img])
        #result = Image.fromarray((x_test[img] * 255).astype(np.uint8))
        #result.save(ok_name)
        cv.imwrite(ok_name,  x_test[img])
        
    # Show the best attempt at a solution (successful or not)
    #helper.plot_image(attack_image, actual_class, class_names, predicted_class)

    return [model.name, pixel_count, img, actual_class, predicted_class, success, cdiff, prior_probs, predicted_probs, attack_result.x]



(x_train, y_train), (x_test, y_test) = cifar10.load_data()
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

K.tensorflow_backend._get_available_gpus()
#lenet = LeNet()
#nin = NetworkInNetwork()
resnet = ResNet()
#densenet = DenseNet()
#models = [resnet]


pixels = int(sys.argv[1]) # Number of pixels to attack
model = densenet
case_path = 'resnet_data_p1/'
for i in range(10000):   
    print(i)
    cls = y_test[i][0]
    image = i    
    _ = attack(image, model, cls, case_path,pixel_count=pixels,verbose=True)

