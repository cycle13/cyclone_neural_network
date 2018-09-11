from skimage.transform import rotate
import numpy as np
import random 
from PIL import Image

def augment(X_train, y_train, X_test, y_test, args, RANDOM_CROP_NUMBER, RESIZE_DIM, RANDOM_CROP_DIM):
    X_tr = X_train
    y_tr = y_train
    X_t = X_test
    y_t = y_test

    if 'translate' in args:
        y_train_new = []
        X_train_new = []
        y_test_new = []
        X_test_new = []

        i = 0
        while i < len(X_train):
            x_data = X_train[i]
            y_data = y_train[i]
            number_of_crops = RANDOM_CROP_NUMBER
            while number_of_crops > 0:
                y_crop = random.randint(0,31)
                x_crop = random.randint(0,31)
                y_train_new.append([y_data[0]-y_crop, y_data[1]-x_crop])
                X_train_new.append(x_data[y_crop:RESIZE_DIM-32+y_crop, 
                                    x_crop:RESIZE_DIM-32+x_crop])
                number_of_crops -= 1
            i+=1

        i = 0
        while i < len(X_test):
            x_data = X_test[i]
            y_data = y_test[i]
            y_crop = random.randint(0,31)
            x_crop = random.randint(0,31)
            y_test_new.append([y_data[0]-y_crop, y_data[1]-x_crop])
            X_test_new.append(x_data[y_crop:RESIZE_DIM-32+y_crop, 
                                x_crop:RESIZE_DIM-32+x_crop])
            i+=1

        X_tr = np.array(X_train_new)
        y_tr = np.array(y_train_new)
        X_t = np.array(X_test_new)
        y_t = np.array(y_test_new)

    if 'flip_h' in args:
        y_train_new = []
        X_train_new = []

        i = 0
        while i < len(X_train):
            x_data = X_train[i]
            y_data = y_train[i]
            
            y_train_new.append([y_data[0], x_data.shape[1] - 1 - y_data[1]])
            X_train_new.append(np.fliplr(x_data))
            y_train_new.append(y_data)
            X_train_new.append(x_data)

            i+=1
            
        X_tr = np.array(X_train_new)
        y_tr = np.array(y_train_new)
        
    if 'flip_v' in args:
        y_train_new = []
        X_train_new = []
       
        i= 0
        while i < len(X_train):
            x_data = X_train[i]
            y_data = y_train[i]
            
            y_train_new.append([x_data.shape[0] - 1 - y_data[0], y_data[1]])
            X_train_new.append(np.flipud(x_data))
            y_train_new.append(y_data)
            X_train_new.append(x_data)
            i+=1
            
        X_tr = np.array(X_train_new)
        y_tr = np.array(y_train_new)

    if 'rotate' in args:
        y_train_new = []
        X_train_new = []
       
        i= 0
        while i < len(X_train):
            original_shape = X_train[i].shape
            x_data = X_train[i].reshape(original_shape[:2])
            y_data = y_train[i]
            
            number_of_rotates = RANDOM_CROP_NUMBER

            while number_of_rotates > 0:
                angle = random.randint(0,360)
                image = Image.fromarray(x_data, 'L')
                rotated = Image.Image.rotate(image, angle)
                rotated_tensor = np.array(rotated)
            
                angle = np.deg2rad(angle)
                CENTRE = x_data.shape[0] / 2
                longitude = y_data[1]
                latitude = y_data[0]
                a = longitude - CENTRE 
                b = CENTRE - latitude
                longitude =  CENTRE + round(a * np.cos(angle) - b * np.sin(angle))
                latitude = CENTRE - round(b * np.cos(angle) + a * np.sin(angle))
                
                y_train_new.append([latitude, longitude])
                X_train_new.append(rotated_tensor.reshape(original_shape))
                
                number_of_rotates -= 1 
            
            y_train_new.append(y_data)
            X_train_new.append(x_data.reshape(original_shape))
            i+=1
            
        X_tr = np.array(X_train_new)
        y_tr = np.array(y_train_new)



    # Apply all augmentations
    if 'all' in args:
        
        # Translation
        y_train_new = []
        X_train_new = []
        y_test_new = []
        X_test_new = []

        i = 0
        while i < len(X_train):
            x_data = X_train[i]
            y_data = y_train[i]
            number_of_crops = RANDOM_CROP_NUMBER
            while number_of_crops > 0:
                y_crop = random.randint(0,31)
                x_crop = random.randint(0,31)
                y_train_new.append([y_data[0]-y_crop, y_data[1]-x_crop])
                X_train_new.append(x_data[y_crop:RESIZE_DIM-32+y_crop, 
                                    x_crop:RESIZE_DIM-32+x_crop])
                number_of_crops -= 1
            i+=1

        print(len(X_train_new))
        print('Done translating')

        i = 0
        while i < len(X_test):
            x_data = X_test[i]
            y_data = y_test[i]
            y_crop = random.randint(0,31)
            x_crop = random.randint(0,31)
            y_test_new.append([y_data[0]-y_crop, y_data[1]-x_crop])
            X_test_new.append(x_data[y_crop:RESIZE_DIM-32+y_crop, 
                                x_crop:RESIZE_DIM-32+x_crop])
            i+=1

        # Flip horizontal and vertical      
        y_train_new_new = []
        X_train_new_new = []

        i = 0
        while i < len(X_train_new):
            x_data = X_train_new[i]
            y_data = y_train_new[i]
            
            # Flip horizontal
            y_train_new_new.append([y_data[0], x_data.shape[1] - 1 - y_data[1]])
            X_train_new_new.append(np.fliplr(x_data))

            # Flip vertical
            y_train_new_new.append([x_data.shape[0] - 1 - y_data[0], y_data[1]])
            X_train_new_new.append(np.flipud(x_data))
            
            # Add original
            y_train_new_new.append(y_data)
            X_train_new_new.append(x_data)
            
            # Add rotation 90, 180, 270
            original_shape = X_train_new[i].shape
            
            if 'colour' in args:
                for angle in [90, 180, 270, random.randint(-20,20), random.randint(-20,20)]:
                    image = Image.fromarray(x_data, 'RGB')
                    rotated = Image.Image.rotate(image, angle)
                    rotated_tensor = np.array(rotated)
                
                    angle = np.deg2rad(angle)
                    CENTRE = x_data.shape[0] / 2
                    longitude = y_data[1]
                    latitude = y_data[0]
                    a = longitude - CENTRE 
                    b = CENTRE - latitude
                    longitude =  CENTRE + round(a * np.cos(angle) - b * np.sin(angle))
                    latitude = CENTRE - round(b * np.cos(angle) + a * np.sin(angle))
                    
                    y_train_new_new.append([latitude, longitude])
                    X_train_new_new.append(rotated_tensor.reshape(original_shape))

            else:
                x_data = X_train_new[i].reshape(original_shape[:2])
                
                for angle in [90, 180, 270, random.randint(-20,20), random.randint(-20,20)]:
                    image = Image.fromarray(x_data, 'L')
                    rotated = Image.Image.rotate(image, angle)
                    rotated_tensor = np.array(rotated)
                
                    angle = np.deg2rad(angle)
                    CENTRE = x_data.shape[0] / 2
                    longitude = y_data[1]
                    latitude = y_data[0]
                    a = longitude - CENTRE 
                    b = CENTRE - latitude
                    longitude =  CENTRE + round(a * np.cos(angle) - b * np.sin(angle))
                    latitude = CENTRE - round(b * np.cos(angle) + a * np.sin(angle))
                    
                    y_train_new_new.append([latitude, longitude])
                    X_train_new_new.append(rotated_tensor.reshape(original_shape))
            i+=1
            
        y_train_new = y_train_new_new
        X_train_new = X_train_new_new
 
        print(len(X_train_new))
        print('Done flips and rotations')
 
        X_tr = np.array(X_train_new)
        y_tr = np.array(y_train_new)
        X_t = np.array(X_test_new)
        y_t = np.array(y_test_new)

     # Apply all augmentations
    if 'classification' in args:
        
        # Translation
        X_train_new = []
        X_test_new = []
        y_train_new = []
        y_test_new = []

        i = 0
        while i < len(X_train):
            x_data = X_train[i]
            y_data = y_train[i]
            
            number_of_crops = RANDOM_CROP_NUMBER
            while number_of_crops > 0:
                y_crop = random.randint(0,31)
                x_crop = random.randint(0,31)
                X_train_new.append(x_data[y_crop:RESIZE_DIM-32+y_crop, 
                                    x_crop:RESIZE_DIM-32+x_crop])
                y_train_new.append(y_data)
                number_of_crops -= 1
            i+=1

        print(len(X_train_new))
        print('Done translating')

        i = 0
        while i < len(X_test):
            x_data = X_test[i]
            y_data = y_test[i]
            y_crop = random.randint(0,31)
            x_crop = random.randint(0,31)
            X_test_new.append(x_data[y_crop:RESIZE_DIM-32+y_crop, 
                                x_crop:RESIZE_DIM-32+x_crop])
            y_test_new.append(y_data)
            i+=1

        # Flip horizontal and vertical      
        X_train_new_new = []
        y_train_new_new = []
        
        i = 0
        while i < len(X_train_new):
            x_data = X_train_new[i]
            y_data = y_train_new[i]

            # Flip horizontal
            X_train_new_new.append(np.fliplr(x_data))
            y_train_new_new.append(y_data)

            # Flip vertical
            X_train_new_new.append(np.flipud(x_data))
            y_train_new_new.append(y_data)

            # Add original
            X_train_new_new.append(x_data)
            y_train_new_new.append(y_data)

            # Add rotation 90, 180, 270
            original_shape = X_train_new[i].shape
            
            if 'colour' in args:
                # for angle in [90, 180, 270, random.randint(-20,20), random.randint(-20,20)]:
                for angle in [90, 180, 270]:
                    image = Image.fromarray(x_data, 'RGB')
                    rotated = Image.Image.rotate(image, angle)
                    rotated_tensor = np.array(rotated)
                
                    angle = np.deg2rad(angle)
                    X_train_new_new.append(rotated_tensor.reshape(original_shape))
                    y_train_new_new.append(y_data)

            else:
                x_data = X_train_new[i].reshape(original_shape[:2])
                
                for angle in [90, 180, 270, random.randint(-20,20), random.randint(-20,20)]:
                    image = Image.fromarray(x_data, 'L')
                    rotated = Image.Image.rotate(image, angle)
                    rotated_tensor = np.array(rotated)
                
                    angle = np.deg2rad(angle)
                    X_train_new_new.append(rotated_tensor.reshape(original_shape))
                    y_train_new_new.append(y_data)
            i+=1
            
        X_train_new = X_train_new_new
        y_train_new = y_train_new_new
 
        print(len(X_train_new))
        print('Done flips and simple rotation')
        X_tr = np.array(X_train_new)
        y_tr = np.array(y_train_new)
        X_t = np.array(X_test_new)
        y_t = np.array(y_test_new)
    return X_tr, y_tr, X_t, y_t