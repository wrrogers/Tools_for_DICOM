import os
import math
import json
import numpy as np
from keras.utils import np_utils
import pandas as pd
import pydicom as dicom
import matplotlib.pyplot as plt
from ipywidgets import FloatProgress
from IPython.display import display
import cv2
from sklearn.preprocessing import MinMaxScaler


basepath = 'C:/Users/wrrog/Desktop/Data/'
patient_list = {}

#Thanks to Howard Chen https://www.raddq.com/dicom-processing-segmentation-visualization-in-python/
def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices

#Thanks to Howard Chen https://www.raddq.com/dicom-processing-segmentation-visualization-in-python/
def get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 1
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope
    
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
        
    image += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)

# Simly returns a sample stak of images
def sample_stack(stack, rows=6, cols=6, start_with=10, show_every=3):
    fig,ax = plt.subplots(rows,cols,figsize=[12,12])
    for i in range(rows*cols):
        ind = start_with + i*show_every
        ax[int(i/rows),int(i % rows)].set_title('slice %d' % ind)
        ax[int(i/rows),int(i % rows)].imshow(stack[ind],cmap='gray')
        ax[int(i/rows),int(i % rows)].axis('off')
    plt.show()

# Retrieves the contours from a given .dcm file
def get_contours(seq,z):
    #ctrs[0]
    #len(ctrs) #Length of 1, all data is in [0]
    #len(ctrs[0].ContourSequence) #The squence is 21 Deep
    #len(ctrs[0].ContourSequence[0].ContourData) #first sequence has 180 numbers
    # The sequence has 3 numbers reating, (x, y, z?)
    #print("Getting countours for sequence", z+1)
    ctrs = np.array(seq.ContourData)
    ctrs3d = []
    for c in range(1,int(len(ctrs)/3)+1):
        ctrs3d.append(list(ctrs[c*3-3:c*3]))
        
    ctrs3d = np.array(ctrs3d)
    # Lets remove the z postion
    #ctrs2d = ctrs3d[:,:2]
    return ctrs3d

# Retrieves all contours from all .dcm files in directories
def get_all_contours(path):
    ds = dicom.read_file(os.path.join(path), force=True)
    ctrs = ds.ROIContourSequence
    ctrs_stack = []
    #print(ctrs)
    #print("There a total of", len(ctrs[0].ContourSequence),"slices of contours")
    for z, c in enumerate(ctrs[0].ContourSequence):
        #print("Getting contours for z number:", z)
        ctrs_slice = get_contours(c,z)
        ctrs_stack.append(ctrs_slice)

    return ctrs_stack

# This is a python generator that pulls each patient data one at a time
def gen_patient(p):
    patients = os.listdir(p)
    patients = sorted(patients)
    skip_list = pd.read_csv(basepath+"ignore.csv", header = None).iloc[:,0].tolist()
    #skip_list.append('LUNG1-176')
    
    for n, patient in enumerate(patients):
        if patient in skip_list:
            path = os.path.join(p, patient)
            nextdir = os.listdir(path)
            nextdir2 = os.listdir(os.path.join(path,nextdir[0]))
    
            missing_ctrs = []
            if len(nextdir2) > 1: # There are some folders that seem to be missing contour data
                folder1 = os.listdir(os.path.join(path,nextdir[0],nextdir2[0]))
                #folder2 = os.listdir(os.path.join(path,nextdir[0],nextdir2[1]))
        
                contents = {'contour':os.path.join(path,nextdir[0],nextdir2[1]),
                            'image':  os.path.join(path,nextdir[0],nextdir2[0])}
                
                if len(folder1)>1:
                    contents = {'contour':os.path.join(path,nextdir[0],nextdir2[1]),
                                'image':  os.path.join(path,nextdir[0],nextdir2[0])}
                else:
                    contents = {'contour':os.path.join(path,nextdir[0],nextdir2[0]),
                                'image':  os.path.join(path,nextdir[0],nextdir2[1])}
                
                patient_list[patient] = contents
                yield patient, contents
            else:
                missing_ctrs.append(np.array([nextdir, nextdir2]))
        np.save('missing_ctrs', np.asarray(missing_ctrs))
            
# This is to retrieve the paths for a patient (to test errors in data)
def get_patient(p, patient):
    path = os.path.join(p, patient)
    nextdir = os.listdir(path)
    nextdir2 = os.listdir(os.path.join(path,nextdir[0]))

    if len(nextdir2) > 1: # There are some folders that seem to be missing contour data
        folder1 = os.listdir(os.path.join(path,nextdir[0],nextdir2[0]))
        #folder2 = os.listdir(os.path.join(path,nextdir[0],nextdir2[1]))

        contents = {'contour':os.path.join(path,nextdir[0],nextdir2[1]),
                    'image':  os.path.join(path,nextdir[0],nextdir2[0])}
        
        if len(folder1)>1:
            contents = {'contour':os.path.join(path,nextdir[0],nextdir2[1]),
                        'image':  os.path.join(path,nextdir[0],nextdir2[0])}
        else:
            contents = {'contour':os.path.join(path,nextdir[0],nextdir2[0]),
                        'image':  os.path.join(path,nextdir[0],nextdir2[1])}
        
        return contents

# Retrieves all data
def get_data(p, verbose = None):
    file = '000000.dcm'
    stack = get_all_contours(os.path.join(p['contour'],file))
    stack2d = [x[:,0:2] for x in stack] # Get all of the positions
    positions = [x[0,2] for x in stack] # Get all of the positions
    path = p['image']
    patient = load_scan(path)
    
    slices = []
    for p in patient:
        if p.ImagePositionPatient[2] in positions:
            slices.append(p)
            
    image = np.stack([s.pixel_array for s in slices])
      
    #I MADE THE FIXES HERE IN THIS FOR LOOP
    for n, s in enumerate(stack2d):
        try:
            s -= slices[n].ImagePositionPatient[0:2] #Subtract the base vector
            s /= [float(slices[n].PixelSpacing[0]), float(slices[n].PixelSpacing[1])] #Correct for spacing
        except IndexError:
            if verbose == True:
                print("There was an IndexError in the get_data function on", path)
            else:
                pass
    
    stack2d = stack2d[::-1] #Put it in the right order as the loop added it in backwards
    
    return image, stack2d

# Passes images and contours and returns 3d masked images
def maskit(images, ctrs2d):
    masked3d = []
    for img, c in zip(images, ctrs2d):
        #print("img",img.shape)
        dst = np.zeros(shape = (512, 512), dtype = 'float32') # An empty mask for CV2 to compute the normalization
        normalized = cv2.normalize(img,dst,0.0,1.0,cv2.NORM_MINMAX,cv2.CV_32FC1)
        #scaler = MinMaxScaler()
        #normalized = scaler.fit_transform(img)
        #print("new",normalized.shape)
        mask = np.zeros((512, 512), dtype=np.int8) # Create an empty mask
        polys = np.array([c] , np.int16) # Convert all the floats to ints (for the sake of plotting)
        # Use cv2 to create the mask using the contours
        for i in polys.tolist():
            cv2.fillConvexPoly(mask, np.array(i), 1)
    
        masked = normalized * mask # Done using a simple multiplication (ones don't change, 0s turn black)
        masked3d.append(masked)
    masked3d = np.array([masked3d])
    masked3d = masked3d[0]
    return masked3d

# Takes in an a 3d masked image and returns it cropped of emptry space around the edges
def cropit(arr):
    zs = []
    # Get all slices on the z axis with no data
    for z in range(arr.shape[0]):
        if arr[z,:,:].sum() == 0:
            zs.append(z)
    
    arr = np.delete(arr,zs,axis=0) # Delete the slices on the z axis

    xs = []     
    # Get all slices on the x axis with no data       
    for x in range(arr.shape[1]):
        if arr[:,x,:].sum() == 0:
            xs.append(x) # Delete the slices on the x axis
    
    arr = np.delete(arr,xs,axis=1)

    ys = []
    # Get all slices on the y axis with no data           
    for y in range(arr.shape[2]):
        if arr[:,:,y].sum() == 0:
            ys.append(y) # Delete the slices on the y axis
    
    arr = np.delete(arr,ys,axis=2)
    
    return arr

# Takes in an a 3d masked image and returns it cropped of emptry space around the edges
def cropitXY(arr, min_image_size = 100):
    zs = []
    # Get all slices on the z axis with no data
    for z in range(arr.shape[0]):
        if arr[z,:,:].sum() < min_image_size: # <---------- Delete slices with small images
            zs.append(z)
    
    arr = np.delete(arr,zs,axis=0) # Delete the slices on the z axis

    xs = []     
    # Get all slices on the x axis with no data       
    for x in range(arr.shape[1]):
        if arr[:,x,:].sum() == 0:
            xs.append(x) # Delete the slices on the x axis
    
    arr = np.delete(arr,xs,axis=1)

    ys = []
    # Get all slices on the y axis with no data           
    for y in range(arr.shape[2]):
        if arr[:,:,y].sum() == 0:  
            ys.append(y) # Delete the slices on the y axis
    
    arr = np.delete(arr,ys,axis=2)
    
    return arr

# Takes in the cropped 3d images and adds padding to make them all the same size 
def pad(cropped, maxes, max):
    zf_max, yf_max, xf_max = (maxes[0], maxes[1], maxes[2]) # The destination size of dimensions
    f = FloatProgress(min = 0, max = max, description = 'Padding:')
    display(f)
    pad_stack = []
    for image in cropped:
        #Subtracting the destination from the current size to find how much padding is needed
        z_diff, y_diff, x_diff = zf_max - image.shape[0], yf_max - image.shape[1], xf_max - image.shape[2]
        z_odd = z_diff%2
        y_odd = y_diff%2
        x_odd = x_diff%2
    
        # Pad all the dimensions as equally as possible on each side (6 sides in a cube)
        if z_odd:
            z_pad = (int((z_diff - 1) / 2)+1, (int((z_diff - 1) / 2)))
        else:
            z_pad = (int(z_diff / 2), int(z_diff / 2)) 
     
        if y_odd:
            y_pad = (int((y_diff - 1) / 2)+1, (int((y_diff - 1) / 2)))
        else:
            y_pad = (int(y_diff / 2), int(y_diff / 2))                 
        
        if x_odd:
            x_pad = (int((x_diff - 1) / 2)+1, (int((x_diff - 1) / 2)))
        else:
            x_pad = (int(x_diff / 2), int(x_diff / 2))
            
        pad_stack.append(np.pad(image, [z_pad, y_pad, x_pad], mode='constant'))
        f.value += 1
    
    pad_stack = np.array([pad_stack])[0] # Convert to a numpy array
    return pad_stack

# Takes in the cropped 3d images and adds padding to make them all the same size 
def padXY(cropped, maxes, max):
    zf_max, yf_max, xf_max = (maxes[0], maxes[1], maxes[2]) # The destination size of dimensions
    f = FloatProgress(min = 0, max = max, description = 'Padding:')
    display(f)
    pad_stack = []
    for image in cropped:
        #Subtracting the destination from the current size to find how much padding is needed
        y_diff, x_diff = yf_max - image.shape[1], xf_max - image.shape[2]
        #z_odd = z_diff%2
        y_odd = y_diff%2
        x_odd = x_diff%2
    
        # Pad all the dimensions as equally as possible on each side (6 sides in a cube)
        #if z_odd:
        #    z_pad = (int((z_diff - 1) / 2)+1, (int((z_diff - 1) / 2)))
        #else:
        #    z_pad = (int(z_diff / 2), int(z_diff / 2)) 
     
        if y_odd:
            y_pad = (int((y_diff - 1) / 2)+1, (int((y_diff - 1) / 2)))
        else:
            y_pad = (int(y_diff / 2), int(y_diff / 2))                 
        
        if x_odd:
            x_pad = (int((x_diff - 1) / 2)+1, (int((x_diff - 1) / 2)))
        else:
            x_pad = (int(x_diff / 2), int(x_diff / 2))
            
        # I MADE SOME CHANGES HERE!
        # first by not padding the z axis
        imgs = np.pad(image, [(0,0), y_pad, x_pad], mode='constant')
        # second by resizing the image for input into a pretrained model
        new_imgs = []
        for img in imgs:
            resized = cv2.resize(img.astype(np.float32), (224, 224), interpolation = cv2.INTER_AREA)
            # Repeat the image to simulate 3 color channels
            new_imgs.append(np.stack((resized,)*3, -1))
        new_imgs = np.asarray(new_imgs)
        pad_stack.append(new_imgs) # Not padding Z
        f.value += 1
    
    pad_stack = np.array([pad_stack])[0] # Convert to a numpy array
    return pad_stack

# Takes in a path and processes the masking and cropping
def mask_crop(path, max = None, verbose = None):
    patients = gen_patient(os.path.join(path, 'NSCLC-Radiomics'))
    if max == None:
        f = FloatProgress(min = 0, max = len(patients), description='Cropping:')
    else:
        f = FloatProgress(min = 0, max = max, description='Cropping:')
    display(f)
    xf_max, yf_max, zf_max = (0,0,0)
    crop_stack = []
    # Cycle through each patient so the they can be masked and cropped
    for n, pat in enumerate(patients):
        p, c = pat 
        if n >= max: break
        image, contours = get_data(c)
        masked = maskit(image, contours)
        cropped = cropit(masked)
        crop_stack.append(cropped)
        # Store the maximum sizes of each dimension
        if xf_max < cropped.shape[2]:
            xf_max = cropped.shape[2]
            #print("X value new max of:", xf_max, "<-------")
        else:
            xf_max
            
        if yf_max < cropped.shape[1]:
            yf_max = cropped.shape[1] 
            #print("Y value new max of:", yf_max, "<-------")
        else:
            yf_max
        
        if zf_max < cropped.shape[0]:
            zf_max = cropped.shape[0]
            #print("Z value new max of:", zf_max, "<-------")
        else:
            zf_max
        
        if verbose == True:
            print(p)
        f.value +=1 # This is for the progress bar
        
    return crop_stack, (zf_max, yf_max, xf_max)

# Takes in a path and processes the masking and cropping
def mask_cropXY(path, max = None, verbose = None):
    patients = gen_patient(os.path.join(path, 'NSCLC-Radiomics'))
    if max == None:
        f = FloatProgress(min = 0, max = len(patients), description='Cropping:')
    else:
        f = FloatProgress(min = 0, max = max, description='Cropping:')
    display(f)
    xf_max, yf_max, zf_max = (0,0,0)
    crop_stack = []
    # Cycle through each patient so the they can be masked and cropped
    for n, pat in enumerate(patients):
        p, c = pat 
        if n >= max: break
        image, contours = get_data(c)
        masked = maskit(image, contours)
        cropped = cropitXY(masked)
        crop_stack.append(cropped)
        # Store the maximum sizes of each dimension
        if xf_max < cropped.shape[2]:
            xf_max = cropped.shape[2]
            #print("X value new max of:", xf_max, "<-------")
        else:
            xf_max
            
        if yf_max < cropped.shape[1]:
            yf_max = cropped.shape[1] 
            #print("Y value new max of:", yf_max, "<-------")
        else:
            yf_max
        
        if zf_max < cropped.shape[0]:
            zf_max = cropped.shape[0]
            #print("Z value new max of:", zf_max, "<-------")
        else:
            zf_max
            
        if verbose == True:
            print(p)
        f.value +=1 # This is for the progress bar
        
    return crop_stack, (zf_max, yf_max, xf_max)


# All put together
def process_all(path, max = 422, verbose = None):
    cropped, maxes = mask_crop(path, max)
    padded = pad(cropped, maxes, len(cropped))
    return padded

def process_all2(path, max = 422, verbose = None):
    cropped, maxes = mask_cropXY(path, max, verbose)
    padded = padXY(cropped, maxes, len(cropped))
    return padded

def pwriter(path,string):
    f = open(path+'/pat_paths.json', 'w')
    f.write(string)
    f.close()
    
def preader(path):
    f = open(path+'pat_paths.json', 'r')
    file = f.read()
    return file

def getXy(path, file, trim = None):
    trim = True
    data = np.load(os.path.join(basepath,file))
    #data = np.load(os.path.join(path,file))
    
    patient_list = json.loads(preader(basepath)) # These are records with out Contour Data
    target = pd.read_csv(basepath+'Lung1.clinical.csv')
    target = target[target['PatientID'].isin(patient_list.keys())]
    
    survivedU2 = target[(target['Survival.time'] < 730)]
    survivedU2 = survivedU2[survivedU2['deadstatus.event'] == 0]
                        
    # Lets save a list of files to ignore during preprocessing
    
    # Let's remove what we can't use from our list
    survivedU2 = survivedU2['PatientID'].tolist()
    target = target[~target["PatientID"].isin(survivedU2)]
    
    binary = lambda x: 0 if x >= 730 else 1
    target['twoyearstatus.event'] = target['Survival.time'].apply(binary)
    
    # There is too many with a deathstatus of one, so lets pull the grey area data
    if trim:
        target = target.reset_index()
        mask = target[(target['Survival.time'] < 500) | (target['Survival.time'] > 730)].index.tolist()
        data = data[mask]
        target = target.loc[mask]
    
    return data, target['twoyearstatus.event'].tolist(), target.as_matrix()

# This was for troubleshooting my network
def random_generator(shape, batch_size):
    while True:
        rand_X = []
        for i in range(batch_size): 
            rand_X.append(np.random.rand(shape[0], shape[1], shape[2]))
        #rand_y = np.random.randint(2, size = batch_size)
        rand_y_ohe = np_utils.to_categorical(np.random.randint(2, size = batch_size))
        rand_X = np.asarray(rand_X)
        rand_X = rand_X.reshape((len(rand_X),shape[0], shape[1], shape[2],1))
        yield rand_X, rand_y_ohe
        
# This was for troubleshooting my network
def random_generator2(shape, batch_size):
    while True:
        rand_X = []
        for i in range(batch_size):
            stack = []
            for j in range(3):
                stack.append(np.random.rand(shape[0], shape[1]))
            stack = np.asarray(stack).reshape((224,224,3))
            rand_X.append(np.asarray(stack))
            
        #rand_y = np.random.randint(2, size = batch_size)
        rand_y_ohe = np_utils.to_categorical(np.random.randint(2, size = batch_size))
        rand_X = np.asarray(rand_X)
        yield rand_X, rand_y_ohe
        
# 3Dto1D with PCA
def convert1d(img, c):
    new_img = img.reshape((img.shape[0]*img.shape[1]), img.shape[2])
    new_img = new_img.transpose()

    from sklearn.decomposition import PCA
    pca = PCA(n_components=c)
    pca.componrnt = True
    new_img = pca.fit_transform(new_img)

    return new_img

#data =  process_all2(basepath, verbose = True)
#np.load('missing_ctrs.npy')
#np.save("data2.npy", data)

#data[0].shape
#data[0][9].shape
#plt.imshow(data[0][9])

#img = load_scan('C:/Users/wrrog/Desktop/Data/NSCLC-Radiomics/LUNG1-012/01-01-2014-StudyID-42151/1-36694')
#ctrs = get_all_contours('C:/Users/wrrog/Desktop/Data/NSCLC-Radiomics/LUNG1-012/01-01-2014-StudyID-42151/1-40242/000000.dcm')

#plt.imshow(img[70].pixel_array)
#plt.plot(ctrs)


#gen = random_generator2((224, 224), 1000)
#X_test, y_test_ohe = next(gen)











