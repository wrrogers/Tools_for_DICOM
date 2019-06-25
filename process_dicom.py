# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 14:48:28 2019

Data Generator from image files

@author: wrrog
"""

import scipy
from scipy.io import loadmat
import scipy.ndimage
from scipy.ndimage.interpolation import zoom
from scipy.ndimage.morphology import binary_dilation,generate_binary_structure
#from imageio import imwrite
from cv2 import imwrite, boundingRect, imread
from PIL import Image

from skimage import measure, morphology
from skimage.morphology import convex_hull_image

from sklearn.preprocessing import minmax_scale
from sklearn.metrics.pairwise import euclidean_distances

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
#import h5py
import os
import time

import SimpleITK as sitk
import pydicom as dicom

from multiprocessing.pool import ThreadPool, Pool
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import warnings
from functools import partial


def binarize_per_slice(image, spacing, intensity_th=-600, sigma=1, area_th=30, eccen_th=0.99, bg_patch_size=10):
    bw = np.zeros(image.shape, dtype=bool)
    
    print("...... image is shape", image.shape, image[0].shape)
    
    # prepare a mask, with all corner values set to nan
    image = np.asarray(image)
    image_size = image.shape[1]
    grid_axis = np.linspace(-image_size/2+0.5, image_size/2-0.5, image_size)
    x, y = np.meshgrid(grid_axis, grid_axis)
    d = (x**2+y**2)**0.5
    nan_mask = (d<image_size/2).astype(float)
    nan_mask[nan_mask == 0] = np.nan
    for i in range(image.shape[0]):
        # Check if corner pixels are identical, if so the slice  before Gaussian filtering
        if len(np.unique(image[i, 0:bg_patch_size, 0:bg_patch_size])) == 1:
            current_bw = scipy.ndimage.filters.gaussian_filter(np.multiply(image[i].astype('float32'), nan_mask), sigma, truncate=2.0) < intensity_th
        else:
            current_bw = scipy.ndimage.filters.gaussian_filter(image[i].astype('float32'), sigma, truncate=2.0) < intensity_th
        
        # select proper components
        label = measure.label(current_bw)
        properties = measure.regionprops(label)
        valid_label = set()
        for prop in properties:
            if prop.area * spacing[1] * spacing[2] > area_th and prop.eccentricity < eccen_th:
                valid_label.add(prop.label)
        current_bw = np.in1d(label, list(valid_label)).reshape(label.shape)
        bw[i] = current_bw
        
    return bw

def all_slice_analysis(bw, spacing, cut_num=0, vol_limit=[0.68, 8.2], area_th=6e3, dist_th=62):
    # in some cases, several top layers need to be removed first
    if cut_num > 0:
        bw0 = np.copy(bw)
        bw[-cut_num:] = False
    label = measure.label(bw, connectivity=1)
    # remove components access to corners
    mid = int(label.shape[2] / 2)
    bg_label = set([label[0, 0, 0], label[0, 0, -1], label[0, -1, 0], label[0, -1, -1], \
                    label[-1-cut_num, 0, 0], label[-1-cut_num, 0, -1], label[-1-cut_num, -1, 0], label[-1-cut_num, -1, -1], \
                    label[0, 0, mid], label[0, -1, mid], label[-1-cut_num, 0, mid], label[-1-cut_num, -1, mid]])
    for l in bg_label:
        label[label == l] = 0
        
    # select components based on volume
    properties = measure.regionprops(label)
    for prop in properties:
        if prop.area * spacing.prod() < vol_limit[0] * 1e6 or prop.area * spacing.prod() > vol_limit[1] * 1e6:
            label[label == prop.label] = 0
            
    # prepare a distance map for further analysis
    x_axis = np.linspace(-label.shape[1]/2+0.5, label.shape[1]/2-0.5, label.shape[1]) * spacing[1]
    y_axis = np.linspace(-label.shape[2]/2+0.5, label.shape[2]/2-0.5, label.shape[2]) * spacing[2]
    x, y = np.meshgrid(x_axis, y_axis)
    d = (x**2+y**2)**0.5
    vols = measure.regionprops(label)
    valid_label = set()
    # select components based on their area and distance to center axis on all slices
    for vol in vols:
        single_vol = label == vol.label
        slice_area = np.zeros(label.shape[0])
        min_distance = np.zeros(label.shape[0])
        for i in range(label.shape[0]):
            slice_area[i] = np.sum(single_vol[i]) * np.prod(spacing[1:3])
            min_distance[i] = np.min(single_vol[i] * d + (1 - single_vol[i]) * np.max(d))
        
        if np.average([min_distance[i] for i in range(label.shape[0]) if slice_area[i] > area_th]) < dist_th:
            valid_label.add(vol.label)
            
    bw = np.in1d(label, list(valid_label)).reshape(label.shape)
    
    # fill back the parts removed earlier
    if cut_num > 0:
        # bw1 is bw with removed slices, bw2 is a dilated version of bw, part of their intersection is returned as final mask
        bw1 = np.copy(bw)
        bw1[-cut_num:] = bw0[-cut_num:]
        bw2 = np.copy(bw)
        bw2 = scipy.ndimage.binary_dilation(bw2, iterations=cut_num)
        bw3 = bw1 & bw2
        label = measure.label(bw, connectivity=1)
        label3 = measure.label(bw3, connectivity=1)
        l_list = list(set(np.unique(label)) - {0})
        valid_l3 = set()
        for l in l_list:
            indices = np.nonzero(label==l)
            l3 = label3[indices[0][0], indices[1][0], indices[2][0]]
            if l3 > 0:
                valid_l3.add(l3)
        bw = np.in1d(label3, list(valid_l3)).reshape(label3.shape)
    
    return bw, len(valid_label)

def fill_hole(bw):
    # fill 3d holes
    label = measure.label(~bw)
    # idendify corner components
    bg_label = set([label[0, 0, 0], label[0, 0, -1], label[0, -1, 0], label[0, -1, -1], \
                    label[-1, 0, 0], label[-1, 0, -1], label[-1, -1, 0], label[-1, -1, -1]])
    bw = ~np.in1d(label, list(bg_label)).reshape(label.shape)
    
    return bw


def two_lung_only(bw, spacing, max_iter=22, max_ratio=4.8):    
    def extract_main(bw, cover=0.95):
        for i in range(bw.shape[0]):
            current_slice = bw[i]
            label = measure.label(current_slice)
            properties = measure.regionprops(label)
            properties.sort(key=lambda x: x.area, reverse=True)
            area = [prop.area for prop in properties]
            count = 0
            sum = 0
            while sum < np.sum(area)*cover:
                sum = sum+area[count]
                count = count+1
            filter = np.zeros(current_slice.shape, dtype=bool)
            for j in range(count):
                bb = properties[j].bbox
                filter[bb[0]:bb[2], bb[1]:bb[3]] = filter[bb[0]:bb[2], bb[1]:bb[3]] | properties[j].convex_image
            bw[i] = bw[i] & filter
           
        label = measure.label(bw)
        properties = measure.regionprops(label)
        properties.sort(key=lambda x: x.area, reverse=True)
        bw = label==properties[0].label

        return bw
    
    def fill_2d_hole(bw):
        for i in range(bw.shape[0]):
            current_slice = bw[i]
            label = measure.label(current_slice)
            properties = measure.regionprops(label)
            for prop in properties:
                bb = prop.bbox
                current_slice[bb[0]:bb[2], bb[1]:bb[3]] = current_slice[bb[0]:bb[2], bb[1]:bb[3]] | prop.filled_image
            bw[i] = current_slice

        return bw
    
    found_flag = False
    iter_count = 0
    bw0 = np.copy(bw)
    while not found_flag and iter_count < max_iter:
        label = measure.label(bw, connectivity=2)
        properties = measure.regionprops(label)
        properties.sort(key=lambda x: x.area, reverse=True)
        if len(properties) > 1 and properties[0].area/properties[1].area < max_ratio:
            found_flag = True
            bw1 = label == properties[0].label
            bw2 = label == properties[1].label
        else:
            bw = scipy.ndimage.binary_erosion(bw)
            iter_count = iter_count + 1
    
    if found_flag:
        d1 = scipy.ndimage.morphology.distance_transform_edt(bw1 == False, sampling=spacing)
        d2 = scipy.ndimage.morphology.distance_transform_edt(bw2 == False, sampling=spacing)
        bw1 = bw0 & (d1 < d2)
        bw2 = bw0 & (d1 > d2)
                
        bw1 = extract_main(bw1)
        bw2 = extract_main(bw2)
        
    else:
        bw1 = bw0
        bw2 = np.zeros(bw.shape).astype('bool')
        
    bw1 = fill_2d_hole(bw1)
    bw2 = fill_2d_hole(bw2)
    bw = bw1 | bw2

    return bw1, bw2, bw

def load_file(path):
    #print('Loading file', path)
    file = sitk.ReadImage(path)
    image = sitk.GetArrayFromImage(file)
    return image

def load_img(path):
    files = os.listdir(path)
    files = [os.path.join(path, f) for f in files]
    with ThreadPool() as p:
        img = np.asarray(list(p.imap(load_file, files))).squeeze()
    return img

def load_png(path, type = 'image'):
    if type == 'image':
        end = 'i.png'
    else:
        end = 'm.png'
    files = os.listdir(path)
    files = [f for f in files if f[-5:] == end]
    files = [os.path.join(path, f) for f in files]
    pool = Pool()
    img = pool.map(Image.open,files)
    img = [np.array(i) for i in img]
    img = np.asarray(img)
    if type == 'mask':
        img = img / 255
    return img

def getScanPath(path):
    cwdpath = os.getcwd()
    os.chdir(os.path.join(path))
    for root, dirs, files in os.walk(".", topdown=False):
        newpath = os.path.join(path, root)
        for name in files:
            ds = dicom.read_file(os.path.join(newpath, name), force=True)
            #print(ds.dir('Modality'))
            #print(ds.dir('SliceLocation'))
            if len(os.listdir(root)) > 60:
                if hasattr(ds, 'Modality'):
                    if ds.Modality == 'CT':
                        if hasattr(ds, 'SliceLocation'):
                            os.chdir(cwdpath)
                            return os.path.join(root)#, origin, spacing
                        else: continue
                    else: continue
                else: continue
            else: continue

def getFirstDicom(path):
    contents = os.listdir(path)
    for c in contents:
        if c[-4:] == '.dcm':
            return c
        else:
            continue

def stringToArray(string):
    elements = []
    position = 0
    while position != -1:
        previous = position
        position = string.find('\\', position+1)
        elements.append(float(string[previous:position].strip(' \\')))
    return np.array(elements)

def getDicomMeta(path):
    reader = sitk.ImageFileReader()
    first_file = getFirstDicom(path)
    filepath = os.path.join(path, first_file)
    reader.SetFileName(filepath)
    reader.LoadPrivateTagsOn();
    reader.ReadImageInformation()
    origin = np.array(list(reversed(reader.GetOrigin())))
    spacing = np.array(list(reversed(reader.GetSpacing())))
    taglib = {'Origin': origin, 'Spacing': spacing}
    for k in reader.GetMetaDataKeys():
        v = reader.GetMetaData(k)
        #print("({0}) == \"{1}\"".format(k,v))
        if k == "0020|000d": taglib['Study Instance UID'] = v
        if k == "0020|000e": taglib['Series Instance UID'] = v
        if k == "0020|0052": taglib['Frame of Reference UID'] = v
        if k == "0040|a124": taglib['UID'] = v
        if k == "0028|0030": taglib['Pixel Spacing'] = stringToArray(v)
        if k == "0020|0030": taglib['Image Position'] = stringToArray(v)
        if k == "0020|0032": taglib['Image Position Patient'] = stringToArray(v)
        if k == "0028|3004": taglib['Modality'] = v
        if k == "0020|1041": taglib['Slice Location'] = v
        if k == "0018|2005": taglib['Slice Location Vector'] = v
        
    return taglib

def processLungs(fpath, ipath = None, file_type = 'png'):
    print("Processing Lungs at ...\n", fpath, '\n')
    spath = getScanPath(fpath)
    print("CT Scan at ...\n", spath, '\n')
    path  = os.path.join(fpath, spath)
    if file_type.upper().strip('.') == 'PNG':
        case_pixels = load_png(ipath)
    else:
        case_pixels = load_img(path)
    meta = getDicomMeta(path)
    print("... image and meta data loaded.")
    bw = binarize_per_slice(case_pixels, meta['Spacing'])
    print("... image binarized.")
    flag = 0
    cut_num = 0
    cut_step = 2
    bw0 = np.copy(bw)
    while flag == 0 and cut_num < bw.shape[0]:
        bw = np.copy(bw0)
        bw, flag = all_slice_analysis(bw, meta['Spacing'], cut_num=cut_num, vol_limit=[0.68,7.5])
        cut_num = cut_num + cut_step

    bw = fill_hole(bw)
    print("... holes filled in mask.")
    bw1, bw2, bw = two_lung_only(bw, meta['Spacing'])
    print("... finished processing. \n")
    return case_pixels, bw1, bw2, meta['Spacing']

def plot_3d(image):
    
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)
    
    #verts, faces = measure.marching_cubes_classic(p, threshold)
    verts, faces = measure.marching_cubes_classic(p)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()
    
def process_mask(mask):
    convex_mask = np.copy(mask)
    for i_layer in range(convex_mask.shape[0]):
        mask1  = np.ascontiguousarray(mask[i_layer])
        if np.sum(mask1)>0:
            mask2 = convex_hull_image(mask1)
            if np.sum(mask2)>2*np.sum(mask1):
                mask2 = mask1
        else:
            mask2 = mask1
        convex_mask[i_layer] = mask2
    struct = generate_binary_structure(3,1)  
    dilatedMask = binary_dilation(convex_mask,structure=struct,iterations=10) 
    return dilatedMask

# def savenpy(id):
id = 1

def lumTrans(img):
    lungwin = np.array([-1200.,600.])
    newimg = (img-lungwin[0])/(lungwin[1]-lungwin[0])
    newimg[newimg<0]=0
    newimg[newimg>1]=1
    newimg = (newimg*255).astype('uint8')
    return newimg

def resample(imgs, spacing, new_spacing,order = 2):
    if len(imgs.shape)==3:
        new_shape = np.round(imgs.shape * spacing / new_spacing)
        true_spacing = spacing * imgs.shape / new_shape
        resize_factor = new_shape / imgs.shape
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imgs = zoom(imgs, resize_factor, mode = 'nearest',order=order)
        return imgs, true_spacing
    elif len(imgs.shape)==4:
        n = imgs.shape[-1]
        newimg = []
        for i in range(n):
            slice = imgs[:,:,:,i]
            newslice,true_spacing = resample(slice,spacing,new_spacing)
            newimg.append(newslice)
        newimg=np.transpose(np.array(newimg),[1,2,3,0])
        return newimg,true_spacing
    else:
        raise ValueError('wrong shape')

def savenpy(id,filelist,prep_folder,data_path,use_existing=True):      
    resolution = np.array([1,1,1])
    name = filelist[id]
    if use_existing:
        if os.path.exists(os.path.join(prep_folder,name+'_label.npy')) and os.path.exists(os.path.join(prep_folder,name+'_clean.npy')):
            print(name+' had been done')
            return
    try:
        im, m1, m2, spacing = processLungs(os.path.join(data_path,name))
        Mask = m1+m2
        
        newshape = np.round(np.array(Mask.shape)*spacing/resolution)
        xx,yy,zz= np.where(Mask)
        box = np.array([[np.min(xx),np.max(xx)],[np.min(yy),np.max(yy)],[np.min(zz),np.max(zz)]])
        box = box*np.expand_dims(spacing,1)/np.expand_dims(resolution,1)
        box = np.floor(box).astype('int')
        margin = 5
        extendbox = np.vstack([np.max([[0,0,0],box[:,0]-margin],0),np.min([newshape,box[:,1]+2*margin],axis=0).T]).T
        extendbox = extendbox.astype('int')

        #convex_mask = m1
        dm1 = process_mask(m1)
        dm2 = process_mask(m2)
        dilatedMask = dm1+dm2
        Mask = m1+m2
        extramask = dilatedMask ^ Mask
        bone_thresh = 210
        pad_value = 170

        im[np.isnan(im)]=-2000
        sliceim = lumTrans(im)
        sliceim = sliceim*dilatedMask+pad_value*(1-dilatedMask).astype('uint8')
        bones = sliceim*extramask>bone_thresh
        sliceim[bones] = pad_value
        sliceim1,_ = resample(sliceim,spacing,resolution,order=1)
        sliceim2 = sliceim1[extendbox[0,0]:extendbox[0,1],
                    extendbox[1,0]:extendbox[1,1],
                    extendbox[2,0]:extendbox[2,1]]
        sliceim = sliceim2[np.newaxis,...]
        np.save(os.path.join(prep_folder,name+'_clean'),sliceim)
        np.save(os.path.join(prep_folder,name+'_label'),np.array([[0,0,0,0]]))
    except:
        print('bug in '+name)
        raise
    print(name+' done')

def stringerate(number, length):
    number = str(number)
    strlen = len(number)
    zeros  = "0" * (length - strlen)
    return zeros + number
    
def save_imgs(arr, path, type = 'm', zeros = 4):
    if type == 'm':
        arr = arr.astype('float64')    
    for n, e in enumerate(arr):
        if type == 'm':
            e = minmax_scale(e, feature_range=(0,255))
        sav_path = os.path.join(path, 'img_{}_{}.png'.format(stringerate(n, zeros), type))
        #img = Image.fromarray(e)
        #rgb = img.convert('RGB')
        print(sav_path)
        imwrite(sav_path, e)
        #rgb.save(sav_path)

def getROI(mask):
    mask = mask.astype(dtype=np.uint8)
    xs, ys = [], []
    z = [0, mask.shape[0]]
    count = 0
    for n in range(mask.shape[0]):
        layer = mask[n, :, :]
        if np.sum(layer) == 0:
            count += 1
        if count == n:
            z[0] = count
            
        rect = boundingRect(layer)
        x = (rect[0], (rect[0]+rect[2]))
        y = (rect[1], (rect[1]+rect[3]))
        if x[0] > 0:
            xs.append(x)
        if y[0] > 0:
            ys.append(y)
    xs, ys = np.array(xs), np.array(ys)
    x = (min(xs[:,0]), max(xs[:,1]))
    y = (min(ys[:,0]), max(ys[:,1]))
    z[1] = count - z[0]
    z = tuple(z)
    return x, y, z

def getFalsePoints(img, mask, num_cubes, true_pos = [], rad = 25):
    #middle = 0
    points = []
    #sums = []
    for n in range(num_cubes):
        #print("... got point", n+1)
        while True:
            rand_x = random.randint(0, img.shape[2]-1)
            rand_y = random.randint(0, img.shape[1]-1)
            rand_z = random.randint(0, img.shape[0]-1)

            if len(true_pos) > 0:
                distances = euclidean_distances(true_pos, [[rand_x, rand_y, rand_z]])
                if np.min(distances) < 100:
                    continue
                
            #proba = random.randint(0,100)/100

            #sums.append(np.sum(mask[rand_z-rad:rand_z+rad, rand_y-rad:rand_y+rad, rand_x-rad:rand_x+rad]))
            #print("The sum is:", sums[n])
            #print("The random point is:", mask[rand_z, rand_y, rand_x])

            if mask[rand_z, rand_y, rand_x] == 1:
                #if np.sum(mask[rand_z-rad:rand_z+rad, rand_y-rad:rand_y+rad, rand_x-rad:rand_x+rad]) == 125000:    
                #    middle += 1
                point = [rand_z, rand_y, rand_x]
                points.append(point)
                break
            
    return points#, middle, np.array(sums)

def cubeEm(img, points, rad):
    cubes = []
    for point in points:
        cubes.append( img[point[0] - rad: point[0] + rad,
                          point[1] - rad: point[1] + rad,
                          point[2] - rad: point[2] + rad] )
    cubes = np.asarray(cubes)        
    return cubes

############################## MAIN ###########################################



if __name__ == "__main__":
    spath = r'G:\Data'
    fpath = r'D:\Data\MILDBL_ANON'
    #true_pos = [[130, 291, 276], [380, 218, 235], [104, 238, 198], [440, 218, 130]]
    
    ids = os.listdir(fpath)

    for id in ids:
        
        id_path = os.path.join(fpath, id)
        img, m1, m2, spacing = processLungs(id_path, file_type = '.dcm')
        mask = m1 + m2
        dmask = process_mask(mask)
        
        img = lumTrans(img)
        
        sav_path = os.path.join(spath, id)
        os.makedirs(sav_path)
        save_imgs(img, sav_path, type = 'i')
        save_imgs(mask, sav_path, type = 'm')

    #    if n > 5: break
    
    #img, m1, m2, spacing = processLungs(fpath, file_type = '.dcm')
    #mask = m1 + m2
    #dmask = process_mask(ms)
    
    #plot_3d(dmask)
    #save_imgs(ms, "f:/test")
    #points = getFalsePoints(img, mask, 1, true_pos = true_pos)
    #cubes = cubeEm(img, points, 25)
    
    #plt.imshow(cubes[0][25,:,:])
    
    
    