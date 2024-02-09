import SimpleITK as sitk
#import pydicom
import numpy as np
#import cupy as cp
import os
import matplotlib.pyplot as plt
import itk
import itkwidgets 
import k3d
from k3d.colormaps import matplotlib_color_maps
#import open3d as o3d
import pandas as pd
#from scipy.spatial import KDTree
from math import *
#from scipy import spatial
import time
from tqdm import tqdm
from itkwidgets import view


def serie_reader(path):
    '''
    path : load dicom series by path to directory
    return volume
    '''
    reader1 = sitk.ImageSeriesReader()
    dicom_names1 = reader1.GetGDCMSeriesFileNames(path)
    reader1.SetFileNames(dicom_names1)
    image = reader1.Execute()
    image = sitk.Cast(image, image.GetPixelIDValue())
    return image

def get_spacing(image):
    '''
    image : volume
    return return spacing matrix
    '''
    spacing=np.array([[image.GetSpacing()[0], 0,0],
                      [0, image.GetSpacing()[1],0], 
                      [0,0, image.GetSpacing()[2]]]).astype(np.float32)
    return spacing


def get_direction(image):
    '''
    image : volume
    return direction matrix (direction cosin matrix)
    '''
    dcm = np.zeros([3,3])
    dcm[0][0] = image.GetDirection()[0]
    dcm[0][1] = image.GetDirection()[1]
    dcm[0][2] = image.GetDirection()[2]
    dcm[1][0] = image.GetDirection()[3]
    dcm[1][1] = image.GetDirection()[4]
    dcm[1][2] = image.GetDirection()[5]
    dcm[2][0] = image.GetDirection()[6]
    dcm[2][1] = image.GetDirection()[7]
    dcm[2][2] = image.GetDirection()[8]
    
    return dcm


def calcul_physicsCoo(ori, A, image):
    '''
    ori : [x,y,z] coordinates of the origin of volume in patient ref.
    A : dot product between spacing and DCM
    image : volume
    return list of coordinates and intensity value
    '''
    phyCoo=[]
    for i in tqdm(range(0,image.shape[0])):
        for j in range(image.shape[1]):
            for k in range(image.shape[2]):
                p_c=ori+np.dot(A, np.array([k,j,i]))
                phyCoo.append([p_c[0], p_c[1], p_c[2]])#, i])
    
    phyCoo=np.array(phyCoo)
    return phyCoo

def define_param(newres, coomin, coomax):
    '''
    newres : float, smallest resolution value
    coomin : [x,y,z] minimum coordinates
    coomax : [x,y,z] maximum coordinates
    
    return 
    new_origin : [x,y,z] new origin coordinates
    new_spacing : new spacing matrix
    dir_new : new direction matrix 
    '''
    new_origin= np.array([coomin[0], coomax[1], coomin[2]])
    x1= np.array([coomax[0], coomax[1], coomin[2]])
    y1= np.array([coomin[0], coomin[1], coomin[2]])
    z1= np.array([coomin[0], coomax[1], coomax[2]])
    
    
    new_spacing=np.array([[newres, 0,0],
                          [0, newres,0], 
                          [0,0, newres]]).astype(np.float32) #cp rajouter get()
    
    rx=np.sqrt((((x1[0]-new_origin[0])**2)+((x1[1]-new_origin[1])**2)+((x1[2]-new_origin[2])**2)))

    #calcul direct cosine matrix 

    #axe x
    Xx= (x1[0]-new_origin[0])/rx
    Xy= (x1[1]-new_origin[1])/rx
    Xz= (x1[2]-new_origin[2])/rx

    #axe y
    ry=np.sqrt((((y1[0]-new_origin[0])**2)+((y1[1]-new_origin[1])**2)+((y1[2]-new_origin[2])**2)))

    Yx= (y1[0]-new_origin[0])/ry
    Yy= (y1[1]-new_origin[1])/ry
    Yz= (y1[2]-new_origin[2])/ry

    #axe z
    rz=np.sqrt((((z1[0]-new_origin[0])**2)+((z1[1]-new_origin[1])**2)+((z1[2]-new_origin[2])**2)))
    Zx= (z1[0]-new_origin[0])/rz
    Zy= (z1[1]-new_origin[1])/rz
    Zz= (z1[2]-new_origin[2])/rz

    dir_new = np.zeros([3,3])
    dir_new[0][0] = Xx
    dir_new[0][1] = Xy
    dir_new[0][2] = Xz
    dir_new[1][0] = Yx
    dir_new[1][1] = Yy
    dir_new[1][2] = Yz
    dir_new[2][0] = Zx
    dir_new[2][1] = Zy
    dir_new[2][2] = Zz
    
    return new_origin, new_spacing, dir_new


    

def itk_view_from_simpleitk(image, sp, direction, origin):
    '''
    Get a view of an ITK image from a SimpleITK image.
    '''
    
    #np_view = sitk.GetArrayViewFromImage(image)
    print(image.shape)
    itk_view = itk.image_view_from_array(image)
    #print(sp.diagonal())
    itk_view.SetSpacing(sp)
    itk_view.SetOrigin(origin)
    
    #itkspacing = itk.matrix_from_array(sp)
    
    itkdir = itk.matrix_from_array(direction)
    
    itk_view.SetDirection(direction)
    #outputImageFileName='new_vol_test'
    
    return itk_view


def save_volume(array, res, origin, dcm, filename, vType):
    '''
    Save Volume
    
    '''
    result_image = sitk.Image(array.shape, vType)
    result_image = sitk.GetImageFromArray(array)
    
    result_image.SetSpacing((res, res, res))
    result_image.SetOrigin(origin)
    result_image.SetDirection(tuple(dcm.flatten()))
    
    # write the image
    sitk.WriteImage(result_image, filename)