#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import pydicom
from pydicom import dcmread
import numpy as np
import cv2
import math
import pandas as pd
from scipy import ndimage
from math import pi
from scipy.ndimage.measurements import label
from scipy import signal
from datetime import datetime
import statsmodels.api as sm
import glob
import statsmodels


# In[2]:


def Reverse(tuples):
    new_tup = tuples[::-1]
    return new_tup

def sort_dicom_files(list_of_files):
    """
    This function sorts DICOM files in a directory based on their Slice Location.

    Args:
        list_of_files (list): The list of DICOM files.

    Returns:
        list: A list of sorted DICOM file.
    """
    # Try to extract the Slice Location header from the first file
    try:
        dcmread(list_of_files[0]).SliceLocation 
        slice_order_exists = True
    except AttributeError:
        slice_order_exists = False

    if slice_order_exists:
        # Go through each DICOM file and extract the Slice Location header
        slice_order = [float(dcmread(f).SliceLocation) for f in list_of_files]

        # Use zip to combine the lists, then sort by the slice order
        combined = sorted(zip(slice_order, list_of_files))

        # The sorted function returns a list of tuples, so you can use zip again to separate them back into two lists
        _, sorted_files = zip(*combined)
    else:
        # If SliceLocation does not exist, sort the files based on their filenames
        sorted_files = sorted(list_of_files)
    if dcmread(list_of_files[0]).Manufacturer == "Philips":
        sorted_files  = Reverse(sorted_files)

    return list(sorted_files)



def read_dicom_series(directory):
    """Reads a DICOM Series files in the given directory.
    Only filesnames matching filepattern will be considered"""
    print(directory)
    # lstFilesDCM = natsort.natsorted(glob.glob(os.path.join(directory, filepattern)))
    # removing dependancy on natsort. Will ask Ali later why he used it.
    lstFilesDCM = glob.glob(directory + "*.dcm")
    if not lstFilesDCM:
        lstFilesDCM = glob.glob(directory + "*.ima") #check for dicom files instead
    if not lstFilesDCM:
        lstFilesDCM = glob.glob(directory+"*")
        
    if not lstFilesDCM: #empty list = no files found in folder/zip file
        raise RuntimeError("Files not found! Check that there are either .IMA or .dcm files within the zip file.")
    # sort the files
    lstFilesDCM = sort_dicom_files(lstFilesDCM)

    # Get ref file
    RefDs = dcmread(lstFilesDCM[0])
    # Load dimensions based on the number of rows, columns, and slices (along the Z axis)
    ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))

    # The array is sized based on 'ConstPixelDims'
    ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)

    # loop through all the DICOM files
    for filenameDCM in lstFilesDCM:
        # read the file
        ds = dcmread(filenameDCM)
        # store the raw image data
        ArrayDicom[:, :, lstFilesDCM.index(filenameDCM)] = ds.pixel_array

    pixel_size = ds.PixelSpacing[0]
    phase_encoding = ds.InPlanePhaseEncodingDirection
    scan_date = ds.ContentDate
    return ArrayDicom, pixel_size, phase_encoding, scan_date



# In[3]:


def angle_profile_radial(image, image_profile, cX, cY):

    # Get indices where the profile image is not zero
    coords = np.argwhere(image_profile == 1)

    # Calculate the profile, distances, and store results
    X, Y = coords[:, 0], coords[:, 1]
    profile = image[X, Y]
    distances = np.sqrt((cX - X) ** 2 + (cY - Y) ** 2)

    # Create the DataFrame
    df = pd.DataFrame({
        'profile': profile,
        'X': X,
        'Y': Y,
        'distance': distances
    })

    # Sort the DataFrame by distance
    df.sort_values(by='distance', inplace=True, ascending=True, ignore_index=True)

    return df


# In[4]:


def template_profile_separate(spoke_number):
    '''
    This function generates expected reference profile depending on the spoke number
    '''
    profile = np.zeros((90,3))
    pixel_size = 0.5
    radi = [12/0.5, 25/0.5, 38/0.5] ## in milimeter
    D = [7, 6.39, 5.78, 5.17, 4.55, 3.94, 3.33, 2.72, 2.11, 1.5]
    d = D[spoke_number-1]
    for i,r in enumerate(radi):
        profile[round(r-d):round(r+d),i] = 1
    return profile


# In[5]:


def truncate_profile(p,M):
    ## This function gets a 1D array of profile (p) and a background threshold (M)
    ## and eliminate backgrounds from the begining and end of the profile
    ## and returns the index of the first and last element with higher value than M

    indices = np.where(p >= M)[0]
    
    # Return the first and last indices
    if indices.size == 0:
        return None, None  # Handle edge case where no value meets the condition
    idx_l, idx_h = indices[0], indices[-1]
    
    return idx_l, idx_h

def angle_image(alpha_degree,cX,cY,X,Y):
    '''
    This function gets an angle and generate a 1D profile of the image values along the angle
    '''
    Im = np.zeros((X, Y))
    alpha = np.radians(alpha_degree)

    if alpha_degree == 90:
        Im[cX, cY:Y] = 1  # Direct assignment for vertical line
    elif 45 <= abs(alpha_degree) < 135:
        y_range = np.arange(cY, Y)
        x_values = cX + (y_range - cY) / np.tan(alpha)
        valid_indices = (x_values >= 0) & (x_values < X-1)
        Im[np.round(x_values[valid_indices]).astype(int), y_range[valid_indices]] = 1
    elif 135 <= abs(alpha_degree) < 225:
        x_range = np.arange(cX, X)
        y_values = cY + np.tan(alpha) * (x_range - cX)
        valid_indices = (y_values >= 0) & (y_values < Y-1)
        Im[x_range[valid_indices], np.round(y_values[valid_indices]).astype(int)] = 1
    elif 225 <= abs(alpha_degree) < 315:
        y_range = np.arange(0, cY)
        x_values = cX + (y_range - cY) / np.tan(alpha)
        valid_indices = (x_values >= 0) & (x_values < X-1)
        Im[np.round(x_values[valid_indices]).astype(int), y_range[valid_indices]] = 1
    else:
        x_range = np.arange(0, cX)
        y_values = cY + np.tan(alpha) * (x_range - cX)
        valid_indices = (y_values >= 0) & (y_values < Y-1)
        Im[x_range[valid_indices], np.round(y_values[valid_indices]).astype(int)] = 1
    
    return Im


# In[6]:


def shift_for_max_corr(x,y,max_shift):
    '''
    This function slightly jitter the generated 1D profile to align with the reference profile of a specific spoke
    '''
    cc_vec = []
    shifts = np.arange(-max_shift, max_shift + 1)
    
    for shift in shifts:
        y_shifted = np.roll(y, shift)
        # Handle boundary conditions
        if shift < 0:
            y_shifted[shift:] = y_shifted[shift - 1]
        elif shift > 0:
            y_shifted[:shift] = y_shifted[shift - 1]
        
        # Compute correlation
        cc_vec.append(np.corrcoef(x, y_shifted)[0, 1])

    # Find the optimal shift with the maximum correlation
    optimal_shift_idx = np.argmax(cc_vec)
    optimal_shift = shifts[optimal_shift_idx]
    y_aligned = np.roll(y, optimal_shift)

    return x, y_aligned, optimal_shift


# In[7]:


def angle_stat_test(image_thresholded,angle,spoke_number,cX,cY):
    '''
    This function performes statistical tests for s specific angle
    '''

    # Normalize the image
    image_min, image_max = np.min(image_thresholded), np.max(image_thresholded)
    image_normalized = (image_thresholded - image_min) / (image_max - image_min)
    [X, Y] = image_thresholded.shape

    # Generate angle image and radial profile
    Im = angle_image(angle, cX, cY,X,Y)
    df = angle_profile_radial(image_normalized, Im, cX, cY)
    p = np.array(df['profile'])

    # Truncate the profile
    idx_l, idx_h = truncate_profile(p, 0.5)
    if idx_l is None or idx_h is None or idx_l >= idx_h:
        return None, None  # Handle invalid profiles gracefully

    # Process the truncated profile
    part = p[idx_l:idx_h]
    part_resampled = signal.resample(part, 90)
    part_short = part_resampled[:-5]

    # Polynomial detrending
    num = len(part_short)
    x_short = np.linspace(0, num, num)
    model = np.polyfit(x_short, part_short, 2)
    predicted = np.polyval(model, np.linspace(0, len(part_resampled), len(part_resampled)))
    part_detrended = part_resampled - predicted

    # Thresholding
    mean_part = np.mean(part_detrended)
    std_part = np.std(part_detrended)
    l_thr, h_thr = mean_part - 2 * std_part, mean_part + 2 * std_part
    part_detrended[-5:] = np.where(
        (part_detrended[-5:] < l_thr) | (part_detrended[-5:] > h_thr), 0, part_detrended[-5:]
    )

    # Smoothing
    kernel = np.ones(3) / 3  # Kernel size 3
    part_smoothed = np.convolve(part_detrended - mean_part, kernel, mode='same')

    # Cross-correlation for alignment
    profile_all = np.sum(template_profile_separate(spoke_number), axis=1)
    _, part_smoothed, _ = shift_for_max_corr(profile_all, part_smoothed, 5)

    # Prepare GLM
    y = part_smoothed.reshape(90, 1)
    predicted = predicted.reshape(90, 1)
    profiles_with_bias = np.append(template_profile_separate(spoke_number), np.ones((90, 1)), axis=1)

    # Fit the GLM
    glm_model = sm.GLM(y, profiles_with_bias)
    glm_results = glm_model.fit()

    # Return statistical results
    return glm_results.pvalues, glm_results.params


# In[8]:


def ACR_LCD(BIN_FILE):
    '''
    This function recieves the directory where the dicom images are store and generates results for the
    low contrast detectability test
    '''
    to_analyze = [10, 9, 8, 7]  # slices to analyze
    output_list = [['pass']*10]*4
    images_all, pixel_size, phase_encoding, scan_date = read_dicom_series(BIN_FILE) # Reading dicom files

    [X, Y, N] = images_all.shape

    angles = [-23, -16, -9, -2] # initial angles of the first spoke for each slice

    # Precompute threshold structure
    structure = np.ones((3, 3), dtype=np.int)
    for ind, slice_num in enumerate(to_analyze):
        image = images_all[:, :, slice_num]

        # Normalize the image
        image_normalized = image / np.max(image)

        #  thresholding to extract central disk
        for thr in np.arange(0.05, 0.65, 0.001):
            ret, thresh = cv2.threshold(image_normalized, thr, 1, 0)
            labeled, ncomponents = label(thresh, structure)
            thresh_inner = labeled == np.max(labeled)
            if np.sum(thresh_inner != 0) / np.sum(image_normalized != 0) > 0.1 and np.sum(thresh_inner != 0) / np.sum(image_normalized != 0) < 0.2:
                break

        # Fill the holes and apply threshold
        thresh_filled = ndimage.binary_fill_holes(thresh, structure=np.ones((5,5))).astype(int)
        image_thresholded = image * thresh_inner
        # find center of the disk
        M_inner = cv2.moments(np.uint8(thresh_inner))
        cY_inner = int(M_inner["m10"] / M_inner["m00"])
        cX_inner = int(M_inner["m01"] / M_inner["m00"])

        Im_all = np.zeros((X, Y))
        pass_vec = np.zeros(10)
        p_vals_vec = []
        params_vec = []
        angle_vec = []
        spoke_vec = []
        the_angle = angles[ind]

        for spoke_number, alpha_degree_initial in enumerate(range(the_angle, -360, -36)):
            for angle in np.arange(alpha_degree_initial - 8, alpha_degree_initial + 8, 1 / (1 + spoke_number)):
                p_vals, params = angle_stat_test(image_thresholded, angle, spoke_number + 1, cX_inner, cY_inner)
                params_vec.extend(params[0:3])
                p_vals_vec.extend(p_vals[0:3])
                spoke_vec.append(spoke_number)
                angle_vec.append(angle)

        # pefrom statistical analysis with FDR correction on all spokes of a slice
        p_vals_vec_fdr = statsmodels.stats.multitest.fdrcorrection(p_vals_vec, alpha=0.0125, method='indep', is_sorted=False)
        pvals_fdr = p_vals_vec_fdr[0].reshape(-1, 3)
        params_vec_fdr = np.array(params_vec).reshape(-1, 3)

        # check if all three p-values of a spoke pass significant threshold
        for i, g in enumerate(pvals_fdr):
            if np.sum(g) == 3 and np.sum(params_vec_fdr[i, :] > 0) == 3:
                Im = angle_image(angle_vec[i], cX_inner, cY_inner,X,Y)
                Im_all = np.logical_or(Im_all, Im)
                pass_vec[spoke_vec[i]] += 1

        # Create output image
        pass1 = np.sum(pass_vec != 0)
        image_normalized = (image_thresholded - np.min(image_thresholded)) / (np.max(image_thresholded) - np.min(image_thresholded))
        Im_combine = 0.1 * Im_all + image_normalized
        plt.figure(figsize = (7,7))
        plt.title("Slice: "+str(slice_num+1))
        plt.imshow(Im_combine)
        plt.show()

        strs = np.array(["Pass" for _ in range(len(pass_vec))])
        strs[pass_vec==0] = 'Fail'
        output_list[ind] = list(strs)

    df_index = ['Slice 11', 'Slice 10', 'Slice 9', 'Slice 8']

    df_output = pd.DataFrame(output_list,columns=['Spoke1','Spoke2','Spoke3','Spoke4','Spoke5','Spoke6','Spoke7','Spoke8','Spoke9'
                                  ,'Spoke10'],index = df_index)

    return df_output


# In[11]:


BIN_FILE = input("Directory of the ACR DICOM files: ")
#BIN_FILE = "L:/AG_MRI/ACR_lowcontrast_study/Datasets/Dataset01/"
output = ACR_LCD(BIN_FILE)
print(output)

# In[ ]:




