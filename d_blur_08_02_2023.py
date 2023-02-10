# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 11:53:54 2023

@author: GKumar
"""

import cv2
import numpy as np
import pytesseract
import glob

def shrap_filter(image):
    # Compute the local mean
    bilateral = cv2.bilateralFilter(image, 9, 75, 75)
    #cv2.imshow('Bilateral Blurring', bilateral)

    kernel_sharpening = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]], np.float32)
    kernel = 1/3 * kernel_sharpening
    sharpened = cv2.filter2D(bilateral, -1, kernel)
    #cv2.imshow('Image Sharpening', sharpened)
    
    weightedSum = cv2.addWeighted(image, 0.3, sharpened, 0.7, 0)
    arr = np.asarray(weightedSum)
    avg = arr * 0.5
    #cv2.imshow('Avg Image', avg)
    
    return avg

def wiener_deconvolution(img, kernel):
    wiener = cv2.filter2D(img, -1, kernel)
    return wiener

def denoise_deblur_OCR(image):
    # Pre-processing
    img = shrap_filter(image)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    #cv2.imshow('blurred', blurred)
    
    # Deblur the image using Wiener deconvolution
    kernel = (4, 6)
    kernel = cv2.getOptimalDFTSize(kernel[0])

    deblurred = wiener_deconvolution(blurred, kernel)
    deblurred = cv2.convertScaleAbs(deblurred)
    #cv2.imshow('deblurred', deblurred)
    
    # Denoising
    denoised = cv2.fastNlMeansDenoising(deblurred, None, 20, 7, 21)
    cv2.imshow('denoised', denoised)
    gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    
    # Text extraction
    _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    license_plate = None
    for contour in contours:
        rect = cv2.boundingRect(contour)
        if rect[2] < 100 or rect[3] < 50:
            continue
        license_plate = denoised[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
        break
    
    # OCR
    text = pytesseract.image_to_string(license_plate, lang='eng')
    with open("output.txt", "w") as file:
        file.writelines(text)
    print(text)
    return text
for img in glob.glob('input/*.jpg'):
    img = cv2.imread(img)
    cv2.imshow('original', img)
    denoise_deblur_OCR(img)
    cv2.waitKey(5)
    cv2.destroyAllWindows()