# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 16:56:30 2020

@author: NI20168201
"""
#Necessary Libraries Imported
import cv2
import subprocess
import imutils
import numpy as np
import glob
import imagepath

def main(output,k,alpha):
    for img_name in output:

        text = "Non-Blurry"

        print("Image Name = ", img_name)

        img = cv2.imread(img_name)
        #Resizing the image to the mentioned width
        img_r = imutils.resize(img, width=500)
        #COnverting the colored image to gray scale one for determining the blurriness
        gray = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

        # Get edge sharpness
        #Laplacian Operator compute the Laplacian of the image and then 
        #return the focus measure, which is 
        #simply the variance of the Laplacian
        laplacian_var = cv2.Laplacian(gray, 50, cv2.CV_64F).var()

        if laplacian_var < 600: #Defining the threshold,
        # The Laplacian highlights regions of an image 
        #containing rapid intensity changes

            text = "Blurry"
            print("Blurry")
            alpha=1

        else:

            print("Non-Blurry")
            alpha=0

        cv2.putText(
            img,
            "{}: {:.2f}".format(text, laplacian_var),
            (500, 500),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (255, 0, 0),
            10)
        """
        If an image contains high variance then 
        there is a wide spread of responses, both edge-like 
        and non-edge like, representative of a normal, 
        in-focus image. But if there is very low variance, 
        then there is a tiny spread of responses, 
        indicating there are very little edges in the image. 
        As we know, the more an image is blurred, the less edges there are.
        """
        cv2.namedWindow("Img", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Img", 600, 600)
        cv2.imshow("Img", img)
        cv2.waitKey(0)
        
        #Saving the output image with bluriness detected 
        strr = "../WCCL/Blur_Output/"  + str(k) + "_1"+".jpg"
        cv2.imwrite(strr, img)
        k=k+1
      



if __name__ == "__main__":
    #Specifying the directory of the image
    # Images Location
    output=[]#Image List
    path = imagepath.test_img_dir

    path = glob.glob(path+"*")
    
    for img in path:
        output.append(img)
    print(output)
    k=0
    alpha=0
    main(output,k,alpha)