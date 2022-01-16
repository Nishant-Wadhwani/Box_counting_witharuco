# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 15:07:32 2021

@author: NI20168201
"""

import os
import cv2
import sys
import csv
import time
import subprocess
import numpy as np
import cv2.aruco as aruco
from scipy.spatial import distance as dist
import glob

# Code to mask images


def get_center(corner):
    """
        Returns center of provided rectangle
    """

    x1 = corner[0][0][0]
    y1 = corner[0][0][1]

    x2 = corner[0][1][0]
    y2 = corner[0][1][1]

    x3 = corner[0][2][0]
    y3 = corner[0][2][1]

    x4 = corner[0][3][0]
    y4 = corner[0][3][1]

    x = int((x1+x2+x3+x4)/4)
    y = int((y1+y2+y3+y4)/4)

    return (x, y)


def order_points_old(pts):
    """
        Returns ordered coordinates
    """
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def pix_to_cm(box_all_cordinates, marker_length):
    """
        Pixel to distance calculator
    """

    pts = np.zeros((4, 2), dtype="float32")
    pts[0, :] = box_all_cordinates[0][0]
    pts[1, :] = box_all_cordinates[0][1]
    pts[2, :] = box_all_cordinates[0][2]
    pts[3, :] = box_all_cordinates[0][3]
    rect = order_points_old(pts)

    corner1 = rect[0, :]
    corner2 = rect[1, :]

    e_distance = dist.euclidean(corner1, corner2)
    one_pix_cm = marker_length / e_distance

    return one_pix_cm


if __name__ == "__main__":

    # Define dictionary
    k=0
    aruco_dict_pallet = aruco.Dictionary_get(aruco.DICT_6X6_1000)  # Pallet dictionary
    aruco_dict_rack = aruco.Dictionary_get(aruco.DICT_7X7_1000)  # Rack dictionary
    parameters = aruco.DetectorParameters_create()

    # Get pallet info
    pallet_width = 121.92  # In cm
    pallet_breadth = 121.92  # In cm
    pallet_info = [pallet_width, pallet_breadth]

    # Get aruco info
    sku_markerLength = 0.05  # 5 cm
    pallet_markerLength = 0.05  # 5 cm
    aruco_info = [sku_markerLength, pallet_markerLength]

    # Get current time stamp
    timeStamp = time.time()
    time_stamp = str(time.time())

    # Get current location
    
    outer_folder_name = ("Output/" +
                         "Mission_" +
                         time_stamp)
    os.mkdir(outer_folder_name) 
    
    # Create sub folder to save cropped images
    folder_name = outer_folder_name + "/" + "MaskedImages"
    os.mkdir(folder_name) 
    
    # Save "main folder" name into "time_stamp.txt"
    file1 = "Output/" +  "time_stamp.txt"
    file2 = open(file1,"w")
    file2.write(folder_name+"\n") 
 
    file2.close()
    # Open output file to write result
    output_file = outer_folder_name + "/" + "crop.csv"
    csv_file = open(output_file, 'w', newline='')

    # Write headings to out file
    fields = list(['Image_Name', 'Pallet_ID', 'Remark'])
    writer = csv.DictWriter(csv_file, fieldnames=fields)
    writer.writeheader()
    output = []
    # Get list of all images from specified location
    
    images = glob.glob("Input/*")
    images.sort()
    for img in images:
        output.append(img)
    # Mask each image - cover unwanted region
    j = 0
    print("Masking Images...\n")

    for image_name in output:

        # Get image name alone
        name_break = image_name.split('/')
        img_name = name_break[-1]
        name = img_name[0:len(img_name)-4]
        print("Image Name = ", img_name)

        # Read image
        frame = cv2.imread(image_name)
        # frame = cv2.blur(frame, (5, 5))
        
        # Convert image to gray scale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        """# Dispay gray scale image
        cv2.namedWindow("blackAndWhiteImage", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("blackAndWhiteImage", 600,600)
        cv2.imshow("blackAndWhiteImage", blackAndWhiteImage)
        cv2.waitKey(0)"""

        # Pallet aruco marker detection + Process pallet's aruco
        cornersP, idsP, rejectedImgPointsP = aruco.detectMarkers(
            gray,
            aruco_dict_pallet,
            parameters=parameters)  # Lists of ids and the corners

        """frame = aruco.drawDetectedMarkers(
            frame,
            cornersP,
            idsP) # Draw detected markers

        # Display the result
        cv2.namedWindow("Pallet Aruco", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Pallet Aruco", 600,600)
        cv2.imshow("Pallet Aruco", frame)
        cv2.waitKey(0)"""

        # Get total number of pallet ids detected
        length_pallet = len(cornersP)

        # Condition to check if any pallet is present
        # in image or not
        if length_pallet == 0:

            print("Error 1 - No pallet detected")

            # Write to csv file
            strr = "No Pallet Detected"
            row = [{'Image_Name': img_name,
                    'Pallet_ID': str(0),
                    'Remark': strr}]
            writer.writerows(row)

            continue

        # Mask unwanted region with respet to each
        # detected pallet marker id
        i = 0
        row = {}
        for corner_pallet in cornersP:

            print("Pallet Id = ", idsP[i][0])

            # Get it's co-ordinates
            (x, y) = get_center(corner_pallet)

            """# Just for visualization purpose
            cv2.putText(
                frame,
                str(i),
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                8,
                (0, 0, 255),
                15)"""

            # CM to Pixel conversion
            one_pix_cm = pix_to_cm(corner_pallet, pallet_markerLength*100)

            # Calculate image dimension that need to be masked
            length_pix = \
                int(pallet_breadth/one_pix_cm)
            height_Pix = \
                int(130/one_pix_cm)  # Assuming rack height is 140 meter
            offset_pix = \
                int(((pallet_markerLength*100)/2)/one_pix_cm)

            # Get top-left co-ordinate
            left_x1 = x - int(length_pix/2)
            top_y1 = y - height_Pix

            # Get bottom-right co-ordinate
            right_x2 = x + int(length_pix/2)
            bottom_y2 = y + 6*offset_pix

            # Get frame dimension
            h, w = frame.shape[:2]

            # Check if complete pallet is covered in image or not
            if ((left_x1 >= 0) and
                    (right_x2 <= w) and
                    (bottom_y2 > int(height_Pix/2))):

                # If depth is in negative,
                # consider depth is zero
                if top_y1 < 0:
                    top_y1 = 0

                # Get masked image
                mask = frame.copy()
                mask[top_y1:bottom_y2, left_x1:right_x2, :] = [0, 0, 0]
                mask[np.where(mask != [0, 0, 0])] = 255
                temp_img = cv2.add(mask, frame)

                # Display masked image
                cv2.namedWindow("temp_img ", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("temp_img ", 400,400)
                cv2.imshow("temp_img ", temp_img)
                cv2.waitKey(0)

                # write masked image
                
                strr = "Output/" +"Mission_" +time_stamp+ "/" + "MaskedImages/" + name + ".jpg"
                cv2.imwrite(strr, temp_img)
                
                print("Output/Mission_" + time_stamp + "/" + name + ".PNG")

                # Write to csv file
                row = [{'Image_Name': img_name,
                        'Pallet_ID': str(idsP[i][0])}]
                writer.writerows(row)

                # Increment image count
                k=k+1
                j = j+1

            else:

                print("No full pallet view for pallet id ", idsP[i][0])
                a = 0

                # Write to csv file
                strr = "No full pallet view - " + str(idsP[i][0])
                row = [{'Image_Name': img_name,
                        'Pallet_ID': str(0),
                        'Remark': strr}]
                writer.writerows(strr)

            # Increment value of "i"
            i = i+1

        print("-----------------------------\n")
    csv_file.close()
    print("Done")
    print("\n")
