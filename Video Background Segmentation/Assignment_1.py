#!/usr/bin/env python
# coding: utf-8

'''.........................Import module.....................................'''

import numpy as np
import cv2
import math
import random

'''........................Import and read video file........................'''


video = cv2.VideoCapture('umcp.mpg')
#return bool(True/False) and frame(width, hight and channels(R,G,B) info)
return_b, frame = video.read()
hight, width, channels=frame.shape #width=352, hight=240


'''...........................Hyperparameters................................'''

alpha=0.05 #Learning rate 
rho=0.1 #Weightage of new parameters
k=3

'''..........................K-Means clustering algorithm.........................'''

def KMC(frame, k):
    hight, width, channels = frame.shape
    mu = [[random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)],
          [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)],
         [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]]
    
    for em in range(20):  
    #Expectation step (calculation of r)
        a=np.zeros(k)
        r=np.zeros((hight,width,k))
        for x in range(0, hight,1):
            for y in range(0, width,1):
                for i in range(0, k, 1):
                    for n in range(0, channels, 1):
                        a[i]+=(mu[i][n]-frame[x][y][n])**2 #n is variable with 3 coordinate
                min_a=[val for val, j in enumerate(a) if j == min(a)] #find index of minimum value of a
                r[x][y][min_a[0]]=1
                a=np.zeros(k)
    # #Maximization step (calculation of mu)
        count=[[0,0,0],[0,0,0],[0,0,0]]
        mu = [[0, 0, 0],[0, 0, 0],[0, 0, 0]]
        for x in range(0, hight,1):
            for y in range(0, width,1):
                for i in range(0, k, 1):
                    for n in range(0, channels, 1):
                        if r[x][y][i]==1:
                            mu[i][n]+=r[x][y][i]*frame[x][y][n]
                            count[i][n]+=1
        for i in range(0,k,1):
            for n in range(0, channels):
                if count[i][n] !=0:
                    mu[i][n]=mu[i][n]/count[i][n] #mean value of mu
                    
                
    #variance caculation
    sigma = [[0, 0, 0],[0, 0, 0],[0, 0, 0]]
    count=[[0,0,0],[0,0,0],[0,0,0]]
    for x in range(0, hight,1):
            for y in range(0, width,1):
                for i in range(0, k, 1):
                    for n in range(0, channels, 1):
                        if r[x][y][i]==1:
                            sigma[i][n]+=((r[x][y][i]*frame[x][y][n])-mu[i][n])**2
                            count[i][n]+=1
    for i in range(0,k,1):
        for n in range(0, channels):
            if count[i][n] !=0:
                sigma[i][n]=(sigma[i][n]/count[i][n])
                
    #pi calculation
    pi=[0,0,0]
    pi[0]=count[0][0]/(count[0][0]+count[1][0]+count[2][0])
    pi[1]=count[1][0]/(count[0][0]+count[1][0]+count[2][0])
    pi[2]=count[2][0]/(count[0][0]+count[1][0]+count[2][0])
    
                
    return mu, sigma, pi


'''..........................SG Background removal algorithm...........................'''

return_b, frame = video.read()
mu, sigma, pi=KMC(frame,3)
k=3
# No of rows column and RGB channels
hight, width, channels = frame.shape

mu_pix = np.zeros((hight, width, k, channels))
sigma_pix = np.zeros((hight, width, k, channels))
pi_pix = np.zeros((hight, width, k, channels))
bg = np.zeros((hight, width, channels), dtype = np.uint8)
fg = np.zeros((hight, width, channels), dtype = np.uint8)

for n in range(0, channels):
    for i in range(0, k):
        for x in range(0,hight):
            for y in range(0,width):
                mu_pix[x][y][i][n] = mu[i][n]
                sigma_pix[x][y][i][n] = sigma[i][n]
                pi_pix[x][y][i][n] = pi[i]


def SGBG(frame, k, mu_pix, sigma_pix, pi_pix, alpha, rho, bg, fg):
    
    # collect information of frames
    hight, width, channels = frame.shape
    pi_sigma = np.zeros(k)

    # Mean variance and pi value calculation for all pixel
    for n in range(0, channels):
        for x in range(0, hight):
            for y in range(0, width):
                val = 0
                tigger = 0
                for i in range(0, k):
                    if abs(frame[x][y][n] - mu_pix[x][y][i][n]) < (2.5 * (sigma_pix[x][y][i][n]) ** (1 / 2.0)):
                        mu_pix[x][y][i][n] = (1 - rho) * mu_pix[x][y][i][n] + rho * frame[x][y][n]
                        sigma_pix[x][y][i][n] = (1 - rho) * sigma_pix[x][y][i][n] + rho * (frame[x][y][n] - mu_pix[x][y][i][n]) ** 2
                        pi_pix[x][y][i][n] = (1 - alpha) * pi_pix[x][y][i][n] + alpha 
                        tigger = 1
                    else:
                        pi_pix[x][y][i][n] = (1 - alpha) * pi_pix[x][y][i][n]  
                    val += pi_pix[x][y][i][n]

                
                # pi value normalization and pi/sigma value calculation
                for i in range(0, k):
                    pi_pix[x][y][i][n] = pi_pix[x][y][i][n] / val
                    pi_sigma[i] = pi_pix[x][y][i][n] / sigma_pix[x][y][i][n]

                
                # mean sigma and pi value rearrangement
                for i in range(0, k):
                    swap = False
                    for m in range(0, k - i - 1):
                        if pi_sigma[m] < pi_sigma[m + 1]:
                            pi_sigma[m], pi_sigma[m + 1] = pi_sigma[m + 1], pi_sigma[m]
                            pi_pix[x][y][m][n], pi_pix[x][y][m + 1][n]= pi_pix[x][y][m + 1][n], pi_pix[x][y][m][n]
                            mu_pix[x][y][m][n], mu_pix[x][y][m + 1][n] = mu_pix[x][y][m + 1][n], mu_pix[x][y][m][n]
                            sigma_pix[x][y][m][n], sigma_pix[x][y][m + 1][n]= sigma_pix[x][y][m + 1][n], sigma_pix[x][y][m][n]
                            swap = True
                    if swap == False:
                        break
                 
                
                # update the last gaussian
                if tigger == 0:
                    mu_pix[x][y][k - 1][n] = frame[x][y][n]
                    sigma_pix[x][y][k - 1][n] = 10000

                # calculation of threshold value for backround and foregroud pixel
                threshold = 0
                k_t = 0
                for i in range(0, k):
                    threshold += pi_pix[x][y][i][n] #least pi
                    if threshold > 0.7:
                        k_t = i
                        break

                # Update foreground and background
                for i in range(0, k_t + 1):
                    if tigger == 0 or abs(frame[x][y][n] - mu_pix[x][y][i][n]) > (2.5 * (sigma_pix[x][y][i][n]) ** (1 / 2.0)):
                        bg[x][y][n] = mu_pix[x][y][i][n]
                        fg[x][y][n] = frame[x][y][n]
                        break
                    else:
                        bg[x][y][n] = frame[x][y][n]
                        fg[x][y][n]= 255
                        
                        
    # fill outer part of foreground with white colour
    for n in range(0, channels):
        for x in range(0, hight):
            for y in range(0, width ):
                if fg[x][y][n] == 255:
                    fg[x][y][0] = fg[x][y][1] = fg[x][y][2] = 255
    return sigma_pix, pi_pix, mu_pix, bg, fg


'''....................background and foreground image conversion.................'''
video= cv2.VideoCapture('umcp.mpg')
i=0
while(video.isOpened()):
    return_b, frame = video.read()
    if return_b == False:
        break
    sigma_pix, pi_pix, mu_pix, bg, fg = SGBG(frame, k, mu_pix, sigma_pix, pi_pix, alpha, rho, bg, fg)
    cv2.imwrite('background_'+str(i)+'.jpg',bg)
    cv2.imwrite('fourground_'+str(i)+'.jpg',fg)
    i+=1

video.release()
cv2.destroyAllWindows()


'''.........................convert foreground images into video....................''' 

fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
video = cv2.VideoWriter('Foreground.avi', fourcc, 30, (width, hight))
for i in range(0,999):
    fg_img = cv2.imread('fourground_'+str(i) + '.jpg')
    video.write(fg_img)
cv2.destroyAllWindows()
video.release()

'''.........................convert background images into video....................''' 

fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
video = cv2.VideoWriter('Background.avi', fourcc, 30, (width, hight))
for i in range(0,999):
    bg_img = cv2.imread('background_'+str(i) + '.jpg')
    video.write(bg_img)
cv2.destroyAllWindows()
video.release()
