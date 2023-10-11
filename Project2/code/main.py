# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from Classifier import MinErrorRate
from matplotlib.patches import Rectangle

folder = "data/"

def Plot_train_Seg(train, test, xs1, ys1, w1, h1, xs2, ys2, w2, h2,
                   xs3, ys3, w3, h3, t1="", t2="", norm=False):
    ### Plot image with training boxes
    plt.imshow(train)  # Plot image

    ax = plt.gca()
    # Make training patches ((x_start,y_start), width, height)
    rect1 = Rectangle((xs1, ys1), w1, h1, linewidth=1, edgecolor="b",
                              facecolor="none")
    rect2 = Rectangle((xs2, ys2), w2, h2, linewidth=1, edgecolor="b",
                              facecolor="none")
    rect3 = Rectangle((xs3, ys3), w3, h3, linewidth=1, edgecolor="b",
                              facecolor="none")

    ax.add_patch(rect1)
    ax.add_patch(rect2)
    ax.add_patch(rect3)
    plt.title(t1, size=14)
    plt.axis("off")
    plt.show()

    ### Plot segmentation
    if norm:
        train = train[:, :, :2] # For normalized image
        test = test[:, :, :2] # For normalized image
    img1 = train[ys1:ys1+h1, xs1:xs1+w1]
    img2 = train[ys2:ys2+h2, xs2:xs2+w2]
    img3 = train[ys3:ys3+h3, xs3:xs3+w3]

    segmentation = MinErrorRate(img1, img2, img3, test)
    plt.imshow(segmentation)
    plt.title(t2, size=14)
    plt.axis("off")
    plt.show()


def Normalize(image):
    ### Normalize image with RGB tristimulus values
    RGB = np.sum(image, axis=2)
    RGB[RGB == 0] = 1
    R = image[:, :, 0]/RGB
    G = image[:, :, 1]/RGB
    B = image[:, :, 2]/RGB
    tr_image = np.dstack([R,G,B])
    return tr_image

#----------------------------------------
if __name__=="__main__":
    # 1)
    ### Train and test the classifier with only one image
    trial = np.array(Image.open(folder+"Bilde1.png").convert("RGB"))
    print(trial.shape)

    ## Plot original image with training boxes and following segmentation
    xs1 = 210; ys1 = 90;  w1 = 135; h1 = 90
    xs2 = 220; ys2 = 310; w2 = 150; h2 = 130
    xs3 = 95;  ys3 = 220; w3 = 55;  h3 = 150
    t1 = "Treningsregioner"
    t2 = "Bilde segmentert uten normalisering"
    norm = False
    Plot_train_Seg(trial, trial, xs1, ys1, w1, h1, xs2, ys2, w2, h2,
               xs3, ys3, w3, h3, t1, t2, norm)

    ## Plot normalized image and following segmentation
    t1 = "Normaliserte RGB-verdier"
    t2 = "Bilde segmentert med normalisering"
    norm = True
    trial_norm = Normalize(trial)
    Plot_train_Seg(trial_norm, trial_norm, xs1, ys1, w1, h1, xs2, ys2, w2, h2,
               xs3, ys3, w3, h3, t1, t2, norm)

    # 2)
    ### Use two separate images for training and testing
    train = np.array(Image.open(folder+"Bilde2.png", mode="r").convert("RGB"))
    test = np.array(Image.open(folder+"Bilde3.png", mode="r").convert("RGB"))

    ## Plot original image with training boxes and following segmentation
    xs1 = 740;  ys1 = 1000; w1 = 600;  h1 = 500
    xs2 = 2150; ys2 = 550;  w2 = 600;  h2 = 600
    xs3 = 10;   ys3 = 10;   w3 = 1000; h3 = 600
    t1 = "Treningsregioner"
    t2 = "Bilde segmentert uten normalisering"
    norm = False
    Plot_train_Seg(train, test, xs1, ys1, w1, h1, xs2, ys2, w2, h2,
               xs3, ys3, w3, h3, t1, t2, norm)

    ## Plot normalized image and following segmentation
    t1 = "Normaliserte RGB-verdier"
    t2 = "Bilde segmentert med normalisering"
    norm = True
    train_norm = Normalize(train)
    test_norm = Normalize(test)
    Plot_train_Seg(train_norm, test_norm, xs1, ys1, w1, h1, xs2, ys2, w2, h2,
               xs3, ys3, w3, h3, t1, t2, norm)
