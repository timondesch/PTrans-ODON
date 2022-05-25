
import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import skimage

"""## Code"""

PATH_IN = "../../IN/PNG/dataset_png/"
NB_TOTAL = 0 
PATH_OUT = "../../OUT/data_aug_treatments/"
NB_IMG = 10
NB_TREAT_MIN, NB_TREAT_MAX = 1, 6

UPPER = [*range(1, 17)]
LOWER = [*range(17, 33)]

WEIGHTS = [*range(1, 9)] + [*range(1, 9)]
WEIGHTS = WEIGHTS / np.sum(WEIGHTS)

for base, dirs, files in os.walk(PATH_IN):
    for directories in dirs:
        NB_TOTAL += 1

def show(img):
    plt.axis('off')
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.show()

def rand_position():
    return np.random.choice(random.choice([UPPER, LOWER]), 1, p=WEIGHTS)[0]

def get_tooth(position, image):
    random.seed()
    np.random.seed()
    try:
        random_tooth = cv2.imread(PATH_IN + image + "/" + image + "_" + str(position) + ".png", cv2.IMREAD_GRAYSCALE)
        random_tooth = cv2.bitwise_xor(random_tooth, np.full_like(random_tooth, 0))
        return random_tooth
    except:
        print("Invalid position.")
        return None

def gen_treatment(position, treated_tooth):
    _, tooth_mask = cv2.threshold(treated_tooth, 1, 255, cv2.THRESH_BINARY)

    contours = cv2.findContours(tooth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    c = max(contours, key=cv2.contourArea)

    left = tuple(c[c[:, :, 0].argmin()][0])
    right = tuple(c[c[:, :, 0].argmax()][0])
    top = tuple(c[c[:, :, 1].argmin()][0])
    bottom = tuple(c[c[:, :, 1].argmax()][0])
    
    bg = np.zeros(tooth_mask.shape)
    
    # finding extreme point of tooth and generate simple treatment
    if position >= 17:
        bg[top[1]][top[0]] = 255
    else :
        bg[bottom[1]][bottom[0]] = 255
    treatment = cv2.dilate(bg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                         (random.randint(20, 40),random.randint(20, 40))),
                                                           iterations=random.randint(3,6))

    points = []
    for i in range(treatment.shape[0]):
        for j in range(treatment.shape[1]):
            if treatment[i][j] == 255:
                if (random.randint(1,150)==1):
                    points.append((j, i))


    bg2 = np.zeros(tooth_mask.shape)
    for point in points:
        bg2[point[1]][point[0]] = 255
      
    newtreatment = cv2.dilate(bg2, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                            (20,20))).astype(np.uint8)

    treat = cv2.bitwise_or(newtreatment.astype(np.uint8), treatment.astype(np.uint8))
    return np.where((treat>0)&(treated_tooth>0), 255, 0)

def initialize_struct():
    global PATH_OUT, PATH_IN
    dirName = str(datetime.now().strftime("%d-%m-%Y_%H-%M"))
    os.makedirs(f"{PATH_OUT}{dirName}")
    PATH_OUT = PATH_OUT + dirName + "/"

    folders = [x for x in os.listdir(PATH_IN) if "_" not in x]
    for folder in folders:
        if not os.path.exists(PATH_OUT + folder): os.makedirs(PATH_OUT + folder)
    return folders

def fuse(base_img, top, bottom):
    treated_img_up = np.zeros_like(base_img)
    treated_img_down = np.zeros_like(base_img)
    for treatment in top:

        noised = skimage.util.random_noise(treatment/255, mode='speckle', var=0.04**2)
        noised = (255*noised).astype(np.uint8)
        
        kernel = np.ones((5,5),np.float32)/25
        blurred_treat = cv2.filter2D(noised, -1, kernel)

        # add all treats together
        treated_img_up = np.bitwise_or(treated_img_up, blurred_treat)

    for treatment in bottom:
        noised = skimage.util.random_noise(treatment/255, mode='speckle', var=0.04**2)
        noised = (255*noised).astype(np.uint8)
        
        kernel = np.ones((5,5),np.float32)/25
        blurred_treat = cv2.filter2D(noised, -1, kernel)

        # add all treats together
        treated_img_down = np.bitwise_or(treated_img_down, blurred_treat)
    
    # fusion of neighbors treatments
    treated_img_up2 = cv2.morphologyEx(treated_img_up, cv2.MORPH_CLOSE, np.ones((5, 15)))
    treated_img_down2 = cv2.morphologyEx(treated_img_down, cv2.MORPH_CLOSE, np.ones((5, 15)))
    treated_img_up = np.array(np.maximum(treated_img_up, treated_img_up2))
    treated_img_down = np.array(np.maximum(treated_img_down, treated_img_down2))
    treated_img = np.array(np.maximum(treated_img_up, treated_img_down))
    treated_img = np.array(np.maximum(treated_img, base_img.copy()))
    return treated_img



def main():
    NB_IMG = 10
    NB_TREAT = 2
    images = initialize_struct()
    for index, image in enumerate(images):
        print(index/NB_TOTAL*100, "%")
        print("Working on image : " + image)
        base_img = cv2.imread(PATH_IN + image + "/" + image + ".png", cv2.IMREAD_GRAYSCALE)
        for i in range(2):
            print("---- creating sub-image nb : " + str(i))
            treated_img_up = np.zeros_like(base_img)
            treated_img_down = np.zeros_like(base_img)
            treatments_up = []
            treatments_down = []
            for _ in range(np.random.randint(NB_TREAT_MIN, NB_TREAT_MAX + 1)):
                treated_tooth = None
                p=-1
                while(treated_tooth is None):
                    p = rand_position()
                    treated_tooth = get_tooth(p, image)
                (treatments_up if p < 17 else treatments_down).append((gen_treatment(p, treated_tooth), treated_tooth))
            
            for treatment in treatments_up:
                treat2 = np.where((treatment[0]>0)&(treatment[1]>0), 255, 0)

                noised = skimage.util.random_noise(treat2/255, mode='speckle', var=0.04**2)
                noised = (255*noised).astype(np.uint8)
                
                kernel = np.ones((5,5),np.float32)/25
                blurred_treat = cv2.filter2D(noised, -1, kernel)

                # add all treats together
                treated_img_up = np.bitwise_or(treated_img_up, blurred_treat)

            for treatment in treatments_down:
                treat2 = np.where((treatment[0]>0)&(treatment[1]>0), 255, 0)

                noised = skimage.util.random_noise(treat2/255, mode='speckle', var=0.04**2)
                noised = (255*noised).astype(np.uint8)
                
                kernel = np.ones((5,5),np.float32)/25
                blurred_treat = cv2.filter2D(noised, -1, kernel)

                # add all treats together
                treated_img_down = np.bitwise_or(treated_img_down, blurred_treat)


            # fusion of neighbors treatments
            treated_img_up2 = cv2.morphologyEx(treated_img_up, cv2.MORPH_CLOSE, np.ones((5, 15)))
            treated_img_down2 = cv2.morphologyEx(treated_img_down, cv2.MORPH_CLOSE, np.ones((5, 15)))
            treated_img_up = np.array(np.maximum(treated_img_up, treated_img_up2))
            treated_img_down = np.array(np.maximum(treated_img_down, treated_img_down2))
            treated_img = np.array(np.maximum(treated_img_up, treated_img_down))
            treated_img = np.array(np.maximum(treated_img, base_img.copy()))
            cv2.imwrite(PATH_OUT + image + "/" + image + "_" + str(i) + ".png", treated_img)

if __name__ == "__main__":
  main()
  