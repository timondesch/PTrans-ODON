from glob import glob
import os
import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt

class dataset_eval :
    def __init__(self, path):

        self.folders = { f : os.path.join(path, f) for f in ["reference", "img", "mask", "inpaintings", "segmentations", "segmentations_img"]}

        for i in self.folders:
            if not os.path.exists(self.folders[i]):
                os.mkdir(self.folders[i])

        self.scores_path = os.path.join(path, "scores.csv")

        get_file = lambda name, ext : {os.path.splitext(os.path.basename(f))[0] : f for f in glob(os.path.join(path,name,"*." + ext))}
        get_file2 = lambda name, ext : {os.path.basename(f) : get_file(os.path.join(name, os.path.basename(f)), ext) for f in glob(os.path.join(path,name,"*"))}

        self.reference = get_file("reference", "png")
        self.img = get_file("img", "png")
        self.mask = get_file("mask", "png")

        self.inpaintings = get_file2("inpaintings", "png")
        self.segmentations = get_file2("segmentations", "csv")
        self.segmentations_img = get_file2("segmentations_img", "png")

        self.scores = {}

    def dice(self, x, y):
        dices = {}
        dice = 0
        
        for i in range(len(x)):
            for j in range(len(x[i])):
                
                a = x[i][j]
                b = y[i][j]
                
                if a not in dices.keys() :
                    dices[a] = [0,0]
                if b not in dices.keys() :
                    dices[b] = [0,0]
                    
                if (a == b) : 
                    dices[a][0] += 2
                    
                dices[a][1] += 1
                dices[b][1] += 1
                
        for k in dices :
            if k != '0':     
                dice += (dices[k][0]/dices[k][1])
    
        dice = dice / (len(dices) - 1)
        
        return dice
        
    def make_masks(self, threshold=250):
        for i in self.img:
            if i not in self.mask :
                
                image = cv2.imread(self.img[i], cv2.IMREAD_COLOR)
                _, dst = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
                
                save_path = os.path.join(self.folders["mask"], i + ".png")
                plt.imsave(save_path, dst)
                self.mask[i] = save_path

        
    def make_inpaintings(self, inpaintings):
        self.make_masks()
        for i in inpaintings:
            if i not in self.inpaintings:
                os.mkdir(os.path.join(self.folders["inpaintings"], i)) 
                self.inpaintings[i] = {}
                
        for i in self.img:
            image = cv2.imread(self.img[i], cv2.IMREAD_COLOR)
            mask = cv2.imread(self.mask[i], cv2.IMREAD_GRAYSCALE)
            
            for j in inpaintings :
                if i not in self.inpaintings[j] :
                    r = inpaintings[j](image, mask)
                    save_path = os.path.join(self.folders["inpaintings"], j, i+".png")
                    plt.imsave(save_path, r)                    
                    self.inpaintings[j][i] = save_path
                
                
    def make_segmentations(self, model) : 
        to_segment = {**self.inpaintings, 'reference' : self.reference}
        segmentations_folders = {i : os.path.join(self.folders["segmentations"], i) for i in to_segment}
        
        for i in to_segment:
            if i not in self.segmentations:
                os.mkdir(os.path.join(self.folders["segmentations"], i)) 
                self.segmentations[i] = {}
        
        for i in to_segment :
            for j in to_segment[i]:
                if j not in self.segmentations[i]:
                    image = cv2.imread(to_segment[i][j], cv2.IMREAD_COLOR)
                    im_input=cv2.resize(image/255.0,(1024,512),cv2.INTER_NEAREST)
                
                    pred = model.predict(np.expand_dims(im_input, 0))[0]
                    result = np.argmax(pred,axis=-1)
                
                
                
                    with open(os.path.join(self.folders["segmentations"], i, j + ".csv"), "w") as f:
                        writer = csv.writer(f)
                        writer.writerows(result)

                    self.segmentations[i][j] = os.path.join(self.folders["segmentations"], i, j + ".csv")
                    
    def make_segmentations_img(self) :
        for i in self.segmentations:
            if i not in self.segmentations_img:
                os.mkdir(os.path.join(self.folders["segmentations_img"], i)) 
                self.segmentations_img[i] = {}
                
        for i in self.segmentations :
            for j in self.segmentations[i] :
                if j not in self.segmentations_img[i]:
                    annotation = np.asarray(self.parse_annotation(self.segmentations[i][j])).astype(int)
                    img = self.give_color_to_annotation(annotation)
                    save_path = os.path.join(self.folders["segmentations_img"], i, j+".png")
                    plt.imsave(save_path, img.astype(np.uint8))                    
                    self.segmentations_img[i][j] = save_path
            
            
    def make_scores(self) : 
        for i in self.segmentations:
            self.scores[i] = {}
            
        for i in self.segmentations['reference']:
            reference = self.parse_annotation(self.segmentations['reference'][i])
                
            for j in self.segmentations : 
                inpainting = self.parse_annotation(self.segmentations[j][i])
                self.scores[j][i] = self.dice(reference, inpainting)
        
        with open(self.scores_path, "w", newline="") as f:
            f.write(" ,")
            for i in self.segmentations['reference']:
                f.write(i + ",")
            
            f.write("\n")
            
            for i2 in self.scores :
                f.write(str(i2) + ", ")
                print(i2)
                for j in self.scores[i2] :
                    f.write(str(self.scores[i2][j]) + ", ")
                f.write("\n")
                
    def parse_annotation(self, path):
        with open(path, 'r') as file:
            csv_reader = csv.reader(file)
            annotation = []
            for row in csv_reader:
                annotation.append(row)
        return annotation
    
    def give_color_to_annotation(self, annotation):
    
        color = [(245/255,245/255,245/255)]+[tuple(np.random.randint(256, size=3) / 255.0) for i in range(32)]
        seg_img = np.zeros( (annotation.shape[0],annotation.shape[1], 3) ).astype('float')
 
        for c in range(33):
            segc = (annotation == c)
            seg_img[:,:,0] += segc*( color[c][0] * 255.0)
            seg_img[:,:,1] += segc*( color[c][1] * 255.0)
            seg_img[:,:,2] += segc*( color[c][2] * 255.0)
     
        return seg_img.astype(int)
