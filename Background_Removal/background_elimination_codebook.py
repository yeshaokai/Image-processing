from __future__ import division
import matplotlib.pyplot as plt
import glob
import numpy as np
from numpy import linalg as LA
import scipy.ndimage
import sys
class CodeWord:
    def __init__(self,rgb_v,aux):
        self.rgb_v = rgb_v
        self.aux = aux
    def wrap_around(self,N):
        self.aux[3] = max(self.aux[3],(N-self.aux[5]+self.aux[4]-1))
    def effective(self,N):
        if self.aux[3]<N:
            return True
        else :
            return False
        
class CodeBook:
    def __init__(self):
        self.codewords = []
    def add_codeword(self,rgb_v,aux):
        cw = CodeWord(rgb_v,aux)
        self.codewords.append(cw)
    def update(self,cw,x,I,t):
        v = cw.rgb_v
        aux = cw.aux
        f = aux[2]
        v = (v*f+x)/(f+1)
        aux[0] = min(aux[0],I) # Imin as aux[0]
        aux[1] = max(aux[1],I) # Imax as aux[1]
        aux[2] = aux[2]+1    # f as aux[2]

        aux[3] = max(aux[3],t-aux[5]) #aux[3] the maximum negative run length aux[5] as q
        aux[4] = aux[4]   # aux[4] as p
        aux[5] = t  

    def size(self):
        return len(self.codewords)
    def colordist(self,cw,xt):
        vi = cw.rgb_v
        v_mag_square = np.linalg.norm(vi)**2
        x_mag_square = np.linalg.norm(xt)**2

        xtvt = np.dot(xt,vi)**2
        
        p_square = xtvt/(v_mag_square)

            
        return np.sqrt(x_mag_square-p_square)
    def brightness(self,cw,I):
        alpha = 0.4
        belta = 1.5
        Ilow = cw.aux[1]*alpha
        Ihigh =min(belta*cw.aux[1],cw.aux[0]/alpha)
        

        if Ilow<=I and I<= Ihigh:
            return True
        else:
            return False
    def match(self,xt,I,ebu1):
        for i in range(len(self.codewords))[::-1]:
            cw = self.codewords[i]
            if self.colordist(cw,xt)< ebu1 and self.brightness(cw,np.linalg.norm(xt)):
#                print (cw.aux)
                return cw

    def getword(self,i):
        return self.codewords[i]
        
class CodeBook2D:
    def __init__(self,shape):
        self.book2D = []
        for i in range(shape[0]):
            self.book2D.append([])
            for j in range(shape[1]):
                c = CodeBook()
                self.book2D[i].append(c)
    def getCodeBook(self,i,j):
        return self.book2D[i][j]
                
    
class Backgroundsubstraction:
    def __init__(self,imglist,ebu1,ebu2):
        self.codebook2D = None
        self.nframes = len(imglist)
        self.ebu1 = ebu1
        self.ebu2 = ebu2
        self.imglist = imglist
        self.shape = self.imglist[0].shape
        self.masks = np.zeros(self.shape[:2])

    def buildCodeBooks(self):
        frameShape = self.shape
        self.codebook2D = CodeBook2D(frameShape)
        for t in range(1,self.nframes+1):
            img = self.imglist[t-1]
            print ("processing frame " + str(t))
            for i in range(frameShape[0]):
                for j in range(frameShape[1]):
                    Xt = img[i][j]
                    I = np.sum(Xt)#Xt[0]+Xt[1]+Xt[2]

                    codebook = self.codebook2D.getCodeBook(i,j)
                    cw = codebook.match(Xt,I,self.ebu1)
                    if not cw:
                        V = Xt
                        aux = [I,I,1,t-1,t,t]
                        codebook.add_codeword(V,aux)
                    else:
                        codebook.update(cw,Xt,I,t)
    def wrap_around(self):
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                codebook = self.codebook2D.getCodeBook(i,j)
                for n in range(codebook.size()):
                    cw = codebook.getword(n)
                    cw.wrap_around(self.nframes-1)
    def effctive_codewords(self):
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                codebook = self.codebook2D.getCodeBook(i,j)
                filtered = []
                for n in range(codebook.size()):
                    cw = codebook.getword(n)
                    if cw.effective((self.nframes-1)/2):
                        filtered.append(cw)
                codebook.codewords = filtered
    def buildMask(self,img,ebu2):
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                codebook = self.codebook2D.getCodeBook(i,j)
                Xt = img[i][j]
                I = sum(Xt)
                for n in range(codebook.size())[::-1]:
                    cw = codebook.getword(n)
                    if codebook.colordist(cw,Xt) < ebu2 and codebook.brightness(cw,I):
                       self.masks[i][j] = 0
                       # 0 represents background 
                    else:
                        self.masks[i][j] = 1
                        # one represents foreground
    def background_substraction(self):
        global imglist

        fig = plt.figure()
        imgindices = [416,514,682,749,590]
        for k in range(1,6):
            img = imglist[imgindices[k-1]]
            self.buildMask(img,self.ebu2)


            result = np.zeros(self.shape[:2])        
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    if self.masks[i][j] ==1:
                        result[i][j] = 255
                    else:
                        result[i][j] = 0
            ax = fig.add_subplot(2,3,k)
            result = self.morphological_operations(result)
            ax.imshow(result,cmap='gray')
        plt.show()
    def morphological_operations(self,img):
        opening = scipy.ndimage.grey_opening(img,size=(1,1))
        closing = scipy.ndimage.grey_closing(opening,size=(2,2))
        return closing
if __name__ == "__main__":    
    
    files = glob.glob('Video1_1/PetsD2TeC1_*')


    imglist = [plt.imread(file) for file in files]


    train = np.array(imglist[200:260])

    ebu1 = 400
    ebu2 = 400
    b = Backgroundsubstraction(imglist = train,ebu1 = ebu1,ebu2 = ebu2)
    print ("ebu1 is "+str(ebu1) + " ebu2 is "+ str(ebu2))
    b.buildCodeBooks()
    b.wrap_around()
    b.effctive_codewords()

    b.background_substraction()



