import numpy as np
import copy
import matplotlib
from scipy import interpolate
matplotlib.use("TkAgg")
import matplotlib.pylab as plt
import tkinter
from tkinter import messagebox
from tkinter import *
from numpy import linalg as LA
from matplotlib.pyplot import plot,ion,show
import _thread
from time import sleep
import collections
import glob

# find the gradient of the whole image
# my test tells me that gradient[0] represents column direction and gradient[1] represents row direction, which is opposite 
# to the newest document of numpy.



def apply_gaussian_filter(data,filter_shape,std):
    fd1 = filter_shape[0]
    fd2 = filter_shape[1]
    if fd1<3 or fd2<3:
        raise Exception("invalid gaussian filter.")

    x = np.arange(fd1) # about to generate samples in x. Make sure they are all positive                                                           \
                                                                                                                                                    
    y = np.arange(fd2)

    mean_x = np.mean(x)# get the value of the discrete values                                                                                      \
                                                                                                                                                    
    mean_y = np.mean(y)


    if fd1%2 == 0 or fd2%2 ==0:  # if it is even                                                                                                   \
                                                                                                                                                    
        raise Exception("i don't want to support even gaussian for computation sake")
    width1 = fd1*std
    width2 = fd2*std

    sample_x = np.linspace(mean_x-((fd1-1)/2),mean_x+((fd1-1)/2),num=fd1)

    sample_y = np.linspace(mean_y-((fd2-1)/2),mean_x+((fd2-1)/2),num=fd2)

    gaussian1d_x = np.exp((-(sample_x-mean_x)**2)/(2.0*std**2)) # feed gaussian function the sample points                                         \
                                                                                                                                                    
    min_gaussian_x = np.min(gaussian1d_x)
    gaussian1d_x = gaussian1d_x/min_gaussian_x # normalize it so the smallest entry is 1                                                           \
                                                                                                                                                    
    gaussian1d_x = np.floor(gaussian1d_x) # round the nearest integer 
    gaussian1d_x = np.floor(gaussian1d_x) # round the nearest integer                                                                              \
                                                                                                                                                    

    gaussian1d_y = np.exp((-(sample_y-mean_y)**2)/(2.0*std**2))
    min_gaussian_y = np.min(gaussian1d_y)
    gaussian1d_y = gaussian1d_y/min_gaussian_y
    gaussian1d_y = np.floor(gaussian1d_y)

#    im = plt.imread(imgname)                                                                                                                       
    filter_shape = gaussian1d_x.shape
    data_shape = data.shape
    datanew = copy.deepcopy(data)
    gaussian1d_x = gaussian1d_x/(np.sum(gaussian1d_x))  # normalize by the sum                                                                     \
                                                                                                                                                    
    gaussian1d_y = gaussian1d_y/(np.sum(gaussian1d_y))
    for c in range(data_shape[0]):
        datanew[c,:] = np.convolve(datanew[c,:],gaussian1d_y,'same')
    for r in range(data_shape[1]):
        datanew[:,r] = np.convolve(datanew[:,r],gaussian1d_x,'same')

    return datanew

def intermediate(threadname,n,coords):
    global pic2
    global pic3
    global belta
    if (n == 2 and pic2):
        return
    if (n == 3 and pic3):
        return 
    if (n ==2):
        pic2 = True
    if (n == 3):
        pic3 = True
    print ("got here")
    global fig
    thread_coords = coords

    ax = fig.add_subplot(2,2,n)
    ax.clear()
    if (n ==2):
        ax.set_title("step 2")
    if (n ==3):
        ax.set_title("step 3")
    ax.imshow(im)
    x = [coord[0] for coord in thread_coords]
    y = [coord[1] for coord in thread_coords]
    corner = []
    for i in range(len(thread_coords)):
        if belta[i] ==0 :
            corner.append(thread_coords[i])
    corner_x = [coord[0] for coord in corner]
    corner_y = [coord[1] for coord in corner]
    ax.plot(y,x,marker='.',c='b',linestyle='None')
    ax.plot(corner_y,corner_x,marker='*',c='w',linestyle='None')
    fig.canvas.draw()
def work(threadname,delay):

#    while (not ready):
#        pass
    global coords
    global fig
    global belta
    global startFile 
    global im
    int_coords = copy.deepcopy(coords)
    int_coords = getIntergerCoords(int_coords)
    coords = int_coords
    new_coords = greedy(threshold1=threshold1,threshold2=threshold2,threshold3 = threshold3)

    ax = fig.add_subplot(1,2,2)
    ax.clear()
    ax.set_title(startFile)
    ax.imshow(im)
    x = [coord[0] for coord in new_coords]
    y = [coord[1] for coord in new_coords]
    corner = []
    for i in range(len(coords)):
        if belta[i] ==0 :
            corner.append(coords[i])
    corner_x = [coord[0] for coord in corner]
    corner_y = [coord[1] for coord in corner]
    ax.plot(corner_y,corner_x,marker='*',c='w',linestyle='None')
    ax.plot(y,x,marker='.',c='b',linestyle='None')
    fig.canvas.draw()
    plt.savefig("attemp1.jpg")
    
    plt.clf()
    coords_map = {}

    coords_map[startFile] = copy.deepcopy(coords)
    
    # we should start doing things from this thread.
    
    global interFiles    
    global gradient_mag
    count = 0
    for file in interFiles:
        count = count + 1
        print ("start processing" + file)
        new_coords = greedy(threshold1=threshold1,threshold2=threshold2,threshold3 = threshold3)

        coords = new_coords

        im = plt.imread(file)
        im = im.astype(float)
        im_gradients = np.gradient(im)
        row_gradient = im_gradients[0]
        column_gradient = im_gradients[1]
        gradient_mag = np.sqrt(row_gradient**2+column_gradient**2)
        ax = fig.add_subplot(1,2,1)
        ax.clear()
        ax.set_title(file)
        ax.imshow(im)
        x = [coord[0] for coord in new_coords]
        y = [coord[1] for coord in new_coords]
        corner = []
        for i in range(len(coords)):
            if belta[i] ==0 :
                corner.append(coords[i])
        corner_x = [coord[0] for coord in corner]
        corner_y = [coord[1] for coord in corner]
        ax.plot(corner_y,corner_x,marker='*',c='w',linestyle='None')
        ax.plot(y,x,marker='.',c='b',linestyle='None')
        fig.canvas.draw()
        file = file.split('/')[-1]
        plt.savefig("result_"+file)
#        show()
        sleep(1)
def add_points(distances,coords_lst):

    global new_coords
    for i in range(len(distances)):
        row_curv = column_cur = 0
        if distances[i]>8*Number_Of_Points_Between_Interpolation:
            if i!= len(coords_lst)-1:
                row_next = coords_lst[i+1][0]
                column_next = coords_lst[i+1][1]
            else:
                row_next = coords_lst[0][0]
                column_next = coords_lst[0][1]
            row_cur = coords_lst[i][0]
            column_cur = coords_lst[i][1]            
            f = interpolate.interp1d([row_cur,row_next],[column_cur,column_next])
            row_interpolated = np.linspace(row_cur,row_next,num=Number_Of_Points_Between_Interpolation)

            column_interpolated = f(row_interpolated)

            temp_coords = [(row_interpolated[i],column_interpolated[i]) for i in range(len(row_interpolated))]        
            for c in temp_coords:
                new_coords.append(c)
        else:
            if i!=len(distances)-1:
                new_coords.append(coords_lst[i])
                new_coords.append(coords_lst[i+1])
            else:
                new_coords.append(coords_lst[i])
                new_coords.append(coords_lst[0])

    temp = copy.deepcopy(new_coords)
    new_coords = sorted(set(new_coords),key=lambda x:temp.index(x))  
    x = [pair[0] for pair in new_coords]
    y = [pair[1] for pair in new_coords]
    

    ax.set_title("original countour")
    ax.plot(y,x,marker='.',c='r',linestyle='None')



    global ready
    ready = True
    global coords 
    coords = new_coords


    fig.canvas.draw()
    _thread.start_new_thread(work,("thread",1,))


def calculate_distances(coords,do_interpolation = True):

    distances = []
    for i in range(len(coords)):
        v_cur = coords[i]
        v_next = coords[0] if i ==len(coords)-1 else coords[i+1]
        v_cur = np.array(v_cur)
        v_next = np.array(v_next)
        distances.append(LA.norm(v_cur-v_next))
    if do_interpolation:
        add_points(distances,coords)
    else:
        return distances
def onclick(event):

    global end    
    if not end:
        global ix, iy
        ix, iy = event.xdata, event.ydata
        
        ax.plot(ix,iy,marker='.',c='r')
        fig.canvas.draw()
        global coords
        coords.append((iy, ix))
    


    if len(coords)> Start_After_N_Click and not end:
        calculate_distances(coords)
        end = True

    return coords
def getNeighbor(coord,m = 9):
    '''
    supposed to return  a list of neighbor on that coordinate
    '''
    ret = []
    if m == 9:
        ret.append((coord[0],coord[1]))
        ret.append((coord[0]-1,coord[1]-1)) #left upper
        ret.append((coord[0]-1,coord[1]+1)) #right upper
        ret.append((coord[0]+1,coord[1]+1)) #right bottom
        ret.append((coord[0]+1,coord[1]-1)) #left bottom
        ret.append((coord[0]-1,coord[1]))  # top 
        ret.append((coord[0],coord[1]+1)) # right
        ret.append((coord[0]+1,coord[1])) # bottom
        ret.append((coord[0],coord[1]-1)) # left
    if m == 5:
        ret.append((coord[0],coord[1]))
        ret.append((coord[0]-1,coord[1]))  # top 
        ret.append((coord[0],coord[1]+1)) # right
        ret.append((coord[0]+1,coord[1])) # bottom
        ret.append((coord[0],coord[1]-1)) # left
    return ret
def calculateEnergyTerms(neighbors,v_prev,v_next,mag_neighbors,alpha=1,belta=1,gamma=1,coords=None):
    # Econt  continuty term   d_aver - magnitute(vi-vi-1)
    # Ecurv  curative term
    # Eimage image term
    # we also need alpha betla..

    Econt = Ecurv = Eimage = 0
    distances = calculate_distances(coords,do_interpolation= False)
    d_aver = np.mean(distances)

    mags =  mag_neighbors

    Econt_lst = []
    Ecurv_lst = []
    Eimage_lst = []
    E = []
    mag_min = min(mags)
    mag_max = max(mags)


    if (mag_max-mag_min)<5: # make sure the difference is not too great
        mag_min = mag_max -5
    for i in range(len(neighbors)):
        mag = mags[i]
        v_cur = np.array(neighbors[i])
        v_prev = np.array(v_prev)
        v_next = np.array(v_next)

        E_cont = abs(d_aver - LA.norm(v_cur-v_prev))
        E_curv = LA.norm(v_prev-2*v_cur+v_next)**2
        E_image = (mag_min -mag)/(mag_max-mag_min)

        Econt_lst.append(E_cont)
        Ecurv_lst.append(E_curv)
        Eimage_lst.append(E_image)
        
    # normlize first two energy term 
    max_Econt = max(Econt_lst)
    max_Ecurv = max(Ecurv_lst)
    for i in range(len(Econt_lst)):

        Econt_lst[i] = Econt_lst[i]/max_Econt

        Ecurv_lst[i] = Ecurv_lst[i]/max_Ecurv



    # calculate the energy term

    Etotal = 0
    for i in range(len(Econt_lst)):

        Etotal = gamma*Eimage_lst[i] + alpha*Econt_lst[i] + belta*Ecurv_lst[i] 
    
        E.append(Etotal)

    return E
def getIntergerCoords(coords):
    '''
    convert coords of float to coords of int in order to do neighbor accessing
    '''
    ret = []
    for i in range(len(coords)):
        ret.append((int(coords[i][0]),int(coords[i][1])))
    return ret

def greedy(threshold1=10,threshold2=10,threshold3 = 50):
    # Econt  continuty term   d_aver - magnitute(vi-vi-1)
    # Ecurv  curative term
    # Eimage image term
    global coords
    ptsmoved = 0

    global gradient_mag
    global alpha
    global belta
    global gamma
    alpha = []
    belta = []
    gamma = []
#    alpha = 1
#    belta = 1
#    gamma = 1.2
    for i in range(len(coords)):
        alpha.append(0.5)
        belta.append(1.5)
        gamma.append(1.2)

    fraction = len(coords)/10.0
    while ptsmoved < threshold3:
        moved_in_iteration = 0
        for i in range(len(coords)):
            Emin = 100*100*100
            v_cur = coords[i]
            neighbors = getNeighbor(v_cur,m=9)
            v_prev = None
            v_next = None

            v_prev = coords[len(coords)-1] if i==0 else coords[i-1]
            v_next = coords[0] if i==len(coords)-1 else coords[i+1]

            mag_neighbors = [gradient_mag[neighbor[0]][neighbor[1]] for neighbor in neighbors]


            E = calculateEnergyTerms(neighbors,v_prev,v_next,mag_neighbors,alpha=alpha[i],belta=belta[i],gamma=gamma[i],coords=coords)
            Emin = np.min(E)
            jmin = np.argmin(E)


            coords[i] = neighbors[jmin]

            if Emin!= E[0]: # E[0] has the energy for current point
                moved_in_iteration = moved_in_iteration + 1
                ptsmoved = ptsmoved +1

        C = [] # better approximation of curvature
      
        for i in range(len(coords)):
            v_prev = None
            v_next = None
            v_cur = coords[i]

            v_prev = coords[len(coords)-1] if i==0 else coords[i-1]
            v_next = coords[0] if i==len(coords)-1 else coords[i+1]
            u_i = (v_cur[0]-v_prev[0],v_cur[1]-v_prev[1])
            u_next = (v_next[0]-v_cur[0],v_next[1]-v_cur[1])
            with np.errstate(divide='ignore',invalid='ignore'):
                c = LA.norm(u_i/LA.norm(u_i) - u_next/LA.norm(u_next))**2
            C.append(c)
        th1_lst= []
        th2_lst = []
        for i in range(len(C)):

            c_prev = C[len(C)-1] if i==0 else C[i-1]
            c_next = C[0] if i==len(C)-1 else C[i+1]

            c_cur = C[i]
            v_cur = coords[i]
            th1_lst.append(c_cur)
            th2_lst.append(gradient_mag[v_cur[0]][v_cur[1]])
            if c_cur == max([c_prev,c_next,c_cur]) and c_cur>threshold1 and gradient_mag[v_cur[0]][v_cur[1]] > threshold2:

                belta[i] = 0
            
        if (ptsmoved > 3):
            pass
#            _thread.start_new_thread(intermediate,("thread",2,copy.deepcopy(coords)))
        if (ptsmoved  >20 ):
            pass
 #           _thread.start_new_thread(intermediate,("thread",3,copy.deepcopy(coords)))

        if (moved_in_iteration<fraction):
            pass


    return coords

    


#global coords

def takeSequence():
    global startFile
    global endFile
    files = glob.glob('./Sequence2/deg*')

    startFile = files[0]
    endFile = files[-1]
    return files[1:]


if __name__ == "__main__":
    # declare global variables
    startFile = None
    endFile = None
    interFiles = None
    interFiles = takeSequence()

    end = False
    filename = startFile
    im = plt.imread(filename)
    fig = plt.figure()
    ax = fig.add_subplot(1,2,1)
    ax.imshow(im)
    data = copy.deepcopy(im)

    im = im.astype(float)
    im = apply_gaussian_filter(data = im,filter_shape=(3,3),std=0.8)
    im_gradients = np.gradient(im)
    row_gradient = im_gradients[0]
    column_gradient = im_gradients[1]
    gradient_mag = np.sqrt(row_gradient**2+column_gradient**2)

    ready = False
    coords = []
    new_coords = []
    alpha = []
    belta = []
    gamma = []
    Number_Of_Points_Between_Interpolation = 3
    Start_After_N_Click = 20
    threshold1 = 0.3
    threshold2 = 7
    threshold3 = 10
    pic2 = False
    pic3 = False

    
    cid = fig.canvas.mpl_connect('button_press_event',onclick)
    
    show()
    


    
