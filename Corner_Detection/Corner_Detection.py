import numpy as np
import matplotlib.pylab as plt
import time
import copy


def apply_gaussian_filter(data,filter_shape,std):
    fd1 = filter_shape[0]
    fd2 = filter_shape[1]
    if fd1<=3 or fd2<=3:
        raise Exception("invalid gaussian filter.")

    x = np.arange(fd1) # about to generate samples in x. Make sure they are all positive                         \
                                                                                                                  
    y = np.arange(fd2)

    mean_x = np.mean(x)# get the value of the discrete values                                                    \
                                                                                                                  
    mean_y = np.mean(y)


    if fd1%2 == 0 or fd2%2 ==0:  # if it is even                                                                 \
                                                                                                                  
        raise Exception("i don't want to support even gaussian for computation sake")
    width1 = fd1*std
    width2 = fd2*std

    sample_x = np.linspace(mean_x-((fd1-1)/2),mean_x+((fd1-1)/2),num=fd1)

    sample_y = np.linspace(mean_y-((fd2-1)/2),mean_x+((fd2-1)/2),num=fd2)

    gaussian1d_x = np.exp((-(sample_x-mean_x)**2)/(2.0*std**2)) # feed gaussian function the sample points       \
                                                                                                                  
    min_gaussian_x = np.min(gaussian1d_x)
    gaussian1d_x = gaussian1d_x/min_gaussian_x # normalize it so the smallest entry is 1                         \
                                                                                                                  
    gaussian1d_x = np.floor(gaussian1d_x) # round the nearest integer                                            \
                                                                                                                  

    gaussian1d_y = np.exp((-(sample_y-mean_y)**2)/(2.0*std**2))
    min_gaussian_y = np.min(gaussian1d_y)
    gaussian1d_y = gaussian1d_y/min_gaussian_y
    gaussian1d_y = np.floor(gaussian1d_y)

    filter_shape = gaussian1d_x.shape

    data_shape = data.shape
    datanew = copy.deepcopy(data)
    gaussian1d_x = gaussian1d_x/(np.sum(gaussian1d_x))  # normalize by the sum                                   \
                                                                                                                  
    gaussian1d_y = gaussian1d_y/(np.sum(gaussian1d_y))


    for c in range(data_shape[0]):
        datanew[c,:] = np.convolve(datanew[c,:],gaussian1d_y,'same')
    for r in range(data_shape[1]):
        datanew[:,r] = np.convolve(datanew[:,r],gaussian1d_x,'same')

    return datanew


# compute gradient over image I
def get_numerical_gradient(data):
    gradients = np.gradient(data)
    return gradients
def rgb2gray(im):
    # my implementation of rgb2gray
    if im.shape[2]!=3:
        raise ValueError("wrong dimension for this image")
    R = im[:,:,0]
    G = im[:,:,1]
    B = im[:,:,2]
    img_gray = R*299./1000 + G*587./1000 + B*114./1000
    return img_gray

im = plt.imread("CheckerBoard.jpg")

im = im.astype(float)
if len(im.shape)==3:    
    im = rgb2gray(im)
print ("the shape of the image")
print (im.shape)
gradients = get_numerical_gradient(im)
# i am still not sure...

gradient_x = gradients[0]
gradient_y = gradients[1]

def is_invalid(row,col,window_size,shape):
    if row<(window_size-1)/2 or row >=shape[0]-(window_size-1)/2 or col<(window_size-1)/2 or col>=shape[1]-(window_size-1)/2:

        return True
    else:
        return False
    

def get_C_matrix(row,col,size):

    global im

    # size is some 2N+1
    # make an exception of it if is not

    if size<1 or size%2==0:
        raise ValueError("invalid size")


    global gradient_x
    global gradient_y

    Ex = gradient_x # in the direction of row
    Ey = gradient_y # in the directino of column

    # get the neighborhood
    origin_x = int(col - (size-1)/2)
    origin_y = int(row - (size-1)/2)
    
    column_range = range(origin_x,origin_x+size)

    row_range = range(origin_y,origin_y+size)

    Ex = Ex[row_range,column_range]
    Ey = Ey[row_range,column_range]
    
    item0 = np.sum(np.square(Ex)) # Sigma(Ex^2)
    item1 = np.sum(Ex*Ey) # Sigma(Ex*Ey)
    item2 = item1
    item3 = np.sum(np.square(Ey))
    C = np.array([item0,item1,item2,item3]).reshape(2,2)
    # calculate the eighValues
    return C
def get_lamda_pairs(C):
    
    eigenvalues = np.linalg.eigh(C)[0]
    lamda1 = eigenvalues[1]
    lamda2 = eigenvalues[0] # lamda2 is the smaller one

    return (lamda1,lamda2)
def within_neighborhood(coord1,coord2,size):
    # as we know, we filter out those invalid points. So we can safely calculate anything
    row0,row1 = coord1[0],coord2[0]
    col0,col1 = coord1[1],coord2[1]
    row_range = [row0-int((size-1)/2),row0+int((size-1)/2)]
    col_range = [col0 - int((size-1)/2),col0+int((size-1)/2)]
#    print (row_range,col_range)
#    print ("top coord"+str((row0,col0)))
#    print ("bot coord"+str((row1,col1)))
    if row1 <=row_range[1] and row1 >= row_range[0] and col1>= col_range[0] and col1<=col_range[1]:

        return True
    else:
        return False
    
def CORNERS(data,window_size,threshold,std):
    # apply gaussian filter
    lst = []
    lamda2_lst = []
    data  =  apply_gaussian_filter(data = data,filter_shape = (5,5),std = std)
    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            if not is_invalid(row,col,window_size,data.shape):
                C = get_C_matrix(row,col,window_size)
                lamda2 = get_lamda_pairs(C)[1]
                lamda2_lst.append(lamda2)
                if lamda2> threshold:
                    lst.append((lamda2,row,col))
    # sort lst and make it descresed
    lst = sorted(lst,reverse = True)

    lst = [(e[1],e[2]) for e in lst]

    dirty_index = set()
    top_coord = lst[0]
    bot_coord = lst[0]
    overlap = True
    for i in range(1,len(lst)): # so it does not index out of the range
        if overlap == False:
            top_coord = lst[i]
            overlap = True
            continue
        bot_coord = lst[i]
        if within_neighborhood(top_coord,bot_coord,window_size):
            dirty_index.add(i)
            overlap = True
        else:
            overlap = False
    retLst = []
    lst.pop() # the last one is not used anyway
    for i in range(len(lst)):
        if i in dirty_index:
            continue
        else:
            retLst.append(lst[i])
    print ("dirty")
    print (len(dirty_index))
    return (retLst,lamda2_lst)
start_time = time.time()
retLst,lamda2_lst = CORNERS(data=im,window_size=9,threshold=1800,std=1.2)
indices = np.random.choice(len(lamda2_lst),100)
fig = plt.figure()
a = fig.add_subplot(1,1,1)
a.set_title("std=1.2,neighbor=9,threshold=1800")
lamda2_lst = np.array(lamda2_lst)
#plt.hist(lamda2_lst[indices],bins=np.arange(0,00,5))
y = [e[0] for e in retLst]
x = [e[1] for e in retLst]
plt.imshow(im,cmap='gray')
plt.plot(x,y,c='w',marker='s',markersize=3,markeredgewidth=1,markerfacecolor='None',linestyle='None')
elapsed_time =time.time()-start_time
print ("lasts "+str(elapsed_time))
plt.show()
