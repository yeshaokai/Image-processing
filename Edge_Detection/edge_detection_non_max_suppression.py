import numpy as np
import matplotlib.pylab as plt
import copy
# my own code on gaussian
def apply_gaussian_filter(data,filter_shape,std):
    fd1 = filter_shape[0]
    fd2 = filter_shape[1]
    if fd1<=3 or fd2<=3:
        raise Exception("invalid gaussian filter.")

    x = np.arange(fd1) # about to generate samples in x. Make sure they are all positive                                                                         
    y = np.arange(fd2)

    mean_x = np.mean(x)# get the value of the discrete values                                                                                                    
    mean_y = np.mean(y)


    if fd1%2 == 0 or fd2%2 ==0:  # if it is even                                                                                                                 
        raise Exception("i don't want to support even gaussian for computation sake")
    width1 = fd1*std
    width2 = fd2*std

    sample_x = np.linspace(mean_x-((fd1-1)/2),mean_x+((fd1-1)/2),num=fd1)

    sample_y = np.linspace(mean_y-((fd2-1)/2),mean_x+((fd2-1)/2),num=fd2)

    gaussian1d_x = np.exp((-(sample_x-mean_x)**2)/(2.0*std**2)) # feed gaussian function the sample points                                                       
    min_gaussian_x = np.min(gaussian1d_x)
    gaussian1d_x = gaussian1d_x/min_gaussian_x # normalize it so the smallest entry is 1                                                                         
    gaussian1d_x = np.floor(gaussian1d_x) # round the nearest integer                                                                                            

    gaussian1d_y = np.exp((-(sample_y-mean_y)**2)/(2.0*std**2))
    min_gaussian_y = np.min(gaussian1d_y)
    gaussian1d_y = gaussian1d_y/min_gaussian_y
    gaussian1d_y = np.floor(gaussian1d_y)

#    im = plt.imread(imgname)
    filter_shape = gaussian1d_x.shape
    data_shape = data.shape
    datanew = copy.deepcopy(data)
    gaussian1d_x = gaussian1d_x/(np.sum(gaussian1d_x))  # normalize by the sum                                                                                   
    gaussian1d_y = gaussian1d_y/(np.sum(gaussian1d_y))


    for c in range(data_shape[0]):
        datanew[c,:] = np.convolve(datanew[c,:],gaussian1d_y,'same')
    for r in range(data_shape[1]):
        datanew[:,r] = np.convolve(datanew[:,r],gaussian1d_x,'same')

    return datanew

def get_numerical_gradient(data):
    #numpy.gradient uses central difference quotient to calculate the gradient
    #d(instensity)/d(pixel), where d(pixel) is assumed to be one by default
    #for first pixel, it uses forward difference, for last pixel, it uses backward difference, for anything in between, it uses central difference
    gradients = np.gradient(data) # np.gradient on 2d gives two arrays.The first array is gradient on x and the second is gradient on y
    
    return gradients
def get_edge_strength(Jx,Jy):
    return np.sqrt(np.square(Jx)+np.square(Jy))
def get_edge_orientation(Jx,Jy):
    with np.errstate(divide='ignore',invalid='ignore'):        
        c = np.true_divide(Jy,Jx)
        c[c==np.inf]=0
        c=np.nan_to_num(c)
        
        return np.arctan(c)

def CANNY_ENHANCER(data,std):
    
    data = apply_gaussian_filter(data,filter_shape=(5,5),std=std)


    gradients = get_numerical_gradient(data)

    Jx = gradients[1] # Jx is gradients on x, Jy is gradients on y
    Jy = gradients[0]
    

    edge_strength = get_edge_strength(Jx,Jy)

    
    edge_orientation = get_edge_orientation(Jx,Jy)
    return (data,edge_strength,edge_orientation)

def is_boundary(i,j,shape):
    # not going to thin the boundary, in case that it causes  out of range exception
    
    if (i!=shape[0]-1 and j!=shape[1]-1) and (i!=0 and j!=0):
        return False
    else :
        return True
def NONMAX_SUPPRESSION(im,es,e0):
    # convert edge normals to degree that is from 0 to 180

    e0 = positive_degree_from_radiant(e0)

    # convert continuous degree to 4 distinct degrees
    e0 = slice_degree(e0)

    retData = copy.deepcopy(im)

    for i in range(es.shape[0]):
        for j in range(es.shape[1]):
            direction = e0[i][j]
            if is_boundary(i,j,es.shape):
                continue
            maxS = 0
            if direction == -45:
                maxS= max([es[i-1][j+1],es[i][j],es[i+1][j-1]])
            if direction == 90:
                maxS= max([es[i-1][j],es[i][j],es[i+1][j]])
            if direction ==0:
                maxS= max([es[i][j-1],es[i][j],es[i][j+1]])
            if direction == 45:
                maxS = max([es[i-1][j-1],es[i][j],es[i+1][j+1]])
            if es[i][j]!=maxS:
                retData[i][j] = 0
            else:
                retData[i][j] = es[i][j]            
    return (e0,retData)
def positive_degree_from_radiant(data):
#    data = np.rad2deg(data)
    data = np.degrees(data)

    return data
def slice_degree(data):
    
    data[np.where(np.logical_and(data<=90, data>=67.5))]= 90
    data[np.where(np.logical_and(data>=22.5 ,data<=67.5))]= 45
    data[np.where(np.logical_and(data>=-22.5 , data<=22.5))]=0
    data[np.where(np.logical_and(data>=-67.5 , data<=-22.5))]=-45
    data[np.where(np.logical_and(data>=-90 , data<=-67.5))]=90

    return data
def traverse(after_nonmax,visited,pass_threshold,i,j,e0,shape,Tl,Th):
    direction = e0[i][j]
    if (i,j) in visited:
        return
    if is_boundary(i,j,shape):
        return 
    visited.add((i,j))
    if after_nonmax[i][j]>=Tl:
        pass_threshold.add((i,j))
    if after_nonmax[i][j]<Tl:
        return
    if direction == 0:        
        traverse(after_nonmax,visited,pass_threshold,i-1,j,e0,shape,Tl,Th)
        traverse(after_nonmax,visited,pass_threshold,i+1,j,e0,shape,Tl,Th)
    if direction == 90:
        traverse(after_nonmax,visited,pass_threshold,i,j-1,e0,shape,Tl,Th)
        traverse(after_nonmax,visited,pass_threshold,i,j+1,e0,shape,Tl,Th)
    if direction == 45:
        traverse(after_nonmax,visited,pass_threshold,i-1,j+1,e0,shape,Tl,Th)
        traverse(after_nonmax,visited,pass_threshold,i+1,j-1,e0,shape,Tl,Th)
    if direction == -45:
        traverse(after_nonmax,visited,pass_threshold,i-1,j-1,e0,shape,Tl,Th)
        traverse(after_nonmax,visited,pass_threshold,i+1,j+1,e0,shape,Tl,Th)
def HYSTERESIS_THRESH(after_nonmax,e0,Tl,Th):
    # after_nonmax : output from nonmax_suppression
    # e0 : edge orientation(note that this e0 is already processed by the previous procedure )
    # Tl : low threshold 
    # Th : high threshold 
    if (Tl>=Th):
        raise ValueError("Tl must be less than Th")
    visited = set()
    pass_threshold = set()
    retData = copy.deepcopy(after_nonmax)
    shape = retData.shape

    for i in range(shape[0]):
        for j in range(shape[1]):
            coord = (i,j)
            # not going to process those on boundary
            if is_boundary(i,j,shape):
                continue
            if coord not in visited:
                if retData[i][j]> Th:
                    # only when that pixel is larger than Th that we process its connected neighbor

                    traverse(after_nonmax,visited,pass_threshold,i,j,e0,shape,Tl,Th)                                                                                    
            else:
                # pass those processed
                continue


    for i in range(retData.shape[0]):
        for j in range(retData.shape[1]):
            if (i,j) not in pass_threshold:
                retData[i][j]=0
    return retData

im = plt.imread("einstein.jpg")
im = im.astype(float)

gaussian,edge_strength,edge_orientation=CANNY_ENHANCER(im,std=2)
fig = plt.figure()
# nonmax_suppression change the value of edge_orientation

edge_orientation,after_nonmax= NONMAX_SUPPRESSION(gaussian,edge_strength,edge_orientation)

after_threshold = HYSTERESIS_THRESH(after_nonmax,edge_orientation,2,5)

a = fig.add_subplot(2,3,1)
a.set_title('after gaussian std =0.8')
plt.imshow(gaussian,cmap='gray')
a = fig.add_subplot(2,3,2)
a.set_title('edge strength')
plt.imshow(edge_strength,cmap='gray')
a = fig.add_subplot(2,3,3)
a.set_title('edge orientation')
plt.imshow(edge_orientation,cmap='gray')    
a = fig.add_subplot(2,3,4)
a.set_title('after thinning')
plt.imshow(after_nonmax,cmap='gray')
a = fig.add_subplot(2,3,5)
a.set_title('after thresholding Tl=2 Th=5')
plt.imshow(after_threshold,cmap='gray')
a = fig.add_subplot(2,3,6)
a.set_title('histogram')




indices = np.random.choice(len(after_nonmax.flatten()),100)
plt.hist(after_nonmax.flatten()[indices],bins=np.arange(0,np.max(after_nonmax),10))

plt.show()


