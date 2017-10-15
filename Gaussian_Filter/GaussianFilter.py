import numpy as np
import copy
import matplotlib.pylab as plt
from theano.tensor.signal import conv
from numpy import linalg as LA
imgname1 ='NoisyImage1.jpg'
imgname2 ='NoisyImage2.jpg'



def test_mean_filter(imgname,filter_shape):

    im = plt.imread(imgname)

    average_filter_shape = filter_shape # tuple

    average_filter = np.ones(average_filter_shape)/np.sum(average_filter_shape) # normalize the filter 
    
    # theano requires two array to have same data type

    average_filter=average_filter.astype(float)

    im=im.astype(float)
    # this function does not have requirement on image size    
    image_shape = im.shape
    # zero padding is used. the step of slide is 1 by default
    conv_out = conv.conv2d(input =im,filters=average_filter,filter_shape =average_filter_shape,image_shape=image_shape)
# need to evaluate it
    # theano's calculation is symbolic. This evaluation step is needed to compile and return the value
    return (im,conv_out.eval())


def test_discrete_gaussian(std):
    fd1 = 5
    fd2 = 5

    x = np.arange(fd1) # about to generate samples in x. Make sure they are all positive
    y = np.arange(fd2) 

    mean_x = np.mean(x)# get the value of the discrete values
    mean_y = np.mean(y)


    if fd1%2 == 0 or fd2%2 ==0:  # if it is even
        raise Exception("i don't want to support even gaussian for computation sake")

    sample_x = mean_x+(x-mean_x)*std #get the sample points,which are mean-2*std,mean-std,mean,mean+std,mean+2std
    sample_y = mean_y+(y-mean_y)*std
    
    gaussian1d_x = np.exp((-(sample_x-mean_x)**2)/(2.0*std**2)) # feed gaussian function the sample points
    
    plt.plot(sample_x,gaussian1d_x)
    plt.show()


def test_gaussian_filter(imgname,filter_shape,std):

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

    im = plt.imread(imgname)
    filter_shape = gaussian1d_x.shape
    image_shape = im.shape
    imnew = copy.deepcopy(im)
    
    ##
#    gaussian1d_x=gaussian1d_y=np.array([1,9,18,9,1])
    ##

    gaussian1d_x = gaussian1d_x/(np.sum(gaussian1d_x))  # normalize by the sum
    gaussian1d_y = gaussian1d_y/(np.sum(gaussian1d_y))


    for c in range(image_shape[1]):
        imnew[c,:] = np.convolve(imnew[c,:],gaussian1d_y,'same')
    for r in range(image_shape[0]):
        imnew[:,r] = np.convolve(imnew[:,r],gaussian1d_x,'same')
    print (gaussian1d_x)
    return (im,imnew)
    
def test_median_filter(imgname,size):
    im = plt.imread(imgname)
    im_original = copy.deepcopy(im)
    image_shape = im.shape
    # one of the ways to handle boundary issue is to avoid boundary elements (cited from wiki:median filter)
    edgex = size[0]//2
    edgey = size[1]//2
    filter = np.zeros(size).flatten()
    for c in range(edgex,image_shape[0]-edgex):
        for r in range(image_shape[1]-edgey):
            i = 0
            for fx in range(size[0]):
                for fy in range(size[1]):
                    filter[i] = im[c+fx-edgex][r+fy-edgey]
                    i = i+1
            filter = sorted(filter)
            im[c,r] = filter[size[0]*size[1]//2]
    return im_original,im
#(imbefore,imafter) = test_mean_filter(imgname,(8,8))


# note that standard deviation should be larger than 0.8 from ... all the math

fig = plt.figure()


'''
imbefore,imafter1 = test_mean_filter(imgname1,(3,3))
imbefore,imafter2 = test_mean_filter(imgname1,(5,5))
a= fig.add_subplot(1,3,1)
a.set_title('original')
plt.imshow(imbefore,cmap='gray')
a= fig.add_subplot(1,3,2)
a.set_title('mean filter with (3,3)')
plt.imshow(imafter1,cmap='gray')
a= fig.add_subplot(1,3,3)
a.set_title('mean filter with (5,5)')
plt.imshow(imafter2,cmap='gray')
'''  # uncomment for testing mean filter


'''
imbefore,imafter1=test_gaussian_filter(imgname2,(5,5),0.8)

imbefore,imafter2=test_gaussian_filter(imgname2,(5,5),2)

a= fig.add_subplot(1,3,1)
a.set_title('original')
plt.imshow(imbefore,cmap='gray')
a= fig.add_subplot(1,3,2)
a.set_title('gaussian smoothing with std 0.8')
plt.imshow(imafter1,cmap='gray')
a= fig.add_subplot(1,3,3)
a.set_title('gaussian smoothing with std 2')
plt.imshow(imafter2,cmap='gray')
#test_discrete_gaussian(0.8)
''' # uncomment for testing gaussian filter


'''
imbefore,imafter1 = test_median_filter(imgname2,(3,3))
imbefore,imafter2 = test_median_filter(imgname2,(7,7))

a= fig.add_subplot(1,3,1)
a.set_title('original')
plt.imshow(imbefore,cmap='gray')
a= fig.add_subplot(1,3,2)
a.set_title('median smoothing (3,3)')
plt.imshow(imafter1,cmap='gray')
a= fig.add_subplot(1,3,3)
a.set_title('median smoothing (7,7)')
plt.imshow(imafter2,cmap='gray')
'''    # uncomment for testing median filter


plt.show()
