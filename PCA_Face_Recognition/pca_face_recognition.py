from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import glob
import numpy as np
from numpy import linalg as LA
from sklearn.preprocessing import normalize
import scipy
import time
# required module -> sklearn
# building training and target as X, y
# image -> "subject #" mapped list

def getSubject(file):

    number = ''
    for s in file:
        if s.isdigit():
            number = number + s

    return number
def rgb2gray(im):

    if (len(im.shape) <3):
        return im

    R = im[:,:,0]
    G = im[:,:,1]
    B = im[:,:,2]
    im_gray = R*299./1000 + G*587./1000 + B*114./1000
    return im_gray

files = glob.glob('yalefaces_centered_small/*')

subject_im = {}
subject_target = {}
subject_train = {}
subject_test = {}

for file  in files:
    if ".db"  in file:
        continue
    im = plt.imread(file)
    im = rgb2gray(im)

    subject = getSubject(file)
    if not subject in subject_im:
        subject_im[subject] = []
    if not subject in subject_target:
        subject_target[subject]  = []

    subject_im[subject].append(np.array(im.flatten()))

    subject_target[subject].append(np.array(subject))

# take 8 pic from each person for training, 2 pic from each person for testing
for key in subject_im.keys():
    # construct dict : subject_train and subject_test
    # the value of those dictionary are randomized result from 10 pictures

    X_train ,X_test,y_train,y_test = train_test_split(subject_im[key],subject_target[key],test_size = 0.2)
#    print (X_train)
    subject_train[key] = X_train
    subject_test[key] = X_test


import sys


X_train = np.array([]).reshape((0,154*116))
X_test = np.array([]).reshape((0,154*116))
y_train = []
y_test = []

# forming  global training_data

for key in subject_train:
    array = np.array(subject_train[key])

    X_train = np.vstack((X_train,array))

# forming  global test_data
for key in subject_test:
    array = np.array(subject_test[key])
    X_test = np.vstack((X_test,array))



average_face_from_train = np.mean(X_train,axis = 0)
average_face_from_test = np.mean(X_test,axis = 0)
average_face_from_train = average_face_from_train.flatten()

'''
ax.imshow(average_face_from_train.reshape(154,116),cmap='gray')
plt.show()
sys.exit()
'''
def PCA(m,k=20):
    # note that m must be a mean subtracted , image as column vector matrix
    # returns eigen vectors
    # because PCA method does returns huge array, we only choose the first few
    X = np.dot(m,m.T)
    
    w,v = np.linalg.eigh(X)
    indices = np.argsort(w)
    indices = indices[::-1]
    indices = indices[:k]
    return v[:,indices]
def SVD(m,k=20):

    n = (m).shape[0]
    
    Y = (m.T)/(np.sqrt(n-1))

#    sys.exit()
    U,s,V = scipy.linalg.svd(Y,full_matrices=False)
    V = V.T
    indices = np.argsort(s)
    indices = indices[::-1]
    indices = indices[:k]

    return V[:,indices]

def TurkPentland(m,k = 20):

    A = m/(np.sqrt(m.shape[1]))
    symmetry_M  =np.dot(A.T,A)
    
    w,v = LA.eig(symmetry_M)
    
    indices = np.argsort(w)
    indices = indices[::-1]
    indices = indices[:k]

    v = np.dot(A,v)
    for i in range(v.shape[1]):
        v[:,i] = v[:,i]/np.linalg.norm(v[:,i])

    return v[:,indices]
def transferBasis(X,eigenfaces):


    return np.dot(eigenfaces.T,X)

# substract average from each face
def getRebaseImage(Y,eigenfaces):

    ret = np.dot(eigenfaces,Y)

#    for i in range(ret.shape[1]):
#        ret[:,i] = ret[:,i]+mean_face

    return ret

    
def reprojectedAndTheirOriginal(eigenfaces,testSize = 10):
    global X_test

    indices = np.random.choice(X_test.shape[1],10,replace=True)

    test_samples = X_test[:,indices]

#    sys.exit(

    Y = transferBasis(test_samples,eigenfaces)
    fig = plt.figure()
    Rebasis = getRebaseImage(Y,eigenfaces)    

    for i in range(1,testSize+1):
        ax = fig.add_subplot(4,5,i)
        image = Rebasis[:,i-1]
        try:
            image = image+ average_face_from_train
            image = image.reshape((154,116))

            ax.imshow(image,cmap='gray')
        except ValueError:
            print (image.shape)

    for i in range(1,testSize+1):
        ax = fig.add_subplot(4,5,10+i)
        image = test_samples[:,i-1]
        image = image+ average_face_from_test
        image = image.reshape((154,116))
        ax.imshow(image,cmap='gray')
        
    plt.show()
def testRunningTime(X_train):
 
    t = time.time()
    print ("entering TurkPentland method")
    eigenfaces = TurkPentland(X_train)
    print ("it took " + str(time.time()-t) + " seconds")
    print ("entering SVD method")
    t = time.time()
    eigenfaces = SVD(X_train)
    print ("it took " + str(time.time()-t) + " seconds")
    print ("entring direct covariance  method")
    t = time.time()
    eigenfaces = PCA(X_train)
    print ("it took " + str(time.time()-t) + " seconds")

def testN(X_train):
    # using turk and pentland's method 
    # pick a single image for testing this 
    global X_test
    testImage = X_test[:,12]
    fig = plt.figure()
    eigenfaces = TurkPentland(X_train,k=1)
    Y = transferBasis(testImage,eigenfaces)
    
    Rebasis = getRebaseImage(Y,eigenfaces)
    
    Rebasis=Rebasis.reshape((154,116))
    ax = fig.add_subplot(2,2,1)
    ax.imshow(Rebasis,cmap='gray')
    ax.set_title("1 eigenface")
    eigenfaces = TurkPentland(X_train,k=10)
    Y = transferBasis(testImage,eigenfaces)
    Rebasis = getRebaseImage(Y,eigenfaces)
    Rebasis=Rebasis.reshape((154,116))
    ax = fig.add_subplot(2,2,2)
    ax.imshow(Rebasis,cmap='gray')
    ax.set_title("10 eigenface")
    eigenfaces = TurkPentland(X_train,k=18)
    Y = transferBasis(testImage,eigenfaces)
    Rebasis = getRebaseImage(Y,eigenfaces)
    Rebasis=Rebasis.reshape((154,116))
    ax = fig.add_subplot(2,2,3)
    ax.set_title("18 eigenface")
    ax.imshow(Rebasis,cmap='gray')
    eigenfaces = TurkPentland(X_train,k=120)
    Y = transferBasis(testImage,eigenfaces)
    Rebasis = getRebaseImage(Y,eigenfaces)
    Rebasis=Rebasis.reshape((154,116))
    ax = fig.add_subplot(2,2,4)
    ax.imshow(Rebasis,cmap='gray')
    ax.set_title("120 eigenface")
    plt.show()
def testEigenFaces(X_train):
    eigenfaces = TurkPentland(X_train,k=18)
    fig = plt.figure()
    
    for i in range(1,19):
        ax = fig.add_subplot(4,5,i)
        image = eigenfaces[:,i-1]
        image = image.reshape((154,116))
        ax.set_title("PC "+str(i))
        ax.imshow(image,cmap='gray')
    plt.show()
def findMatches(X_train,eigenfaces):
    global X_test
    indices = np.random.choice(X_test.shape[1],20,replace=True)
    testImages = X_test[:,indices]
    # find new basis

    Y_test = transferBasis(testImages,eigenfaces)
    print (testImages.shape)
    # calculate distance between them and training set images
    print (X_train.shape)
    Y_train = transferBasis(X_train,eigenfaces)

    min_j_list = []
    for i in range(Y_test.shape[1]):
        min_j_list.append([])
        min_dist = 100000
        min_j = 0
        for j in range(Y_train.shape[1]):
            dist = np.linalg.norm(Y_test[:,i]-Y_train[:,j])


            if dist <min_dist:
                min_dist = dist
                min_j = j
        min_j_list[i].append(min_j)
    print (min_j_list)
    fig = plt.figure()


    for i in range(0,10):
        # show the test image first
        ax = fig.add_subplot(4,5,i+1)
        image = testImages[:,i]+ average_face_from_test
        image = image.reshape((154,116))
        ax.imshow(image,cmap='gray')
    for i in range(0,10):
        ax = fig.add_subplot(4,5,10+i+1)
        index = min_j_list[i][0]
        image = X_train[:,index] +average_face_from_train        
        image = image.reshape((154,116))
        ax.imshow(image,cmap='gray')
    plt.show()
def testNonFaces(eigenfaces):
    files = glob.glob('NonfaceImages/*')
    nonFaceSet = []
    nonFaces = []
    for file in files:
        if ".db" in file:
            continue
        im = plt.imread(file)
        im = rgb2gray(im)
        im = im.flatten()
        subject = getSubject(file)
        nonFaceSet.append((subject,im))
        nonFaces.append(im)
    nonFaces = np.array(nonFaces)
    nonFaces = nonFaces.T
    average_nonFaces = np.mean(nonFaces,axis=1)
    for i in range(nonFaces.shape[1]):
        nonFaces[:,i] = nonFaces[:,i]-average_nonFaces

    print (average_nonFaces.shape)
    fig = plt.figure()
    '''
    for i in range(len(nonFaceSet)):
        ax = fig.add_subplot(4,5,i+1)
        X = nonFaceSet[i][1]
        X = X + average_nonFaces
        image = X.reshape((154,116))
        ax.imshow(image,cmap='gray')
    '''
    '''
    for i in range(len(nonFaceSet)):
        X = nonFaceSet[i][1]

        subject = nonFaceSet[i][0]
#        ax = fig.add_subplot(4,5,i+1)
        Y = transferBasis(X,eigenfaces)
        Projected = getRebaseImage(Y,eigenfaces)

        difference = X - Projected
        print (np.linalg.norm(Projected))
        Projected = Projected+average_nonFaces

        norm = np.linalg.norm(difference)
        image = Projected.reshape((154,116))
        difference = difference.reshape((154,116))
        plt.scatter(i,norm)
        #ax.imshow(difference,cmap='gray')
    plt.xlabel('image number')
    plt.ylabel('Frobenius norm')

    indices = np.random.choice(X_test.shape[1],10,replace=True)
        
    test_samples = X_test[:,indices]

    Y = transferBasis(test_samples,eigenfaces)
    Projected = getRebaseImage(Y,eigenfaces)

    for i in range(test_samples.shape[1]):
        ax = fig.add_subplot(4,5,i+1)
        image = test_samples[:,i]
        image = image+ average_face_from_train
        image = image.reshape((154,116))
        ax.imshow(image,cmap='gray')

    for i in range(test_samples.shape[1]):
#        ax = fig.add_subplot(4,5,10+i+1)
        reconstructed = Projected[:,i]
        difference = test_samples[:,i]-reconstructed 
        difference = difference.reshape((154,116))
        norm = np.linalg.norm(difference)
        plt.scatter(i,norm)
        #ax.imshow(difference,cmap='gray')
    plt.xlabel('test number')
    plt.ylabel('Frobenius norm')
    plt.show()
    '''
X_train = X_train - average_face_from_train
#sys.exit()
X_train = X_train.T
X_test = X_test - average_face_from_test
X_test = X_test.T

'''

testRunningTime(X_train)

'''
'''

testN(X_train)

'''
'''

eigenfaces = TurkPentland(X_train)
reprojectedAndTheirOriginal(eigenfaces)

'''
'''
testEigenFaces(X_train)
'''
'''
eigenfaces = TurkPentland(X_train)
findMatches(X_train,eigenfaces)
'''
#eigenfaces = TurkPentland(X_train)
#testNonFaces(eigenfaces)
#testRunningTime(X_train)
#eigenfaces = TurkPentland(X_train)
#eigenfaces = SVD(X_train)

#eigenfaces = PCA(X_train)
#showTestImages(eigenfaces)
#SVD(X_train)
#np.dot(X_train,V)

