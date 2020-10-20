import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import matplotlib.ticker as ticker


########################################################################
from matplotlib.patches import Ellipse
from sklearn import mixture

import multiprocessing
from multiprocessing import Process
import sys
import os
lock = multiprocessing.Lock()

def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))
        
def plot_gmm(gmm, X, ww, label=True, ax=None):
   # ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
   # if label:
   #    ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
   # else:
   #   ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
   # ax.axis('equal')
    
   # w_factor = 0.2 / gmm.weights_.max()

    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
      kernel = multivariate_normal(mean=pos, cov=covar)
      ww += kernel.pdf(xxyy)  
      #temp = kernel.pdf(xxyy)
      #draw_ellipse(pos, covar, alpha=w * w_factor)

########################################################################
#blur the curve Horizontally
from scipy import signal

def blurCurve(inputData, windowSize, stdv, uRes, sRes):
  img = inputData.reshape(uRes,sRes)
  window = signal.gaussian(windowSize, std=stdv)
  filteredData = [float(0.0) for j in range(uRes*sRes)]
  imgFiltered = np.asarray(filteredData).reshape(uRes,sRes)

  for i in range(1, sres):
    imgFiltered[i] = signal.convolve(img[i], window, mode='same') / sum(window)

  img = np.transpose(img)
  #imgFiltered = np.transpose(imgFiltered)
  filteredData = imgFiltered.reshape(uRes*sRes)
  return filteredData
########################################################################

#Generate a distribution of points matcthing the curve
def fitDistr(inputDistr, uRes, vRes, sRes, nGMM, blockOffX, blockOffY, filehandle):
  inputDistr = inputDistr / sum(inputDistr)

  points = np.random.choice(a = uRes*vRes*sRes*sRes, size = 500000, p = inputDistr)
  number_points = len(points)
  pointList = []

  #from 1D index to 4D index
  for i in range(number_points):
    pointIdx = points[i]
    idx4 = pointIdx / (ures*vres * sres)
    idx30ff = pointIdx % (ures*vres * sres)
    idx3 = idx30ff / (ures*vres)
    idx20ff = idx30ff % (ures*vres)
    idx2 = idx20ff / (ures)
    idx1 = idx20ff % (ures)

    temp = [idx1 * ustep, idx2 * vstep, idx3 / float(sres) * 2.0 - 1.0, idx4 / float(sres) * 2.0 - 1.0]
    pointList.append(temp)

  pointArray = np.asarray(pointList)
  fittedData = np.asarray(zzList)

  print("Generate a distribution of points matcthing the curve DONE!")

  gmm = mixture.GaussianMixture (n_components=nGMM, covariance_type='full', random_state=0)
  #plot_gmm(gmm, pointArray, ww)
  #ax = plt.gca()
  #ax.scatter(pointArray[:, 0], pointArray[:, 1], s=40, zorder=2)
  #ax.axis('equal')

  gmm.fit(pointArray).predict(pointArray)
	
  for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
    pos[0] += blockOffX
    pos[1] += blockOffY
    for i in range(4):
      filehandle.write('%s\n' % pos[i])
    for i in range(4):
      for j in range(4):
        filehandle.write('%s\n' % covar[i][j])
    filehandle.write('%s\n' % w)


    #print(w)
   # kernel = multivariate_normal(mean=pos, cov=covar)
   # fittedData += kernel.pdf(uvstArray)  
  #return fittedData
########################################################################
########################################################################

#Generate a distribution of points matcthing the curve
def fitDistr2(pointArray, uRes, vRes, sRes, nGMM, blockOffX, blockOffY, meanPoints, filehandle):

  gmm = mixture.GaussianMixture (n_components=nGMM, covariance_type='full', random_state=0, weights_init=np.ones(nGMM) / nGMM,
      means_init=np.asarray(meanPoints))
  #plot_gmm(gmm, pointArray, ww)
  #ax = plt.gca()
  #ax.scatter(pointArray[:, 0], pointArray[:, 1], s=40, zorder=2)
  #ax.axis('equal')

  gmm.fit(pointArray).predict(pointArray)
	
  for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
    #pos[0] += blockOffX
    #pos[1] += blockOffY
    for i in range(4):
      filehandle.write('%s\n' % pos[i])
    for i in range(4):
      for j in range(4):
        filehandle.write('%s\n' % covar[i][j])
    filehandle.write('%s\n' % w)
  print("Fit distribution done!")
########################################################################
#Generate a distribution of points matcthing the curve

def fitDistr2D(inputDistr, uRes, sRes, nGMM, blockOffX, blockOffY, usArray):
  inputDistr = inputDistr / sum(inputDistr)

  points = np.random.choice(a = uRes*sRes, size = 10000, p = inputDistr)
  number_points = len(points)
  pointList = []

  #from 1D index to 4D index
  for i in range(number_points):
    pointIdx = points[i]
    idx2 = pointIdx / (ures)
    idx1 = pointIdx % (ures)

    temp = [blockOffX + idx1 * ustep, idx2 / float(sres) * 2.0 - 1.0]
    pointList.append(temp)

  pointArray = np.asarray(pointList)

  print("Generate a distribution of points matcthing the curve DONE!")

  gmm = mixture.GaussianMixture (n_components=nGMM, covariance_type='full', random_state=0)
  #plot_gmm(gmm, pointArray, ww)
  #ax = plt.gca()
  #ax.scatter(pointArray[:, 0], pointArray[:, 1], s=40, zorder=2)
  #ax.axis('equal')

  gmm.fit(pointArray).predict(pointArray)	

  fittedData2D = [float(0.0) for j in range(ures*sres)]
  fittedData = np.asarray(fittedData2D)

  for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
    	kernel = multivariate_normal(mean=pos, cov=covar)
    	fittedData += kernel.pdf(usArray)  
  return fittedData
########################################################################
data = np.loadtxt('E:/Projects/MitsubaSSS/xml/kettle/textureTest/pythonCode/data/Normal4DBlockCurve.txt')
  #np.loadtxt('C:/Users/beibe/Desktop/gaussian/NormalJacobianCurve_256_step_2.txt')
col0 = data[:,0] # u
col1 = data[:,1] # v
col2 = data[:,2] # normal.x 
col3 = data[:,3] # normal.y
#get the Jacobian
J1=data[:,4]
J2=data[:,5]
J3=data[:,6]
J4=data[:,7]

blockX = int(J1[0])
blockY = int(J2[0])

inputULength = int(col0[0] / blockX)
inputVLength = int(col2[0] / blockY)

blockSizeX = int(col0[0] / blockX)
blockSizeY = int(col2[0] / blockY)

inputStep = col1[0]

realUres = int(inputULength / inputStep)
realVres = int(inputVLength / inputStep)

scale = 24

sres = int(realUres * scale) #256
ures = int(realUres * scale) #256 * 5
vres = int(realVres * scale)

urange = (0, inputULength)
vrange = (0, inputVLength)
srange = (-1, 1)

ustep = (urange[1] - urange[0]) / (ures)
vstep = (vrange[1] - vrange[0]) / (vres)
sstep = (srange[1] - srange[0]) / (sres)

realUstep = (urange[1] - urange[0]) / (realUres)
print(realUstep)

#######################################################################
sigmaH = realUstep * 1.5 / np.sqrt(12.0)#realUstep / np.sqrt(8 * np.log(2))
sigmaR = 0.01
sigmaRSqu = float(sigmaR*sigmaR)

c1 =  1.0 / (sigmaH * sigmaH)
c2 =  1.0 / sigmaRSqu

factor = 64
targetGMMCount = int(realUres * realVres / factor)
inputGMMCount = int(realUres * realVres)
print(blockX)

#zzList = [float(0.0) for j in range(ures*vres * sres*sres)]

#define a place to stored the learned gaussian

def learnBlock(iblock, jblock):
	print("Learn Block!")

	idxBlock = (iblock * blockY + jblock) * inputGMMCount
	print(idxBlock)
	#######################################################################
	#uvstList = []
	#for g in range(sres):
	#    t = srange[0] + g * sstep
	#    for k in range(sres):
	#        s = srange[0] + k * sstep
	#        for j in range(vres):
	#        	v = vrange[0] + (j + jblock * vres) * vstep
	#        	for i in range(ures):
	#        		u = urange[0] + (i + iblock * ures) * ustep
	#        		temp = [u,v,s,t]
	#        		uvstList.append(temp)
	#
	#uvstArray = np.asarray(uvstList)
	#print(uvstArray)

	#zz = np.asarray(zzList)

	#w = 1.0

	########################################################################
	#2D vis stuff
	#uvstList2D = []	
	#for j in range(sres):
	#    s = srange[0] + j * sstep
	#    for i in range(ures):
	#    	u = urange[0] + (i + iblock * ures) * ustep	   
	#    	temp = [u,s]
	#    	uvstList2D.append(temp)
	#
	#uvstArray2D = np.asarray(uvstList2D)
	#print(uvstArray2D)
#
	#zzList2D = [float(0.0) for j in range(ures*sres)]
	#zz2D = np.asarray(zzList2D)
#
	#x1,y1 = np.mgrid[urange[0] + iblock * ures* ustep : urange[1] + iblock * ures* ustep : ustep,
    #             srange[0]:srange[1]:sstep]
#
	#xxyy = np.c_[np.transpose(x1).ravel(), np.transpose(y1).ravel()]
	#print(xxyy)
	########################################################################
	#get input distribution
	path = 	'E:/Projects/MitsubaSSS/xml/kettle/textureTest/pythonCode/blurred16Fit64/fittedGaussian4D' + '_' +str(iblock)  + '_'+ str(jblock) + '.txt'
	#with open(path, 'w') as filehandle:
	#	filehandle.write('%d\n' % targetGMMCount)
	gausPoints = []
	meanPoints = []

	for i in range(inputGMMCount):
	  idx = int(idxBlock + i + 1)

	  mu = (col0[idx], col1[idx], col2[idx], col3[idx])  
	  iFlake = int(idx / realVres)
	  jFlake = int(idx % realVres)
	  #print(iFlake)
	  factorD = int(np.sqrt(factor))
	  if iFlake % factorD == int(factorD/2) :
	  	if jFlake % factorD == int(factorD/2) :
	  		meanPoints.append(mu)
	  print(mu)
	  matrixJ = np.mat([[J1[idx],J2[idx]],[J3[idx],J4[idx]]])
	  matrixJT = matrixJ.T
	  matrixJTJ = matrixJT * matrixJ
	  M = np.mat([[1,0],[0,1]])
	  Z = np.mat([[0,0],[0,0]])
	
	  invCMatrix = c1 * np.r_[np.c_[M,Z],np.c_[Z,Z]] + c2 * np.r_[np.c_[matrixJTJ, -matrixJT], np.c_[-matrixJ, M]] 
	  invCMatrix2D = np.mat([[c1,0],[0,c2]])

	  #invCMatrix =  np.mat([[c1,0,0,0],[0,c1,0,0],[0,0,c2,0],[0,0,0,c2]])
	  #print(invCMatrix)
	  sigma = np.array(invCMatrix.I)
	  sigma2D = np.array(invCMatrix2D.I)

	  blurSize = 16 * sigmaH
	  blurMatrix = np.matrix([[blurSize * blurSize,0,0,0],[0,blurSize * blurSize,0,0],[0,0,0,0],[0,0,0,0]])
	  tempBlur = np.random.multivariate_normal(mean=mu, cov=blurMatrix, size =int(100000 / inputGMMCount))
	

	  #blurSigma = sigma
	  #blurSigma[0,0] = 2 * blurSigma[0,0]
	  #blurSigma[1,1] = 2 * blurSigma[1,1]
	  temp = np.random.multivariate_normal(mean=mu, cov=sigma, size =int(100000 / inputGMMCount))
	  temp += (tempBlur - mu)
	  gausPoints.append(temp[:])


	gausPointsArray3D = np.asarray(gausPoints)
	#print(gausPointsArray3D.shape)
	gausPoints4D = gausPointsArray3D[0]
	for i in range(1, inputGMMCount):
	  gausPoints4D = np.concatenate((gausPoints4D, gausPointsArray3D[i]), axis=0)
	#print(np.asarray(gausPoints4D).shape)
	
	  #kernel = multivariate_normal(mean=mu, cov=sigma)
	  #zz += kernel.pdf(uvstArray)

	  #kernel2D = multivariate_normal(mean=(col0[idx], col2[idx]), cov=sigma2D)
	  #zz2D += kernel2D.pdf(uvstArray2D)
	  #print(zz)
#
	  #for k in range(4):
	  #	filehandle.write('%s\n' % mu[k])
	  #for k in range(4):
	  #	for j in range(4):
	  #		filehandle.write('%s\n' % sigma[k][j])
	  #filehandle.write('%f\n' % w)

	#zzStored = zz  
	
	##comments this, if want to fit the input distribution
	#zzFiltered = blurCurve(zz,32,10,ures,sres)
	#zz = zzFiltered
	#for j in range(ures*sres):
	#	print(zz2D[j])

	#fittedData = fitDistr2D(zz2D, ures, sres, targetGMMCount, iblock * inputULength, jblock * inputVLength, uvstArray2D)
	#img = zz2D.reshape((ures,sres))
	#img = np.transpose(img)
	#im = plt.pcolormesh(x1, y1,img)
	#plt.colorbar()

	with open(path, 'w') as filehandle:
		filehandle.write('%d\n' % targetGMMCount)
		#fitDistr(zz, ures, vres, sres, targetGMMCount, iblock * inputULength, jblock * inputVLength,  filehandle)
		fitDistr2(gausPoints4D, ures, vres, sres, targetGMMCount, iblock * inputULength, jblock * inputVLength,meanPoints, filehandle)
	filehandle.close()

#######################################################################

realBlockx = blockX
realBlocky = blockY

countProcess = 32
countIts = int(realBlockx * realBlockx / countProcess)
#learnBlock(15,11)
#learnBlock(6,7)
#learnBlock(7,6)
#learnBlock(7,7)
#
#######################################################################
#only train part of the flake
#count = 4
#countIts = int(count * count / countProcess)
#centerBlock = blockX / 2
#startX = centerBlock - int(count / 2)
#startY = startX
#blockList = []
#
#for i in range(count):
#	for j in range(count):
#		blockList.append((startX + i ) * blockY + startY + j)
#print(blockList)
#
#for i in range(countIts):
#	processes = []
#	for j in range(countProcess):
#		idxBlockTotal = blockList[i * countProcess + j]
#		iblock = int(idxBlockTotal / realBlocky)
#		jblock = int(idxBlockTotal % realBlocky)
#		if __name__ == '__main__':
#  			p = Process(target=learnBlock, args=(iblock, jblock))
#  			p.start()
#  			processes.append(p)
#	if __name__ == '__main__':
#		for p in processes:
#		    p.join()

for i in range(countIts):
	processes = []
	for j in range(countProcess):
		idxBlockTotal = i * countProcess + j
		iblock = int(idxBlockTotal / realBlocky)
		jblock = int(idxBlockTotal % realBlocky)
		if __name__ == '__main__':
  			p = Process(target=learnBlock, args=(iblock, jblock))
  			p.start()
  			processes.append(p)
	if __name__ == '__main__':
		for p in processes:
		    p.join()

#for some blocks
#processes = []
#for iblock in range(realBlockx - 4, realBlockx):
#  for jblock in range(realBlockx - 4, realBlocky):  
#  	if __name__ == '__main__':
#  		p = Process(target=learnBlock, args=(iblock, jblock))
#  		p.start()
#  		processes.append(p)
#
#if __name__ == '__main__':
#	for p in processes:
#	    p.join()
	
########################################################################
#discard some very large values
#ww *= factor
#
#clamp = min(zzStored.max(),ww.max())
#
#for i in range(1, ures * vres * sres * sres):
#    if zzStored[i] > clamp:
#        zzStored[i] = clamp
#    if ww[i] > clamp:
#        ww[i] = clamp      
#          
##print(zzFiltered.sum())
##print(ww.sum())
#
#zzStored /= zzStored.max()
#ww /= ww.max()
#
##zstored for input distribution
##zzFiltered for blured distribution
##ww for fit blured distribution

#img = ww.reshape((ures,sres))
#img = np.transpose(img)
#im = plt.pcolormesh(x1, y1,img)

########################################################################
#plt.colorbar()
plt.show() 


