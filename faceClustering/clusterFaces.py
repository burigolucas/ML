## Dependencies
import os
import dlib

import PIL
from PIL import Image, ImageOps
PIL.Image.LOAD_TRUNCATED_IMAGES = True

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import pickle

def plot_dataset(filePaths,sampleSize=20,rects=None,title=None,random=False,output=None):
    '''
    Plot sample of dataset
    INPUT:
        filePaths (list)
        sampleSize (int)
        rects (list)
        title (str)
        random (boolean)
    '''
    sampleSize = np.min([sampleSize,len(filePaths)])
    #sampleSize = sampleSize - sampleSize%2 # ensure it is even for ploting in two rows
    if random:
        sampleIx = np.random.choice(len(filePaths),sampleSize)
    else:
        sampleIx = np.arange(0,sampleSize)
    
    fig = plt.figure(figsize=(20, 4))
    
    if title is not None:
        fig.suptitle(f"{title}")
    
    ncols = np.ceil(sampleSize/2)
    for idx in np.arange(sampleSize):
        
        ax = fig.add_subplot(2, ncols, idx+1, xticks=[], yticks=[])
        
        filepath = filePaths[sampleIx[idx]]
        pil_image = Image.open(filepath).convert('RGB')
        # fix orientation for images from iPhone, iPad...
        pil_image = ImageOps.exif_transpose(pil_image)
        # resize image        
        basewidth = 256
        wpercent = (basewidth/float(pil_image.size[0]))
        hsize = int((float(pil_image.size[1])*float(wpercent)))
        pil_image = pil_image.resize((basewidth,hsize), Image.ANTIALIAS)
        
        npimg = np.array(pil_image)  
        ax.imshow(npimg)
        label = filepath.split('/')[-2]
        if len(label) > 10:
            label = label[:10] + "..."
        ax.set_title(label)
        if rects is not None:
            for (center_x,center_y,x,y,w,h,coord1,coord2,coord3,coord4) in rects[sampleIx[idx]]:
                ax.add_patch(patches.Rectangle((coord1*pil_image.size[0],coord2*pil_image.size[1]),
                                               (coord3-coord1)*pil_image.size[0],(coord4-coord2)*pil_image.size[1],
                                               linewidth=1,
                                               edgecolor='r',
                                               facecolor='none'))
    if output is not None:
        fig.savefig(
            output,
            bbox_inches='tight',
            dpi=200)


def generate_dataframe_face_detection(data,
                                      addGroundTruthLabel=False):
    '''
    Generate pandas dataframe with the results of face detection
    INPUT:
        data (dict)
    OUTPUT:
        df (pandas dataframe)
    '''
    filePaths = data['filePaths']
    nbOfFaces = data['nbOfFaces']
    locations = data['locations']
    encodings = data['encodings']    

    labels = [None] * len(filePaths)

    return pd.DataFrame({'filePath': filePaths,
                         'label': labels,
                         'nbOfFaces': nbOfFaces,
                         'locations': locations,
                         'encodings': encodings})

def generate_thumbnails(filePaths,rects,outputDir=None):
    '''
    Create image with thumbnails of faces
    '''

    thumbnails = []
    for filepath,locations in zip(filePaths,rects):
        for ix,location in enumerate(locations):

            pil_image = Image.open(filepath).convert('RGB')
            # fix orientation for images from iPhone, iPad...
            pil_image = ImageOps.exif_transpose(pil_image)
            
            w = pil_image.size[0]
            h = pil_image.size[1]
            [x0,y0,x1,y1] = location[6:]
            
            pil_image = pil_image.crop((x0*w,y0*h,x1*w,y1*h))
            pil_image = pil_image.resize((256,256))
            thumbnails.append(np.array(pil_image))
            if outputDir is not None:
                pil_image.save(os.path.join(outputDir,f"{filepath.replace('/','_')}_{ix}.jpg"))
    return thumbnails

# Load face locations and encodings
print(f"[INFO] Loading face encodings...")
inputfile = f"output_images-faceEncodings.pickle"
data = pickle.loads(open(inputfile, "rb").read())

## Face Detection
df = generate_dataframe_face_detection(data,addGroundTruthLabel=False)

sampleSize = df.shape[0]

det_accuracy = np.mean(np.array(df.nbOfFaces)>0)
print(f"[INFO] Detection efficiency: {det_accuracy:.3f}")

nbDetectedZeroFaces = np.sum(np.array(df.nbOfFaces)==0)
nbDetectedOneFace = np.sum(np.array(df.nbOfFaces)==1)
nbDetectedMultipleFaces = np.sum(np.array(df.nbOfFaces)>1)

print(f"[INFO] Found 0 faces: {nbDetectedZeroFaces} ({nbDetectedZeroFaces/sampleSize:.3f})")
print(f"[INFO] Found 1 face: {nbDetectedOneFace} ({nbDetectedOneFace/sampleSize:.3f})")
print(f"[INFO] Found >1 faces: {nbDetectedMultipleFaces} ({nbDetectedMultipleFaces/sampleSize:.3f})")

### Explore Detected Faces
filterIx = (df.nbOfFaces>0)
print(f'[INFO] {filterIx.sum()} images with identified faces')
files = df[filterIx].filePath.values
recs = df[filterIx].locations.values
encs = df[filterIx].encodings.values

plot_dataset(filePaths=files,
             rects=recs,
             title=f'Detected faces',
             random=False,
             output=f'output_images-sample-detectedFaces.png')

thumbnails = generate_thumbnails(
    filePaths=files,
    rects=recs,
    outputDir='thumbnails')

print('[INFO] Thumbnails generated')

ncols=20
fig = plt.figure(figsize=(20, 4))

for idx,npimg in enumerate(thumbnails):

    ax = fig.add_subplot(2, ncols, idx+1, xticks=[], yticks=[])
    ax.imshow(npimg)

fig.savefig(
   'output_images-thumbnails.png',
   bbox_inches='tight',
   dpi=200)

## Face Clustering

encodings = []

nbFaces = len(thumbnails)
embDim = 128

emb_array = np.zeros((nbFaces,embDim))

faceIx = 0

for file_encs in encs:
    
    for enc_array in file_encs:
        dlib_vec = dlib.vector(enc_array)
        emb_array[faceIx,:] = enc_array
        faceIx = faceIx+1
        encodings.append(dlib_vec)

print("[INFO] Clustering faces with Chinese Whispers algorithm")
labels_pred = dlib.chinese_whispers_clustering(encodings, 0.5)

ncols=20
fig = plt.figure(figsize=(20, 4))

thumbnails_labels = sorted(zip(thumbnails,labels_pred), key = lambda t: t[1])

for idx,(npimg,label) in enumerate(thumbnails_labels):

    ax = fig.add_subplot(2, ncols, idx+1, xticks=[], yticks=[])
    ax.imshow(npimg)
    ax.set_title(label)

fig.savefig(
   'output_images-thumbnails-classes.png',
   bbox_inches='tight',
   dpi=200)