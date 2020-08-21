## Dependencies
import os
import dlib
import cv2

import PIL
from PIL import Image, ImageOps
PIL.Image.LOAD_TRUNCATED_IMAGES = True

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from tqdm import tqdm
import pickle

import bz2
import urllib.request
from urllib.parse import urlparse

## Helper Functions

def download_dlib_data(dataDir='models'):
    '''
    Download dlib model data

    INPUT:
        dataDir (str): path to model data
    '''
    urls = ['http://dlib.net/files/mmod_human_face_detector.dat.bz2',
            'http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2',
            'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2',
            'http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2']

    for url in urls:
        filename = os.path.join(dataDir,os.path.basename(urlparse(url).path).replace('.bz2',''))
        if not os.path.exists(filename):
            print(f"[INFO] Downloading {url}")
            tmp_file = urllib.request.urlretrieve(url)
            with open(tmp_file[0],'rb') as f_in:
                zipfile = bz2.BZ2File(f_in)
                print(f"[INFO] Extracting {filename}")
                with open(filename, 'wb') as f_out:
                    f_out.write(zipfile.read())
                    f_out.close()
                zipfile.close()
                f_in.close()


def find_images(top):
    '''
    Walk down the top dir path and yield any valid image
    
    INPUT:
        top (str): top directory of the tree
    OUTPUT:
        filepath (str): filepath of a valid image
    '''
    
    for (dirpath, dirnames, filenames) in os.walk(top):
         for filename in filenames:
            
             filepath = os.path.join(dirpath, filename)

             # check if file is a valid image
             # using imghdr to find images is about 3 times faster than using PIL
             # however, imghdr has a bug to detect some jpeg images (see https://bugs.python.org/issue16512)
             # PIL supports a larger number of image formats
             # To use imghdr, firts import imghdr
             #if imghdr.what(filepath) is not None:               
             #    yield filepath

             # check if file is a valid image
             try:
                 img = Image.open(filepath)
                 img.verify()
                
                 yield filepath
             except IOError:
                 # not a valid image
                 continue
                

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

# Initialize models data
download_dlib_data()
model_data_shape_predictor_5_face_landmarks = 'models/shape_predictor_5_face_landmarks.dat'
model_data_shape_predictor_68_face_landmarks = 'models/shape_predictor_68_face_landmarks.dat'
model_data_face_detector_cnn = 'models/mmod_human_face_detector.dat'
model_data_face_descriptor = 'models/dlib_face_recognition_resnet_model_v1.dat'

# Face detectors
dlib_face_detector_hog = dlib.get_frontal_face_detector()
dlib_face_detector_cnn = dlib.cnn_face_detection_model_v1(model_data_face_detector_cnn)

cv_haar_cascade = cv2.CascadeClassifier(
    os.path.join(cv2.data.haarcascades,'haarcascade_frontalface_default.xml')
)

# Face landmark predictors
dlib_face_predictor_5_landmarks = dlib.shape_predictor(model_data_shape_predictor_5_face_landmarks)
dlib_face_predictor_68_landmarks = dlib.shape_predictor(model_data_shape_predictor_68_face_landmarks)

# Face embedding
dlib_face_encoder = dlib.face_recognition_model_v1(model_data_face_descriptor)

def detect_faces(filepath,
                 model_detector='dlib_cnn',
                 model_landmarks='dlib_68',
                 model_encoder='dlib'):
    '''
    Detect faces, predict landmarks and compute face embedding
    
    INPUT:
        filepath (str):        path of image
        model_detector (str):  face detection model
        model_landmarks (str): face landmarks detection model
        model_encoder (str):   facial embedding enconder model
    OUTPUT:
        locations:
        landmarks:
        encodings:
    '''
    
    pil_image = Image.open(filepath).convert('RGB')
    # fix orientation for images from iPhone, iPad...
    pil_image = ImageOps.exif_transpose(pil_image)
    # resize image        
    basewidth = 800
    wpercent = (basewidth/float(pil_image.size[0]))
    hsize = int((float(pil_image.size[1])*float(wpercent)))
    pil_image = pil_image.resize((basewidth,hsize), Image.ANTIALIAS)
    
    img = np.array(pil_image)

    # detect faces
    if model_detector == 'dlib_hog':
        faces = dlib_face_detector_hog(img,1)
    elif model_detector == 'dlib_cnn':
        faces = [loc.rect for loc in dlib_face_detector_cnn(img,1)]
    elif model_detector == 'cv_haar-cascade':
       
        pil_image = Image.open(filepath).convert('L')
        pil_image = ImageOps.exif_transpose(pil_image)
        # resize image        
        basewidth = 800
        wpercent = (basewidth/float(np.array(pil_image).size[0]))
        hsize = int((float(np.array(pil_image).size[1])*float(wpercent)))
        pil_image = pil_image.resize((basewidth,hsize), Image.ANTIALIAS)
        img_gray = np.array(pil_image)
        
        faces = cv_haar_cascade.detectMultiScale(img_gray, scaleFactor=1.3, minNeighbors=5)
        faces = [dlib.rectangle(left=x,top=y,right=x+w,bottom=y+h) for (x,y,w,h) in faces]
    else:
        raise ValueError("model_detector",model_detector)
    
    locations = []
    landmarks = []
    encodings = []
    
    for face in faces:

        # locations
        x0, y0, w, h = face.left(), face.top(), face.width(), face.height()
        center_x = (x0+0.5*w)/pil_image.size[0]
        center_y = (y0+0.5*h)/pil_image.size[1]
        coord = [x0/pil_image.size[0],
                 y0/pil_image.size[1],
                 (x0+w)/pil_image.size[0],
                 (y0+h)/pil_image.size[1]]
        locations.append((center_x,center_y,x0,y0,w,h,*coord))
        
        # landmarks
        if model_landmarks == 'dlib_5':
            shape = dlib_face_predictor_5_landmarks(img, face)
        elif model_landmarks == 'dlib_68':
            shape = dlib_face_predictor_68_landmarks(img, face)
        else:
            raise ValueError("model_landmarks",model_landmarks)
            
        points = [(point.x,point.y) for point in shape.parts()]
        landmarks.append(points)

        # 128D vector encoding
        if model_encoder == 'dlib':
            description = dlib_face_encoder.compute_face_descriptor(img, shape)
            encoding = np.array(description)
        else:
            raise ValueError("model_encoder",model_encoder)

        encodings.append(encoding)

    return (locations,landmarks,encodings)

def extract_face_encodings(outputfile,
                           filePaths,
                           model_detector='dlib_cnn',
                           model_landmarks='dlib_68',
                           model_encoder='dlib',
                           sampleSize=-1):
    '''
    Extract face encodings and store serialized data to a pickle file
    
    INPUT:
        filePaths (list) 
        model_detector (str):  face detection model
        model_landmarks (str): face landmarks detection model
        model_encoder (str):   facial embedding enconder model
        sampleSize (int):      set value greater than 0 to only detect faces from a sample
        
    '''
    
    if sampleSize > 0 and sampleSize < len(filePaths):
        filePaths = np.random.choice(filePaths,sampleSize)

    print(f"[INFO] Detecting faces for {len(filePaths)} images")

    outputfile = outputfile.replace('.pickle','')
    outputfile = f"{outputfile}.pickle"

    list_filepaths = []
    list_nbOfFaces = []
    list_locations = []
    list_landmarks = []
    list_encodings = []

    if os.path.exists(outputfile):
        print(f"[WARNING] Output file {outputfile} already exists. Loading data")
        print(f"[INFO] Loading data from {outputfile}")
        data = pickle.loads(open(outputfile, "rb").read())

        list_filepaths = data['filePaths']
        list_nbOfFaces = data['nbOfFaces']
        list_locations = data['locations']
        list_landmarks = data['landmarks']
        list_encodings = data['encodings']

    for filepath in tqdm(filePaths):
               
        if filepath in list_filepaths:
            continue
        print(f"[INFO] Detecting faces in {filepath}")
        (locations,landmarks,encodings) = detect_faces(
            filepath,
            model_detector=model_detector,
            model_landmarks=model_landmarks,
            model_encoder=model_encoder)
        
        list_filepaths.append(filepath)
        list_nbOfFaces.append(len(locations))
        list_locations.append(locations)
        list_landmarks.append(landmarks)
        list_encodings.append(encodings)

        if len(list_filepaths)%500 == 0:          
            data = {'filePaths': list_filepaths,
                    'nbOfFaces': list_nbOfFaces,
                    'locations': list_locations,
                    'landmarks': list_landmarks,
                    'encodings': list_encodings}
            print(f"[INFO] Temporary serializing encodings")
            f = open(outputfile, "wb")
            f.write(pickle.dumps(data))
            f.close()
    
    data = {'filePaths': list_filepaths,
            'nbOfFaces': list_nbOfFaces,
            'locations': list_locations,
            'landmarks': list_landmarks,
            'encodings': list_encodings}
    print(f"[INFO] Serializing encodings")
    f = open(outputfile, "wb")
    f.write(pickle.dumps(data))
    f.close()

    print('[INFO] Face detection completed')



## ETL Pipeline for Face Detection and Embedding

rootDir = 'data/' # path to the root directory where the images are stored

# Search images
listFiles = list(find_images(f'{rootDir}'))
print(f"[INFO] Number of images:",len(listFiles))

# Sample visualization
plot_dataset(
    listFiles,
    title='Sample of images',
    random=False,
    sampleSize=20,
    output=f'output_images-sample.png')

# Detect faces and extract facial embeddings
extract_face_encodings(
    outputfile=f'output_images-faceEncodings',
    filePaths=listFiles,
    model_detector='dlib_cnn')
