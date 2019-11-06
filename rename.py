import os 
import tqdm as tqdm

#constants-directory names
TRAINING_DIR = '/Users/athreya/desktop/GAN_Exploration/PicassoArtClassifier/Training'
TESTING_DIR = '/Users/athreya/desktop/GAN_Exploration/PicassoArtClassifier/Testing'
PICASSO_DIR = TRAINING_DIR + '/' + 'Picasso_training'
GOGH_DIR = TRAINING_DIR + '/' + 'vangogh_training'
PICASSO_DIR_T = TESTING_DIR + '/' + 'Picasso_testing'
GOGH_DIR_T = TESTING_DIR + '/' + 'vangogh_testing'

#renaming all the files in the picasso and vangogh training directories
#picasso
n = 0
for i in os.listdir(PICASSO_DIR):
    os.rename(PICASSO_DIR + '/'+ i,PICASSO_DIR + '/' + 'PC.{}'.format(n) + '.jpg' )
    n=n+1
n = 0
#van gogh
for i in os.listdir(GOGH_DIR):
    os.rename(GOGH_DIR + '/'+ i,GOGH_DIR + '/' + 'VG.{}'.format(n) + '.jpg' )
    n=n+1
#renaming all of the files in the picasso and vangogh testing directories
#picasso
n = 0
for i in os.listdir(PICASSO_DIR_T):
    os.rename(PICASSO_DIR_T + '/'+ i,PICASSO_DIR_T + '/' + '{}'.format(n) + '.jpg' )
    n=n+1
#van gogh
for i in os.listdir(GOGH_DIR_T):
    os.rename(GOGH_DIR_T + '/'+ i,GOGH_DIR_T + '/' + '{}'.format(n) + '.jpg' )
    n=n+1
