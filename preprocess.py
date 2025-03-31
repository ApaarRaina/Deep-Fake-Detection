import pandas as pd
import cv2 as cv
import os

df=pd.read_csv('test.csv')

df.drop(columns=['Unnamed: 0','original_path','id'],inplace=True)


clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

output_dir = 'C:/programs/Deep_Fake_detection/processed_test'

# Create the directory if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for i in range(len(df)):
    path=df.loc[i,'path']

    img=cv.imread(path,cv.IMREAD_GRAYSCALE) #grayscale conversiom
    img=cv.resize(img,(256,256),interpolation=cv.INTER_AREA)  #resize

    img=clahe.apply(img) #brightness adjust


    relative_path = os.path.relpath(path, start='test')  # e.g., 'real/31355.jpg'
    new_path = os.path.join(output_dir, relative_path)  # e.g., 'processed_train/real/31355.jpg'


    os.makedirs(os.path.dirname(new_path), exist_ok=True)  # Creates 'processed_train/real/'

    # Save the processed image
    success = cv.imwrite(new_path, img)


