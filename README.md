# hw3
![image](https://github.com/kappamoss/hw3/blob/main/Gunsciz.jpg)
# Requirements
detectron2   
(https://detectron2.readthedocs.io/en/latest/tutorials/install.html)  
if you can't bulit it well, you need to install vs 2015 c++ tool   
  
python == 3.6  

pytorch == 1.8  

opencv 

# Environment
GUP: GTX2080  
CUDA: 11.3

# Training  
(you need to  download the dataset from here:https://drive.google.com/drive/folders/1-3oUs2aqWN8F8Gsl9LfPjDLlIBWwSuR6?usp=sharing, and put them into dataset folder, no external data be used.)  
To train the model:  
python train_nu.py

# Pre-trained Models
Download pretrained models here:
https://drive.google.com/file/d/1Tci61PIIFRDw-MFz-5XN2SZJqeHFhyET/view

you need to put this model into input_model folder

# To produce submission file

python inference.py  

Note: if you only want to generate answer.json, you don't need to put train and val folders to ./hw3/dataset/(only test folder and test_img_ids.json needed.), and put the model into output_model folder.

