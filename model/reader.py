import numpy as np
import imageio
from skimage.transform import resize
from sklearn.model_selection  import train_test_split
import os
import random
from tqdm import tqdm_notebook

class SB2_SB3_reader(object):
    
    def __init__(self,
                 train_images_SB2, test_images_SB2, train_images_SB3,
                 eval_on,
                 batch_size, shape = (256,256),
                 classes_SB2 = {'BUBBLES_v3' : 0,'CLEAR_BLOB' : 1,'DILATED' : 2,'TURBID_v2': 3,'UNDEFINED' : 4,
                 'WALL' :5,'WRINKLES_v2' : 6},
                 classes_SB3 = {'bubbles' : 0,'clearBlob' : 1,'dilated':2,'turbid' : 3,'undefined': 4, 'wall' : 5,'wrinkles' :6},
                 path_SB2='../../data/intestins/original/', path_SB3='../sb3_CAPRI_DATASET_6Classes_2/sb3_CAPRI_DATASET_6Classes/',
                 extensions_SB2 = ['jpg', 'png', 'bmp'], extensions_SB3 = ['jpg'],
                 partition=42):
        
        #SB2
        if path_SB2[-1] != '/':
            path_SB2 += '/'
        self._path_SB2 = path_SB2
        self._extensions_SB2 = extensions_SB2
        self.dataset_SB2 = {}
        self.classes_SB2 = classes_SB2
        self.train_images_SB2 = train_images_SB2
        self.test_images_SB2 = test_images_SB2
        self.train_images_SB3 = train_images_SB3
        
        self.__findAllImages_SB2()
        
        
        self.prefetch_SB2 = {} 
        
        
        #both
        self.shape = shape
        self.batch_size = batch_size
        
        ##SB3
        if path_SB3[-1] != '/':
            path_SB3 += '/'
        self._path_SB3 = path_SB3
        self._extensions_SB3 = extensions_SB3
        self.dataset_SB3 = {}
        self.classes_SB3 = classes_SB3
        self.prefetch_SB3 = {} 
        
        self.__findAllImages_SB3()
        
        self.SB3_train = []
        if train_images_SB3 != 0:
            with open('./SB3_big_traintest/train.txt') as f:
                SB3_train = f.read().splitlines()
            #create subset according to the number of SB3 images for training
            clas_compt = np.zeros(len(self.classes_SB3))
            
            path0 = SB3_train[0] #../sb3_CAPRI_DATASET_6Classes_2/sb3_CAPRI_DATASET_6Classes/bubbles/noPolyp_027_1195.jpg
            clas0 = path0.split('/')[-2]
            self.SB3_train.append(path0)
            clas_compt[self.classes_SB3[clas0]] += 1
            
            for path in SB3_train[1:]:
                clas = path.split('/')[-2]
                if clas_compt[self.classes_SB3[clas]] < train_images_SB3:
                    self.SB3_train.append(path)
                    clas_compt[self.classes_SB3[clas]] += 1

        self.SB3_test = []
        if eval_on =='SB3':
            with open('./SB3_big_traintest/test.txt') as f:
                self.SB3_test = f.read().splitlines()
                
        self.partition = partition
        random.seed(self.partition)
                
        self.eval_on = eval_on
    
    def __findAllImages_SB2(self):
        """
            __findAllImages function
            
            Look for the images in each of the folders inside the path
            
            :param : No inputs params
            :return: None
            """
        for _dirpath, _dirnames, _filenames in tqdm_notebook(os.walk(self._path_SB2)):
            class_act = _dirpath.split('/')[-1]
            if class_act not in self.classes_SB2.keys():
                continue
            else:
                for _file in _filenames:
                    if not "r180" in _file and not "r90" in _file:
                        for _extension in self._extensions_SB2:
                            if _extension in _file:
                                if _dirpath in self.dataset_SB2:
                                    self.dataset_SB2[_dirpath].append(_dirpath +'/' + _file)
                                else:
                                    self.dataset_SB2[_dirpath] = [_dirpath +'/' + _file]

    def __findAllImages_SB3(self):
        """
            __findAllImages function
            
            Look for the images in each of the folders inside the path
            
            :param : No inputs params
            :return: None
            """
        for _dirpath, _dirnames, _filenames in tqdm_notebook(os.walk(self._path_SB3)):
            class_act = _dirpath.split('/')[-1]
            if class_act not in self.classes_SB3.keys():
                continue
            else:
                for _file in _filenames:
                    for _extension in self._extensions_SB3:
                        if _extension in _file:
                            if _dirpath in self.dataset_SB3:
                                self.dataset_SB3[_dirpath].append(_dirpath +'/' + _file)
                            else:
                                self.dataset_SB3[_dirpath] = [_dirpath +'/' + _file]

    def _resize(self, img): #same for SB2 and SB3 
        return np.clip(resize(img, self.shape, mode='symmetric', preserve_range = True), 0., 1.)

    def load_img(self, path):
        return self._resize(imageio.imread(path).astype(np.float32)/255.)

    def define_train_test_SB2(self, partition):
        self.train_key_SB2, self.test_key_SB2 = [], []
        compt = True
        for lesion in self.dataset_SB2.keys():
            list_images = sorted(list(self.dataset_SB2[lesion]))
            train, test = train_test_split(list_images, random_state = partition,
                                           test_size = self.test_images_SB2, train_size = self.train_images_SB2)
            self.train_key_SB2 += train
            self.test_key_SB2 += test
        
        random.seed(partition)
        random.shuffle(self.train_key_SB2)
        random.shuffle(self.test_key_SB2)
    
    def iterate_train(self):
        print('iterating...')
        random.shuffle(self.train_key_SB2)
        if self.train_images_SB3 == 0:
            for im in self.train_key_SB2:
                img = self.load_img(im)
                yield (img, [self.classes_SB2[im.split('/')[-2]]])
        else:
            random.shuffle(self.SB3_train)
            compt_SB3 = 0
            for i,im in enumerate(self.train_key_SB2):
                if i>0 and i%50 == 0:
                    for j in range(14):
                        im_SB3 = self.SB3_train[compt_SB3]
                        label_SB3 = self.classes_SB3[im_SB3.split('/')[-2]]
                        img_SB3 = self.prefetch_SB3[im_SB3] if im in self.prefetch_SB3 else self.load_img(im_SB3)
                        
                        #data augmentation
                        flips = [(slice(None, None, random.choice([-1, None])), #axis y
                                  slice(None, None, random.choice([-1, None]))) #axis x
                                 for _ in range(img.shape[0])]
                        img_SB3 = img_SB3[flips[1]]
                        img_SB3 = img_SB3[flips[0]]
                        img_SB3 = np.rot90(img_SB3, random.randint(0,4))

                        img_SB3 = img_SB3.reshape([self.shape[0],self.shape[1],3])
                        yield (img_SB3, [label_SB3])
                        compt_SB3 += 1
                        if compt_SB3 == len(self.SB3_train):
                            compt_SB3 = 0
                            random.shuffle(self.SB3_train)
                img = self.load_img(im)
                img = img.reshape([self.shape[0],self.shape[1],3])
                yield (img, [self.classes_SB2[im.split('/')[-2]]])


    def iterate_eval(self):
        if self.eval_on == 'SB3':
            for im in self.SB3_test:
                label = self.classes_SB3[im.split('/')[-2]]
                img = self.prefetch_SB3[im] if im in self.prefetch_SB3 else self.load_img(im)
                img = img.reshape([self.shape[0],self.shape[1],3])
                yield (img, [label])
        elif self.eval_on == 'SB2':
            for im in self.test_key_SB2:
                label = self.classes_SB2[im.split('/')[-2]]
                img = self.prefetch_SB2[im] if im in self.prefetch_SB2 else self.load_img(im)
                img = img.reshape([self.shape[0],self.shape[1],3])
                yield (img, [label])
        else:
            raise ValueError("Change eval to SB2 or SB3.")
