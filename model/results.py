import numpy as np
import random
import imageio
import umap
from tqdm import tqdm_notebook
import os
import pickle

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import seaborn as sns

from sklearn.model_selection  import train_test_split

class get_results(object):
    
    def __init__(self, net_results):
        self.test_size = len(net_results['embeddings'])
        self.embedding_size = net_results['embeddings'][0].shape[1]
        
        self.confusion_matrix = net_results['confusion_matrix'][-1][0]
        self.confusion_matrix_norm = self.conf_matrix_norm()
        
        self.embeddings = np.zeros((self.test_size,self.embedding_size))
        for i,emb in enumerate(net_results['embeddings']):
            self.embeddings[i,:] = net_results['embeddings'][i]
        
        self.labels = np.array([i[0] for i in net_results['labels']])
        self.predictions = np.array([i[0] for i in net_results['predictions']])
         
        
        self.seen_classes = []
        reducer = umap.UMAP()
        self.umap_embeddings = reducer.fit_transform(self.embeddings)
    
    def conf_matrix_norm(self):
        conf_matrix_norm = self.confusion_matrix.astype(np.float)
        s = conf_matrix_norm.sum(axis=1)
        for i,si in enumerate(s):
            if si != 0:
                conf_matrix_norm[i,:] = conf_matrix_norm[i,:]/si
        return conf_matrix_norm
    
    def drawPoint(self, pred, label, point, ax):
        x,y = point[0], point[1]
        _label = str(label == pred) + ' ' + self.int_classes[pred]

        alpha = 1 if label == pred else 0.5

        if not _label in self.seen_classes:
            ax.scatter(x, y, c = self.colours[pred], marker = self.markers[label == pred], s=15,
                       alpha = alpha, label = _label)
            self.seen_classes.append(_label)

        else:
            ax.scatter(x, y, c = self.colours[label], marker = self.markers[label == pred], s=15,
                       alpha = alpha)
    
    def plot_colours(self, save = False, legend=False, filename = None):

        id_sort = np.argsort(self.predictions)
        umap_embeddings_sort = self.umap_embeddings[id_sort]
        labels_sort = self.labels[id_sort]
        predictions_sort = self.predictions[id_sort]

        classes = {'Bubbles' : 0,'Clear Blob' : 1,'Dilated' : 2,'Turbid': 3,'Undefined' : 4,
                     'Wall' :5,'Wrinkle' : 6}
        self.int_classes = {y:x for x,y in classes.items()} #switch keys and values

        self.markers = {True: 'o', False: 'x'}
        self.colours = {}
        for i in range(7):
            self.colours[i] = sns.color_palette("hls", 7)[i]

        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111)

        for point, label, pred in zip(umap_embeddings_sort, labels_sort, predictions_sort):
            self.drawPoint(pred, label, point, ax)

        ax.set_xlabel('1st Component', fontsize = 9.75)
        ax.set_ylabel('2nd Component', fontsize = 9.75)
        ax.grid(True)
        
        if legend:
            lgd = plt.legend(bbox_to_anchor=(1.2, 0.65),loc='center')
            #lgd = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
            #  fancybox=True, shadow=True, ncol=7)
        if save:
            if legend:
                fig.savefig('writing/{}.png'.format(filename), bbox_inches='tight', bbox_extra_artist=[lgd])
            else:
                fig.savefig('writing/{}.png'.format(filename), bbox_inches='tight')
        if save:
            fig.savefig('res_experiments/{}.png'.format(filename), bbox_inches='tight')
        plt.show()
        plt.close()
    
    def imscatter(self, x, y, image, ax=None, zoom=1):
        if ax is None:
            ax = plt.gca()
        try:
            image = plt.imread(image)
        except TypeError:
            # Likely already an array...
            pass
        im = OffsetImage(image, zoom=zoom)
        x, y = np.atleast_1d(x, y)
        artists = []
        for x0, y0 in zip(x, y):
            ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
            artists.append(ax.add_artist(ab))
        ax.update_datalim(np.column_stack([x, y]))
        ax.autoscale()
        ax.set_xlabel('1st Component', fontsize = 9.75)
        ax.set_ylabel('2nd Component', fontsize = 9.75)
        ax.grid(True)

        return artists

    def plot_images(self,save,train_size,test_size,eval_on,filename):
        
        if eval_on=='SB3':
            path_SB3='../sb3_CAPRI_DATASET_6Classes_2/sb3_CAPRI_DATASET_6Classes/'
            extensions_SB3 = ['jpg']
            classes_SB3 = {'bubbles' : 0,'clearBlob' : 1,'dilated':2,'turbid' : 3,'undefined': 4, 'wall' : 5,'wrinkles' :6}
            dataset_SB3 = {}

            for _dirpath, _dirnames, _filenames in tqdm_notebook(os.walk(path_SB3)):
                class_act = _dirpath.split('/')[-1]
                if class_act not in classes_SB3.keys():
                    continue
                else:
                    for _file in _filenames:
                        for _extension in extensions_SB3:
                            if _extension in _file:
                                if _dirpath in dataset_SB3:
                                    dataset_SB3[_dirpath].append(_dirpath +'/' + _file)
                                else:
                                    dataset_SB3[_dirpath] = [_dirpath +'/' + _file]
            
            images = []
            with open('./SB3_big_traintest/test.txt') as f:
                SB3_test = f.read().splitlines()
            for im in SB3_test:
                img = imageio.imread(im)
                images.append(img)
            
            fig = plt.figure(figsize=(6,6))
            ax = fig.add_subplot(111)
            
            for i,im in enumerate(images):
                self.imscatter(self.umap_embeddings[i,0], self.umap_embeddings[i,1], im, zoom=0.04, ax=ax)
            plt.show()
            if save:
                fig.savefig('res_experiments/{}.png'.format(filename), bbox_inches='tight')
        elif eval_on=='SB2':
            classes_intestins = {'BUBBLES_v3' : 0,'CLEAR_BLOB' : 1,'DILATED' : 2,'TURBID_v2': 3,'UNDEFINED' : 4,
                 'WALL' :5,'WRINKLES_v2' : 6}
            path_intestins='../../data/intestins/original/'
            extensions_intestins = ['jpg', 'png', 'bmp']
            dataset_intestins = {}
            for _dirpath, _dirnames, _filenames in tqdm_notebook(os.walk(path_intestins)):
                class_act = _dirpath.split('/')[-1]
                if class_act not in classes_intestins.keys():
                    continue
                else:
                    for _file in _filenames:
                        if not "r180" in _file and not "r90" in _file:
                            for _extension in extensions_intestins:
                                if _extension in _file:
                                    if _dirpath in dataset_intestins:
                                        dataset_intestins[_dirpath].append(_dirpath +'/' + _file)
                                    else:
                                        dataset_intestins[_dirpath] = [_dirpath +'/' + _file]
            partition = 42
            train_key_intestins, test_key_intestins = [], []
            compt = True
            for lesion in dataset_intestins.keys():
                list_images = sorted(list(dataset_intestins[lesion]))
                train, test = train_test_split(list_images, random_state = partition,
                                               test_size = test_size, train_size = train_size)
                train_key_intestins += train
                test_key_intestins += test
        
            random.seed(partition)
            random.shuffle(train_key_intestins)
            random.shuffle(test_key_intestins)

            images = []                            
            for im in test_key_intestins:
                img = imageio.imread(im)
                images.append(img)
            
            fig = plt.figure(figsize=(6,6))
            ax = fig.add_subplot(111)

            for i,im in enumerate(images):
                self.imscatter(self.umap_embeddings[i,0], self.umap_embeddings[i,1], im, zoom=0.04, ax=ax)
            plt.show()
            if save:
                fig.savefig('res_experiments/{}.png'.format(filename), bbox_inches='tight')