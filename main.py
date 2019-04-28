import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization
from keras.optimizers import Adam
from keras import regularizers
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import pandas as pd
import itertools
from sys import exit


class TwoGenRec(object):
    def __init__(self, train_path=None, valid_path=None, test_path=None, learning_rate=0.0001):
        self.model = Sequential()
        self.define_model()
        self.learning_rate = learning_rate
        self.model.load_weights("model.h5")
        self.model.compile(Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
        self.gen_args = dict(featurewise_center=True,
                             featurewise_std_normalization=True,
                             rotation_range=10.,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2)
        self.img_generator = None
        self.define_generator()
        self.target_size = (224, 224)
        self.train_path = train_path
        self.valid_path = valid_path
        self.test_path = test_path
        self.test_y = None
        self.pred_y = None
        self.rocstat = None
        self.aucstat = None
        self.threshold = 0.20
        self.classes = ['man', 'woman']

    def define_model(self):
        """Creates model of CNN neural network"""
        model = self.model
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(224, 224, 64)))
        model.add(MaxPooling2D(pool_size=(2, 2), input_shape=(224, 224, 64)))
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(112, 112, 64)))
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(112, 112, 128)))
        model.add(MaxPooling2D(pool_size=(2, 2), input_shape=(112, 112, 128)))
        model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(56, 56, 128)))
        model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(56, 56, 256)))
        model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(56, 56, 256)))
        model.add(MaxPooling2D(pool_size=(2, 2), input_shape=(56, 56, 256)))
        model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(28, 28, 256)))
        model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(28, 28, 512)))
        model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(28, 28, 512)))
        model.add(MaxPooling2D(pool_size=(2, 2), input_shape=(28, 28, 512)))
        model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(14, 14, 512)))
        model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(14, 14, 512)))
        model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(14, 14, 512)))
        model.add(MaxPooling2D(pool_size=(2, 2), input_shape=(14, 14, 512)))
        model.add(Flatten())
        model.add(BatchNormalization())
        model.add(Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Dense(2, activation='softmax'))

    def define_generator(self):
        """Defines ImageDataGenerator object in self.img_generator class field"""
        image_gen = ImageDataGenerator(**self.gen_args)
        # Mean per channel based on the training set used for the standardization
        image_gen.mean = np.array([[[151.79312, 117.222755, 100.84372]]])
        # Variance per channel based on the training set used for the standardization
        image_gen.std = np.array([[[72.83951, 64.24882, 61.288933]]])
        self.img_generator = image_gen

    def clas_response(self, prob_matrix):
        """Assigns to man for probability lower or equal threshold
        The input have to be (n, 2) or (n) dimension matrix"""
        if prob_matrix.ndim == 2:
            womorman = (prob_matrix[:, 1] <= self.threshold).astype(int)
        elif prob_matrix.ndim == 1:
            womorman = (prob_matrix > self.threshold).astype(int)
        else:
            raise Exception("Number of dimensions in clas_response method does not fit demands")
        return womorman

    def test_statistics(self, test_steps=26, batch_size=14):
        """Calculates ROC and AUC for the given test set, further it calculates
        proposed threshold of classifier response based on the maximum sum of sensitivity
        and specificity"""
        if self.model is None or self.img_generator is None:
            raise Exception("The model and image generator should not have value of {0:} and {1:}".
                            format(self.model, self.img_generator))
        if self.test_path is None:
            raise Exception("The path for test models is not provided")
        test_batches = self.img_generator.flow_from_directory(self.test_path, target_size=self.target_size,
                                                              classes=self.classes, batch_size=batch_size,
                                                              shuffle=False)
        y_pred = self.model.predict_generator(test_batches, steps=test_steps, verbose=0)
        test_y = test_batches.classes
        self.test_y = test_y
        y_pred = np.array([i[1] for i in y_pred])
        self.pred_y = y_pred
        fpr, tpr, thresholds = roc_curve(test_y, y_pred)
        prop_thres = np.argmax(1 - fpr + tpr)
        print("Proposed threshold: {0:} for sensitivity {1:} and specificity {2:}".
              format(thresholds[prop_thres], tpr[prop_thres], 1 - fpr[prop_thres]))
        self.aucstat = roc_auc_score(test_y, y_pred)
        # Create Dataframe from ROC data
        self.rocstat = pd.DataFrame(np.array([tpr, fpr]).T, columns=['True Positive Rate', 'False Positive Rate'])

    def plot_roc(self):
        if self.rocstat is None or self.aucstat is None:
            raise Exception("Before printing the ROC curve please test_statistics")
        sns.set_style('darkgrid')
        ax = sns.lineplot(x='False Positive Rate', y='True Positive Rate', data=self.rocstat)
        l1 = ax.lines[0]
        x1 = l1.get_xydata()[:, 0]
        y1 = l1.get_xydata()[:, 1]
        anchored_text = AnchoredText("Area under curve: "+str(self.aucstat), loc='lower right')
        ax.fill_between(x1, y1, color="blue", alpha=0.2)
        ax.add_artist(anchored_text)
        plt.show()

    def calc_specsens(self):
        """Calculates sensitivity and specificity for current threshold"""
        pred_y = self.clas_response(self.pred_y)
        # True positive is woman and true negative is man
        cm = confusion_matrix(self.test_y, pred_y)
        tp = cm[1, 1]
        fp = cm[0, 1]
        fn = cm[1, 0]
        tn = cm[0, 0]
        print("Sensitivity: {}".format(tp / (tp + fn)))
        print("Specificity: {}".format(tn / (tn + fp)))

    def plot_confmatrix(self):
        pred_y = self.clas_response(self.pred_y)
        cm = confusion_matrix(self.test_y, pred_y)
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion Matrix for test set")
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, self.classes, rotation=45)
        plt.yticks(tick_marks, self.classes)
        tresh = cm.max() / 2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment='center',
                     color='white' if cm[i, j] > tresh else 'black')
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    def train_modem(self, steps_per_epoch=1, epochs=1, validation_steps=1):
        """Training of the model for given validation and training set"""
        if self.train_path is None or self.valid_path is None:
            raise Exception("The model and image generator should not have value of {0:} and {1:}".
                            format(self.train_path, self.valid_path))
        train_batches = self.img_generator.flow_from_directory(self.train_path, target_size=self.target_size,
                                                               classes=self.classes, batch_size=400,
                                                               shuffle=True)
        valid_batches = self.img_generator.flow_from_directory(self.valid_path, target_size=self.target_size,
                                                               classes=self.classes, batch_size=160,
                                                               shuffle=True)
        self.model.summary()
        self.model.fit_generator(train_batches, steps_per_epoch=steps_per_epoch, epochs=epochs, verbose=2,
                                 validation_data=valid_batches, validation_steps=validation_steps)
        self.model.save_weights("model.h5")

    def get_img_stdmean(self, directory_train='',
                        calc_steps=40, batch_size=200,
                        target_size=(224, 224),
                        shuffle=True):
        """Calculates statistics required for standarization of the images in the training set"""
        mean = 0
        mean_std = 0
        for i in range(calc_steps):
            image_gen = ImageDataGenerator(**self.img_generator)
            batch_gen = image_gen.flow_from_directory(directory=directory_train,
                                                      batch_size=batch_size,
                                                      target_size=target_size,
                                                      shuffle=shuffle)
            imgs, labels = next(batch_gen)
            image_gen.fit(imgs)
            mean = mean + image_gen.mean
            mean_std = mean_std + image_gen.std
        mean = mean/calc_steps
        mean_std = mean_std/calc_steps
        print("Mean for each channel: ", mean)
        print("Standard deviation for each channel: ", mean_std)
        return mean, mean_std

    def predictimage(self, image_path=None):
        """Classifies given image to one of two classes: man or woman"""
        img = image.load_img(image_path, target_size=self.target_size)
        img = image.img_to_array(img)
        img = self.img_generator.standardize(img)
        img = np.expand_dims(img, axis=0)
        prob = self.model.predict_proba(img)
        # index 0 is man, index 1 is woman
        asgn_class = int(not self.clas_response(prob)[0])
        print(prob[0])
        print(self.classes[asgn_class])

    def scanforthreshold(self):
        oldthreshold = self.threshold
        for i in np.linspace(0.0, 1, 50):
            self.threshold = i
            print("#########")
            print("Threshold: {}".format(i))
            self.calc_specsens()
            print("#########")
        self.threshold = oldthreshold


if __name__ == '__main__':
    print("Initialization ...")
    predictor = TwoGenRec(test_path='data/test')
    while True:
        print("Hello, please choose an option: \n",
              "1. Predict image's gender \n",
              "2. Exit\n")
        menu_inp = input("Choose an option[1]: ") or "1"
        if menu_inp == "1":
            while True:
                print("Type image's path to assess the gender.")
                path = input("Please type absolute path or relative path: ")
                if isinstance(path, str) and path.split('.')[-1] not in ('jpg', 'png', 'bmp'):
                    print("Please type correct path to the image or correct format (jpg, png or bmp)")
                    if input("Type 'EXIT' if you want to exit the program") == 'EXIT':
                        exit()
                else:
                    break
            predictor.predictimage(image_path=path)
        elif menu_inp == "2":
            exit()
        else:
            print("Option {} does not exists".format(menu_inp))
