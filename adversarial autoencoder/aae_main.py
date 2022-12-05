#!/usr/bin/env python
# coding: utf-8


from __future__ import print_function
try:
	raw_input
except:
	raw_input = input


# import modules 
import numpy as np
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense
from keras.utils import plot_model
from keras.datasets import mnist
from keras.optimizers import Adam
import argparse
import matplotlib.pyplot as plt
from matplotlib import gridspec, colors
from datetime import datetime
from sklearn.manifold import TSNE
from absl import flags
from absl import app
import os, sys
import cv2
import random
import glob
from absl import flags
from absl import app
import csv
os.environ['CUDA_VISIBLE_DEVICES']= '1'

from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.layers import Conv2D, MaxPooling2D, Flatten, ZeroPadding2D, GlobalAveragePooling2D, AveragePooling2D
from keras.layers import BatchNormalization, ReLU, Activation, Concatenate, Conv2DTranspose, Reshape


FLAGS = flags.FLAGS

# General
flags.DEFINE_string("desc", "aae", "Name of the model to be saved")
flags.DEFINE_bool("train", False, "Train")
flags.DEFINE_bool("reconstruct", False, "Reconstruct image")
flags.DEFINE_bool("generate", False, "Generate image from latent")
flags.DEFINE_bool("generate_grid", False, "Generate grid of images from latent space (only for 2D latent)")
flags.DEFINE_bool("plot", False, "Plot latent space")
flags.DEFINE_integer("latent_dim", 2, "Latent dimension")

# Train
flags.DEFINE_integer("epochs", 50, "Number of training epochs")
# flags.DEFINE_integer("train_samples", 10000, "Number of training samples from MNIST")
flags.DEFINE_integer("batchsize", 100, "Training batchsize")

# Test
# flags.DEFINE_integer("test_samples", 10000, "Number of test samples from MNIST")
flags.DEFINE_list("latent_vec", None, "Latent vector (use with --generate flag)")


def create_model(input_shape, latent_dim, verbose=False, save_graph=False):

    autoencoder_input = Input(shape=input_shape)
    generator_input = Input(shape=input_shape)

    # encoder = Sequential()
    # encoder.add(Dense(1000, input_shape=(input_dim,), activation='relu'))
    # encoder.add(Dense(1000, activation='relu'))
    # encoder.add(Dense(latent_dim, activation=None))
    
    encoder = Sequential()

    encoder.add(Conv2D(filters=32,kernel_size=(3,3),padding='same',activation='relu'))
    encoder.add(MaxPooling2D(pool_size=(2,2)))
    encoder.add(Conv2D(filters=64,kernel_size=(3,3),padding='same',activation='relu'))
    encoder.add(MaxPooling2D(pool_size=(2,2)))
    encoder.add(Conv2D(filters=128,kernel_size=(3,3),padding='same',activation='relu'))
    encoder.add(MaxPooling2D(pool_size=(2,2)))
    encoder.add(Conv2D(filters=256,kernel_size=(3,3),padding='same',activation='relu'))
    encoder.add(MaxPooling2D(pool_size=(2,2)))
    encoder.add(Conv2D(filters=512,kernel_size=(3,3),padding='same',activation='relu'))
    encoder.add(MaxPooling2D(pool_size=(2,2)))
    encoder.add(Flatten())                  #flatten image to vector
    encoder.add(Dense(latent_dim))           #actual encoder

    # decoder = Sequential()
    # decoder.add(Dense(1000, input_shape=(latent_dim,), activation='relu'))
    # decoder.add(Dense(1000, activation='relu'))
    # decoder.add(Dense(input_dim, activation='sigmoid'))
    
    decoder = Sequential(name = 'reconstruction')
    decoder.add(Input((latent_dim,)))
    decoder.add(Dense(2*2*256))
    decoder.add(Reshape((2,2,256)))
    decoder.add(Conv2DTranspose(filters=512, kernel_size=(3, 3), strides=2, activation='relu', padding='same'))
    decoder.add(Conv2DTranspose(filters=256, kernel_size=(3, 3), strides=2, activation='relu', padding='same'))
    decoder.add(Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=2, activation='relu', padding='same'))
    decoder.add(Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=2, activation='relu', padding='same'))
    decoder.add(Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=2, activation='relu', padding='same'))
    decoder.add(Conv2DTranspose(filters=1, kernel_size=(3, 3), strides=2, activation=None, padding='same', name = 'reconstruction'))

    discriminator = Sequential()
    discriminator.add(Dense(128, input_shape=(latent_dim,), activation='relu'))
    discriminator.add(Dense(128, activation='relu'))
    discriminator.add(Dense(1, activation='sigmoid'))
    
    kl_grader = Sequential(name = 'kl_grade')
    kl_grader.add(Dense(128, input_shape=(latent_dim,), activation ='relu'))
    kl_grader.add(Dense(64, activation='relu'))
    kl_grader.add(Dense(5, activation='softmax'))
    
    autoencoder = Model(autoencoder_input, [decoder(encoder(autoencoder_input)), kl_grader(encoder(autoencoder_input))])
    autoencoder.compile(loss= ['mean_squared_error', 'categorical_crossentropy'], loss_weights = [1, 0.7], 
                        optimizer = "adamax", metrics = {'kl_grade': 'categorical_accuracy'})
    # autoencoder.compile(optimizer=Adam(lr=1e-4), loss="mean_squared_error")


    discriminator.compile(optimizer=Adam(lr=1e-4), loss="binary_crossentropy")
    discriminator.trainable = False
    generator = Model(generator_input, discriminator(encoder(generator_input)))
    generator.compile(optimizer=Adam(lr=1e-4), loss="binary_crossentropy")
    
    print("Autoencoder Architecture")
    print(autoencoder.summary())
    print("Discriminator Architecture")
    print(discriminator.summary())
    print("Generator Architecture")
    print(generator.summary())


    return autoencoder, discriminator, generator, encoder, decoder


IMSIZE = 128, 128
def ImagePreprocessing(input_img):
    h, w = IMSIZE
    print('Preprocessing ...')
    img=[]
    for i, im, in enumerate(input_img):
        tmp = cv2.resize(im, dsize=(w, h), interpolation=cv2.INTER_AREA)
        tmp = cv2.equalizeHist(tmp)
        img.append(tmp)
    print(len(img), 'images processed!')
    return np.array(img)


DATASET_PATH = '/home/centos/knee/kneeKL224'

def Class2Label(cls):
    lb = [0] * 5
    lb[int(cls)] = 1
    return lb

def DataLoad(imdir):
    folders = glob.glob(os.path.join(imdir,'*'))
    img = []
    lb = []
    cls = []
    for folder in folders:
        impath = glob.glob(os.path.join(folder,'*.png'))
        label = folder[-1]
        print('Loading', len(impath), label, 'images ...', 'HOHO')
        for i, p in enumerate(impath):
            img_whole = cv2.imread(p,0)
            h, w = img_whole.shape
            if impath[0][-5] == 'L':
                img.append(img_whole)
                lb.append(Class2Label(label))
                cls.append(int(label))
            elif impath[0][-5] == 'R':
                img.append(img_whole)
                lb.append(Class2Label(label))
                cls.append(int(label))
    return np.array(img), np.array(lb)

def load_knee_test_data():
    x_test, y_test = DataLoad(os.path.join(DATASET_PATH, 'test'))
    return (x_test, y_test)

def load_knee_data():
    x_test, y_test = DataLoad(os.path.join(DATASET_PATH, 'test'))
    x_train, y_train = DataLoad(os.path.join(DATASET_PATH, 'train'))
    x_val, y_val = DataLoad(os.path.join(DATASET_PATH, 'val'))
    x_train = np.concatenate([x_train, x_val])
    y_train = np.concatenate([y_train, y_val])
    return (x_train, y_train), (x_test, y_test)

def _load_knee_train_val_data():
    x_train, y_train = DataLoad(os.path.join(DATASET_PATH, 'train'))
    x_val, y_val = DataLoad(os.path.join(DATASET_PATH, 'val'))
    rand_x = np.random.RandomState(42)
    rand_y = np.random.RandomState(42)
    x = np.concatenate([x_train, x_val])
    y = np.concatenate([y_train, y_val])
    rand_x.shuffle(x)
    rand_y.shuffle(y)
    x_classes = {}
    y_classes = {}
    x_val_classes = {}
    y_val_classes = {}
    for i in range(5):
        x_classes[i] = x[np.where(y.argmax(1) == i)][:int(0.909*len(np.where(y.argmax(1) == i)[0]))]
        y_classes[i] = y[np.where(y.argmax(1) == i)][:int(0.909*len(np.where(y.argmax(1) == i)[0]))]
        x_val_classes[i] = x[np.where(y.argmax(1) == i)][:int(0.909*len(np.where(y.argmax(1) == i)[0]))]
        y_val_classes[i] = y[np.where(y.argmax(1) == i)][:int(0.909*len(np.where(y.argmax(1) == i)[0]))]
    x_train = np.concatenate((list(x_classes.values())))
    y_train = np.concatenate((list(y_classes.values())))
    x_val = np.concatenate((list(x_val_classes.values())))
    y_val = np.concatenate((list(y_val_classes.values())))
    return (x_train, y_train), (x_val, y_val)

def load_knee_train_val_data():
    x_train, y_train = DataLoad(os.path.join(DATASET_PATH, 'train'))
    x_val, y_val = DataLoad(os.path.join(DATASET_PATH, 'val'))
    return (x_train, y_train), (x_val, y_val)

def train(batch_size, n_epochs, latent_dim):
    autoencoder, discriminator, generator, encoder, decoder = create_model(input_shape = (128, 128, 1), latent_dim=latent_dim)
    (x_train, y_train), (x_val, y_val) = load_knee_train_val_data()

    x = x_train
    x = ImagePreprocessing(x)
    normalize = colors.Normalize(0., 255.)
    x = normalize(x)
    x = np.expand_dims(x, axis=-1)
    y = y_train
    
    x_val = ImagePreprocessing(x_val)
    normalize = colors.Normalize(0., 255.)
    x_val = normalize(x_val)
    x_val = np.expand_dims(x_val, axis=-1)
    
    # x.shape = (1000, 28, 28)
    # x = x.reshape(-1, 784)


    rand_x = np.random.RandomState(42)
    rand_y = np.random.RandomState(42)
    
    col_names = ['autoencoder_loss', 'reconstruction_loss', 'discriminator_loss', 'generator_loss', 'kl_grade_loss', 
                 'kl_grade_accuracy', 'val_kl_grade_loss', 'val_kl_grade_accuracy']

    with open(f'{desc}.csv', "w") as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(col_names)

    past = datetime.now()
    for epoch in np.arange(1, n_epochs + 1):
        autoencoder_losses = []
        reconstruction_losses = [] 
        discriminator_losses = []
        generator_losses = []
        kl_grade_losses = []
        kl_grade_accuracies = []
        
        rand_x.shuffle(x)
        rand_y.shuffle(y)
        
        # dataset = [[X, Y] for X, Y in zip(x_val, y_val)]
        # random.shuffle(dataset)
        # x_val = np.array([n[0] for n in dataset])
        # y_val = np.array([n[1] for n in dataset])
        
        for batch in np.arange((len(x) // batch_size) + 1):
            start = int(batch * batch_size)
            end = int(start + batch_size)
            if batch == (len(x) // batch_size):
                samples = x[start:]
                labels = y[start:]
                batch_size = len(x)%batch_size
            else:
                samples = x[start:end]
                labels = y[start:end]
            
            autoencoder_history = autoencoder.fit(x=samples, y=[samples, labels], epochs=1, batch_size=batch_size, validation_split=0.0, verbose=0)
    
            fake_latent = encoder.predict(samples)
            discriminator_input = np.concatenate((fake_latent, np.random.randn(batch_size, latent_dim) * 5.))
            discriminator_labels = np.concatenate((np.zeros((batch_size, 1)), np.ones((batch_size, 1))))
            discriminator_history = discriminator.fit(x=discriminator_input, y=discriminator_labels, epochs=1, batch_size=batch_size, validation_split=0.0, verbose=0)
            generator_history = generator.fit(x=samples, y=np.ones((batch_size, 1)), epochs=1, batch_size=batch_size, validation_split=0.0, verbose=0)
        
            autoencoder_losses.append(autoencoder_history.history["loss"])
            reconstruction_losses.append(autoencoder_history.history["reconstruction_loss"])
            discriminator_losses.append(discriminator_history.history["loss"])
            generator_losses.append(generator_history.history["loss"])
            kl_grade_losses.append(autoencoder_history.history["kl_grade_loss"])
            kl_grade_accuracies.append(autoencoder_history.history["kl_grade_categorical_accuracy"])
            
        now = datetime.now()
        print("\nEpoch {}/{} - {:.1f}s".format(epoch, n_epochs, (now - past).total_seconds()))
        print("Autoencoder Loss: {}".format(np.mean(autoencoder_losses)))
        print("Reconstruction Loss: {}".format(np.mean(reconstruction_losses)))
        

        print("Discriminator Loss: {}".format(np.mean(discriminator_losses)))
        print("Generator Loss: {}".format(np.mean(generator_losses)))
        
        print("KL grade loss: {}".format(np.mean(kl_grade_losses)))
        print("KL grade accuracy: {}".format(np.mean(kl_grade_accuracies)))
        
        val = autoencoder.evaluate(x_val, [x_val, y_val], verbose = 0)
        print(f"Validation KL grade accuracy: {val[3]}")
        print(f"Validation KL grade loss: {val[2]}")
        past = now
        
        log = [np.mean(autoencoder_losses), np.mean(reconstruction_losses), np.mean(discriminator_losses), 
               np.mean(generator_losses), np.mean(kl_grade_losses), np.mean(kl_grade_accuracies), val[2], val[3]]
        
        with open(f'{desc}.csv', "a") as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(log)

        if epoch % 50 == 0:
            print("\nSaving models...")
            # autoencoder.save('{}_autoencoder.h5'.format(desc))
            encoder.save('{}_encoder.h5'.format(desc))
            decoder.save('{}_decoder.h5'.format(desc))
            autoencoder.save('{}_autoencoder.h5'.format(desc))
            # if FLAGS.adversarial:
            # 	discriminator.save('{}_discriminator.h5'.format(desc))
            # 	generator.save('{}_generator.h5'.format(desc))

    # autoencoder.save('{}_autoencoder.h5'.format(desc))
    encoder.save('{}_encoder.h5'.format(desc))
    decoder.save('{}_decoder.h5'.format(desc))
    autoencoder.save('{}_autoencoder.h5'.format(desc))
    # if FLAGS.adversarial:
        # discriminator.save('{}_discriminator.h5'.format(desc))
        # generator.save('{}_generator.h5'.format(desc))

def reconstruct(n_samples):
    encoder = load_model('{}_encoder.h5'.format(desc))
    decoder = load_model('{}_decoder.h5'.format(desc))
    (x_train, y_train), (x_test, y_test) = load_knee_data()
    choice = np.random.choice(np.arange(n_samples))
    original = x_test[choice]
    original = np.array([original])
    original = ImagePreprocessing(original)
    normalize = colors.Normalize(0., 255.)
    original = normalize(original)
    original = np.expand_dims(original, axis=-1)
    latent = encoder.predict(original)
    reconstruction = decoder.predict(latent)
    draw([{"title": "Original", "image": np.squeeze(original)}, {"title": "Reconstruction", "image": np.squeeze(reconstruction)}])

def generate(latent=None):
    decoder = load_model('{}_decoder.h5'.format(desc))
    if latent is None:
        latent = np.random.randn(1, latent_dim)
    else:
        latent = np.array(latent)
    sample = decoder.predict(latent.reshape(1, latent_dim))
    draw([{"title": "Sample", "image": np.squeeze(sample)}])

def draw(samples):
    fig = plt.figure(figsize=(5 * len(samples), 5))
    gs = gridspec.GridSpec(1, len(samples))
    for i, sample in enumerate(samples):
        ax = plt.Subplot(fig, gs[i])
        # ax.imshow((sample["image"] * 255.).reshape(28, 28), cmap='gray')
        ax.imshow((sample["image"] * 255.), cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        ax.set_title(sample["title"])
        fig.add_subplot(ax)
    plt.show(block=False)
    # raw_input("Press Enter to Exit")

def generate_grid(latent=None):
    decoder = load_model('{}_decoder.h5'.format(desc))
    samples = []
    for i in np.arange(400):
        latent = np.linspace((i % 20) * 1.5 , (i / 20) * 1.5, latent_dim)
        samples.append({
            "image": np.squeeze(decoder.predict(latent.reshape(1, latent_dim)))
        })
    draw_grid(samples)

def draw_grid(samples):
    fig = plt.figure(figsize=(15, 15))
    gs = gridspec.GridSpec(20, 20, wspace=-.5, hspace=0)
    for i, sample in enumerate(samples):
        ax = plt.Subplot(fig, gs[i])
        # ax.imshow((sample["image"] * 255.).reshape(28, 28), cmap='gray')
        ax.imshow((sample["image"] * 255.), cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        # ax.set_title(sample["title"])
        fig.add_subplot(ax)
    plt.show(block=False)
    raw_input("Press Enter to Exit")
    # fig.savefig("images/{}_grid.png".format(desc), bbox_inches="tight", dpi=300)

def plot(n_samples):
    encoder = load_model('{}_encoder.h5'.format(desc))
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x = x_test[:n_samples].reshape(n_samples, 784)
    y = y_test[:n_samples]
    normalize = colors.Normalize(0., 255.)
    x = normalize(x)
    latent = encoder.predict(x)
    if latent_dim > 2:
        tsne = TSNE()
        print("\nFitting t-SNE, this will take awhile...")
        latent = tsne.fit_transform(latent)
    fig, ax = plt.subplots()
    for label in np.arange(10):
        ax.scatter(latent[(y_test == label), 0], latent[(y_test == label), 1], label=label, s=3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    ax.set_aspect('equal')
    ax.set_title("Latent Space")
    plt.show(block=False)
    raw_input("Press Enter to Exit")
    # fig.savefig("images/{}_latent.png".format(desc), bbox_inches="tight", dpi=300)

def main(argv):
    global desc
    desc = FLAGS.desc
    if FLAGS.train:
        train(batch_size=FLAGS.batchsize, n_epochs=FLAGS.epochs, latent_dim = FLAGS.latent_dim)
    elif FLAGS.reconstruct:
        reconstruct(n_samples=FLAGS.test_samples)
    elif FLAGS.generate:
        if FLAGS.latent_vec:
            assert len(FLAGS.latent_vec) == FLAGS.latent_dim, "Latent vector provided is of dim {}; required dim is {}".format(len(FLAGS.latent_vec), FLAGS.latent_dim)
            generate(FLAGS.latent_vec)
        else:
            generate()
    elif FLAGS.generate_grid:
        generate_grid()
    elif FLAGS.plot:
        plot(FLAGS.test_samples)


if __name__ == "__main__":
    app.run(main)




