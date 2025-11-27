from tqdm import tqdm
import tensorflow as tf
import numpy as np
import datetime
import time
import os
# from skimage.metrics import structural_similarity as ssim
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score
# import csv
# import pandas as pd
import matplotlib.pyplot as plt
# import matplotlib.patheffects as PathEffects

import warnings
warnings.filterwarnings("error")
# from scipy.stats.stats import PearsonRConstantInputWarning, PearsonRNearConstantInputWarning


class CGAN:
    def __init__(self, output_path='./', gen_lr=2e-4, dis_lr=2e-4, kernel_size=4, strides=2, n_slices=1, fat_masked=False, histEq=False):
        self.OUTPUT_CHANNELS = 1
        self.kernel_size = kernel_size
        self.strides = strides
        self.n_slices = n_slices
        self.output_path = output_path
        self.fat_masked = fat_masked
        self.histEQ = histEq
        
        self.generator = self.__Generator()
        self.discriminator = self.__Discriminator()
        
        self.generator_optimizer = tf.keras.optimizers.Adam(gen_lr, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(dis_lr, beta_1=0.5)
        
        if self.output_path is not None:
            self.log_dir = os.path.join(self.output_path, "logs/")
            self.__summary_writer_train = tf.summary.create_file_writer(self.log_dir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/train/")
            self.__summary_writer_valid = tf.summary.create_file_writer(self.log_dir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/valid/")

            self.checkpoint_dir = os.path.join(self.output_path, "training_checkpoints/")
            self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")

            os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.__checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                                discriminator_optimizer=self.discriminator_optimizer,
                                                generator=self.generator,
                                                discriminator=self.discriminator)
        
    def plot_generator(self):
        return tf.keras.utils.plot_model(self.generator, show_shapes=True, dpi=64)
        
    def plot_discriminator(self):
        return tf.keras.utils.plot_model(self.discriminator, show_shapes=True, dpi=64)

    def __plot_regr(self, ax, xdata, ydata, title, xlabel, ylabel):
        
        p = [0, 0]
        try:
            p = pearsonr(xdata, ydata)
        except Exception as e:
            print(e)

        ax.scatter(xdata, ydata, color='violet', s=50, alpha=0.7, edgecolors='blue')

        min_ix = np.argmin(xdata)
        max_ix = np.argmax(xdata)
        ax.plot([xdata[min_ix], xdata[max_ix]], [xdata[min_ix], xdata[max_ix]], "k-.")
        
        slope = 0
        try:
            regr = LinearRegression().fit(xdata.reshape(-1, 1), ydata.reshape(-1, 1))
            adjust_points = regr.predict(xdata.reshape(-1, 1))
        

            slope = float(regr.coef_)
            inter = float(regr.intercept_)
            ax.plot([xdata[min_ix], xdata[max_ix]], [adjust_points[min_ix], adjust_points[max_ix]], "g-.")
        except Exception as e:
            print(e)

        textstr = '\n'.join((
            r'$\rho=%.2f$' % (p[0], ),
            r'$Slope=%.2f$' % (slope,)))

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        ax.text(xdata[min_ix], xdata[max_ix], textstr, fontsize=10, verticalalignment='top', bbox=props)
        ax.set_title(title, fontsize=20, fontweight="bold")
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    def generate_images(self, test_input, target, mask, means, progress=None, output_path=None):        
        prediction = self.generator(test_input, training=False).numpy()
        
#         print(prediction.min(), prediction.max())
#         print(np.histogram(prediction))
        
        fig = plt.figure(constrained_layout=True, figsize=(15,10))
        gs = fig.add_gridspec(4, 5, height_ratios=[1, 1, 1, 1], width_ratios=[1, 1, 1, 1, 2])

        for i in range(4):

            ax1 = fig.add_subplot(gs[i, 0])
            ax1.imshow(test_input[i, :, :, int(test_input.shape[-1]/2.0)], cmap='gray', vmin=-3, vmax=15)
            ax1.axis('off')
                    
            vmin = 0
            vmax = 1.5

            ax2 = fig.add_subplot(gs[i, 1])
            ax2.imshow(target[i, :, :, 0], cmap='inferno', vmin=vmin, vmax=vmax)
            ax2.axis('off')

            ax3 = fig.add_subplot(gs[i, 2])
            ax3.imshow(prediction[i, :, :, 0], cmap='inferno', vmin=vmin, vmax=vmax)
            ax3.axis('off')

            fat_pet = np.array(target[i, :, :, 0])
            fat_pet[~mask[i, :, :, 1]] = 0
            fat_pred = np.array(prediction[i, :, :, 0])
            fat_pred[~mask[i, :, :, 1]] = 0
            fat_error = np.abs(fat_pred - fat_pet)
            ax4 = fig.add_subplot(gs[i, 3])
            ax4.imshow(fat_error, cmap='inferno', vmin=0, vmax=1)
            ax4.axis('off')

            if i == 0:
                ax1.set_title("CT")
                ax2.set_title("PET")
                ax3.set_title("Prediction")
                ax4.set_title("Fat Error")

        ax5 = fig.add_subplot(gs[0:2,4])
        self.__plot_regr(ax5,
                         xdata=means["target_taorta_means"],
                         ydata=means["predict_taorta_means"],
                         title="Transversal Aorta",
                         xlabel="Reference",
                         ylabel="Predicted")

        ax10 = fig.add_subplot(gs[2:4,4])
        self.__plot_regr(ax10,
                         xdata=means["target_neck_means"],
                         ydata=means["predict_neck_means"],
                         title="Neck",
                         xlabel="Reference",
                         ylabel="Predicted")

        image_path = os.path.join(output_path,"images/")
        if not os.path.exists(image_path):
            os.makedirs(image_path)
        plt.savefig(os.path.join(image_path,"%04d.png" % progress), bbox_inches='tight', dpi=100)
        plt.close(fig)
    
    def load_checkpoint(self, path):
        self.__checkpoint.restore(path).expect_partial()
    
    def predict(self, test_input):
        return self.generator(test_input, training=False)
    
    def eval_network(self, eval_ds, label, nsamples=None):        
        means = {'target_neck_means': np.empty([0]),
                 'predict_neck_means': np.empty([0]),
                 'target_taorta_means': np.empty([0]),
                 'predict_taorta_means': np.empty([0])}
        summary = {'gen_total_loss_%s' % label: 0,
                   'gen_gan_loss_%s' % label: 0,
                   'gen_l1_loss_%s' % label: 0,
                   'disc_loss_%s' % label: 0}
        if nsamples is not None:
            sample_idxs = np.linspace(0, eval_ds.GetTotalSamples()-1, nsamples, dtype="int")
        else:
            sample_idxs = range(eval_ds.GetTotalSamples())
        
        for n in tqdm(sample_idxs):
            test_input, target, mask, slices_pos, case_id = eval_ds.GetDataSlicedByIndex(n, normalized=True, preprocess=False)
            
#             input_neck = test_input[slices_pos[1]:slices_pos[2],...]
#             input_taorta = test_input[slices_pos[1]:slices_pos[2],...]
                                    
            gen_output = self.generator(test_input, training=False)
            disc_real_output = self.discriminator([test_input, target], training=False)
            disc_generated_output = self.discriminator([test_input, gen_output], training=False)
            
            target = tf.boolean_mask(target, mask[:,:,:,1:2])
            gen_output = tf.boolean_mask(gen_output, mask[:,:,:,1:2])
            
            gen_total_loss, gen_gan_loss, gen_l1_loss = self.__generator_loss(disc_generated_output, gen_output, target)
            disc_loss = self.__discriminator_loss(disc_real_output, disc_generated_output)
            
            means['target_neck_means'] = np.append(means['target_neck_means'], np.mean(target.numpy()[slices_pos[1]:slices_pos[2],...]))
            means['predict_neck_means'] = np.append(means['predict_neck_means'], np.mean(gen_output.numpy()[slices_pos[1]:slices_pos[2],...]))
            
            means['target_taorta_means'] = np.append(means['target_taorta_means'], np.mean(target.numpy()[slices_pos[0],...]))
            means['predict_taorta_means'] = np.append(means['predict_taorta_means'], np.mean(gen_output.numpy()[slices_pos[0],...]))
            
            summary['gen_total_loss_%s' % label] += gen_total_loss/float(eval_ds.GetTotalSamples())
            summary['gen_gan_loss_%s' % label] += gen_gan_loss/float(eval_ds.GetTotalSamples())
            summary['gen_l1_loss_%s' % label] += gen_l1_loss/float(eval_ds.GetTotalSamples())
            summary['disc_loss_%s' % label] += disc_loss/float(eval_ds.GetTotalSamples())
                
#             prediction = self.generator(test_input, training=False).numpy()
  
#             predict_neck = prediction[slices_pos[1]:slices_pos[2],...]
#             target_neck = target[slices_pos[1]:slices_pos[2],...]
#             mask_neck = mask[slices_pos[1]:slices_pos[2],...]

#             target_neck_means = np.append(target_neck_means, np.mean(target_neck[mask_neck[...,1]]))
#             predict_neck_means = np.append(predict_neck_means, np.mean(predict_neck[mask_neck[...,1]]))

#             predict_taorta = prediction[slices_pos[0],...]
#             target_taorta = target[slices_pos[0],...]
#             mask_taorta = mask[slices_pos[0],...]

#             target_taorta_means = np.append(target_taorta_means, np.mean(target_taorta[mask_taorta[...,1]]))
#             predict_taorta_means = np.append(predict_taorta_means, np.mean(predict_taorta[mask_taorta[...,1]]))
            
        return means, summary
    
            
    def fit(self, train_ds, epochs, valid_ds):
        example_idxs = np.ones((6,2), dtype="int") * 5
        example_idxs[:,0] = np.arange(6)

        if valid_ds is not None:
            example_input, example_target, example_mask = valid_ds.take(example_idxs)
            
        best_correlation_neck = 0
        best_correlation_taorta = 0
        for epoch in range(epochs):
            start = time.time()
            print("Epoch: ", epoch)

            for n in tqdm(range(train_ds.GetTotalBatches())):
                input_image, target, mask = train_ds.get_batch()
                self.__train_step(input_image, target, mask)
            print()
            
            # train_means, train_summary = self.eval_network(train_ds, "train", nsamples=valid_ds.GetTotalSamples())
            
            # neck_correlation_train = 0
            # try:
            #     neck_correlation_train, _ = pearsonr(train_means["predict_neck_means"], train_means["target_neck_means"])
            #     neck_correlation_train = abs(neck_correlation_train)
            #     with self.__summary_writer_train.as_default():
            #         tf.summary.scalar('neck_correlation_train', neck_correlation_train, step=epoch)
            # except Exception as e:
            #     print("neck_correlation_train")
            #     print(e)
                
            # taorta_correlation_train = 0
            # try:
            #     taorta_correlation_train, _ = pearsonr(train_means["predict_taorta_means"], train_means["target_taorta_means"])
            #     taorta_correlation_train = abs(taorta_correlation_train)
            #     with self.__summary_writer_train.as_default():
            #         tf.summary.scalar('taorta_correlation_train', taorta_correlation_train, step=epoch)
            # except Exception as e:
            #     print("taorta_correlation_train")
            #     print(e)
            
            # with self.__summary_writer_train.as_default():
            #     tf.summary.scalar('gen_total_loss_train', train_summary["gen_total_loss_train"], step=epoch)
            #     tf.summary.scalar('gen_gan_loss_train', train_summary["gen_gan_loss_train"], step=epoch)
            #     tf.summary.scalar('gen_l1_loss_train', train_summary["gen_l1_loss_train"], step=epoch)
            #     tf.summary.scalar('disc_loss_train', train_summary["disc_loss_train"], step=epoch)
                

            if valid_ds is not None:
                valid_means, valid_summary = self.eval_network(valid_ds, "valid")
                self.generate_images(example_input, example_target, example_mask,
                                     valid_means, epoch, self.output_path)
                
                
                neck_correlation_valid = 0
                try:
                    neck_correlation_valid, _ = pearsonr(valid_means["predict_neck_means"], valid_means["target_neck_means"])
                    neck_correlation_valid = abs(neck_correlation_valid)
                    
                    if best_correlation_neck <= neck_correlation_valid:
                        best_correlation_neck = neck_correlation_valid
                        self.__checkpoint.write(file_prefix=self.checkpoint_prefix+"_correlation_neck")
                        
                    with self.__summary_writer_valid.as_default():
                        tf.summary.scalar('neck_correlation_valid', neck_correlation_valid, step=epoch)
                except Exception as e:
                    print("neck_correlation_valid")
                    print(e)
                
                taorta_correlation_valid = 0
                try:
                    taorta_correlation_valid, _ = pearsonr(valid_means["predict_taorta_means"], valid_means["target_taorta_means"])
                    taorta_correlation_valid = abs(taorta_correlation_valid)
                    
                    if best_correlation_taorta <= taorta_correlation_valid:
                        best_correlation_taorta = taorta_correlation_valid
                        self.__checkpoint.write(file_prefix=self.checkpoint_prefix+"_correlation_taorta")
                    
                    with self.__summary_writer_valid.as_default():
                        tf.summary.scalar('taorta_correlation_valid', taorta_correlation_valid, step=epoch)
                except Exception as e:
                    print("taorta_correlation_valid")
                    print(e)
            
                with self.__summary_writer_valid.as_default():
                    tf.summary.scalar('gen_total_loss_valid', valid_summary["gen_total_loss_valid"], step=epoch)
                    tf.summary.scalar('gen_gan_loss_valid', valid_summary["gen_gan_loss_valid"], step=epoch)
                    tf.summary.scalar('gen_l1_loss_valid', valid_summary["gen_l1_loss_valid"], step=epoch)
                    tf.summary.scalar('disc_loss_valid', valid_summary["disc_loss_valid"], step=epoch)

            print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time()-start))
        self.__checkpoint.save(file_prefix=self.checkpoint_prefix)
        
#     @tf.function
#     @tf.autograph.experimental.do_not_convert
    def __train_step(self, input_image, target, mask):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator(input_image, training=True)

            disc_real_output = self.discriminator([input_image, target], training=True)
            disc_generated_output = self.discriminator([input_image, gen_output], training=True)

            # Adding a threshold to focus on fat tissue:
            # ++++++++++++++++++++++++++++++++++++++++++   
            if self.fat_masked:         
                target = tf.boolean_mask(target, mask[:,:,:,1:2])
                gen_output = tf.boolean_mask(gen_output, mask[:,:,:,1:2])
            else:
                target = tf.boolean_mask(target, mask[:,:,:,0:1])
                gen_output = tf.boolean_mask(gen_output, mask[:,:,:,0:1])
            # ------------------------------------------
            
            gen_total_loss, gen_gan_loss, gen_l1_loss = self.__generator_loss(disc_generated_output, gen_output, target)
            disc_loss = self.__discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_total_loss, self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients, self.discriminator.trainable_variables))

    def __sigmoid(self, x, L, x0, k, b):
        y = L / (1 + tf.math.exp(-k*(x-x0))) + b
        return y

    def __histogram_equalization(self, data):
        popt = [0.98857345, 0.3881542, 8.27075912, -0.01479465]
        cdf = self.__sigmoid(data, *popt)
        h = (cdf - tf.math.reduce_min(cdf))*tf.math.reduce_max(data)/(tf.math.reduce_max(cdf)-tf.math.reduce_min(cdf))
        return h

    def __generator_loss(self, disc_generated_output, gen_output, target):
        if self.histEQ:
            gen_output = self.__histogram_equalization(gen_output)
            target = self.__histogram_equalization(target)

        LAMBDA = 100
        loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
        total_gen_loss = gan_loss + (LAMBDA * l1_loss)
        return total_gen_loss, gan_loss, l1_loss
    
    def __discriminator_loss(self, disc_real_output, disc_generated_output):
        loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
        generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
        total_disc_loss = real_loss + generated_loss
        return total_disc_loss
        
    def __downsample(self, filters, size, apply_batchnorm=True):
        initializer = tf.random_normal_initializer(0., 0.02)
        result = tf.keras.Sequential()
        result.add(tf.keras.layers.Conv2D(filters, size, strides=self.strides,
                                          padding='same',
                                          kernel_initializer=initializer,
                                          use_bias=False))
        if apply_batchnorm:
            result.add(tf.keras.layers.BatchNormalization())
        result.add(tf.keras.layers.LeakyReLU())
        return result
    
    def __upsample(self, filters, size, apply_dropout=False):
        initializer = tf.random_normal_initializer(0., 0.02)
        result = tf.keras.Sequential()
        result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=self.strides,
                                                   padding='same',
                                                   kernel_initializer=initializer,
                                                   use_bias=False))
        result.add(tf.keras.layers.BatchNormalization())
        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))
        result.add(tf.keras.layers.ReLU())
        return result
    
    def __Generator(self):
        inputs = tf.keras.layers.Input(shape=[512, 512, self.n_slices])

        down_stack = [self.__downsample(64, self.kernel_size, apply_batchnorm=False),  # (batch_size, 256, 256, 64)
                      self.__downsample(128, self.kernel_size),  # (batch_size, 128, 128, 128)
                      self.__downsample(256, self.kernel_size),  # (batch_size, 64, 64, 256)
                      self.__downsample(512, self.kernel_size),  # (batch_size, 32, 32, 512)
                      self.__downsample(512, self.kernel_size),  # (batch_size, 16, 16, 512)
                      self.__downsample(512, self.kernel_size)]  # (batch_size, 8, 8, 512)

        up_stack = [self.__upsample(512, self.kernel_size, apply_dropout=True),  # (batch_size, 16, 16, 512)
                    self.__upsample(512, self.kernel_size),  # (batch_size, 32, 32, 512)
                    self.__upsample(256, self.kernel_size),  # (batch_size, 64, 64, 256)
                    self.__upsample(128, self.kernel_size),  # (batch_size, 128, 128, 128)
                    self.__upsample(64, self.kernel_size)]  # (batch_size, 256, 256, 64)

        initializer = tf.random_normal_initializer(0., 0.02)
        lastDown = tf.keras.layers.Conv2D(self.OUTPUT_CHANNELS, self.kernel_size,
                                          strides=(2, 2),
                                          padding='same',
                                          kernel_initializer=initializer,
                                          activation='relu')
#                                           activation='linear')  # (batch_size, 128, 128, 1)

        x = inputs

        # Downsampling through the model
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = tf.keras.layers.Concatenate()([x, skip])

        x = lastDown(x)

        return tf.keras.Model(inputs=inputs, outputs=x)
    
    def __Discriminator(self):
        initializer = tf.random_normal_initializer(0., 0.02)

        inp = tf.keras.layers.Input(shape=[512, 512, self.n_slices], name='input_image')
        tar = tf.keras.layers.Input(shape=[128, 128, 1], name='target_image')

        inp128 = tf.keras.layers.MaxPool2D(pool_size=(4,4))(inp)

        x = tf.keras.layers.concatenate([inp128, tar])  # (batch_size, 128, 128, self.n_slices+1)

        down1 = self.__downsample(64, 4, False)(x)  # (batch_size, 64, 64, 64)
        down2 = self.__downsample(128, 4)(down1)  # (batch_size, 32, 32, 128)
        down3 = self.__downsample(256, 4)(down2)  # (batch_size, 16, 16, 256)

        zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 18, 18, 256)
        conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                      kernel_initializer=initializer,
                                      use_bias=False)(zero_pad1)  # (batch_size, 15, 15, 512)

        batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
        leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
        zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 17, 17, 512)
        last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(zero_pad2)  # (batch_size, 14, 14, 1)

        return tf.keras.Model(inputs=[inp, tar], outputs=last)
