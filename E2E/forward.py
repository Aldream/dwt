import tensorflow as tf
import os
import math
import sys
import scipy.io as sio
import skimage.io as skio
import scipy.ndimage.interpolation
from post_process import *

def forward_model(model, feeder, outputSavePath, batchSize=1):
    with tf.compat.v1.Session() as sess:
        tfBatchImages = tf.compat.v1.placeholder(dtype="float", shape=[None, None, None, None])
        tfBatchSS = tf.compat.v1.placeholder(dtype="float", shape=[None, None, None])
        tfBatchSSMask = tf.compat.v1.placeholder(dtype="float", shape=[None, None, None])
        keepProb = tf.compat.v1.placeholder("float")

        with tf.name_scope("model_builder"):
            print("attempting to build model")
            model.build(tfBatchImages, tfBatchSS, tfBatchSSMask, keepProb=keepProb)
            print("built the model")

        init = tf.compat.v1.initialize_all_variables()

        sess.run(init)

        if not os.path.exists(outputSavePath):
            os.makedirs(outputSavePath)

        for i in range(int(math.floor(feeder.total_samples() / batchSize))):
            imageBatch, ssBatch, ssMaskBatch, idBatch = feeder.next_batch()
            print(imageBatch.shape, ssBatch.shape, ssMaskBatch.shape)

            outputBatch = sess.run(model.outputDataArgMax, feed_dict={tfBatchImages: imageBatch,
                                                                      tfBatchSS: ssBatch,
                                                                      tfBatchSSMask: ssMaskBatch,
                                                                      keepProb: 1.0})
            outputBatch = outputBatch.astype(np.uint8)



            for j in range(len(idBatch)):
                outputFilePath = os.path.join(outputSavePath, idBatch[j] + '.png')
                outputFilePathMat = os.path.join(outputSavePath, idBatch[j] + '.mat')
                outputFileDir = os.path.dirname(outputFilePath)

                if not os.path.exists(outputFileDir):
                    os.makedirs(outputFileDir)

                outputImage = watershed_cut(outputBatch[j], ssMaskBatch[j])
                skio.imsave(outputFilePath, scipy.ndimage.interpolation.zoom(outputImage, 2.0, mode='nearest', order=0))

                sio.savemat(outputFilePathMat, {"depth_map": outputBatch[j]}, do_compression=True)

                print("processed image %d out of %d" % (j + batchSize * i + 1, feeder.total_samples()))
                sys.stdout.flush()
