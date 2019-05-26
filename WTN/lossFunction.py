import tensorflow as tf

def depthCELoss2(pred, gt, weight, ss, outputChannels=16):
    with tf.name_scope("depth_CE_loss"):
        pred = tf.reshape(pred, (-1, outputChannels))
        epsilon = tf.constant(value=1e-25)
        predSoftmax = tf.cast(tf.nn.softmax(pred), dtype=tf.float32)

        gt = tf.one_hot(indices=tf.cast(tf.squeeze(tf.reshape(gt, (-1, 1))), dtype=tf.int32), depth=outputChannels, dtype=tf.float32)
        ss = tf.cast(tf.reshape(ss, (-1, 1)), dtype=tf.float32)
        weight = tf.cast(tf.reshape(weight, (-1, 1)), dtype=tf.float32)

        crossEntropyScaling = tf.cast([3.0, 3.0, 3.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=tf.float32)

        crossEntropy = -tf.reduce_sum(input_tensor=((1-gt)*tf.math.log(tf.maximum(1-predSoftmax, epsilon))
                                       + gt*tf.math.log(tf.maximum(predSoftmax, epsilon)))*ss*crossEntropyScaling*weight,
                                      axis=[1])

        crossEntropySum = tf.reduce_sum(input_tensor=crossEntropy, name="cross_entropy_sum")
        return crossEntropySum

def depthCELoss(pred, gt, ss, outputChannels=16):
    with tf.name_scope("depth_CE_loss"):
        pred = tf.reshape(pred, (-1, outputChannels))
        epsilon = tf.constant(value=1e-25)
        #pred = pred + epsilon
        predSoftmax = tf.cast(tf.nn.softmax(pred), dtype=tf.float32)
        predSoftmax = predSoftmax + epsilon

        gt = tf.one_hot(indices=tf.cast(tf.squeeze(tf.reshape(gt, (-1, 1))), dtype=tf.int32), depth=outputChannels, dtype=tf.float32)
        ss = tf.cast(tf.reshape(ss, (-1, 1)), dtype=tf.float32)

        crossEntropy = -tf.reduce_sum(input_tensor=gt * tf.math.log(predSoftmax) * ss, axis=[1])

        crossEntropySum = tf.reduce_sum(input_tensor=crossEntropy, name="cross_entropy_sum")
        return crossEntropySum

def modelTotalLoss(pred, gt, weight, ss, outputChannels=1):
    lossDepthTotal = depthCELoss2(pred=pred, gt=gt, weight=weight, ss=ss,
                                  outputChannels=outputChannels) / (countTotalWeighted(ss, weight) + 1)

    tf.compat.v1.add_to_collection('losses', lossDepthTotal)

    totalLoss = tf.add_n(tf.compat.v1.get_collection('losses'), name='total_loss')

    return totalLoss

def countTotal(ss):
    with tf.name_scope("total"):
        ss = tf.cast(tf.reshape(ss, (-1, 1)), dtype=tf.float32)
        total = tf.reduce_sum(input_tensor=ss)

        return total

def countCorrect(pred, gt, ss, k, outputChannels):
    with tf.name_scope("correct"):
        pred = tf.argmax(input=tf.reshape(pred, (-1, outputChannels)), axis=1)
        gt = tf.one_hot(indices=tf.cast(tf.squeeze(tf.reshape(gt, (-1, 1))), dtype=tf.int32), depth=outputChannels, dtype=tf.float32)

        ss = tf.cast(tf.reshape(ss, (-1, 1)), dtype=tf.float32)

        correct = tf.reduce_sum(input_tensor=tf.mul(tf.reshape(tf.cast(tf.nn.in_top_k(predictions=gt, targets=pred, k=k), dtype=tf.float32), (-1, 1)), ss), axis=[0])
        return correct

def countTotalWeighted(ss, weight):
    with tf.name_scope("total"):
        ss = tf.cast(tf.reshape(ss, (-1, 1)), dtype=tf.float32)
        weight = tf.cast(tf.reshape(weight, (-1, 1)), dtype=tf.float32)
        total = tf.reduce_sum(input_tensor=ss * weight)

        return total