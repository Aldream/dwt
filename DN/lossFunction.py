import tensorflow as tf

def angularErrorTotal(pred, gt, weight, ss, outputChannels=2):
    with tf.name_scope("angular_error"):
        pred = tf.reshape(pred, (-1, outputChannels))
        gt = tf.cast(tf.reshape(gt, (-1, outputChannels)), dtype=tf.float32)
        weight = tf.cast(tf.reshape(weight, (-1, 1)), dtype=tf.float32)
        ss = tf.cast(tf.reshape(ss, (-1, 1)), dtype=tf.float32)

        pred = tf.nn.l2_normalize(pred, 1) * 0.999999
        gt = tf.nn.l2_normalize(gt, 1) * 0.999999

        errorAngles = tf.acos(tf.reduce_sum(input_tensor=pred * gt, axis=[1], keepdims=True))

        lossAngleTotal = tf.reduce_sum(input_tensor=(tf.abs(errorAngles*errorAngles))*ss*weight)

        return lossAngleTotal

def angularErrorLoss(pred, gt, weight, ss, outputChannels=2):
        lossAngleTotal = angularErrorTotal(pred=pred, gt=gt, ss=ss, weight=weight, outputChannels=outputChannels) \
                         / (countTotal(ss)+1)

        tf.compat.v1.add_to_collection('losses', lossAngleTotal)

        totalLoss = tf.add_n(tf.compat.v1.get_collection('losses'), name='total_loss')

        return totalLoss


def exceedingAngleThreshold(pred, gt, ss, threshold, outputChannels=2):
    with tf.name_scope("angular_error"):
        pred = tf.reshape(pred, (-1, outputChannels))
        gt = tf.cast(tf.reshape(gt, (-1, outputChannels)), dtype=tf.float32)
        ss = tf.cast(tf.reshape(ss, (-1, 1)), dtype=tf.float32)

        pred = tf.nn.l2_normalize(pred, 1) * 0.999999
        gt = tf.nn.l2_normalize(gt, 1) * 0.999999

        errorAngles = tf.acos(tf.reduce_sum(input_tensor=pred * gt, axis=[1], keepdims=True)) * ss

        exceedCount = tf.reduce_sum(input_tensor=tf.cast(tf.less(threshold/180*3.14159, errorAngles), dtype=tf.float32))

        return exceedCount

def countCorrect(pred, gt, ss, k, outputChannels):
    with tf.name_scope("correct"):
        pred = tf.argmax(input=tf.reshape(pred, (-1, outputChannels)), axis=1)
        gt = tf.reshape(gt, (-1, outputChannels))

        ss = tf.cast(tf.reshape(ss, (-1, 1)), dtype=tf.float32)

        correct = tf.reduce_sum(input_tensor=tf.mul(tf.reshape(tf.cast(tf.nn.in_top_k(predictions=gt, targets=pred, k=k), dtype=tf.float32), (-1, 1)), ss), axis=[0])
        return correct


def countTotal(ss):
    with tf.name_scope("total"):
        ss = tf.cast(tf.reshape(ss, (-1, 1)), dtype=tf.float32)
        total = tf.reduce_sum(input_tensor=ss)

        return total

def countTotalWeighted(ss, weight):
    with tf.name_scope("total"):
        ss = tf.cast(tf.reshape(ss, (-1, 1)), dtype=tf.float32)
        weight = tf.cast(tf.reshape(weight, (-1, 1)), dtype=tf.float32)
        total = tf.reduce_sum(input_tensor=ss * weight)

        return total