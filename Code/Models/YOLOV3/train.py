#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : train.py
#   Author      : YunYang1994
#   Created date: 2019-07-18 09:18:54
#   Description :
#
#================================================================

import os
import shutil
import numpy as np
import tensorflow as tf
from core.dataset import Dataset
from core.yolov3 import YOLOv3, decode, compute_loss
from core.config import cfg

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

global_steps = None
warmup_steps = None
total_steps = None
model = None
optimizer = None
writer = None
trainset = None


def setup(log_dir):
    global global_steps
    global warmup_steps
    global total_steps
    global model
    global optimizer
    global writer
    global trainset

    trainset = Dataset('train')
    logdir = log_dir
    steps_per_epoch = len(trainset)

    global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
    warmup_steps = cfg.TRAIN.WARMUP_EPOCHS * steps_per_epoch
    total_steps = cfg.TRAIN.EPOCHS * steps_per_epoch

    input_tensor = tf.keras.layers.Input([416, 416, 3])
    conv_tensors = YOLOv3(input_tensor)

    output_tensors = []
    for i, conv_tensor in enumerate(conv_tensors):
        pred_tensor = decode(conv_tensor, i)
        output_tensors.append(conv_tensor)
        output_tensors.append(pred_tensor)

    model = tf.keras.Model(input_tensor, output_tensors)
    if (cfg.TRAIN.LOAD_WEIGHTS):
        model.load_weights('./yolov3')

    optimizer = tf.keras.optimizers.Adam()
    if os.path.exists(logdir): shutil.rmtree(logdir)
    #writer = tf.contrib.summary.create_file_writer(logdir)
    writer = tf.contrib.summary.create_file_writer(logdir)

def train_step(image_data, target):
    global global_steps
    global warmup_steps
    global total_steps
    global model
    global optimizer
    global writer
    global trainset
    with tf.GradientTape() as tape:
        pred_result = model(image_data, training=True)
        giou_loss = conf_loss = prob_loss = 0

        # optimizing process
        for i in range(3):
            conv, pred = pred_result[i*2], pred_result[i*2+1]
            loss_items = compute_loss(pred, conv, *target[i], i)
            giou_loss += loss_items[0]
            conf_loss += loss_items[1]
            prob_loss += loss_items[2]

        total_loss = giou_loss + conf_loss + prob_loss

        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        tf.print("=> STEP %4d   lr: %.6f   giou_loss: %4.2f   conf_loss: %4.2f   "
                 "prob_loss: %4.2f   total_loss: %4.2f" %(global_steps, optimizer.lr.numpy(),
                                                          giou_loss, conf_loss,
                                                          prob_loss, total_loss))

        # tf.print("=> STEP %4d   lr: %.6f   giou_loss: %4.2f   conf_loss: %4.2f   "
        #          "prob_loss: %4.2f   total_loss: %4.2f" %(global_steps, optimizer.lr.eval(session=tf.compat.v1.Session()),
        #                                                   giou_loss, conf_loss,
        #                                                   prob_loss, total_loss))

        # update learning rate
        global_steps.assign_add(1)
        if global_steps < warmup_steps:
            lr = global_steps / warmup_steps *cfg.TRAIN.LR_INIT
        else:
            lr = cfg.TRAIN.LR_END + 0.5 * (cfg.TRAIN.LR_INIT - cfg.TRAIN.LR_END) * (
                (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi))
            )
        optimizer.lr.assign(lr.numpy())

        # writing summary data
        with writer.as_default(), tf.contrib.summary.always_record_summaries():

            # tf.compat.v1.summary.scalar(name="lr", tensor=optimizer.lr) #, step=global_steps)
            # tf.compat.v1.summary.scalar("loss/total_loss", total_loss)#, step=global_steps)
            # tf.compat.v1.summary.scalar("loss/giou_loss", giou_loss)#, step=global_steps)
            # tf.compat.v1.summary.scalar("loss/conf_loss", conf_loss)#, step=global_steps)
            # tf.compat.v1.summary.scalar("loss/prob_loss", prob_loss)#, step=global_steps)
            tf.contrib.summary.scalar("lr", optimizer.lr, step=global_steps)
            tf.contrib.summary.scalar("loss/total_loss", total_loss, step=global_steps)
            tf.contrib.summary.scalar("loss/giou_loss", giou_loss, step=global_steps)
            tf.contrib.summary.scalar("loss/conf_loss", conf_loss, step=global_steps)
            tf.contrib.summary.scalar("loss/prob_loss", prob_loss, step=global_steps)
            writer.flush()



def run(log_dir, output_dir):
    tf.summary.FileWriterCache.clear()
    setup(log_dir)
    for epoch in range(cfg.TRAIN.EPOCHS):
        for image_data, target in trainset:
            train_step(image_data, target)
    model.save_weights(output_dir + r'\yolov3')


'''
for tests - needs to be removed 
'''
if __name__ == "__main__":
    run(r"./data/log/", r'E:\FinalProject\temp')


