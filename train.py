import dataset
import tensorflow as tf
import time
from datetime import timedelta
import math
import random
import numpy as np
import os


from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)


batch_size = 32


classes = os.listdir('training_data')
num_classes = len(classes)


validation_size = 0.2
img_size = 128
num_channels = 3
train_path='training_data'


data = dataset.read_train_sets(train_path, img_size, classes, validation_size=validation_size)


print("Complete reading input data. Will Now print a snippet of it")
print("Number of files in Training-set:\t\t{}".format(len(data.train.labels)))
print("Number of files in Validation-set:\t{}".format(len(data.valid.labels)))



session = tf.Session()
x = tf.placeholder(tf.float32, shape=[None, img_size,img_size,num_channels], name='x')

## labels
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)



##Network graph params
filter_size_conv1 = 3 
num_filters_conv1 = 32

filter_size_conv2 = 3
num_filters_conv2 = 32

filter_size_conv3 = 3
num_filters_conv3 = 64



fc_layer_size = 128

def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))



def create_convolutional_layer(input,
               num_input_channels, 
               conv_filter_size,        
               num_filters):  

    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])

    biases = create_biases(num_filters)

    layer = tf.nn.conv2d(input=input,
                     filter=weights,
                     strides=[1, 1, 1, 1],
                     padding='SAME')

    layer += biases

    layer = tf.nn.max_pool(value=layer,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME')
    layer = tf.nn.relu(layer)

    return layer

def create_con_layer(input,
               num_input_channels,
               conv_filter_size,
               num_filters):

    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])

    biases = create_biases(num_filters)

    layer = tf.nn.conv2d(input=input,
                     filter=weights,
                     strides=[1, 1, 1, 1],
                     padding='SAME')

    layer += biases
    layer = tf.nn.relu(layer)

    return layer

def create_decon_layer(input,
               num_input_channels,
               conv_filter_size,
               num_filters):

    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])


    layer = tf.nn.conv2d_transpose(value=input,
                     filter=weights,
                     strides=[1, 1, 1, 1],
                     padding='SAME')

    return layer

def create_flatten_layer(layer):

    layer_shape = layer.get_shape()


    num_features = layer_shape[1:4].num_elements()


    layer = tf.reshape(layer, [-1, num_features])

    return layer



def create_fc_layer(input,          
             num_inputs,    
             num_outputs,
             use_relu=True):
    

    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)

    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer


def create_fc_layer_group(input,
                    num_inputs,
                    num_outputs,
                    use_relu=True):
    weights = create_weights(shape=[num_inputs, num_outputs*4])
    biases = create_biases(num_outputs*4)

    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer



def cnn(x):



    layer_conv1_1 = create_con_layer(input=x,
                   num_input_channels=num_channels,
                   conv_filter_size=filter_size_conv1,
                   num_filters=64)
    layer_conv1_2 = create_convolutional_layer(input=layer_conv1_1,
                   num_input_channels=64,
                   conv_filter_size=filter_size_conv2,
                   num_filters=64)

    layer_conv2_1 = create_con_layer(input=layer_conv1_2,
                   num_input_channels=64,
                   conv_filter_size=filter_size_conv1,
                   num_filters=128)


    layer_conv2_2 = create_convolutional_layer(input=layer_conv2_1,
                   num_input_channels=128,
                   conv_filter_size=filter_size_conv1,
                   num_filters=128)

    layer_conv3_1= create_con_layer(input=layer_conv2_2,
                   num_input_channels=128,
                   conv_filter_size=filter_size_conv3,
                   num_filters=256)

    layer_conv3_2= create_con_layer(input=layer_conv3_1,
                   num_input_channels=256,
                   conv_filter_size=filter_size_conv3,
                   num_filters=256)

    layer_conv3_3= create_con_layer(input=layer_conv3_2,
                   num_input_channels=256,
                   conv_filter_size=filter_size_conv3,
                   num_filters=256)

    layer_conv3_4= create_convolutional_layer(input=layer_conv3_3,
                   num_input_channels=256,
                   conv_filter_size=filter_size_conv3,
                   num_filters=256)

    layer_conv4_1= create_con_layer(input=layer_conv3_4,
                   num_input_channels=256,
                   conv_filter_size=filter_size_conv3,
                   num_filters=512)

    layer_conv4_2= create_con_layer(input=layer_conv4_1,
                   num_input_channels=512,
                   conv_filter_size=filter_size_conv3,
                   num_filters=512)

    layer_conv4_3= create_con_layer(input=layer_conv4_2,
                   num_input_channels=512,
                   conv_filter_size=filter_size_conv3,
                   num_filters=512)

    layer_conv4_4= create_convolutional_layer(input=layer_conv4_3,
                   num_input_channels=512,
                   conv_filter_size=filter_size_conv3,
                   num_filters=512)

    layer_conv5_1= create_con_layer(input=layer_conv4_4,
                   num_input_channels=512,
                   conv_filter_size=filter_size_conv3,
                   num_filters=512)

    layer_conv5_2= create_con_layer(input=layer_conv5_1,
                   num_input_channels=512,
                   conv_filter_size=filter_size_conv3,
                   num_filters=512)

    layer_conv5_3= create_con_layer(input=layer_conv5_2,
                   num_input_channels=512,
                   conv_filter_size=filter_size_conv3,
                   num_filters=512)

    layer_conv5_4= create_con_layer(input=layer_conv5_3,
                   num_input_channels=512,
                   conv_filter_size=filter_size_conv3,
                   num_filters=512)

    return layer_conv5_4


layer_data = cnn(x)


layer_data = tf.nn.avg_pool(value=layer_data,
                            ksize=[1, 32, 32, 1],
                            strides=[1, 32, 32, 1],
                            padding='SAME')

num_cols = int(layer_data.get_shape()[2])
#layer_data = tf.matmul(layer_data,create_weights(shape=[num_cols, num_cols*4]))
layer_data=tf.concat([layer_data, layer_data,layer_data,layer_data], 1)
# layer_data = tf.matmul(layer_data,create_weights(shape=[4,4]))
split4, split1, split2,split3 = tf.split(layer_data, num_or_size_splits=4, axis=1)


layer_data_final_1 = tf.nn.avg_pool(value=split1,
                            ksize=[1, 32, 32, 1],
                            strides=[1, 32, 32, 1],
                            padding='SAME')
layer_data_final_2 = tf.nn.avg_pool(value=split2,
                            ksize=[1, 32, 32, 1],
                            strides=[1, 32, 32, 1],
                            padding='SAME')
layer_data_final_3 = tf.nn.avg_pool(value=split3,
                            ksize=[1, 32, 32, 1],
                            strides=[1, 32, 32, 1],
                            padding='SAME')
layer_data_final_4 = tf.nn.avg_pool(value=split4,
                            ksize=[1, 32, 32, 1],
                            strides=[1, 32, 32, 1],
                            padding='SAME')

layer_t_1= tf.nn.tanh(layer_data_final_1)
layer_t_2= tf.nn.tanh(layer_data_final_2)
layer_t_3= tf.nn.tanh(layer_data_final_3)
layer_t_4= tf.nn.tanh(layer_data_final_4)

mask_1 = tf.nn.sigmoid(layer_t_1)


mask_2 = tf.nn.sigmoid(layer_t_2)


mask_3= tf.nn.sigmoid(layer_t_3)


mask_4 = tf.nn.sigmoid(layer_t_4)


part_1 = tf.multiply(layer_data,mask_1)

part_1 = tf.nn.max_pool(value=part_1,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME')



part_2 = tf.multiply(layer_data,mask_2)

part_2 = tf.nn.max_pool(value=part_2,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME')


part_3 = tf.multiply(layer_data,mask_3)

part_3 = tf.nn.max_pool(value=part_3,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME')


part_4 = tf.multiply(layer_data,mask_4)

part_4 = tf.nn.max_pool(value=part_4,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME')


prediction = []

layer_flat_1 = create_flatten_layer(part_1)

layer_fc1_1 = create_fc_layer(input=layer_flat_1,
                     num_inputs=layer_flat_1.get_shape()[1:4].num_elements(),
                     num_outputs=fc_layer_size,
                     use_relu=True)

layer_fc2_1 = create_fc_layer(input=layer_fc1_1,
                     num_inputs=fc_layer_size,
                     num_outputs=num_classes,
                     use_relu=False) 

y_pred_1 = tf.nn.softmax(layer_fc2_1,name='y_pred_1')
y_pred_cls_1 = tf.argmax(y_pred_1, dimension=1)

prediction.append(y_pred_cls_1)
session.run(tf.global_variables_initializer())
cross_entropy_1 = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2_1,
                                                    labels=y_true)


layer_flat_2 = create_flatten_layer(part_2)

layer_fc1_2 = create_fc_layer(input=layer_flat_2,
                     num_inputs=layer_flat_2.get_shape()[1:4].num_elements(),
                     num_outputs=fc_layer_size,
                     use_relu=True)

layer_fc2_2 = create_fc_layer(input=layer_fc1_2,
                     num_inputs=fc_layer_size,
                     num_outputs=num_classes,
                     use_relu=False)

y_pred_2 = tf.nn.softmax(layer_fc2_2,name='y_pred_2')
y_pred_cls_2 = tf.argmax(y_pred_2, dimension=1)

prediction.append(y_pred_cls_2)
session.run(tf.global_variables_initializer())
cross_entropy_2 = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2_2,
                                                    labels=y_true)


layer_flat_3 = create_flatten_layer(part_3)

layer_fc1_3 = create_fc_layer(input=layer_flat_3,
                     num_inputs=layer_flat_3.get_shape()[1:4].num_elements(),
                     num_outputs=fc_layer_size,
                     use_relu=True)

layer_fc2_3 = create_fc_layer(input=layer_fc1_3,
                     num_inputs=fc_layer_size,
                     num_outputs=num_classes,
                     use_relu=False)

y_pred_3 = tf.nn.softmax(layer_fc2_3,name='y_pred_3')
y_pred_cls_3 = tf.argmax(y_pred_3, dimension=1)

prediction.append(y_pred_cls_3)
session.run(tf.global_variables_initializer())
cross_entropy_3 = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2_3,
                                                    labels=y_true)


layer_flat_4 = create_flatten_layer(part_4)

layer_fc1_4 = create_fc_layer(input=layer_flat_4,
                     num_inputs=layer_flat_4.get_shape()[1:4].num_elements(),
                     num_outputs=fc_layer_size,
                     use_relu=True)

layer_fc2_4 = create_fc_layer(input=layer_fc1_4,
                     num_inputs=fc_layer_size,
                     num_outputs=num_classes,
                     use_relu=False)

y_pred_4 = tf.nn.softmax(layer_fc2_4,name='y_pred_4')
y_pred_cls_4 = tf.argmax(y_pred_4, dimension=1)

prediction.append(y_pred_cls_4)
session.run(tf.global_variables_initializer())
cross_entropy_4 = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2_4,
                                                    labels=y_true)

y_pred_cls = max(set(prediction), key=prediction.count)

cross_entropy = cross_entropy_1+cross_entropy_2+cross_entropy_3+cross_entropy_4


cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-7).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


session.run(tf.global_variables_initializer()) 


def show_progress(epoch, feed_dict_train, feed_dict_validate, val_loss):
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    msg = "Training Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f}"
    print(msg.format(epoch + 1, acc, val_acc, val_loss))

total_iterations = 0

saver = tf.train.Saver()

def train(num_iteration):
    global total_iterations
    
    for i in range(total_iterations,
                   total_iterations + num_iteration):

        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(batch_size)
        x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(batch_size)

        
        feed_dict_tr = {x: x_batch,
                           y_true: y_true_batch}
        feed_dict_val = {x: x_valid_batch,
                              y_true: y_valid_batch}

        session.run(optimizer, feed_dict=feed_dict_tr)

        if i % int(data.train.num_examples/batch_size) == 0: 
            val_loss = session.run(cost, feed_dict=feed_dict_val)
            epoch = int(i / int(data.train.num_examples/batch_size))
            show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss)
            saver.save(session, './model')


    total_iterations += num_iteration

train(num_iteration=5000)
