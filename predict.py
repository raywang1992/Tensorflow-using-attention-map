import tensorflow as tf
import numpy as np
import os,glob,cv2
import sys,argparse



dir_path = os.path.dirname(os.path.realpath(__file__))
image_path=sys.argv[1] 
filename = dir_path +'/' +image_path
image_size=128
num_channels=3
images = []

image = cv2.imread(filename)

image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
images.append(image)
images = np.array(images, dtype=np.uint8)
images = images.astype('float32')
images = np.multiply(images, 1.0/255.0) 

x_batch = images.reshape(1, image_size,image_size,num_channels)


sess = tf.Session()

saver = tf.train.import_meta_graph('model.meta')

saver.restore(sess, tf.train.latest_checkpoint('./'))


graph = tf.get_default_graph()


y_pred = graph.get_tensor_by_name("y_pred:0")


x= graph.get_tensor_by_name("x:0") 
y_true = graph.get_tensor_by_name("y_true:0") 
y_test_images = np.zeros((1, len(os.listdir('training_data')))) 



feed_dict_testing = {x: x_batch, y_true: y_test_images}
result=sess.run(y_pred, feed_dict=feed_dict_testing)

print(result)
