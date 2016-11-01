
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imresize

class sillyToHuman:
  def __init__(self):
    synset_f = open('imagenet_lsvrc_2015_synsets.txt.txt', 'r')
    to_human_f = open('imagenet_metadata.txt', 'r')
    self.synset = self.get_synset(synset_f)
    self.to_human = self.get_to_human(to_human_f)

  def __call__(self, index):
    s = self.synset[index]
    return self.to_human[s]

  def get_to_human(self, file):
    a = {}
    for line in file:
      l = line.split('\t')
      a[l[0]] = l[1]
    return a

  def get_synset(self, file):
    a = []
    for line in file:
      a.append(line.strip())
    return a


human = sillyToHuman()

sess = tf.Session()

saver = tf.train.import_meta_graph('inception_cpu4.meta')
saver.restore(sess, "model.ckpt-157585")
graph = tf.get_default_graph()

print map(lambda x: x.name, filter(lambda x: x.name.find('predictions')>-1, graph.get_operations()))
print map(lambda x: x.name, graph.get_operations())
images = graph.get_tensor_by_name('batch_processing/Reshape:0')
pred = graph.get_operation_by_name('inception_v3/logits/predictions')
pred_t = pred._outputs[0]

img = imresize(plt.imread('cat.jpg'), [299, 299]).astype(np.float32)/255.0
a = np.zeros([32, 299, 299, 3], dtype=np.float32)
a[0] = img
p = sess.run(pred_t, {images: a})
print human(np.argmax(p[0]))

print("DONE")
