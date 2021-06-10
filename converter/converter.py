
import glob,os


import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.framework.graph_util import convert_variables_to_constants
from tensorflow.tools.graph_transforms import TransformGraph
from tensorflow.python.tools import optimize_for_inference_lib


import tflearn
from tflearn.layers.core import *
from tflearn.layers.conv import *
from tflearn.layers.normalization import *
from tflearn.layers.estimator import regression

def convert_to_pb(model, path, input_layer_name,  output_layer_name, pbfilename, verbose=False):

  model.load(path,weights_only=True)
  print("[INFO] Loaded CNN network weights from " + path + " ...")

  print("[INFO] Re-export model ...")
  del tf.get_collection_ref(tf.GraphKeys.TRAIN_OPS)[:]
  model.save("model-tmp.tfl")

  
  print("[INFO] Re-import model ...")

  input_checkpoint = "model-tmp.tfl"
  saver = tf.train.import_meta_graph(input_checkpoint + '.meta', True)
  sess = tf.Session();
  saver.restore(sess, input_checkpoint)

  # print out all layers to find name of output

  if (verbose):
      op = sess.graph.get_operations()
      [print(m.values()) for m in op][1]

  print("[INFO] Freeze model to " +  pbfilename + " ...")

  

  minimal_graph = convert_variables_to_constants(sess, sess.graph_def, [output_layer_name])

  graph_def = optimize_for_inference_lib.optimize_for_inference(minimal_graph, [input_layer_name], [output_layer_name], tf.float32.as_datatype_enum)
  graph_def = TransformGraph(graph_def, [input_layer_name], [output_layer_name], ["sort_by_execution_order"])
  with tf.gfile.GFile(pbfilename, 'wb') as f:
      f.write(graph_def.SerializeToString())


  if (verbose):
      writer = tf.summary.FileWriter('logs', graph_def)
      writer.close()

  

  for f in glob.glob("model-tmp.tfl*"):
      os.remove(f)

  os.remove('checkpoint')

