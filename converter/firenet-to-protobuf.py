
import glob,os
import sys
sys.path.append('..')



from firenet import construct_firenet
from converter import convert_to_pb



if __name__ == '__main__':

    

    model = construct_firenet (224, 224, False)
    print("[INFO] Constructed FireNet ...")

    path = "../models/FireNet/firenet"; # path to tflearn checkpoint including filestem
    input_layer_name = 'InputData/X'                  # input layer of network
    output_layer_name= 'FullyConnected_2/Softmax'     # output layer of network
    pbfilename = "firenet.pb"        # output pb format filename

    convert_to_pb(model, path, input_layer_name,  output_layer_name, pbfilename)

