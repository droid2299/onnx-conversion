import get_params
import os
import yaml
import torch
import numpy as np
import subprocess
import tf2onnx
import torch.onnx 
#from backbone import EfficientDetBackbone
import yolo_darknet
import model_architecture



def onnx_export(trained_model , dummy_input , output_name , opset):
    print("OPSET = "+str(opset))
    if(opset < 7):
        print("Unable to convert model as all opset_version are Unsupported")
        return

    #try:
    torch.onnx.export(trained_model, dummy_input, output_name , export_params=True, opset_version=opset)
    '''
    except Exception as e:
        print("Exception: "+str(e))
        opset -= 1
        print("TRYING FOR OPSET_VERSION: "+str(opset))
        onnx_export(trained_model , dummy_input , output_name , opsmap_location=et)
    '''



    

def main():
    architecture = FLAGS.model_architecture
    input_path = FLAGS.input_path
    opset = FLAGS.opset_version
    print(architecture , input_path)

    if(architecture == 'tensorflow'):
        try:
            print("In try")
            command = 'python -m tf2onnx.convert --saved-model '+str(input_path)+' --output '+str(input_path)+'_tf2onnx.onnx --opset '+str(opset)
            subprocess.call(command, shell=True)
            #model_proto, _ = tf2onnx.convert(model, input_signature=spec, opset=13, output_path=output_path)

        except Exception as e:
            print("Exception caught")
            print("Exception: "+str(e))



    elif(architecture == 'pytorch'):
        try:
            
            trained_model = model_architecture.model()
            trained_model.load_state_dict(torch.load(input_path , map_location='cpu'))
            trained_model.eval()

            batch_size = FLAGS.batch_size
            dummy_ip_flag = FLAGS.dummy_input
            dummy_ip_list = dummy_ip_flag.split(",")
            print(dummy_ip_list)
            if(len(dummy_ip_list) > 3):
                print("Input in Wrong Format")

        except Exception as e:
            print("Exception caught")
            print("Exception: "+str(e))

        output_name = input_path.split(".")
        

        try:
            dummy_input = torch.randn((batch_size, int(dummy_ip_list[0]), int(dummy_ip_list[1]), int(dummy_ip_list[2])))
            onnx_export(trained_model , dummy_input , output_name[0]+"_"+architecture+".onnx" , opset)
            #torch.onnx.export(trained_model, dummy_input, "yolov4_pytorch.onnx" , export_params=True, opset_version=opset, dynamic_axes={'input' : {0 : 'batch_size'},   'output' : {0 : 'batch_size'}})
        except Exception as e:
            print("Exception caught")
            print("Exception: "+str(e))

        



    elif(architecture == 'darknet'):
        cfg = FLAGS.cfg_path
        num_classes = FLAGS.num_classes
        output_name = input_path.split(".")

        if(cfg == None or num_classes == None):
            print("Please provide CFG file and/or number of classes")
            
        else:
            try:
                trained_model = yolo_darknet.Darknet(cfg)
                trained_model.load_weights(input_path)
                dummy_input = torch.randn((1, 3, trained_model.height, trained_model.width), requires_grad=True)
                onnx_export(trained_model , dummy_input , output_name[0]+"_"+architecture+".onnx" , opset)
            except Exception as e:
                print("Exception caught")
                print("Exception: "+str(e))

    else:
        print("Unsupported architecture")   


if __name__ == '__main__':

    FLAGS = get_params.parse_arguments()
    main()
