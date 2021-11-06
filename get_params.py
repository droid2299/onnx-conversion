import argparse
import sys

def parse_arguments():
    parser = argparse.ArgumentParser(description='Model conversion to ONNX format')

    parser.add_argument(
            '--model_architecture',
            type=str,
            required=True,
            default='tensorflow',
            help='''
                Model architecture. Tensorflow, Pytorch or Darknet''')

    parser.add_argument(
            '--input_path',
            type=str,
            required=True,
            default='efficientdet_d0',
            help='''
                Path to  weights file''')

    parser.add_argument(
            '--opset_version',
            type=int,
            #required=True,
            default=15,
            help='''
                opset_version''')

    parser.add_argument(
            '--batch_size',
            type=int,
            #required=True,
            default=1,
            help='''
                Batch size of the input''')
    parser.add_argument(
            '--dummy_input',
            type=str,
            #required=True,
            default='3,512,512',
            help='''
                Expected input size to the model (For eg. channel,width,height)''')

    parser.add_argument(
            '--cfg_path',
            type=str,
            #required=True,
            #default='yolo.cfg',
            help='''
                Path to cfg file if converting yolo models''')

    parser.add_argument(
            '--num_classes',
            type=int,
            #required=True,
            default=80,
            help='''
                Number of classes in the input dataset''')
            


    
    return parser.parse_args()

