import argparse
import re
from ultralytics import YOLO
import torch

def main(args):


    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'
    model = YOLO(args.model).to(device)  # load the specified model
    
    # Extract image size from model name if it exists
    match = re.search(r'imgsz(\d+)', args.model)
    if match:
        imgsz = int(match.group(1))
    else:
        imgsz = args.imgsz
        
    # Export the model using the provided arguments
    model.export(format=args.format,
                 imgsz=[imgsz,imgsz],
                 half=args.half,
                 int8=args.int8,
                 dynamic=args.dynamic,
                 simplify=args.simplify)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Export a PyTorch YOLO model to other formats.")
    
    # Define the arguments
    parser.add_argument("--model", type=str, required=True, help="Path to the .pt format model.")
    parser.add_argument("--gpu", action="store_true", default=True, help="Use GPU if available.")
    parser.add_argument("--format", type=str, default="engine", choices=["onnx", "engine"], help="Export format: 'onnx' or 'engine'.")
    parser.add_argument("--imgsz", type=int, default=320, help="Image size.")
    parser.add_argument("--half", action="store_true", default=False, help="FP16 quantization.")
    parser.add_argument("--int8", action="store_true", default=False, help="INT8 quantization.")
    parser.add_argument("--dynamic", action="store_true", default=False, help="ONNX/TensorRT: dynamic axes.")
    parser.add_argument("--simplify", action="store_true", default=False, help="ONNX/TensorRT: simplify model.")
    
    args = parser.parse_args()

    
    main(args)
