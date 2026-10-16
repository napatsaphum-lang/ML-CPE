import argparse
from ultralytics import YOLO
from .data_loader import ROOT

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--weights',default=str(ROOT/'outputs'/'model_A'/'weights'/'best.pt'))
    p.add_argument('--camera',type=int,default=0)
    args=p.parse_args()
    model=YOLO(args.weights)
    model.predict(source=args.camera,show=True,conf=.25)
    print('Close camera window or press Q to stop')

if __name__=='__main__': main()
