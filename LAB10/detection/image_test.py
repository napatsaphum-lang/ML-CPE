import argparse
from ultralytics import YOLO
from .data_loader import ROOT

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--weights',default=str(ROOT/'outputs'/'model_A'/'weights'/'best.pt'))
    p.add_argument('--source',default=str(ROOT/'dataset'/'test'/'images'))
    args=p.parse_args()
    YOLO(args.weights).predict(source=args.source,save=True,conf=.25,project=str(ROOT/'outputs'),name='image_predictions',exist_ok=True)

if __name__=='__main__': main()
