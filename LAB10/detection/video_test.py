import argparse
from ultralytics import YOLO
from .data_loader import ROOT

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',required=True,help='Path to video file, e.g. demo.mp4')
    p.add_argument('--weights',default=str(ROOT/'outputs'/'model_A'/'weights'/'best.pt'))
    args=p.parse_args()
    YOLO(args.weights).predict(source=args.source,save=True,project=str(ROOT/'outputs'),name='video_predictions',exist_ok=True)

if __name__=='__main__': main()
