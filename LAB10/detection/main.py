import argparse
from .data_loader import prepare_dataset
from .train import EXPERIMENTS, train_one
from .evaluate import evaluate_one, save_table

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--quick',action='store_true',help='Run one epoch with model A')
    parser.add_argument('--models',nargs='+',choices=list(EXPERIMENTS),default=list(EXPERIMENTS))
    args=parser.parse_args()
    data_yaml=prepare_dataset()
    rows=[]
    for name in (['A'] if args.quick else args.models):
        print(f'--- Training model {name} ---')
        weights=train_one(data_yaml,name,quick=args.quick)
        print(f'--- Evaluating model {name} ---')
        rows.append(evaluate_one(weights,data_yaml))
    save_table(rows)
    print('Done! Graphs are inside outputs/.')

if __name__=='__main__':
    main()
