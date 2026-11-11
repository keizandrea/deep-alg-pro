import argparse
import Train
import Test


def main():
    argparser = argparse.ArgumentParser(
        description="DeepAlgPro Network for predicting allergens.")
    argparser.add_argument('-i', '--inputs', default='./', type=str)
    argparser.add_argument('--epochs', default=100, type=int,
                           metavar='N',
                           help='number of total epochs to run')
    argparser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                           metavar='LR', help='learning rate', dest='lr')
    argparser.add_argument('-b', '--batch-size', default=72, type=int,
                           metavar='N')
    argparser.add_argument('--mode', default='train', type=str,
                           choices=['train', 'test'])
    argparser.add_argument('--model', default='convATTnet', type=str,
                           choices=['convATTnet', 'convNet', 'ATTnet'])
    argparser.add_argument('--weights', default='./model.pt', type=str,
                           help='checkpoint to load in test mode')
    # adding an embedding argument
    argparser.add_argument('--in-channels', default=21, type=int,
                           help='embedding dimension (21 for one-hot)')
    argparser.add_argument('--embeddings', default=None, type=str,
                           help='directory of precomputed embeddings; omit for one-hot')
    argparser.add_argument('--tag', default=None, type=str,
                           help='checkpoint/log name prefix; defaults to --model')
    argparser.add_argument('--folds', default=10, type=int,
                           help='stop after this many CV folds')
    argparser.add_argument('--patience', default=0, type=int,
                           help='stop a fold after N epochs with no F1 gain; 0 = never')



    args = argparser.parse_args()
    if args.mode == 'train':
        Train.train(args)
    if args.mode == 'test':
        Test.test(args)


if __name__ == '__main__':
    main()
