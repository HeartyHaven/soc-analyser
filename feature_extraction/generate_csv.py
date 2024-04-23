import os
import csv
import random
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--feature_path", default = './training_set/sample/feature', type=str, help = 'dir name')
    parser.add_argument("--label_path", default = './training_set/sample/label', type=str, help = 'path to the decompressed dataset')
    parser.add_argument("--out_train_csv",  default = './train.csv', type=str, help = 'path to save training set')
    parser.add_argument('--out_test_csv', default='./test.csv', help='number of process for multi process')
    parser.add_argument('--split_ratio', default=0.85, help='number of process for multi process')
    
    args = parser.parse_args()                                       
    return args

if __name__ == '__main__':
    args=parse_args()
    random.seed(230907)
    features = os.listdir(args.feature_path)
    labels = os.listdir(args.label_path)
    
    features = [v for v in features]
    labels = [ v for v in labels]

    assert len(features) == len(labels)

    with open(args.out_train_csv, 'w') as f_train:
        with open(args.out_test_csv, 'w') as f_test:
            f_train_csv = csv.writer(f_train, delimiter=',')
            f_test_csv = csv.writer(f_test, delimiter=',')

            for i, features_name in enumerate(features):
                features_path = 'training_set/sample/feature/{}'.format(features_name)
                labels_path = 'training_set/sample/label/{}'.format(features_name)
                instance_count_path = 'out/features/instance_count/{}'.format(features_name)
                instance_IR_drop_path = 'out/features/instance_IR_drop/{}'.format(features_name)
                instance_name_path = 'out/features/instance_name/{}z'.format(features_name[:-1])
                if len(features) == 2:
                    if i == 0: 
                        f_train_csv.writerow([features_path, labels_path])
                        f_test_csv.writerow([features_path, labels_path, instance_count_path, instance_IR_drop_path, instance_name_path])
                        print('train: {}'.format(i))
                    else:
                        f_test_csv.writerow([features_path, labels_path, instance_count_path, instance_IR_drop_path, instance_name_path])
                        print('test: {}'.format(i))
                else:
                    if random.random() <= args.split_ratio:
                        f_train_csv.writerow([features_path, labels_path])
                        print('train: {}'.format(i))
                    else:
                        f_test_csv.writerow([features_path, labels_path, instance_count_path, instance_IR_drop_path, instance_name_path])
                        print('test: {}'.format(i))
