import argparse
from dataset import EEGImageNetDataset
from utilities import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_dir", required=True, help="directory name of EEG-ImageNet dataset path")
    parser.add_argument("-g", "--granularity", required=True, help="choose from coarse, fine0-fine4 and all")
    parser.add_argument("-s", "--subject", default=0, type=int, help="subject from 0 to 15")
    parser.add_argument("-o", "--output_dir", required=True, help="directory to save results")
    args = parser.parse_args()
    print(args)

    dataset = EEGImageNetDataset(args.dataset_dir, args.subject, args.granularity)
    with open(os.path.join(args.output_dir, f"s{args.subject}_path.txt"), "w") as f:
        dataset.use_image_label = True
        for data in dataset:
            f.write(f"{data[1]}\n")
    with open(os.path.join(args.output_dir, f"s{args.subject}_label.txt"), "w") as f:
        dataset.use_image_label = False
        for idx, data in enumerate(dataset):
            if idx % 50 == 0:
                label_wnid = dataset.labels[data[1]]
                f.write(f"{idx + 1}-{idx + 50}: {wnid2category(label_wnid, 'ch')}\n")
