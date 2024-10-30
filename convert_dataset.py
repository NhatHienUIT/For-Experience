import argparse, os, torch, cv2
import torchvision.transforms as transforms
import webdataset as wds
from torch.utils.data import Dataset
import numpy as np
import medmnist
from medmnist import PathMNIST

# base_pattern: base folder for storing the dataset
# dataset: dataset (pytorch format)
# m: average (later used for normalization)
# s: standard deviation (later used for normalization)
# n: number of classes
def create_webdataset(base_pattern, dataset, m, s, n, names):
    # Save mean, standard deviation, and other metadata
    torch.save(m, os.path.join(base_pattern, "m.pt"))
    torch.save(s, os.path.join(base_pattern, "s.pt"))
    torch.save(n, os.path.join(base_pattern, "n.pt"))
    torch.save(names, os.path.join(base_pattern, "names.pt"))

    with torch.no_grad():
        pattern = os.path.join(base_pattern, f"data-%06d.tar")
        sink = wds.ShardWriter(pattern, maxcount=10000)

        for i, (data, label) in enumerate(dataset):
            key = "%.6d" % i

            # Convert torch.Tensor to numpy.ndarray and transpose to H x W x C
            if isinstance(data, torch.Tensor):
                data = data.permute(1, 2, 0).numpy()  # Transpose from C x H x W to H x W x C

            sample = {
                "__key__": key,
                "ppm": data,
                "cls": label,
                "pyd": i
            }
            sink.write(sample)

            if i == 0:
                torch.save(np.shape(data), os.path.join(base_pattern, "size.pt"))

        sink.close()

    # Save the number of samples
    torch.save(i + 1, os.path.join(base_pattern, "num_samples.pt"))


    return

class FontDataset(Dataset):
    def __init__(self, folder='./fonts'):
        self.num_fonts = 1
        self.xs = np.zeros((62 * self.num_fonts, 1, 128, 128), dtype=np.uint8)
        self.ys = np.zeros(62 * self.num_fonts, dtype=np.int32)
        fonts_list = ['comicsansms', 'couriernew', 'verdana', 'arial', 'latinmodernmonolight', 'chilanka', 'freemono', 'impact', 'jamrul', 'uroob']
        idx = 0
        for n in range(62):
            for font in fonts_list[0:self.num_fonts]:
                filename = folder + '/%.3d_' % n + font + '.png'
                img = 255 - cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2GRAY)
                self.xs[idx] = img
                self.ys[idx] = n
                idx += 1

    def __len__(self):
        return 62 * self.num_fonts

    def __getitem__(self, idx):
        return (self.xs[idx, 0], self.ys[idx])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', type=str, choices=['mnist', 'fashionmnist', 'cifar10', 'fonts', 'pathmnist'], default='mnist', help='dataset')
    parser.add_argument('--output-folder', type=str, default=None, help=' output folder')
    args = parser.parse_args()

    if args.output_folder is None:
        print("Passing an output folder is mandatory. Returning.")
        return

    if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)
        train_folder = os.path.join(args.output_folder, 'train')
        validation_folder = os.path.join(args.output_folder, 'validation')
        test_folder = os.path.join(args.output_folder, 'test')
        os.mkdir(train_folder)
        os.mkdir(validation_folder)
        os.mkdir(test_folder)
    else:
        print("The output folder already exists. Please delete it. Returning.")
        return

    if args.dataset_name == 'mnist':
        data = datasets.MNIST("./data", train=True, download=True)
        (dtrain_data, dvalidation_data) = torch.utils.data.random_split(data, [int(0.95*len(data)), int(0.05*len(data))], generator=torch.Generator().manual_seed(42))
        dtest_data = datasets.MNIST("./data", train=False, download=True)
        m = torch.tensor([0.0])
        s = torch.tensor([1.0])
        n = torch.tensor([10])
        names = [str(i) for i in range(10)]
    elif args.dataset_name == 'fashionmnist':
        data = datasets.FashionMNIST("./data", train=True, download=True)
        (dtrain_data, dvalidation_data) = torch.utils.data.random_split(data, [int(0.95 * len(data)), int(0.05 * len(data))], generator=torch.Generator().manual_seed(42))
        dtest_data = datasets.FashionMNIST("./data", train=False, download=True)
        m = torch.tensor([0.5])
        s = torch.tensor([2.0])
        n = torch.tensor([10])
        names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    elif args.dataset_name == 'cifar10':
        data = datasets.CIFAR10("./data", train=True, download=True)
        (dtrain_data, dvalidation_data) = torch.utils.data.random_split(data, [int(0.95 * len(data)), int(0.05 * len(data))], generator=torch.Generator().manual_seed(42))
        dtest_data = datasets.CIFAR10("./data", train=False, download=True)
        m = torch.tensor([0.4914, 0.4822, 0.4465])
        s = torch.tensor([0.2023, 0.1994, 0.2010])
        n = torch.tensor([10])
        names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    elif args.dataset_name == 'fonts':
        dtrain_data = FontDataset("./fonts")
        dvalidation_data = FontDataset("./fonts")
        dtest_data = FontDataset("./fonts")
        m = torch.tensor([0.5])
        s = torch.tensor([1.0])
        n = torch.tensor([62])
        names = [chr(i) for i in range(65, 91)] + [chr(i) for i in range(97, 123)] + [chr(i) for i in range(48, 58)]
    elif args.dataset_name == 'pathmnist':
        dtrain_data = PathMNIST(split='train', download=True, transform=transforms.ToTensor())
        dvalidation_data = PathMNIST(split='val', download=True, transform=transforms.ToTensor())
        dtest_data = PathMNIST(split='test', download=True, transform=transforms.ToTensor())
        m = torch.tensor([0.485, 0.456, 0.406])  # Mean for RGB channels
        s = torch.tensor([0.229, 0.224, 0.225])  # Standard deviation for RGB channels
        n = torch.tensor([9])         
        names = [f'Class_{i}' for i in range(9)]
    else:
        print("Incorrect dataset name. Returning.")
        return

    create_webdataset(base_pattern=train_folder, dataset=dtrain_data, m=m, s=s, n=n, names=names)
    create_webdataset(base_pattern=validation_folder, dataset=dvalidation_data, m=m, s=s, n=n, names=names)
    create_webdataset(base_pattern=test_folder, dataset=dtest_data, m=m, s=s, n=n, names=names)

    print("Length of the training dataset: %d." % (len(dtrain_data)))
    print("Length of the validation dataset: %d." % (len(dvalidation_data)))
    print("Length of the testing dataset: %d." % (len(dtest_data)))

    return

if __name__ == "__main__":
    main()
