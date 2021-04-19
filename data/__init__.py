from importlib import import_module
from torch.utils.data import DataLoader

class Data:
    def __init__(self, args):
        self.args = args
        self.data_train = args.data_train
        self.data_test = args.data_test
        self.args.task = args.task

        list_benchmarks_image = ['BLUR_IMAGE']
        benchmark_image = self.data_test in list_benchmarks_image

        # load training dataset
        if not self.args.test_only:
            if self.args.task == "Deblurring":
                m_train = import_module('data.' + self.data_train.lower())
                trainset = getattr(m_train, self.data_train)(self.args)
                self.loader_train = DataLoader(
                    trainset,
                    batch_size=self.args.batch_size,
                    shuffle=False,
                    pin_memory=not self.args.cpu,
                    num_workers=self.args.n_threads,
                )
        else:
            self.loader_train = None

        if benchmark_image:
            if self.args.task == "Deblurring":
                m_test = import_module('data.benchmark')
                testset = getattr(m_test, 'Benchmark')(self.args, name=args.data_test, train=False)

        # load testing dataset
        self.loader_test = DataLoader(
            testset,
            batch_size=1,
            shuffle=False,
            pin_memory=not self.args.cpu,
            num_workers=self.args.n_threads,
        )

