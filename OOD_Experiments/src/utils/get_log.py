import logging  
import os.path
import time
import os


class log_recorder:

    def __init__(self, dataset_name, classifier_name, method_name, proportion, mission, batch_size, epoch,  lr):
        
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)  
        rq = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
        log_name = '../Logs/' + rq[:8] + '/' + rq[8:] + '.log'
        path = '../Logs/' + rq[:8]
        if os.path.exists(path):
            pass
        else:
            os.mkdir(path)
        logfile = log_name
        self.fh = logging.FileHandler(logfile, mode='w')
        self.fh.setLevel(logging.DEBUG)

        formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
        self.fh.setFormatter(formatter)

        self.logger.addHandler(self.fh)

        self.dataset_name = dataset_name
        self.classifier = classifier_name
        self.method_name = method_name
        self.proportion = proportion
        self.mission = mission
        self.batch_size = batch_size
        self.epoch = epoch
        self.lr = lr

        self.logger.info('info default content')
        self.logger.info('dataset_name = {}'.format(self.dataset_name))
        self.logger.info('classifier_name = {}'.format(self.classifier))
        self.logger.info('method_name = {}'.format(self.method_name))
        self.logger.info('proportion = {}'.format(self.proportion))
        self.logger.info('mission = {}'.format(self.mission))
        self.logger.info('batch_size = {}'.format(self.batch_size))
        self.logger.info('EPOCH = {}'.format(self.epoch))
        self.logger.info('learning_rate = {}'.format(self.lr))

        self.logger.info('Type default num1 num2')

    def log_train_loss(self, loss):
        self.logger.info('train_loss = {} {}'.format(loss, 2.71828))

    def log_train_acc(self, acc):
        self.logger.info('train_acc = {} {}'.format(acc, 2.71828))

    def log_eval_loss(self, loss):
        self.logger.info('eval_loss = {} {}'.format(loss, 2.71828))

    def log_eval_acc(self, acc):
        self.logger.info('eval_acc = {} {}'.format(acc, 2.71828))

    def log_test_loss(self, loss):
        self.logger.info('test_loss = {} {}'.format(loss, 2.71828))

    def log_test_acc(self, acc):
        self.logger.info('test_acc = {} {}'.format(acc, 2.71828))

    def log_test_label(self, truth, predict):
        self.logger.info('truth/predict = {} {}'.format(truth, predict))

    def log_close(self):
        self.logger.removeHandler(self.fh)
