import logging  # 引入logging模块
import os.path
import time
import os


class log_recorder:

    def __init__(self, dataset_name, classifier_name, method_name, batch_size, epoch,  lr, select_type,select_ratio):
        # 第一步，创建一个logger
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)  # Log等级总开关
        # 第二步，创建一个handler，用于写入日志文件
        rq = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
        log_name = './Logs/{}/{}/{}/{}_{}_{}_{}.log'.format(rq[:8], dataset_name, classifier_name, rq[8:], method_name,
                                                            select_type, select_ratio)
        path = './Logs/{}/{}/{}/'.format(rq[:8], dataset_name, classifier_name)
        if os.path.exists(path):
            pass
        else:
            os.makedirs(path)
        logfile = log_name
        self.fh = logging.FileHandler(logfile, mode='w')
        self.fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
        # 第三步，定义handler的输出格式
        formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
        self.fh.setFormatter(formatter)
        # 第四步，将logger添加到handler里面
        self.logger.addHandler(self.fh)

        self.dataset_name = dataset_name
        self.classifier = classifier_name
        self.method_name = method_name
        # self.proportion = proportion
        # self.mission = mission
        self.batch_size = batch_size
        self.epoch = epoch
        self.lr = lr
        # 日志
        self.logger.info('info default content _')
        self.logger.info('dataset_name = {} _'.format(self.dataset_name))
        self.logger.info('classifier_name = {} _'.format(self.classifier))
        self.logger.info('method_name = {} _'.format(self.method_name))
        # self.logger.info('proportion = {}'.format(self.proportion))
        # self.logger.info('mission = {}'.format(self.mission))
        self.logger.info('batch_size = {} _'.format(self.batch_size))
        self.logger.info('EPOCH = {} _'.format(self.epoch))
        self.logger.info('learning_rate = {} _'.format(self.lr))

        # 2020-09-02 16:35:48,795 - get_log.py[line:25] - INFO: train_loss = 0.7142857142857143
        # num1 和 num2 两个参数表示在记录日志过程中需要记录的信息
        # 对于loss和acc两种待记录信息，num1表示其信息，num2用自然对数占位以保证格式规整
        # 对于label待记录信息， num1表示真值，num2表示预测值
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
