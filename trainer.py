from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import os, logging, warnings
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import numpy as np
import pandas as pd
import torch.nn.functional as F

logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "true"
warnings.filterwarnings('ignore')


def metric(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    macro_precision = precision_score(y_true, y_pred, average='macro')  # 每一类预测对的占比取平均
    macro_recall = recall_score(y_true, y_pred, average='macro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    return {
        "accuracy": accuracy,
        "precision": macro_precision,
        "recall": macro_recall,
        "f1": macro_f1
    }


def compute_kl_loss(p, q, pad_mask=None):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()

    loss = (p_loss + q_loss) / 2
    return loss


def exp_decay(current, start=1.0, end=0.0, exp=3, rampup_length=100):  # 从1指数衰减到0
    current = np.clip(current / rampup_length, 0.0, 1.0)  # 限制到0-1之间
    return end + (start - end) * np.exp(-current * exp)


class SelfMixTrainer:
    def __init__(self, model, train_data=None, eval_data=None, model_args=None, training_args=None):
        self.model = model.cuda()
        self.train_data = train_data
        self.eval_data = eval_data
        self.model_args = model_args
        self.training_args = training_args

        if self.training_args is not None:
            self.optimizer = Adam(self.model.parameters(), lr=training_args.lr)

    def warmup_original(self):
        # used for draw figures
        logger.info("***** Warmup stage *****")

        train_loader = self.train_data.run("all")
        eval_loader = self.eval_data.run("all")

        loss_func = nn.CrossEntropyLoss()
        records = {'clean_right': [], 'clean_wrong': [], 'noisy_right': [], 'noisy_noise': [], 'noisy_other': [], 'train_acc':[], 'test_acc': []}
        iter_len = len(train_loader)
        eval_steps = []
        for tt in range(self.training_args.split_num):
            eval_steps.append(int(tt * iter_len / self.training_args.split_num))
        train_iter = iter(train_loader)
        now_samples = 0

        train_loss, train_acc = 0., 0.

        for i in range(0, len(train_loader)*self.training_args.warmup_epochs):

            if (i % iter_len) in eval_steps:
                eval_acc, _, _ = self.evaluate(eval_loader)
                clean_right, clean_wrong, noisy_right, noisy_noise, noisy_other, train_acc = self.eval_train(
                    train_loader)
                records['clean_right'].append(clean_right)
                records['clean_wrong'].append(clean_wrong)
                records['noisy_right'].append(noisy_right)
                records['noisy_noise'].append(noisy_noise)
                records['noisy_other'].append(noisy_other)
                records['train_acc'].append(train_acc)
                records['test_acc'].append(eval_acc)

            self.model.train()
            try:
                data = train_iter.next()
            except:
                train_iter = iter(train_loader)
                data = train_iter.next()

            input_ids, att_mask, labels, _, _ = [Variable(elem.cuda()) for elem in data]
            logits = self.model(input_ids, att_mask)
            loss = loss_func(logits, labels)
            train_loss += loss.item()

            pred = logits.argmax(dim=-1).cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
            train_acc += (pred == labels).sum()

            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            now_samples += input_ids.size(0)

        eval_acc, _, _ = self.evaluate(eval_loader)
        clean_right, clean_wrong, noisy_right, noisy_noise, noisy_other, train_acc = self.eval_train(train_loader)
        records['clean_right'].append(clean_right)
        records['clean_wrong'].append(clean_wrong)
        records['noisy_right'].append(noisy_right)
        records['noisy_noise'].append(noisy_noise)
        records['noisy_other'].append(noisy_other)
        records['train_acc'].append(train_acc)
        records['test_acc'].append(eval_acc)
        records = pd.DataFrame(records)
        # 提取文件夹路径
        folder_path = os.path.dirname(self.training_args.record_path)
        # 判断文件夹路径是否存在，不存在则创建
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"文件夹 '{folder_path}' 已创建.")
        else:
            print(f"文件夹 '{folder_path}' 已存在.")
        records.to_csv(self.training_args.record_path, index=False)

    def dynamic_train(self):
        test_best_l = []
        test_last_l = []

        test_best = 0.0

        train_loader = self.train_data.run("all")
        eval_loader = self.eval_data.run("all")  # 之前没有

        train_iters = self.training_args.train_epochs * len(train_loader)

        train_iter = iter(train_loader)

        # used for draw figures
        # records = {'clean_right': [], 'clean_wrong': [], 'noisy_right': [], 'noisy_noise': [], 'noisy_other': [],
        #            'train_acc': [], 'test_acc': []}
        # iter_len = len(train_loader)
        # eval_steps = []
        # for tt in range(self.training_args.split_num):
        #     eval_steps.append(int(tt * iter_len / self.training_args.split_num))

        logger.info("Training begin...")

        for i in range(0, train_iters):

            # used for draw figures
            # if (i % iter_len) in eval_steps:
            #     eval_acc, _, _ = self.evaluate(eval_loader)
            #     clean_right, clean_wrong, noisy_right, noisy_noise, noisy_other, train_acc = self.eval_train(
            #         train_loader)
            #     records['clean_right'].append(clean_right)
            #     records['clean_wrong'].append(clean_wrong)
            #     records['noisy_right'].append(noisy_right)
            #     records['noisy_noise'].append(noisy_noise)
            #     records['noisy_other'].append(noisy_other)
            #     records['train_acc'].append(train_acc)
            #     records['test_acc'].append(eval_acc)

            self.model.train()

            try:
                data = train_iter.next()
            except:
                train_iter = iter(train_loader)
                data = train_iter.next()

            input_ids, att_mask, labels_id, _, index = [Variable(elem.cuda()) for elem in data]
            labels = F.one_hot(labels_id, num_classes=self.model_args.num_classes)

            self.model.eval()
            with torch.no_grad():
                # Predict labels for all data.
                out_x = self.model(input_ids, att_mask)
                p_x = torch.softmax(out_x, dim=1)
                # p_threshold用来控制下限 exp控制函数衰减幅度
                w_x = exp_decay(i, start=1, end=self.model_args.p_threshold, exp=self.model_args.exp,
                                rampup_length=self.training_args.train_epochs * len(
                                    train_loader))  # 在整个训练期间衰减逐渐衰减
                p_x = (1 - w_x) * p_x + w_x * labels

                pt_x = p_x ** (1 / self.model_args.temp)
                targets_x = pt_x / pt_x.sum(dim=1, keepdim=True)
                targets_x = targets_x.detach()

            self.model.train()

            # norm train
            sents_x = self.model.get_sentence_embedding(input_ids, att_mask)
            sents_x2 = self.model.get_sentence_embedding(input_ids, att_mask)
            logits_x = self.model.classify(sents_x)
            logits_x2 = self.model.classify(sents_x2)
            loss_cl = -torch.mean(torch.sum(F.log_softmax(logits_x, dim=-1) * targets_x, dim=-1))
            kl_loss = compute_kl_loss(logits_x, logits_x2)

            # 避免崩溃
            prior = torch.ones(self.model_args.num_classes) / self.model_args.num_classes
            prior = prior.cuda()
            pred_mean = torch.softmax(torch.cat([logits_x, logits_x2], dim=0), dim=1).mean(0)
            penalty = torch.sum(prior * torch.log(prior / pred_mean))

            loss = loss_cl + kl_loss * self.training_args.lambda_r + penalty
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            if (i + 1) % len(train_loader) == 0:
                logger.info("Dynamic training Stage in %d epoch", int((i + 1) / len(train_loader)))
                self.evaluate(train_loader)
                test_last, _, _ = self.evaluate(eval_loader)
                test_best = max(test_best, test_last)
                test_best_l.append(test_best)
                test_last_l.append(test_last)

        # used for draw figures
        # eval_acc, _, _ = self.evaluate(eval_loader)
        # clean_right, clean_wrong, noisy_right, noisy_noise, noisy_other, train_acc = self.eval_train(train_loader)
        # records['clean_right'].append(clean_right)
        # records['clean_wrong'].append(clean_wrong)
        # records['noisy_right'].append(noisy_right)
        # records['noisy_noise'].append(noisy_noise)
        # records['noisy_other'].append(noisy_other)
        # records['train_acc'].append(train_acc)
        # records['test_acc'].append(eval_acc)
        # records = pd.DataFrame(records)
        # # 提取文件夹路径
        # folder_path = os.path.dirname(self.training_args.record_path)
        # # 判断文件夹路径是否存在，不存在则创建
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)
        #     print(f"文件夹 '{folder_path}' 已创建.")
        # else:
        #     print(f"文件夹 '{folder_path}' 已存在.")
        # records.to_csv(self.training_args.record_path, index=False)

        return test_best_l, test_last_l

    def eval_train(self, train_loader):
        # 分干净噪声的
        self.model.eval()
        y_nois, y_true, y_pred = np.zeros(len(train_loader.dataset), dtype=int), np.zeros(len(train_loader.dataset), dtype=int), np.zeros(len(train_loader.dataset), dtype=int)
        with torch.no_grad():
            for i, data in enumerate(train_loader):
                input_ids, att_mask, labels, true_labels, index = [Variable(elem.cuda()) for elem in data]
                outputs = self.model(input_ids, att_mask)
                pred = torch.argmax(outputs, dim=-1).cpu().detach().numpy()
                index = index.long().cpu().detach().numpy()
                labels = labels.cpu().detach().numpy()
                true_labels = true_labels.cpu().detach().numpy()
                y_nois[index] = labels
                y_true[index] = true_labels
                y_pred[index] = pred
        clean_index = y_nois == y_true
        noisy_index = ~clean_index

        # 正确的里面分类正确的和错误的
        clean_right = (y_pred[clean_index] == y_nois[clean_index]).sum()/len(y_nois[clean_index])
        clean_wrong = (y_pred[clean_index] != y_nois[clean_index]).sum()/len(y_nois[clean_index])
        # 错误的里面分类对应正确标签的、对应噪声标签的和都不对应的
        noisy_right = (y_pred[noisy_index] == y_true[noisy_index]).sum()/len(y_true[noisy_index])
        noisy_noise = (y_pred[noisy_index] == y_nois[noisy_index]).sum()/len(y_nois[noisy_index])
        noisy_other = ((y_pred[noisy_index] != y_nois[noisy_index])*(y_pred[noisy_index] != y_true[noisy_index])).sum()/len(y_nois[noisy_index])

        eval_res = metric(y_nois, y_pred)  # 训练集上的准确率
        return clean_right, clean_wrong, noisy_right, noisy_noise, noisy_other, eval_res['accuracy']

    def evaluate(self, eval_loader=None):
        if eval_loader is None:
            eval_loader = self.eval_data.run("all")
        self.model.eval()
        y_true, y_pred = np.zeros(len(eval_loader.dataset), dtype=int), np.zeros(len(eval_loader.dataset), dtype=int)
        for j, data in enumerate(eval_loader):
            val_input_ids, val_att, val_labels, _, index = [Variable(elem.cuda()) for elem in data]
            with torch.no_grad():
                index = index.long().cpu().detach().numpy()
                pred = self.model(val_input_ids, val_att).argmax(dim=-1).cpu().detach().numpy()
                val_labels = val_labels.cpu().detach().numpy()
            y_true[index] = val_labels
            y_pred[index] = pred

        eval_res = metric(y_true, y_pred)
        logger.info("Eval Results: Accuracy: {:.2%}, Precision: {:.2%}, Recall: {:.2%}, F1: {:.2%}"
                    .format(eval_res['accuracy'], eval_res['precision'], eval_res['recall'], eval_res['f1']))
        return eval_res['accuracy'], eval_res['f1'], eval_res['recall']

    def save_model(self, comm=None):
        suffix = '.pt'
        if comm:
            suffix = '_' + str(comm) + suffix

        path = self.training_args.model_save_path + suffix
        dir = os.path.dirname(path)
        folder = os.path.exists(dir)

        # 判断是否存在文件夹如果不存在则创建为文件夹
        if not folder:
            os.makedirs(dir)  # makedirs 创建文件时如果路径不存在会创建这个路径
            # print('文件夹创建成功：', dir)
        self.model.save_model(path)

    def load_model(self, comm=None):
        print('模型重载中...')
        suffix = '.pt'
        if comm:
            suffix = '_' + str(comm) + suffix
        path = self.training_args.model_save_path + suffix
        model_state_dict = torch.load(path)
        self.model.load_state_dict(model_state_dict)
