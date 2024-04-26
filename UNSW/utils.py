import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np


def plot_accuracy(iterations, train_accuracy, val_accuracy, save_path=None):
    # 画图
    plt.plot(iterations, train_accuracy, '-', label='Training Set')  # 训练集准确率曲线
    plt.plot(iterations, val_accuracy, '-', label='Validation Set')  # 验证集准确率曲线
    # 添加标签和标题
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Iterations')
    plt.legend()  # 显示图例
    # 保存图表到文件
    if save_path:
        plt.savefig(save_path)  # 文件名可以根据你的需要自定义
    # 显示图表
    plt.show()



def plot_f1_score(iterations, f1_scores, save_path=None):
    # 画图
    plt.plot(iterations, f1_scores, '-')
    # 添加标签和标题
    plt.xlabel('Iterations')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs. Iterations')
    # 保存图表到文件
    if save_path:
        plt.savefig(save_path)
    # 显示图表
    plt.show()

def plot_matrix(conf_matrix, dev_list, save_path):
    plt.figure(figsize=(20, 16))
    plt.imshow(conf_matrix, cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, conf_matrix[i, j],
                     ha="center", va="center",
                     color="white" if conf_matrix[i, j] > thresh else "black", fontsize=6)

    tick_marks = np.arange(len(dev_list))
    plt.xticks(tick_marks, dev_list)
    plt.yticks(tick_marks, dev_list)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(save_path)
    plt.show()