import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from GoogLeNet_model import GoogLeNet
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from utils import plot_accuracy, plot_f1_score, plot_matrix
from sklearn.metrics import confusion_matrix

# 自定义数据集类
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.categories = sorted(os.listdir(data_dir))
        self.data = []
        self.transform = transform
        for category in self.categories:
            category_dir = os.path.join(data_dir, category)
            category_data = sorted(os.listdir(category_dir))
            self.data.extend([(os.path.join(category_dir, file), self.categories.index(category)) for file in category_data])

    def __getitem__(self, index):
        file_path, label = self.data[index]
        data = np.load(file_path)
        image = Image.fromarray(data.astype(np.uint8))
        image = transform(image)
        return image, label

    def __len__(self):
        return len(self.data)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))

transform = transforms.Compose([ transforms.Grayscale(num_output_channels = 1),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5), (0.5))])

train_dataset = CustomDataset("features/train_npy", transform=transform)
train_num = len(train_dataset)
dev_list = train_dataset.categories

batch_size = 32
train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=0)

validate_dataset = CustomDataset("features/val_npy", transform=transform)
val_num = len(validate_dataset)
validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=0)

print("using {} images for training, {} images for validation.".format(train_num, val_num))

net = GoogLeNet(num_classes=27, aux_logits=True, init_weights=True)
net.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)

epochs = 100
save_path = './GoogLeNet_parameters.pth'
best_f1 = 0.0
train_accurate_list = []
val_accurate_list = []
f1_list = []
recall_list = []
val_last = torch.tensor([])
pre_last = torch.tensor([])

for epoch in range(epochs):
    # train
    net.train()
    running_loss = 0.0
    train_acc = 0.0
    for step, data in enumerate(train_loader, start=0):
        images, labels = data
        images = images.reshape(images.shape[0], 1, 1503)
        optimizer.zero_grad()
        logits, aux_logits2, aux_logits1 = net(images.to(device))
        predict_y = torch.max(logits, dim=1)[1]
        train_acc += torch.eq(predict_y, labels.to(device)).sum().item()
        loss0 = loss_function(logits, labels.to(device))
        loss1 = loss_function(aux_logits1, labels.to(device))
        loss2 = loss_function(aux_logits2, labels.to(device))
        loss = loss0 + loss1 * 0.3 + loss2 * 0.3
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        # print train process
        rate = (step + 1) / len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss:{:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
    print()
    train_accurate = train_acc / train_num
    train_accurate_list.append(train_accurate)
    # validate
    net.eval()
    acc = 0.0  # accumulate accurate number / epoch
    val = torch.tensor([])
    pre = torch.tensor([])
    with torch.no_grad():
        for val_data in validate_loader:
            val_images, val_labels = val_data
            val_images = val_images.reshape(val_images.shape[0], 1, 1503)
            outputs = net(val_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            pre = torch.cat([pre.to(device), predict_y.to(device)])
            val = torch.cat([val.to(device), val_labels.to(device)])
            acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
    val_accurate = acc / val_num
    val_accurate_list.append(val_accurate)
    f1 = f1_score(val.cpu(), pre.cpu(), average='macro')
    recall = recall_score(val.cpu(), pre.cpu(), average='macro')
    pre_last = pre
    val_last = val
    f1_list.append(f1)
    recall_list.append(recall)
    if f1 > best_f1:
        best_f1 = f1
        torch.save(net.state_dict(), save_path)
    print('[epoch %d] train_loss: %.3f train_accuracy: %.3f val_accuracy: %.3f  recall: %.3f  f1: %.3f' %
              (epoch + 1, running_loss / step, train_accurate, val_accurate, recall, f1))
    with open("GoogLeNet_result_npy.txt", 'a') as file:
        file.write("[epoch " + str(epoch + 1) + "]" + "  " + "train_accuracy:" + str(train_accurate) + "  " + "val_accuracy:" + str(val_accurate) + "  " + "recall:" + str(recall) + "  " + "f1:" + str(f1) + '\n')
print('Finished Training')
torch.save(pre_last, 'pre_val_label/pre_last_GoogLeNet.pt')
torch.save(val_last, 'pre_val_label/val_last_GoogLeNet.pt')
# 迭代次数列表作为横坐标
iterations = range(1, len(train_accurate_list) + 1)
with open("GoogLeNet_npy_plt_data.txt", 'a') as file:
    file.write("iterations:" + str(iterations) +
               "train_accurate_list:" + str(train_accurate_list) +
               "val_accurate_list:" + str(val_accurate_list) +
               "f1_list:" + str(f1_list) +
               "recall_list:" + str(recall_list) +
               "dev_list:" + str(dev_list) + '\n')
plot_accuracy(iterations, train_accurate_list, val_accurate_list, save_path='GoogLeNet_accuracy_plot_npy.png')
plot_f1_score(iterations, f1_list, save_path='GoogLeNet_f1_score_plot_npy.png')
conf_matrix = confusion_matrix(val_last.cpu(), pre_last.cpu())
plot_matrix(conf_matrix, dev_list, "GoogLeNet_confusion_matrix_npy.png")

