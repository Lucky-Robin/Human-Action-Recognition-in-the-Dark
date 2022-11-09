import os
from torch.utils.data import DataLoader
from torch import nn
import torch.optim
from torch.utils.tensorboard import SummaryWriter
from network import model
from dataset import data

# Define training device
device = torch.device('cuda')

# Reading mapping table
f = open('dataset/mapping_table.txt', 'r')
mapping_number = []  # ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
mapping_label = []  # ['Drink', 'Jump', 'Pick', 'Pour', 'Push', 'Run', 'Sit', 'Stand', 'Turn', 'Walk']
length = len(open('dataset/mapping_table.txt', 'r').readlines())
for i in range(length):
    line = f.readline()
    a = line.split()
    mapping_number.append(a[0])
    mapping_label.append(a[1])
# print(mapping_number)
# print(mapping_label)
f.close()

# Reading validate mapping table
f = open('dataset/raw/validate.txt', 'r')
validation_serial = []
validation_label = []
validation_filename = []
length = len(open('dataset/raw/validate.txt', 'r').readlines())
for i in range(length):
    line = f.readline()
    a = line.split()
    validation_serial.append(a[0])
    validation_label.append(a[1])
    validation_filename.append(a[2])
# print(validation_serial)
# print(validation_label)
# print(validation_filename)
f.close()

# Reading test mapping table
f = open('dataset/raw/test.txt', 'r')
test_serial = []
test_label = []
test_filename = []
length = len(open('dataset/raw/test.txt', 'r').readlines())
for i in range(length):
    line = f.readline()
    a = line.split()
    test_serial.append(a[0])
    test_label.append(a[1])
    test_filename.append(a[2])
# print(validation_serial)
# print(validation_label)
# print(validation_filename)
f.close()

# Dataset
# Training Dataset
root_dir = 'dataset/preprocessing/train'
label_dir = os.listdir(root_dir)
train_dataset = data.ARID(root_dir, label_dir[0], mapping_label)
for label_dir in os.listdir(root_dir)[1:]:
    label_dataset = data.ARID(root_dir, label_dir, mapping_label)
    train_dataset += label_dataset

# Validation Dataset
root_dir = 'dataset/preprocessing/validate'
validation_dataset = data.ARID_Val(root_dir, validation_label)


# Dataloader
batch_size = 16
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
validate_dataloader = DataLoader(validation_dataset, batch_size=batch_size)


# Dataset length
train_data_size = len(train_dataset)
validation_data_size = len(validation_dataset)
# validation_data_size = len(validation_dataset.img_path)
print("Length of training dataset is {}".format(train_data_size))
print("Length of validation dataset is {}".format(validation_data_size))
print("Total training steps in one epoch are {}".format(int(train_data_size / batch_size)))
print("Total validation steps in one epoch are {}".format(int(validation_data_size / batch_size)))


# Network
network = model.model
network.to(device)


# loss function
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)


# Optimizer
learning_rate = 1e-4
optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)


# Other parameters
# Number of training
total_train_step = 0
# Number of validation
total_val_step = 0
# Rounds of training
epoch = 20


# Tensorboard
writer = SummaryWriter('./logs')


for i in range(epoch):
    print("-------Training round {} begins-------".format(i+1))
    # Training
    network.train()
    for data in train_dataloader:
        imgs, targets, imgs_name = data
        imgs = imgs.to(device)
        outputs = network(imgs)
        targets_int = []
        for j in range(batch_size):
            a = int(targets[j])
            targets_int.append(a)
        target_tensor = torch.tensor(targets_int)
        target_tensor = target_tensor.to(device)
        loss = loss_fn(outputs, target_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 25 == 0:  # print loss every 25 steps
            print("Training step: {}, Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)
            f = open('./train_loss.txt', 'a')
            f.write(str(total_train_step))
            f.write('\t')
            f.write(str(loss.item()))
            f.write('\n')
            f.close()

    # Model Save
    # torch.save(network, 'network_{}.pth'.format(i))
    torch.save(network.state_dict(), 'network_parameter_{}.pth'.format(i))
    print("Model is saved!")

    # Validating
    network.eval()
    validation_filenum = 320
    total_validate_loss = 0
    accuracy = 0
    validation_list = [0 for row in range(validation_filenum)]
    validation_output = [[0 for row in range(10)] for column in range(validation_filenum)]
    validation_output_txt = [0 for row in range(validation_filenum)]
    with torch.no_grad():
        for data in validate_dataloader:
            imgs, targets, img_name, video_name = data
            imgs = imgs.to(device)
            outputs = network(imgs)
            targets_int = []
            for j in range(batch_size):
                a = int(targets[j])
                targets_int.append(a)
            target_tensor = torch.tensor(targets_int)
            target_tensor = target_tensor.to(device)
            loss = loss_fn(outputs, target_tensor)
            total_validate_loss = total_validate_loss + loss.item()

            total_val_step += 1
            if total_val_step % 10 == 0:    # print loss every 10 steps
                print("Val step: {}, Loss: {}".format(total_val_step, loss.item()))
                writer.add_scalar("val_loss", loss.item(), total_val_step)
                writer.add_scalar("val_total_loss", total_validate_loss, i+1)
                f = open('./val_loss.txt', 'a')
                f.write(str(total_val_step))
                f.write('\t')
                f.write(str(loss.item()))
                f.write('\n')
                f.close()
                f = open('./val_total_loss.txt', 'a')
                f.write(str(i+1))
                f.write('\t')
                f.write(str(total_validate_loss))
                f.write('\n')
                f.close()

            for j in range(batch_size):
                classification_max = max(outputs[j])
                classification_index = torch.argmax(outputs[j])
                validation_output[int(video_name[j])][classification_index] += 1

                if classification_index == int(targets[j]):
                    validation_list[int(video_name[j])] = validation_list[int(video_name[j])] + 1
                else:
                    validation_list[int(video_name[j])] = validation_list[int(video_name[j])] - 1

        # Output File
        # f = open("./Validation_output.txt", 'a')
        # for j in range(validation_filenum):
        #     validation_output_max = max(validation_output[j])
        #     validation_output_index = validation_output[j].index(validation_output_max)
        #     validation_output_txt[j] = validation_output_index
        #     f.write(str(j))
        #     f.write('\t')
        #     f.write(str(validation_output_txt[j]))
        #     f.write('\n')
        # f.close()

    for j in range(validation_filenum):
        if validation_list[j] > 0:
            accuracy = accuracy + 1
    val_acc = (accuracy * 100) / 320
    print("Total validation loss: {}".format(total_validate_loss))
    print("Total validation acc: {}%".format(val_acc))
    writer.add_scalar("val_acc", val_acc, i+1)
    f = open('./val_acc.txt', 'a')
    f.write(str(i+1))
    f.write('\t')
    f.write(str(val_acc))
    f.write('\n')
    f.close()
writer.close()
