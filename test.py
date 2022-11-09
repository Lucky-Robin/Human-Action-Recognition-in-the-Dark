from torch.utils.data import DataLoader
from torch import nn
import torch.optim
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
f.close()



# Testing Dataset
root_dir = 'dataset/preprocessing/test'
test_dataset = data.ARID_Val(root_dir, test_label)


# Dataloader
batch_size = 1
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)


# Dataset length
test_data_size = len(test_dataset)


# Network
network = model.model
network.to(device)


# loss function
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)



# Other parameters
# Number of test
total_test_step = 0
# Rounds of test
epoch = 1


for i in range(epoch):
    print("-------Testing begins-------")

    # Testing
    network.eval()
    test_filenum = 450
    accuracy = 0
    test_list = [0 for row in range(test_filenum)]
    test_output = [[0 for row in range(10)] for column in range(test_filenum)]
    test_output_txt = [0 for row in range(test_filenum)]
    with torch.no_grad():
        for data in test_dataloader:
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

            total_test_step += 1
            if total_test_step % 1 == 0:
                print("Val step: {}, Loss: {}".format(total_test_step, loss.item()))

            for j in range(batch_size):
                classification_max = max(outputs[j])
                classification_index = torch.argmax(outputs[j])
                test_output[int(video_name[j])][classification_index] += 1

                if classification_index == int(targets[j]):
                    test_list[int(video_name[j])] = test_list[int(video_name[j])] + 1
                else:
                    test_list[int(video_name[j])] = test_list[int(video_name[j])] - 1

        # Output File
        f = open("./Test_output.txt", 'a')
        for j in range(test_filenum):
            test_output_max = max(test_output[j])
            test_output_index = test_output[j].index(test_output_max)
            test_output_txt[j] = test_output_index
            f.write(str(j))
            f.write('\t')
            f.write(str(test_output_txt[j]))
            f.write('\n')
        f.close()

    for j in range(test_filenum):
        if test_list[j] > 0:
            accuracy = accuracy + 1
    test_acc = (accuracy * 100) / 320
    print("Total test acc: {}%".format(test_acc))

