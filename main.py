import argparse
import itertools
import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable
import time
import dataloader
import models.basic_cnn
import matplotlib.pyplot as plt
#from torchvision import models
from sklearn.metrics import confusion_matrix, classification_report

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)

torch.cuda.manual_seed(SEED)
torch.manual_seed(7)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusionmatrix.png')

def plot_training_statistics(train_stats, model_name):
    fig, axes = plt.subplots(2, figsize=(15, 15))
    axes[0].plot(train_stats[f'{model_name}_Training_Loss'], label=f'{model_name}_Training_Loss')
    axes[0].plot(train_stats[f'{model_name}_Validation_Loss'], label=f'{model_name}_Validation_Loss')
    axes[1].plot(train_stats[f'{model_name}_Training_Acc'], label=f'{model_name}_Training_Acc')
    axes[1].plot(train_stats[f'{model_name}_Validation_Acc'], label=f'{model_name}_Validation_Acc')

    axes[0].set_xlabel("Number of Epochs"), axes[0].set_ylabel("Loss")
    axes[1].set_xlabel("Number of Epochs"), axes[1].set_ylabel("Accuracy in %")

    axes[0].legend(), axes[1].legend()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim = True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc


def train(model, iterator, optimizer, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for (x, y) in iterator:
        x = Variable(torch.FloatTensor(np.array(x))).to(device)
        y = Variable(torch.LongTensor(y)).to(device)

        optimizer.zero_grad()

        y_pred = model(x)

        loss = criterion(y_pred, y)

        acc = calculate_accuracy(y_pred, y)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for (x, y) in iterator:
            x = Variable(torch.FloatTensor(np.array(x))).to(device)
            y = Variable(torch.LongTensor(y)).to(device)

            y_pred = model(x)

            loss = criterion(y_pred, y)

            acc = calculate_accuracy(y_pred, y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def main(args):

    data_dir = 'data/'
    labels = pd.read_csv(os.path.join(data_dir, 'labels.csv'))
    le = LabelEncoder()
    labels.breed = le.fit_transform(labels.breed)
    labels.head()
    X = labels.id
    y = labels.breed
    # assert (len(os.listdir(os.path.join(data_dir, 'train'))) == len(labels))
    print(labels.head())
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    X_valid, X_test, y_valid, y_test = train_test_split(X_valid, y_valid, test_size=0.7, random_state=SEED,
                                                        stratify=y_valid)
    # transform_train = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.08, 1.0),
    #                                                                    ratio=(3.0 / 4.0, 4.0 / 3.0)),
    #                                       transforms.ToTensor(),
    #                                       normalize
    #                                       ])
    #
    # transform_test = transforms.Compose([
    #     transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
    #     normalize])
    transform_train = transforms.Compose([transforms.RandomResizedCrop(32, scale=(0.08, 1.0),
                                                                       ratio=(3.0 / 4.0, 4.0 / 3.0)),
                                          transforms.ToTensor(),
                                          normalize
                                          ])

    transform_test = transforms.Compose([
        transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor(),
        normalize])

    train_data = dataloader.Dataset_Interpreter(data_path=data_dir + 'train/', file_names=X_train, labels=y_train,
                                                transforms=transform_train)
    valid_data = dataloader.Dataset_Interpreter(data_path=data_dir + 'train/', file_names=X_valid, labels=y_valid,
                                                transforms=transform_test)
    test_data = dataloader.Dataset_Interpreter(data_path=data_dir + 'train/', file_names=X_test, labels=y_test,
                                               transforms=transform_test)

    train_loader = DataLoader(train_data, shuffle=True, num_workers=args.num_workers, batch_size=args.batch_size)
    val_loader = DataLoader(valid_data, num_workers=args.num_workers, batch_size=args.batch_size)
    test_loader = DataLoader(test_data, num_workers=args.num_workers, batch_size=args.batch_size)
    print(f'Number of training examples: {len(train_data)}')
    print(f'Number of validation examples: {len(valid_data)}')
    print(f'Number of testing examples: {len(test_data)}')

    model = models.basic_cnn.LeNet().to(device)
    # model = models.resnet18(pretrained=True).to(device)
    # for name, param in model.named_parameters():
    #     if ("bn" not in name):
    #         param.requires_grad = False
    # model.fc = nn.Linear(model.fc.in_features, 120).to(device)


    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-2)
    #optimizer = torch.optim.Adam(model.parameters())
    best_valid_loss = float('inf')

    train_losses = []
    valid_losses = []
    train_accs = []
    valid_accs = []
    BEST_ACC=0
    for epoch in range(args.num_epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        valid_loss, valid_acc = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        train_accs.append(train_acc * 100)
        valid_accs.append(valid_acc * 100)
        is_best = valid_loss > BEST_ACC
        best_loss = max(valid_loss, BEST_ACC)
        state = {
            'epoch': epoch + 1,
            'arch': "abcd",
            'state_dict': model.state_dict(),
            'best_loss': BEST_ACC,
        }
        # torch.save(state, 'checkpoint.pth.tar')
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'model_best_{}.pth.tar'.format(epoch))
        print(f'Epoch: {epoch + 1:02}')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')
        break
    # Test the model
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        all_labels = []
        all_preds = []
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels)
            all_preds.extend(predicted)


        print('Test Accuracy of the model on the 2000 test images: {} %'.format(100 * correct / total))
        classes = ['affenpinscher','afghan_hound','african_hunting_dog','airedale','american_staffordshire_terrier','appenzeller','australian_terrier','basenji','basset','beagle','bedlington_terrier','bernese_mountain_dog','black-and-tan_coonhound','blenheim_spaniel','bloodhound','bluetick','border_collie','border_terrier','borzoi','boston_bull','bouvier_des_flandres','boxer','brabancon_griffon','briard','brittany_spaniel','bull_mastiff','cairn','cardigan','chesapeake_bay_retriever','chihuahua','chow','clumber','cocker_spaniel','collie','curly-coated_retriever','dandie_dinmont','dhole','dingo','doberman','english_foxhound','english_setter','english_springer','entlebucher','eskimo_dog','flat-coated_retriever','french_bulldog','german_shepherd','german_short-haired_pointer','giant_schnauzer','golden_retriever','gordon_setter','great_dane','great_pyrenees','greater_swiss_mountain_dog','groenendael','ibizan_hound','irish_setter','irish_terrier','irish_water_spaniel','irish_wolfhound','italian_greyhound','japanese_spaniel','keeshond','kelpie','kerry_blue_terrier','komondor','kuvasz','labrador_retriever','lakeland_terrier','leonberg','lhasa','malamute','malinois','maltese_dog','mexican_hairless','miniature_pinscher','miniature_poodle','miniature_schnauzer','newfoundland','norfolk_terrier','norwegian_elkhound','norwich_terrier','old_english_sheepdog','otterhound','papillon','pekinese','pembroke','pomeranian','pug','redbone','rhodesian_ridgeback','rottweiler','saint_bernard','saluki','samoyed','schipperke','scotch_terrier','scottish_deerhound','sealyham_terrier','shetland_sheepdog','shih-tzu','siberian_husky','silky_terrier','soft-coated_wheaten_terrier','staffordshire_bullterrier','standard_poodle','standard_schnauzer','sussex_spaniel','tibetan_mastiff','tibetan_terrier','toy_poodle','toy_terrier','vizsla','walker_hound','weimaraner','welsh_springer_spaniel','west_highland_white_terrier','whippet','wire-haired_fox_terrier','yorkshire_terrier']

        all_pred = torch.tensor(all_labels)
        all_label = torch.tensor(all_labels)
        print(classification_report(all_label, all_pred, target_names=classes))
        confusion_mat = confusion_matrix(y_true=all_label, y_pred=all_pred)
        print(confusion_mat)
        plot_confusion_matrix(cm=confusion_matrix(y_true=all_label, y_pred=all_pred),
                              classes=classes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='path for saving trained models')
    parser.add_argument('--image_dir', type=str, default='images/', help='directory for resized images')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--crop_size', type=int, default=224, help='size for randomly cropping images')

    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    #
    # parser.add_argument('--log_step', type=int, default=10, help='step size for prining log info')
    # parser.add_argument('--save_step', type=int, default=5, help='step size for saving trained models')

    args = parser.parse_args()
    print(args)
    main(args)
