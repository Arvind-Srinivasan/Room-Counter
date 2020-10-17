import torch
from data import get_dataloader
from retinanet import model

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

learning_rate = 1e-4
max_epochs = 100

# Parameters
params = {'batch_size': 8,
          'shuffle': True,
          'num_workers': 6}

training_generator = get_dataloader(params)
retinanet = model.resnet18(num_classes=3, pretrained=True).to(device)
optimizer = torch.optim.Adam(retinanet.parameters(), lr=learning_rate)


def train():
    # Loop over epochs
    for epoch in range(max_epochs):


        # Training
        for i, (local_batch, local_labels) in enumerate(training_generator):
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)

            # Model computations
            classification_loss, regression_loss = retinanet([local_batch, local_labels])

            classification_loss = classification_loss.mean()
            regression_loss = regression_loss.mean()

            loss = classification_loss + regression_loss

            if bool(loss == 0):
                continue

            loss.backward()

            torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

            optimizer.step()

            print(
                'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f}'.format(
                    epoch, i, float(classification_loss), float(regression_loss)))


        torch.save(retinanet.state_dict(), "weights/" + str(epoch) + "_weights.pt")


if __name__ == "__main__":
    train()
