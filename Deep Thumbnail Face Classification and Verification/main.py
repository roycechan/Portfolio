import torch
import glob

import data, classification_routine, verification_routine,config, paths
from models import ShuffleNetV2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_classification():
    train_dataloader_class = torch.utils.data.DataLoader(data.train_dataset, batch_size=config.batch_size,
                                                   shuffle=True, num_workers=16)

    val_dataloader_class = torch.utils.data.DataLoader(data.val_dataset, batch_size=config.batch_size,
                                                 shuffle=True, num_workers=24)

    # if resuming from checkpoint
    # last_checkpoint = max([int(list(filter(str.isdigit, i))[0]) for i in glob.glob('output/*.pth')])
    # model_path = str(paths.model+str(last_checkpoint)+'.pth')
    # if device:
    #     network = torch.load(model_path)
    #
    # else:
    #     # Load GPU model on CPU
    #     network = torch.load(model_path, map_location=lambda storage, loc: storage)

    # if start from scratch
    last_checkpoint = 0
    network = ShuffleNetV2.ShuffleNetV2(n_class=config.num_classes, net_size=config.net_size_chosen)

    classification_routine.train(network, train_dataloader_class, val_dataloader_class, last_checkpoint = last_checkpoint)

    test_dataset = data.ClassificationTestDataset(paths.test_data_class)
    test_output = classification_routine.predict(network, test_dataset)
    print(test_output)

def run_verification():
    test_dataloader_verification = data.VerificationTestDataset(paths.test_data_verify)
    test_labels_verification = data.VerificationTrials(paths.test_order_verify)

    # if resuming from checkpoint
    last_checkpoint = max([int(list(filter(str.isdigit, i))[0]) for i in glob.glob('output/*.pth')])
    model_path = str(paths.model+str(last_checkpoint)+'.pth')
    if device:
        network = torch.load(model_path)

    else:
        # Load GPU model on CPU
        network = torch.load(model_path, map_location=lambda storage, loc: storage)

    embeddings = verification_routine.get_embedding(network, test_dataloader_verification)
    idx_list = [idx for tensor, idx in test_dataloader_verification]
    verification_routine.predict(embeddings, idx_list, test_labels_verification)

if __name__ == "__main__":
    run_classification()
    run_verification()
