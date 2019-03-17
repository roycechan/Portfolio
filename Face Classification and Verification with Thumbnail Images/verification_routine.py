import numpy as np
import torch
from torch.utils import data
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import paths

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_embedding(model, data_loader):
    params = {'batch_size': 256,
              'shuffle': False,
              'num_workers': 16}

    with torch.no_grad():
        model.train()
        model.to(device)
        model = torch.nn.DataParallel(model)

        training_generator = data.DataLoader(data_loader, **params)

        embeddings = []

        for batch_num, (img, idx) in enumerate(training_generator):
            img = img.to(device)
            output = model(img)
            output = output.cpu()
            embeddings.append(output)

            if batch_num % 100 == 99:
                print("Processed batches:",batch_num+1)

    return embeddings

def predict(embeddings, idx_list, test_labels):
    embeddings_np = np.vstack(np.array([emb.numpy() for emb in embeddings]))

    embedding_dict = {}
    for idx, i in enumerate(idx_list):
        embedding_dict[i] = embeddings_np[idx]

    scores = []

    for row in range(len(test_labels)):
        a, b = test_labels[row]
        a_embedding = embedding_dict[a].reshape(1, -1)
        b_embedding = embedding_dict[b].reshape(1, -1)
        score = cosine_similarity(a_embedding, b_embedding)
        scores.append(score)

        if row % 100000 == 0:
            print("Processed",row,"trials")

    scores_np = np.array(scores).flatten()
    df = pd.DataFrame([i for i in open(paths.test_order_verify).read().splitlines()])
    df['score'] = scores_np
    df.columns = ["trial", "score"]
    df.to_csv(paths.output_verification, index=False)
