import os

import numpy as np
import torch
import torch.optim as optim
import tqdm
from model import WhisperCNN
from torch.utils.data import DataLoader, Dataset


class WhisperDataset(Dataset):
    def __init__(self, units_dir, audio_feature_dir, partition):
        self.units_dir = units_dir
        self.audio_feature_dir = audio_feature_dir
        self.partition = partition
        self.data = self._parse_units_file()

    def _parse_units_file(self):
        data = []
        for part in self.partition:
            units_file = os.path.join(self.units_dir, f"{part}.unit")
            with open(units_file, "r") as f:
                for line in f:
                    parts = line.strip().split("|")
                    file_name = parts[0].strip()
                    tokens = list(map(int, parts[1].strip().split()))
                    npy_path = os.path.join(
                        self.audio_feature_dir, part, file_name + ".npy"
                    )
                    if os.path.exists(npy_path):
                        data.append({"npy_path": npy_path, "tokens": tokens})
                    else:
                        print(f"File {npy_path} not found.")
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        npy_path = item["npy_path"]
        tokens = item["tokens"]
        features = np.load(npy_path)
        features = features.reshape(-1, features.shape[-1])
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(
            features, dtype=torch.float32
        )


def collate_fn(batch):
    def inplace_stack(TL, pad=0, offset=None):
        shape = list(map(max, zip(*(T.shape for T in TL))))
        shape[0] = (shape[0] + pad + 31) // 32 * 32
        tensor = TL[0].new_zeros((len(TL), *shape))
        for i, T in enumerate(TL):
            slices = (i,) + tuple(slice(dim) for dim in T.shape)
            tensor[slices] = T if offset is None else T + offset
        return tensor

    tokens, features = zip(*batch)
    feature_lengths = [f.shape[0] for f in features]

    tokens_padded = inplace_stack(tokens, pad=3, offset=1)
    features_padded = inplace_stack(features, pad=0)
    feature_mask = (
        torch.arange(features_padded.size(1))[None, :]
        < torch.tensor(feature_lengths)[:, None]
    )

    return tokens_padded, features_padded, feature_mask


def train(model, dataloader, num_epochs, learning_rate, device, output_dir):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for tokens, features, mask in tqdm.tqdm(dataloader, mininterval=10, ncols=50):
            tokens = tokens.to(device)
            features = features.to(device)
            mask = mask.to(device)
            scale = mask.sum() / mask.numel()

            optimizer.zero_grad()

            outputs = model(tokens)
            slices = tuple(slice(None, s) for s in features.shape)
            diff = outputs[slices] - features
            l2loss = (diff.square() * mask.unsqueeze(-1)).mean() / scale
            l1loss = (diff.abs() * mask.unsqueeze(-1)).mean() / scale

            loss = l1loss * 0.1 + l2loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}"
        )

    torch.save(model.state_dict(), os.path.join(output_dir, "whisper_cnn.pth"))
    print(f"Model saved to {os.path.join(output_dir, 'whisper_cnn.pth')}")


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for tokens, features, mask in tqdm.tqdm(dataloader, mininterval=10, ncols=50):
            tokens = tokens.to(device)
            features = features.to(device)
            mask = mask.to(device)

            outputs = model(tokens)
            slices = tuple(slice(None, s) for s in features.shape)
            loss = (
                (outputs[slices] - features).square() * mask.unsqueeze(-1)
            ).mean() / (mask.sum() / mask.numel())
            total_loss += loss.item()

    average_loss = total_loss / len(dataloader)
    print(f"Validation Loss: {average_loss:.4f}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # Training parameters
    partition_train = ["short-pretrain"]
    partition_eval = ["trainval"]
    units_dir = "/mnt/hard3/rhs/intern/av_unit/"
    audio_feature_dir = "/mnt/hard3/rhs/intern/audio_feature"
    output_dir = "./model_output"
    num_tokens = 1001  # Set appropriate number of tokens
    embedding_dim = 128
    hidden_dim = 512
    num_epochs = 20
    learning_rate = 0.003
    batch_size = 32

    # Create training and evaluation datasets and dataloaders
    train_dataset = WhisperDataset(units_dir, audio_feature_dir, partition_train)
    eval_dataset = WhisperDataset(units_dir, audio_feature_dir, partition_eval)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    # Initialize model
    output_dim = train_dataset[0][1].shape[-1]  # Set appropriate output dimension
    model = WhisperCNN(num_tokens, embedding_dim, hidden_dim, output_dim, 5, 5).to(
        device
    )

    # Train model
    train(model, train_dataloader, num_epochs, learning_rate, device, output_dir)

    # Evaluate model
    evaluate(model, eval_dataloader, device)