import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from model import WhisperCNN
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter


class WhisperDataset(Dataset):
    def __init__(
        self, units_dir, audio_feature_dir, partition, max_token_length: int = None
    ):
        self.units_dir = units_dir
        self.audio_feature_dir = audio_feature_dir
        self.partition = partition
        self.data = self._parse_units_file()
        self.max_token_length = max_token_length
        self.generator = torch.Generator().manual_seed(42)

    def _parse_units_file(self):
        data = []
        for part in self.partition:
            units_file = os.path.join(self.units_dir, f"{part}.unit")
            with open(units_file, "r") as f:
                for line in f:
                    parts = line.strip().split("|")
                    file_name = parts[0].strip()
                    tokens = np.array(list(map(int, parts[1].strip().split())))
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

    def _crop_tokens_and_features(self, tokens, features):
        start_token_idx = torch.randint(
            0, len(tokens) - self.max_token_length, (1,), generator=self.generator
        ).item()
        end_token_idx = start_token_idx + self.max_token_length
        tokens = tokens[start_token_idx:end_token_idx]
        start_feature_idx = start_token_idx * 5
        end_feature_idx = min((end_token_idx + 3) * 5, len(features))
        features = features[start_feature_idx:end_feature_idx]
        return tokens, features

    def __getitem__(self, idx):
        item = self.data[idx]
        npy_path = item["npy_path"]
        tokens = item["tokens"]
        features = np.load(npy_path)
        features = features.reshape(-1, features.shape[-1])
        if self.max_token_length is not None and len(tokens) > self.max_token_length:
            tokens, features = self._crop_tokens_and_features(tokens, features)
        return tokens, features


def collate_fn(batch):
    def inplace_stack(TL, pad=0, offset=None):
        shape = list(map(max, zip(*(T.shape for T in TL))))
        shape[0] = (shape[0] + pad + 31) // 32 * 32
        tensor = np.zeros_like(TL[0], shape=(len(TL), *shape))
        for i, T in enumerate(TL):
            slices = (i,) + tuple(slice(dim) for dim in T.shape)
            tensor[slices] = T if offset is None else T + offset
        return torch.from_numpy(tensor)

    tokens, features = zip(*batch)
    feature_lengths = [f.shape[0] for f in features]

    tokens_padded = inplace_stack(tokens, pad=3, offset=1)
    features_padded = inplace_stack(features, pad=0)
    feature_mask = (
        torch.arange(features_padded.size(1))[None, :]
        < torch.tensor(feature_lengths)[:, None]
    )

    return tokens_padded, features_padded, feature_mask


def train(
    model,
    train_dataloader,
    eval_dataloader,
    num_epochs,
    learning_rate,
    device,
    output_dir,
    writer,
):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_l1_loss = 0.0
        running_l2_loss = 0.0
        running_cosine_similarity = 0.0

        for i, (tokens, features, mask) in enumerate(
            tqdm.tqdm(train_dataloader, mininterval=10, ncols=50)
        ):
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
            cosine_similarity = F.cosine_similarity(outputs[slices], features, dim=-1)
            cosine_similarity = cosine_similarity[mask].mean()

            loss = l1loss * 0.1 + l2loss
            loss.backward()
            optimizer.step()

            loss = loss.item()
            l1loss = l1loss.item()
            l2loss = l2loss.item()
            cosine_similarity = cosine_similarity.item()

            running_loss += loss
            running_l1_loss += l1loss
            running_l2_loss += l2loss
            running_cosine_similarity += cosine_similarity

            rstep = epoch * len(train_dataloader) + i
            writer.add_scalar("Loss/Train", loss, rstep)
            writer.add_scalar("Loss/L1_Train", l1loss, rstep)
            writer.add_scalar("Loss/L2_Train", l2loss, rstep)
            writer.add_scalar("Metric/CosineSimilarity_Train", cosine_similarity, rstep)

        avg_loss = running_loss / len(train_dataloader)
        avg_l1_loss = running_l1_loss / len(train_dataloader)
        avg_l2_loss = running_l2_loss / len(train_dataloader)
        avg_cosine_similarity = running_cosine_similarity / len(train_dataloader)

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, L1 Loss: {avg_l1_loss:.4f}, L2 Loss: {avg_l2_loss:.4f}, Cosine Similarity: {avg_cosine_similarity:.4f}"
        )

        # Evaluate after each epoch
        eval_loss, eval_l1_loss, eval_l2_loss, eval_cosine_similarity = evaluate(
            model, eval_dataloader, device
        )

        writer.add_scalars("Loss/Epoch", {"Train": avg_loss, "Eval": eval_loss}, epoch)
        writer.add_scalars(
            "Loss/L1_Epoch", {"Train": avg_l1_loss, "Eval": eval_l1_loss}, epoch
        )
        writer.add_scalars(
            "Loss/L2_Epoch", {"Train": avg_l2_loss, "Eval": eval_l2_loss}, epoch
        )
        writer.add_scalars(
            "Metric/CosineSimilarity_Epoch",
            {"Train": avg_cosine_similarity, "Eval": eval_cosine_similarity},
            epoch,
        )

        save_path = os.path.join(output_dir, f"whisper_cnn_{epoch:04d}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_l1_loss = 0.0
    total_l2_loss = 0.0
    total_cosine_similarity = 0.0
    total_mask = 0

    with torch.no_grad():
        for tokens, features, mask in tqdm.tqdm(dataloader, mininterval=10, ncols=50):
            tokens = tokens.to(device)
            features = features.to(device)
            total_mask += mask.sum().item()
            mask = mask.to(device)

            outputs = model(tokens)
            slices = tuple(slice(None, s) for s in features.shape)
            diff = outputs[slices] - features
            l2loss = (diff.square() * mask.unsqueeze(-1)).sum().item()
            l1loss = (diff.abs() * mask.unsqueeze(-1)).sum().item()
            cosine_similarity = F.cosine_similarity(outputs[slices], features, dim=-1)
            cosine_similarity = cosine_similarity[mask].sum().item()

            loss = l1loss * 0.1 + l2loss
            total_loss += loss
            total_l1_loss += l1loss
            total_l2_loss += l2loss
            total_cosine_similarity += cosine_similarity

    avg_loss = total_loss / total_mask / features.size(-1)
    avg_l1_loss = total_l1_loss / total_mask / features.size(-1)
    avg_l2_loss = total_l2_loss / total_mask / features.size(-1)
    avg_cosine_similarity = total_cosine_similarity / total_mask

    print(
        f"Validation Loss: {avg_loss:.4f}, L1 Loss: {avg_l1_loss:.4f}, L2 Loss: {avg_l2_loss:.4f}, Cosine Similarity: {avg_cosine_similarity:.4f}"
    )
    return avg_loss, avg_l1_loss, avg_l2_loss, avg_cosine_similarity


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
    batch_size = 128
    train_max_token_length = 256
    eval_max_token_length = None

    # Create training and evaluation datasets and dataloaders
    train_dataset = WhisperDataset(
        units_dir, audio_feature_dir, partition_train, train_max_token_length
    )
    eval_dataset = WhisperDataset(
        units_dir, audio_feature_dir, partition_eval, eval_max_token_length
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        prefetch_factor=2,
        num_workers=4,
        generator=torch.Generator().manual_seed(0),
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        prefetch_factor=2,
        num_workers=4,
    )

    # Initialize model
    output_dim = train_dataset[0][1].shape[-1]  # Set appropriate output dimension
    model = WhisperCNN(num_tokens, embedding_dim, hidden_dim, output_dim, 5, 5)
    model.to(device)

    # Initialize TensorBoard writer
    writer = SummaryWriter()

    # Train model
    train(
        model,
        train_dataloader,
        eval_dataloader,
        num_epochs,
        learning_rate,
        device,
        output_dir,
        writer,
    )

    # Close the writer
    writer.close()
