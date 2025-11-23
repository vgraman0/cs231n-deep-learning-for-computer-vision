import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CocoCaptions


class CocoDataset(Dataset):
    def __init__(self, root_dir, ann_file, transform, vocab):
        self.coco = CocoCaptions(root=root_dir, annFile=ann_file, transform=transform)

        self.vocab = vocab
        self.sos_idx = self.vocab.stoi[self.vocab.sos_token]
        self.eos_idx = self.vocab.stoi[self.vocab.eos_token]

    def __len__(self):
        return len(self.coco)

    def __getitem__(self, idx):
        img, raw_captions = self.coco[idx]
        raw_caption = raw_captions[0]

        token_ids = (
            [self.sos_idx] + self.vocab.numericalize(raw_caption) + [self.eos_idx]
        )
        caption = torch.tensor(token_ids, dtype=torch.long)
        return img, caption


class Collator:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        images = []
        captions = []
        lengths = []

        for image, caption in batch:
            images.append(image)
            captions.append(caption)
            lengths.append(len(caption))

        images_tensor = torch.stack(images, dim=0)  # (B,3,H,W)
        captions_tensor = pad_sequence(
            captions, batch_first=True, padding_value=self.pad_idx
        )  # (B, T_max)
        lengths_tensor = torch.tensor(lengths, dtype=torch.long)  # (B,)

        return images_tensor, captions_tensor, lengths_tensor


def get_loader(
    root_dir,
    ann_file,
    transform,
    vocab,
    batch_size=32,
    num_workers=2,
    shuffle=True,
    pin_memory=True,
):
    dataset = CocoDataset(root_dir, ann_file, transform, vocab)

    pad_idx = vocab.stoi[vocab.pad_token]
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=Collator(pad_idx),
    )

    return dataset, loader
