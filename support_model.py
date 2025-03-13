import torch
import torch.nn as nn
import numpy as np

from einops import repeat, rearrange
from einops.layers.torch import Rearrange

from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block

def random_indexes(size: int):
    forward_indexes = torch.randperm(size).cpu().numpy()  # still convert to numpy for compatibility below
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes

def take_indexes(sequences, indexes):
    # Assert that sequences is 3D: (T, B, C)
    assert sequences.dim() == 3, "sequences tensor must be 3-dimensional (T, B, C)"
    return torch.gather(sequences, 0, repeat(torch.tensor(indexes, device=sequences.device), 't b -> t b c', c=sequences.shape[-1]))

class PatchShuffle(torch.nn.Module):
    def __init__(self, ratio) -> None:
        super().__init__()
        self.ratio = ratio

    def forward(self, patches: torch.Tensor):
        T, B, C = patches.shape
        remain_T = int(T * (1 - self.ratio))
        # For each batch sample, generate a random permutation of indices
        forward_indexes_list = []
        backward_indexes_list = []
        for _ in range(B):
            f_idx, b_idx = random_indexes(T)
            forward_indexes_list.append(f_idx)
            backward_indexes_list.append(b_idx)
        forward_indexes = torch.as_tensor(np.stack(forward_indexes_list, axis=-1), dtype=torch.long).to(patches.device)  # shape: (T, B)
        backward_indexes = torch.as_tensor(np.stack(backward_indexes_list, axis=-1), dtype=torch.long).to(patches.device)  # shape: (T, B)

        selected_forward_indexes = forward_indexes[:remain_T]  # shape: (remain_T, B)
        selected_backward_indexes = torch.argsort(selected_forward_indexes, dim=0)

        patches = take_indexes(patches, selected_forward_indexes)
        return patches, selected_forward_indexes, selected_backward_indexes

class MAE_Encoder(torch.nn.Module):
    def __init__(self,
                 sequ_len=64,
                 n_channel=9,
                 patch_size=2,
                 emb_dim=192,
                 num_layer=12,
                 num_head=3,
                 mask_ratio=0.75,
                 ) -> None:
        super().__init__()

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros(sequ_len // patch_size, 1, emb_dim))
        self.shuffle = PatchShuffle(mask_ratio)

        self.patchify = torch.nn.Conv1d(n_channel, emb_dim, patch_size, patch_size)

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        self.layer_norm = torch.nn.LayerNorm(emb_dim)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, sequ):
        patches = self.patchify(sequ)  # (B, emb_dim, T')
        patches = rearrange(patches, 'b c t -> t b c')  # now (T, B, C)
        # Ensure the number of patches equals the expected dimension for positional embedding
        assert patches.shape[0] == self.pos_embedding.shape[0], "Mismatch between number of patches and positional embedding length"
        patches = patches + self.pos_embedding
        patches, forward_indexes, backward_indexes = self.shuffle(patches)
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')

        return features, backward_indexes

class MAE_Decoder(torch.nn.Module):
    def __init__(self,
                 sequ_len=64,
                 n_channel=9,
                 patch_size=2,
                 emb_dim=192,
                 num_layer=4,
                 num_head=3,
                 ) -> None:
        super().__init__()

        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros(sequ_len // patch_size + 1, 1, emb_dim))

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        self.head = torch.nn.Linear(emb_dim, n_channel * patch_size)
        self.patch2sequ = Rearrange('h b (c p1) -> b c (h p1)', p1=patch_size, h=sequ_len//patch_size)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, features, backward_indexes):
        T = features.shape[0]
        backward_indexes = torch.cat([torch.zeros(1, backward_indexes.shape[1], dtype=torch.long, device=backward_indexes.device), backward_indexes + 1], dim=0)
        extra_tokens = self.mask_token.expand(backward_indexes.shape[0] - features.shape[0], features.shape[1], -1)
        features = torch.cat([features, extra_tokens], dim=0)
        features = take_indexes(features, backward_indexes)
        # Ensure that features and pos_embedding are aligned
        assert features.shape[0] == self.pos_embedding.shape[0], "Mismatch between features and positional embeddings in decoder"
        features = features + self.pos_embedding

        features = rearrange(features, 't b c -> b t c')
        features = self.transformer(features)
        features = rearrange(features, 'b t c -> t b c')
        features = features[1:]  # remove global feature

        patches = self.head(features)
        mask = torch.zeros_like(patches)
        num_masked = patches.shape[0] - T + 1
        if num_masked > 0:
            mask[-num_masked:] = 1
        else:
            pass

        new_backward_indexes = backward_indexes[1:] - 1
        mask = take_indexes(mask, new_backward_indexes)
        sequ = self.patch2sequ(patches)
        mask = self.patch2sequ(mask)

        return sequ, mask

class MAE_ViT(torch.nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

        self.encoder = MAE_Encoder(args.n_length, args.n_channel, args.patch_size, args.emb_dim, args.encoder_layer, args.encoder_head, args.mask_ratio)
        self.decoder = MAE_Decoder(args.n_length, args.n_channel, args.patch_size, args.emb_dim, args.decoder_layer, args.decoder_head)

    def forward(self, img):
        features, backward_indexes = self.encoder(img)
        predicted_img, mask = self.decoder(features, backward_indexes)
        return predicted_img, mask

class ViT_Classifier(torch.nn.Module):
    def __init__(self, encoder: MAE_Encoder, num_classes=10) -> None:
        super().__init__()
        self.cls_token = encoder.cls_token
        self.pos_embedding = encoder.pos_embedding
        self.patchify = encoder.patchify
        self.transformer = encoder.transformer
        self.layer_norm = encoder.layer_norm
        self.head = torch.nn.Linear(self.pos_embedding.shape[-1], num_classes)

    def forward(self, sequ):
        patches = self.patchify(sequ)
        patches = rearrange(patches, 'b c h -> h b c')
        patches = patches + self.pos_embedding
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        # features = rearrange(features, 'b t c -> t b c')[0]
        features = features.mean(dim=1)
        return features

class MLP_Classifier(nn.Module):
    def __init__(self, n_features, n_classes):
        super(MLP_Classifier, self).__init__()

        if n_features % n_classes != 0:
            raise ValueError("n_features must be divisible by n_classes for this MLP architecture.")
        n_dim = int(n_features / n_classes / 2) * n_classes

        self.model = nn.Sequential(
            nn.Linear(n_features, n_dim),
            nn.BatchNorm1d(n_dim),
            nn.ReLU(),
            nn.Linear(n_dim, n_classes)
        )

    def forward(self, x):
        return self.model(x)

# save model
def save_model(sa_fo, mstr, model):
    mo_pa = sa_fo + mstr + '.pth'
    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module.state_dict(), mo_pa)
    else:
        torch.save(model.state_dict(), mo_pa)

# load model
def load_model(sa_fo, mstr, model):
    mo_pa = sa_fo + mstr + '.pth'
    states = torch.load(mo_pa, map_location=lambda storage, loc: storage)
    model.load_state_dict(states)
    return

if __name__ == '__main__':
    shuffle = PatchShuffle(0.75)
    a = torch.rand(16, 2, 10)
    b, fwd_idx, bwd_idx = shuffle(a)
    print("Shuffled patches shape:", b.shape)

    img = torch.rand(2, 9, 64)
    encoder = MAE_Encoder()
    decoder = MAE_Decoder()
    features,
