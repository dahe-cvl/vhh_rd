import vhh_rd.Feature_Extractor as FE
import vhh_rd.RD as RD
import vhh_rd.Transformations as Transformations
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
import os, sys, glob, random, itertools
import cv2
from torchvision import transforms
import torch.optim as optim
import wandb
import imgaug as ia
import math

config_path = "./config/config_rd.yaml"
lr = 1e-5

# Do not use NA directory
dirs_to_use = ["CU", "ELS", "LS", "MS", "I"]

class TriplesDataset(Dataset):
    """
    Dataset of images
    """
    def __init__(self, path, transforms, augmentations):
        """
            image_folder must be a folder containing the dataset as png images
        """
        self.transforms = transforms
        self.augmentations = augmentations
        self.left_dir = os.path.join(path, "left")
        self.right_dir = os.path.join(path, "right")
        self.images_paths_left = list(glob.glob(os.path.join(self.left_dir, "*.jpg")))
        self.images_paths_right = list(glob.glob(os.path.join(self.right_dir, "*.jpg")))

    def __len__(self):
        return len(self.images_paths_left) * 2

    def __getitem__(self, idx):
        if idx % 2 == 0:
            anchor_dir = self.images_paths_left
            others_dir = self.images_paths_right
        else:
            others_dir = self.images_paths_left
            anchor_dir = self.images_paths_right

        idx = math.floor(idx / 2.0)

        img_anchor = cv2.imread(anchor_dir[idx]).astype(np.uint8)
        img_pos = cv2.imread(others_dir[idx]).astype(np.uint8)

        path_neg = random.choice(others_dir)
        img_neg = cv2.imread(path_neg).astype(np.uint8)
 
        augment = lambda x: self.augmentations(x).astype(np.float32)
        postprocessing = lambda x: self.transforms(torch.from_numpy(x.transpose((2,0,1))))
        processing = lambda x: postprocessing(augment(x))

        # img_aug_neg = cv2.resize(augment(img_neg), (256,256), interpolation = cv2.INTER_AREA)
        # img_aug_pos = cv2.resize(augment(img_pos), (256,256), interpolation = cv2.INTER_AREA)
        # img_aug_anchor = cv2.resize(augment(img_anchor), (256,256), interpolation = cv2.INTER_AREA)
        # cv2.imshow("positives", np.hstack((img_aug_anchor.astype(np.uint8), img_aug_pos.astype(np.uint8), img_aug_neg.astype(np.uint8))))
        # cv2.waitKey(0)

        img_anchor = processing(img_anchor)
        img_pos = processing(img_pos)
        img_neg = processing(img_neg)

        # Note that original is not augmented
        return {"original":img_anchor, "positive": img_pos, "negative": img_neg}

    def set_seed(self):
        """
            Call this before iterating through the dataloader wrapper to ensure to always get the same data
        """
        random.seed(0)
        ia.seed(0)

def evaluate(data_loader, model, criterion_cosine, criterion_similarity, device):
    model.eval()
    curr_loss_cos = 0
    curr_loss_sim = 0
    dataset = data_loader.dataset
    dataset.set_seed()
    for i, batch in enumerate(tqdm(data_loader)):
        images = batch["original"].to(device)
        batch_length = images.shape[0]
        out_images = model(images)
        del images
        
        images_pos = batch["positive"].to(device)
        out_images_pos = model(images_pos)
        del images_pos

        images_neg = batch["negative"].to(device)
        out_images_neg = model(images_neg)
        del images_neg

        loss_sim = criterion_similarity(out_images, out_images_pos, out_images_neg).item()
        loss_cos = criterion_cosine(out_images, out_images_pos, torch.ones(batch_length).to(device)).item() + criterion_cosine(out_images, out_images_neg, -torch.ones(batch_length).to(device)).item()

        curr_loss_cos += loss_cos
        curr_loss_sim += loss_sim

    return curr_loss_cos / len(dataset), curr_loss_sim / len(dataset)


def main():
    rd = RD.RD(config_path)
    loss_type = rd.config["LOSS_TYPE"]

    model = FE.FeatureExtractor(rd.config["MODEL"], evaluate=False)
    preprocess = model.get_preprocessing(siamese=True)
    modelPath = os.path.join(rd.models_path, rd.config["MODEL"])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    model.model.to(device) 

    # WandB setup
    wandb.init(
    entity="cvl-vhh-research",
    project="Train Siamese Net (vhh_rd)",
    notes="",
    tags=[],
    config=rd.config
    )

    dataset = TriplesDataset("...", preprocess, Transformations.get_augmentations()) 
    val_test_length = int(0.1*len(dataset))

    create_data_loader = lambda path, batchsize: DataLoader(
        TriplesDataset(path, preprocess, Transformations.get_augmentations()), 
        batch_size=batchsize, 
        num_workers=rd.config["NUM_WORKERS_TRAINING"],
        shuffle=True)

    train_loader = create_data_loader("/caa/Homes01/fjogl/vhh_rd/datasets/train", rd.config["BATCHSIZE_TRAIN"])
    val_loader = create_data_loader("/caa/Homes01/fjogl/vhh_rd/datasets/val", rd.config["BATCHSIZE_EVALUATE"])
    test_loader = create_data_loader("/caa/Homes01/fjogl/vhh_rd/datasets/test", rd.config["BATCHSIZE_EVALUATE"])

    cos = torch.nn.CosineSimilarity()
    def neg_CosineSimilarity(a,b):
        return -1*cos(a,b)

    criterion_cos = nn.CosineEmbeddingLoss(reduction='sum')
    criterion_sim =  torch.nn.TripletMarginWithDistanceLoss(distance_function=neg_CosineSimilarity, reduction='sum')

    if loss_type == "cosine":
        criterion = criterion_cos
    elif loss_type == "triplet":
        criterion = criterion_sim

    optimizer = optim.AdamW(model.parameters(), lr=lr)

    best_validation_loss = sys.float_info.max

    for epoch in range(rd.config["NR_EPOCHS"]):
        curr_loss = 0
        print("Epoch {0}".format(epoch))
        model.train()
        for i, batch in enumerate(train_loader):
            images = batch["original"].to(device)
            batch_length = images.shape[0]
            out_images = model(images)
            del images

            images_pos = batch["positive"].to(device)
            out_images_pos = model(images_pos)
            del images_pos
            
            images_neg = batch["negative"].to(device)
            out_images_neg = model(images_neg)
            del images_neg

            if loss_type == "triplet":
                loss = criterion(out_images, out_images_pos, out_images_neg)
                wandb.log({'Train\BatchLoss': loss.item() / batch_length})
            elif loss_type == "cosine":
                c1 = criterion(out_images, out_images_pos, torch.ones(batch_length).to(device))
                c2 = criterion(out_images, out_images_neg, -torch.ones(batch_length).to(device))
                loss = c1 + c2
                wandb.log({'Train\BatchLoss': loss.item() / batch_length, "Train\BatchPossSampleError": c1 , "Train\BatchNegativeSampleError": c2})

            
            loss.backward()
            optimizer.step()
            curr_loss += loss.item()
            print("\t{0} / {1}\t Loss {2}".format(i+1, len(train_loader), loss.item() / batch_length))

        curr_loss = curr_loss / len(train_loader.dataset)
        print("Epoch {0}\nTraining Loss {1}".format(epoch, curr_loss))

        # Validate 
        val_cos_loss, val_tri_loss = evaluate(val_loader, model, criterion_cos, criterion_sim, device)
        print("Validation triplet loss: {0}\t Cosine loss: {1}".format(val_tri_loss, val_cos_loss))
        wandb.log({'Train\Loss': curr_loss, 'Val\LossTriplet': val_tri_loss, "Val\LossCos": val_cos_loss})

         # Early stopping
        if ((loss_type == "triplet" and val_tri_loss <= best_validation_loss) or (loss_type == "cosine" and val_cos_loss <= best_validation_loss)):
            if loss_type == "triplet":
                best_validation_loss = val_tri_loss
            elif loss_type == "cosine":
                best_validation_loss = val_cos_loss

            epochs_since_last_improvement = 0

            print("Saving to ", modelPath)
            torch.save(model.state_dict(), modelPath)
        else:
            epochs_since_last_improvement += 1
            if epochs_since_last_improvement >= rd.config["EPOCHS_EARLY_STPOPPING"]:
                print("No improvement since {0} epochs, stopping training.".format(epochs_since_last_improvement))
                print("Loading final model")
                model.load_state_dict(torch.load(modelPath))
                break

    print("\n\n\nFINAL EVALUATION:")

    val_cos_loss, val_tri_loss = evaluate(val_loader, model, criterion_cos, criterion_sim, device)
    test_cos_loss, test_tri_loss = evaluate(test_loader, model, criterion_cos, criterion_sim, device)

    print("Validation triplet loss: {0}\t Cosine loss: {1}".format(val_tri_loss, val_cos_loss))
    wandb.log({'Final\ValLossTriplet': val_tri_loss, "Final\ValLossCos": val_cos_loss})

    print("Test triplet loss: {0}\t Cosine loss: {1}".format(test_tri_loss, test_cos_loss))
    wandb.log({'Final\TestLossTriplet': test_tri_loss, "Final\TestLossCos": test_cos_loss})

if __name__ == "__main__":
    main()
    