from models.vit_model_face import ViT_face_model, Hybrid_ViT
from models.resnet import resnet_face18
import torch
import sys
from models import ViT_face
import argparse
from util.utils import (
    perform_val_buffalo,
    AverageMeter,
)

import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image


def main(args):
    GPU_ID = [0]
    device = torch.device("cuda:%d" % GPU_ID[0])
    torch.backends.cudnn.benchmark = True
    NUM_CLASS = 10575  # for LFW # 93431 for casia

    channels = 1
    use_scale = True
    grayscale = True

    out_dim = 512
    model_name = "buffalo_sc"

    name = args.name  #'talfw' #'mlfw' # # # #'glfw'  # #

    model = FaceAnalysis(
        name=model_name,
        root="/home/jeans/internship/face-vit/results",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    model.prepare(ctx_id=0, det_size=(640, 640))

    print("=" * 60)
    print("model path: {}".format(model_name))

    BATCH_SIZE = 64
    EMBEDDING_SIZE = out_dim

    MULTI_GPU = False

    print(
        "Process [{}] dataset, model \nemb={}, channels={}, \nuse_scale={}, grayscale={}".format(
            name,
            EMBEDDING_SIZE,
            channels,
            use_scale,
            grayscale,
        )
    )
    print("=" * 60)
    # name, data_set, issame = vers[0]
    # accuracy, std, xnorm, best_threshold, roc_curve = perform_val(MULTI_GPU, device, EMBEDDING_SIZE, BATCH_SIZE, model, data_set, issame)

    # accuracy, std, xnorm, best_threshold, roc_curve = perform_val_resnet_color_images(MULTI_GPU, device, EMBEDDING_SIZE, BATCH_SIZE, model.face_model, grayscale=True, size=size, use_scale=use_scale, target=name)

    accuracy, std, xnorm, best_threshold, roc_curve = perform_val_buffalo(
        MULTI_GPU,
        device,
        EMBEDDING_SIZE,
        model,
        target=name,
    )

    print("[%s]XNorm: %1.5f" % (name, xnorm))
    print("[%s]Accuracy-Flip: %1.5f+-%1.5f" % (name, accuracy, std))
    print("[%s]Best-Threshold: %1.5f" % (name, best_threshold))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="", help="training set directory")
    parser.add_argument("--name", default="lfw", help="test set")
    parser.add_argument("--network", default="VITs", help="training set directory")
    parser.add_argument(
        "--target", default="lfw,talfw,sllfw,calfw,cplfw,cfp_fp,agedb_30", help=""
    )
    parser.add_argument("--batch_size", type=int, help="", default=20)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))
