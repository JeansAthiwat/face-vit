from models.vit_model_face import ViT_face_model, Hybrid_ViT
from models.resnet import resnet_face18
import torch
import sys
from models import ViT_face
import argparse
from util.utils import (
    get_val_data,
    perform_val_color_images_cls,
    perform_val_resnet_color_images,
    perform_val_color_images,
    perform_val_color_images_hybrid_vit,
    AverageMeter,
)

import insightface


def main(args):
    GPU_ID = [0]
    device = torch.device("cuda:%d" % GPU_ID[0])
    torch.backends.cudnn.benchmark = True
    NUM_CLASS = 10575  # for LFW # 93431 for casia

    channels = 1
    use_scale = True
    grayscale = True

    out_dim = 512
    model_path = "XXX"
    name = args.name  #'talfw' #'mlfw' # # # #'glfw'  # #

    model = ""

    print("=" * 60)
    print("model path: {}".format(model_path))

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
        BATCH_SIZE,
        model,
        grayscale=grayscale,
        use_scale=use_scale,
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
