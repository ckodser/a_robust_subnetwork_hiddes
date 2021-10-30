# code from https://github.com/facebookresearch/fvcore/blob/main/docs/flop_count.md

import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table, flop_count_str


def flop_count(model, train_loader):
    model.eval()

    for test_images, test_labels in train_loader:
        sample_image = test_images[0:1]

        flop = FlopCountAnalysis(model, sample_image)
        print(flop_count_table(flop, max_depth=4))

        break
