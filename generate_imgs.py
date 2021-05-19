import numpy as np
import torch
from eval_inception import load_model
from torch.autograd import Variable
import torchvision.utils as vutils
import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--model_in', type=str, required=True)
parser.add_argument('--nz', type=int, required=True)
parser.add_argument('--ngf', type=int, required=True)
parser.add_argument('--nclass', type=int, required=True)
parser.add_argument('--nimgs', type=int, required=True)
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--start_width', type=int, required=True)
parser.add_argument('--splits', type=int, required=True)
parser.add_argument('--ngpu', type=int, required=True)
opt = parser.parse_args()

assert opt.nimgs % opt.splits == 0, "ERR: opt.nimgs must be divided by opt.splits"
assert (opt.nimgs // opt.splits) % opt.batch_size == 0, "ERR: opt.nimgs//opt.splits \
        must be divided by opt.batch_size"

def gen_imgs():
    gen = load_model()
    # buffer:
    # gaussian noise
    z = torch.FloatTensor(opt.batch_size, opt.nz).cuda()
    # random label
    y_fake = torch.LongTensor(opt.batch_size).cuda()
    imgs = []
    with torch.no_grad():
        for i in range(0, opt.nimgs, opt.batch_size):
            z.normal_(0, 1)
            y_fake.random_(0, to=opt.nclass)
            v_z = Variable(z)
            v_y_fake = Variable(y_fake)
            x_fake = gen(v_z, y=v_y_fake)
            x = x_fake.data.cpu().numpy()
            imgs.append(x)
    import pdb;pdb.set_trace()
    imgs = np.asarray(imgs, dtype=np.float32)

    fake_img = np.transpose(vutils.make_grid(
        imgs, padding=2, nrow=3, normalize=True), (1, 2, 0))
    plt.imshow(fake_img)

    file_name = f"./orignal_sngan_cifar10.png"
    plt.savefig(fname=file_name)
