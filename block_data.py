
import numpy as np
import matplotlib.pyplot as plt

import grpc

import minecraft_pb2_grpc
from minecraft_pb2 import *

channel = grpc.insecure_channel('localhost:5001')
client = minecraft_pb2_grpc.MinecraftServiceStub(channel)


a = ['10', 
     '0 1 2 3 4 5 7 9 11 12', 
     '2 6 7 8 9 12 13 14 15', 
     '0 1 10 11 12 13 15', 
     '0 1 3 4 5 8 13', 
     '6 9 11 12 13 14 15', 
     '0 1 2 3 4 5 10 12 14', 
     '4 5 10 11 13 14', 
     '0 1 3 4 5 6 7 8 11 12 13 14 15', 
     '3 4 6 7 8 10 11', 
     '0 5 6 8 9 10 11 12 15', 
     '0 3 4 5 9 11 12 13 15', '1 3 6 7 8 10 11 13 14',
     '5 9 10', '1 4', '4 10 12']


a = [[int(ii) for ii in i.split(' ')] for i in a]

block_ids = []
for x, data in enumerate(a):
    for z in data:
        # print(x, z)
        block_id = x*16+z
        # print(x, z, block_id)
        block_ids.append(block_id)
all_mc_block_ids = block_ids
all_mc_block_ids = np.array(all_mc_block_ids)



raw_block_img = plt.imread('raw_block_img.png')

p1 = np.array([41, 836])
p2 = np.array([1621, 832])
p3 = np.array([39, 146])
p4 = np.array([1620, 142])

def calc_sub_image(x, z):
    i = np.array([(p4[0]-p1[0])/16, 0])
    j = np.array([0, (p4[1]-p1[1])/7 ])
    
    
    s = p1 + z*i + x*j
    e = p1 + (z+1)*i + (x+1)*j
    
    s, e = (s+.5).astype(int), (e+.5).astype(int)
    
    pad = 2
    return raw_block_img[e[1]+pad:s[1]-pad, s[0]+pad: e[0]-pad]
    

all_mc_block_cols = []

for pos, block_id in enumerate(all_mc_block_ids):
    # names = [name for name in dir(minecraft_pb2) if getattr(minecraft_pb2, name)==block_id]
    # print(block_id, names)
    
    x = pos//16
    z = pos%16
    
    
    subimg = calc_sub_image(x, z)
    solidimg = np.zeros_like(subimg)
    solidcol = subimg.mean(axis=(0, 1))
    all_mc_block_cols.append(solidcol)
    solidimg[:, :] = [solidcol]
    
all_mc_block_cols = np.array(all_mc_block_cols)
    
    
def get_mc_structure(img):
    # shape = img.shape
    x = all_mc_block_cols[:, None, None, :] - img
    x = np.linalg.norm(x, ord=2, axis=-1)
    x = x.argmin(axis=0)
    return all_mc_block_cols[x], all_mc_block_ids[x]
    
    
def mc_spawn_block_ids(block_ids, x, y, z):
    xs = np.arange(x, x+block_ids.shape[0], 1)[::-1]
    ys = np.arange(y, y+1, 1)
    zs = np.arange(z, z+block_ids.shape[1], 1)

    xs, ys, zs = np.meshgrid(xs, ys, zs, indexing='ij')
    xs, ys, zs = xs.flatten(), ys.flatten(), zs.flatten()
    
    block_ids = block_ids.flatten()
    
    blocks = [Block(position=Point(x=x, y=y, z=z), type=block_id, orientation=UP) for x, y, z, block_id in zip(xs, ys, zs, block_ids)]
    client.spawnBlocks(Blocks(blocks=blocks))
    
    
def mc_spawn_img(img, x, y, z):
    cols, block_ids = get_mc_structure(img)
    mc_spawn_block_ids(block_ids, x, y, z)
    
    