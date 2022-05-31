
from turtle import clear
import numpy as np
import matplotlib.pyplot as plt

import grpc

import minecraft_pb2_grpc
from minecraft_pb2 import *
import minecraft_pb2

import os
import time

import torch


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
    
a = [getattr(minecraft_pb2, i) for i in dir(minecraft_pb2) if i.endswith('COTTA')]
b = [getattr(minecraft_pb2, i) for i in dir(minecraft_pb2) if i.endswith('BLOCK')][:-1]
a = a+b

idx = np.array([i for i in np.arange(len(all_mc_block_ids)) if all_mc_block_ids[i] in a])
# print(all_mc_block_ids)
# all_mc_block_ids = all_mc_block_ids[idx]
# all_mc_block_cols = all_mc_block_cols[idx]

# print(all_mc_block_ids)


    
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
    
    
def mc_spawn_img(img, x, y, z, center=False):
    if center:
        x=x-img.shape[0]//2
        z=z-img.shape[0]//2
    cols, block_ids = get_mc_structure(img)
    mc_spawn_block_ids(block_ids, x, y, z)

def clear_area(minx, maxx, miny, maxy, minz, maxz):
    client.fillCube(FillCubeRequest(
        cube=Cube(
            min=Point(x=minx, y=miny, z=minz),
            max=Point(x=maxx, y=maxy, z=maxz)
        ),
        type=AIR
    ))
def clear_area_workspace():
    clear_area(-256, 256, 4, 15, -256, 256)
    
def draw_line(x1, y1, z1, x2, y2, z2):
    coors = set()
    dist = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
    for t in np.linspace(0, 1, int(dist*1.5)):
        x = x1 + t*(x2-x1)
        y = y1 + t*(y2-y1)
        z = z1 + t*(z2-z1)
        coors.add((int(x), int(y), int(z)))
    
    blocks = [Block(position=Point(x=x, y=y, z=z), type=OBSIDIAN, orientation=UP) for x, y, z in coors]
    client.spawnBlocks(Blocks(blocks=blocks))

def main():
    # img = plt.imread('init_image_tensor.png')
    # img = img[::img.shape[0]//16, ::img.shape[1]//16, :]
    clear_area_workspace()

    imgi = 0
    pop = torch.load('outputs/pop.pth')
    parent_data = torch.load('outputs/parent_data.pth')

    name2coor = {}

    for circle_id in range(4):
        r = circle_id*23
        print(r)
        n_pics = max(1, int(2*np.pi*r / 25))
        print(n_pics)


        for theta in np.linspace(0, 2*np.pi, n_pics+1)[:-1]:
            x = int(r*np.cos(theta))
            z = int(r*np.sin(theta))
            print(theta)
            print(x, z)

            name = pop[imgi]
            imgi+=1

            name2coor[name] = (x, z)

            img = plt.imread(f'outputs/{name}/ACTUAL/output.png')
            img = img[::img.shape[0]//16, ::img.shape[1]//16, :]

            mc_spawn_img(img, x, 5, z, center=True)

            if name in parent_data:
                parent_name = parent_data[name]
                px, pz = name2coor[parent_name]
                draw_line(x, 4, z, px, 4, pz)

            time.sleep(1)

        print()


def animate_img_path(name, x, y, z, center=True, time_step=0.1):
    path = f'outputs/{name}/ACTUAL/steps'
    img_names = os.listdir(path)
    img_names.sort()
    for img_name in img_names:
        img = plt.imread(f'{path}/{img_name}')
        img = img[::img.shape[0]//16, ::img.shape[1]//16, :]
        mc_spawn_img(img, x, y, z, center=center)
        y+=1
        time.sleep(time_step)


if __name__ == "__main__":
    clear_area_workspace()
    # animate_img_path('desk', 0, 4, 0, center=False, time_step=1.)
    main()