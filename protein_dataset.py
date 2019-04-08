# -*- coding: utf-8 -*
import math
import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pathlib
from scipy.stats import norm as nm
from multiprocessing import Pool
import argparse

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--resolution', type=int, default='256',
                    help='output resolution')
parser.add_argument('--dataset_path', type=str, default='D:\protein structure prediction\data\dataset',
                    help='path of dataset')
parser.add_argument('--output_path', type=str, default='D:\protein structure prediction\data\dataset\processed data',
                    help='path of output')
parser.add_argument('--dataset', type=str, default='bc-30-1_CA',
                    help='name of dataset folder')
parser.add_argument('--input_type', type=str, default='cif',
                    help='type of input file')
parser.add_argument('--output_type', type=str, default='image',
                    help='image or distance_map, default: images')
parser.add_argument('--map_range', type=int, default='42',
                    help='map range of structures, default: -42 to 42')
parser.add_argument('--multi_process', type=bool, default=False,
                    help='multi_process or not')
parser.add_argument('--multi_atom', type=bool, default=False,
                    help='input with 4 atoms and not just CA only')
parser.add_argument('--move2center', type=bool, default=False,
                    help='relocate the center of proteins to the center of coordinate system')
parser.add_argument('--redistribute', type=bool, default=False,
                    help='redistribute the original distribution according to normal distribution')
parser.add_argument('--relative_number', type=bool, default=True,
                    help='mark dots with relative serial number')
parser.add_argument('--draw_connection', type=bool, default=False,
                    help='draw dots connection or not')
parser.add_argument('--redistribute_rate', type=float, default='1.4',
                    help='coefficient of redistribution amplitude')
args = parser.parse_args()

res = args.resolution
mr = args.map_range
s = mr / res  # scale=map_range/resolution
input_folder = args.dataset_path + '\\' + args.dataset
AMINO_ACIDS = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS',
               'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
               'LEU', 'LYS', 'MET', 'PHE', 'PRO',
               'SER', 'THR', 'TRP', 'TYR', 'VAL']
AMINO_ACID_NUMBERS = {}
for aa in range(len(AMINO_ACIDS)):
    AMINO_ACID_NUMBERS.update({AMINO_ACIDS[aa]: aa+20.5})


class Atom(object):
    def __init__(self, aminoacid, index, x, y, z, atom_type='CA', element='C'):
        self.index = int(index)
        self.aa = aminoacid
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.type = atom_type
        self.element = element


def readfile(filename, path):
    file = open(path + '\\' + filename, 'r')
    message = file.readlines()
    file.close()
    return message


def extract_cif(cif_message):
    atoms = []
    for line in cif_message:
        line = line.split()
        atoms.append(Atom(line[5], line[8], line[10],
                          line[11], line[12], line[3], line[2]))
    return atoms


def extract_ca_cif(cif_message):
    atoms = []
    for line in cif_message:
        if line[11] == 'CA':
            line = line.split()
            atoms.append(Atom(line[5], line[8], line[10], line[11], line[12]))
    return atoms


def extract_pdb(pdb_message):
    atoms = []
    for line in pdb_message:
        atoms.append(Atom(line[17:20], line[13:16], line[30:38],
                          line[38:46], line[46:54], line[13:16], line[77]))
    return atoms


def extract_ca_pdb(pdb_message):
    atoms = []
    for line in pdb_message:
        if line[13:15] == 'CA':
            atoms.append(Atom(line[17:20], line[13:16], line[30:38], line[38:46], line[46:54]))
    return atoms


def extract_message(message, message_type):
    if message_type == 'pdb':
        return extract_pdb(message)
    elif message_type == 'cif':
        return extract_cif(message)


def rotation_axis(first_residue):
    x = first_residue.x
    y = first_residue.y
    z = first_residue.z
    c = ((y - x) ** 2 /
         ((y * res * (x ** 2 + y ** 2 + z ** 2 - 2 * s ** 2) ** 0.5 / mr - z) ** 2
           + (x * res * (x ** 2 + y ** 2 + z ** 2 - 2 * s ** 2) ** 0.5 / mr - z) ** 2
           + (y - x) ** 2)
         ) ** 0.5
    a = (y * res * (x ** 2 + y ** 2 + z ** 2 - 2 * s ** 2) ** 0.5 / mr - z) / (x - y) * c
    b = (x * res * (x ** 2 + y ** 2 + z ** 2 - 2 * s ** 2) ** 0.5 / mr - z) / (y - x) * c
    return [(a, b, c), (-a, -b, -c)]  # 转轴


def rotation_angle(first_residue):
    x = first_residue.x
    y = first_residue.y
    z = first_residue.z
    return math.acos(
        ((x + y) * s + z * (x ** 2 + y ** 2 + z ** 2 - 2 * s ** 2) ** 0.5) /
        (x ** 2 + y ** 2 + z ** 2)
    )  # 转角


def rotation(u, v, w, t, axis):  # 原始坐标
    (a, b, c) = axis
    # 罗德里格旋转公式：
    rx = u*math.cos(t)+(b*w-c*v)*math.sin(t)+a*(a*u+b*v+c*w)*(1-math.cos(t))
    ry = v*math.cos(t)+(c*u-a*w)*math.sin(t)+b*(a*u+b*v+c*w)*(1-math.cos(t))
    rz = w*math.cos(t)+(a*v-b*u)*math.sin(t)+c*(a*u+b*v+c*w)*(1-math.cos(t))
    return rx, ry, rz  # 旋转所得坐标


def relocate(atoms):
    x_o = (atoms[0].x + atoms[-1].x) / 2
    y_o = (atoms[0].y + atoms[-1].y) / 2
    z_o = (atoms[0].z + atoms[-1].z) / 2
    for atom in atoms:
        atom.x -= x_o
        atom.y -= y_o
        atom.z -= z_o
    vs = rotation_axis(atoms[0])
    t = rotation_angle(atoms[0])
    atom_v = []
    for v in vs:
        atom_v.append(rotation(atoms[0].x, atoms[0].y, atoms[0].z, t, v))
    if abs(atom_v[0][0] - s) + abs(atom_v[0][1] - s) < abs(atom_v[1][0] - s) + abs(atom_v[1][1] - s):
        k = 0
    else:
        k = 1
    for atom in atoms:
        (atom.x, atom.y, atom.z) = rotation(atom.x, atom.y, atom.z, t, vs[k])
    return atoms


def move2center(atoms):
    coordinates = []
    for atom in atoms:
        coordinates.append([atom.x, atom.y, atom.z])
    coordinates = np.array(coordinates)
    center = tf.Variable(tf.zeros([1, 3]))
    distances = coordinates-center
    loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(distances), 1)))
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train = optimizer.minimize(loss)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    losses = []
    for step in range(10):
        sess.run(train)
        losses.append(sess.run(loss))
    while losses[-1] != losses[-5]:
        sess.run(train)
        losses.append(sess.run(loss))
    coordinates -= sess.run(center)
    for i in range(len(atoms)):
        atoms[i].x = coordinates[i][0]
        atoms[i].y = coordinates[i][1]
        atoms[i].z = coordinates[i][2]
    tf.reset_default_graph()
    return atoms


def sign(x):
    if x < 0:
        return -1
    else:
        return 1


def close_neibor(array, x_ary, y_ary, dot, dis_x, dis_y, rec):
    x_step = sign(dis_x)
    y_step = sign(dis_y)
    if abs(dis_x) < abs(dis_y):
        neibors = [(0, y_step), (x_step, 0), (x_step, y_step), (-x_step, 0),
                   (0, -y_step), (-x_step, y_step), (x_step, -y_step), (-x_step, -y_step)]
    else:
        neibors = [(x_step, 0), (0, y_step), (x_step, y_step), (0, -y_step),
                   (-x_step, 0), (x_step, -y_step), (-x_step, y_step), (-x_step, -y_step)]
    step = 1
    while True:
        for (i, j) in neibors:
            try:
                if array[x_ary + i * step, y_ary + j * step, 2] == 0:
                    array[x_ary + i * step, y_ary + j * step] = [dot.z, dot.index, AMINO_ACID_NUMBERS.get(dot.aa)]
                    rec.update({(x_ary + i * step, y_ary + j * step): dot})
                    # print('dot%d:%d,%d->%d,%d'%(dot[6],x,y,x+i*step,y+j*step))
                    return array
            except IndexError:
                print('dot(%d+%d,%d+%d) is out of the edge' % (x_ary, i * step, y_ary, j * step))
        # print('%d step neibor of dot%d(%d,%d) is full!'%(step,dot_i,x,y))
        step += 1


def lattice_battle(array, x_ary, y_ary, dot1, dot2, rec):  # dot1 is old; dot2 is new
    dis1_x = dot1.x / (2 * s) % 1 - 0.5
    dis1_y = dot1.y / (2 * s) % 1 - 0.5
    dis2_x = dot2.x / (2 * s) % 1 - 0.5
    dis2_y = dot2.y / (2 * s) % 1 - 0.5
    if dis1_x ** 2 + dis1_y ** 2 > dis2_x ** 2 + dis2_y ** 2:
        # print('%d / %d swap!'%(dot1[6],dot2[6]))
        array = close_neibor(array, x_ary, y_ary, dot1, dis1_x, dis1_y, rec)
        array[x_ary, y_ary] = [dot2.z, dot2.index, AMINO_ACID_NUMBERS.get(dot2.aa)]
        rec.update({(x_ary, y_ary): dot2})
    else:
        array = close_neibor(array, x_ary, y_ary, dot2, dis2_x, dis2_y, rec)
    return array


def arraylize(atoms):
    array = np.zeros([res, res, 3], dtype=float, order='C')
    rec = {}  # atoms record
    for atom in atoms:
        x_ary = int((atom.x + mr) // (2 * s))
        y_ary = int((atom.y + mr) // (2 * s))
        if rec.get((x_ary, y_ary)):
            array = lattice_battle(array, x_ary, y_ary, rec[(x_ary, y_ary)], atom, rec)
        else:
            array[x_ary, y_ary] = [atom.z, atom.index, AMINO_ACID_NUMBERS.get(atom.aa)]
            rec.update({(x_ary, y_ary): atom})
    return array, rec


# def values_sta(path):
#     xs = []
#     ys = []
#     for filename in os.listdir(path):
#         atoms = move2center(relocate(extract_cif(readfile(filename, path))))
#         for atom in atoms:
#             xs.append(atom.x)
#             ys.append(atom.y)
#     return xs, ys


def normal_dis(values, var, coefficient):
    dis = []
    values.sort()
    mark = 0
    idx = 0
    for i in range(res):
        cut_point = nm.ppf((i + 1) / res, 0, var**0.5 * coefficient)
        if idx == len(values):
            dis.append([])
            mark = int(idx)
        else:
            while values[idx] < cut_point:
                idx += 1
                if idx == len(values):
                    dis.append(values[mark:idx])
                    mark = int(idx)
                    break
            else:
                dis.append(values[mark:idx])
                mark = int(idx)
    return dis


# def redistribute():


def visual_values_dis(values):
    mark = 0
    idx = 0
    dis = []
    dis_count = []
    axis_length = 2*mr
    for i in range(1, res+1):
        cut_point = (i-res/2)*axis_length/res
        if idx == len(values):
            dis.append([])
        else:
            while values[idx] < cut_point:
                idx += 1
                if idx == len(values):
                    dis.append(values[mark:idx])
                    break
            else:
                dis.append(values[mark:idx])
                mark = int(idx)
    for i in range(res):
        dis_count.append(len(dis[i]))
    plt.bar(range(res), dis_count)
    plt.show()


def vis_normal_dis(values, var, coefficient):
    dis = []
    values.sort()
    mark = 0
    idx = 0
    dis_count = []
    for i in range(res):
        cut_point = nm.ppf((i+1)/res, 0, var**0.5*coefficient)
        if idx == len(values):
            dis.append([])
            mark = int(idx)
        else:
            while values[idx] < cut_point:
                idx += 1
                if idx == len(values):
                    dis.append(values[mark:idx])
                    mark = int(idx)
                    break
            else:
                dis.append(values[mark:idx])
                mark = int(idx)
        dis_count.append(len(dis[i]))
    plt.bar(range(res), dis_count)
    plt.show()


def dots_connection(dot1, dot2, array, site):
    path = [site[dot2][0] - site[dot1][0], site[dot2][1] - site[dot1][1]]

    if path[0] > 0:
        i_move = 1
    elif path[0]==0:
        i_move = 0
    else:
        i_move = -1

    if path[1] > 0:
        j_move = 1
    elif path[1] == 0:
        j_move = 0
    else:
        j_move = -1
    moves_count = abs(path[0]) + abs(path[1]) - 2

    if moves_count >= 0:
        for i in range(max(abs(path[0]),abs(path[1]))):
            if abs(path[0]) > abs(path[1]):
                iter = abs(path[1])
                if i<iter-1:
                    if array[site[dot1][0] + (i + 1) * i_move, site[dot1][1]+i*j_move, 2] == 0:
                        array[site[dot1][0] + (i + 1) * i_move, site[dot1][1]+i*j_move] = [
                            dot1.z + (dot2.z - dot1.z) * (i + 1) / (abs(path[0]) + 1),
                            dot1.index + (dot2.index - dot1.index) * (i + 1) / (abs(path[0]) + 1), 0]
                    if array[site[dot1][0]  + (i + 1) * i_move, site[dot1][1] + (i + 1) * j_move, 2] == 0:
                        array[site[dot1][0] + (i + 1) * i_move, site[dot1][1] + (i + 1) * j_move] = [
                            dot1.z + (dot2.z - dot1.z) * (i + 1) / (abs(path[1]) + 1),
                            dot1.index + (dot2.index - dot1.index) * (i + 1) / (abs(path[1]) + 1), 0]
                else:
                    if array[site[dot1][0] + (i + 1) * i_move, site[dot1][1]+(iter-1)*j_move, 2] == 0:
                        array[site[dot1][0] + (i + 1) * i_move, site[dot1][1]+(iter-1)*j_move] = [
                            dot1.z + (dot2.z - dot1.z) * (i + 1) / (abs(path[0]) + 1),
                            dot1.index + (dot2.index - dot1.index) * (i + 1) / (abs(path[0]) + 1), 0]
            else:
                iter = abs(path[0])
                if i < iter - 1:
                    if array[site[dot1][0] + (i + 1) * i_move, site[dot1][1] + i * j_move, 2] == 0:
                        array[site[dot1][0] + (i + 1) * i_move, site[dot1][1] + i * j_move] = [
                            dot1.z + (dot2.z - dot1.z) * (i + 1) / (abs(path[0]) + 1),
                            dot1.index + (dot2.index - dot1.index) * (i + 1) / (abs(path[0]) + 1), 0]
                    if array[site[dot1][0] + (i + 1) * i_move, site[dot1][1] + (i + 1) * j_move, 2] == 0:
                        array[site[dot1][0] + (i + 1) * i_move, site[dot1][1] + (i + 1) * j_move] = [
                            dot1.z + (dot2.z - dot1.z) * (i + 1) / (abs(path[1]) + 1),
                            dot1.index + (dot2.index - dot1.index) * (i + 1) / (abs(path[1]) + 1), 0]
                else:
                    if array[site[dot1][0] + (iter - 1) * i_move, site[dot1][1] + (i + 1) * j_move, 2] == 0:
                        array[site[dot1][0] + (iter - 1) * i_move, site[dot1][1] + (i + 1)  * j_move] = [
                            dot1.z + (dot2.z - dot1.z) * (i + 1) / (abs(path[0]) + 1),
                            dot1.index + (dot2.index - dot1.index) * (i + 1) / (abs(path[0]) + 1), 0]



def draw_connection(atoms, array, rec):
    site = {}
    for (x, y) in rec.keys():
        site.update({rec[(x,y)]: [x,y]})
    for i in range(len(atoms)-1):
        dots_connection(atoms[i], atoms[i+1], array, site)


def write_log(path):
    arg_name_list = ['dataset', 'resolution', 'input_type', 'output_type', 'map_range', 'multi_atom',
                     'move2center', 'redistribute', 'redistribute_rate', 'relative_number', 'draw_connection']
    arg_list = [args.dataset, args.resolution, args.input_type, args.output_type, args.map_range, args.multi_atom,
                args.move2center, args.redistribute, args.redistribute_rate, args.relative_number, args.draw_connection]
    write_list = [time.strftime("%Y%m%d_%H%M", time.localtime())]
    for i in range(len(arg_name_list)):
        print("%s = %s" % (arg_name_list[i], str(arg_list[i])))
        write_list.append("%s = %s" % (arg_name_list[i], str(arg_list[i])))
    write_list.append('\n\n\n')
    with open(path + '\\args_log.txt', 'a') as log_writer:
        log_writer.write('\n'.join(write_list))


def process():
    log_dir = args.output_path + '\\' + args.dataset
    output_dir = args.output_path + '\\' + args.dataset + '\\' + time.strftime("%Y%m%d_%H%M", time.localtime())
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    write_log(log_dir)
    if args.output_type == 'image':
        if args.redistribute:
            atoms_dic = {}
            xs = []
            ys = []
            for filename in os.listdir(input_folder):
                atoms = relocate(extract_message(readfile(filename, input_folder), args.input_type))
                if args.move2center:
                    atoms = move2center(atoms)
                for atom in atoms:
                    xs.append(atom.x)
                    ys.append(atom.y)
                atoms_dic.update({filename: atoms})
            # var_sta = max(np.var(xs), np.var(ys))
        else:
            for filename in os.listdir(input_folder):
                atoms = relocate(extract_message(readfile(filename, input_folder), args.input_type))
                if args.move2center:
                    atoms = move2center(atoms)
                if args.draw_connection:
                    array, rec = arraylize(atoms)
                    draw_connection(atoms, array, rec)
                else:
                    array, _ = arraylize(atoms)
                if args.relative_number:
                    array[:, :, 1] /= (len(atoms) + 1)
                output_name = filename.replace('.cif', '.npy')
                np.save(output_dir + '\\' + output_name, array)
    elif args.output_type == 'distance_map':
        if args.multi_atom:
            for filename in os.listdir(input_folder):
                atoms = extract_message(readfile(filename, input_folder), args.input_type)


if __name__ == '__main__':
    if args.multi_process:
        print('Parent process %s.' % os.getpid())
        p = Pool(60)
        p.apply_async(process())
        print('Waiting for all subprocesses done...')
        p.close()
        p.join()
        print('All subprocesses done.')
    else:
        process()
