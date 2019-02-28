import math
import numpy as np
import matplotlib.pyplot as plt
import os
import gzip
import sys
from math import isnan
class Protein_screening(object):
    def __init__(self,path='D:\pdb_screening'):
        self.path=path
        
    def path_global(self):
        global path_n
        path_n = os.path.join(self.path,"bc_30.txt")
        global path_cif
        path_cif=os.path.join(self.path,"CIF\\")
        global save_path
        save_path=os.path.join(self.path,"cif_list\\")
        global path_filtered
        path_filtered=save_path.replace("cif_list","cif_filtered")
        global path_method
        path_method=os.path.join(self.path,"method.txt")
        
    #将服务器获取的蛋白质文件解压   
    def zip_transform(self):
        path_z=os.path.join(self.path,'mmCIF')
        #path_cif=path_z.replace("mmCIF","CIF")
        #os.mkdir(path_cif)
        for parent,dirnames,filenames in os.walk(path_z): 
            path_cif_p=parent.replace("mmCIF","CIF")
            os.mkdir(path_cif_p)
            for f in filenames:
                path = os.path.join(parent,f)
                f = gzip.open(path, 'rb')
                file_content = f.read()
                path_w = path.replace(".gz", "")
                path_p = path_w.replace("mmCIF","CIF")
                fw = open(path_p, 'wb') # 若是'wb'就表示写二进制文件
                fw.write(file_content)
                fw.close()
                f.close()
                
                    
    #筛选实验方法 最低分辨率 实验方法打分
    def get_attributes(self,lines):
        method=''
        resolution='?'
        r_work='?'
        r_free='?'
        for i in range(len(lines)):
            if lines[i][0]=='_':
                if method:
                    if method=="'X-RAY DIFFRACTION'":
                        #if lines[i][:24]=='_reflns.d_resolution_low':
                            #resolution_low=lines[i].split()[1]
                        if lines[i][:25]=='_reflns.d_resolution_high':
                            resolution=lines[i].split()[1]
                        if lines[i][:26]=='_refine.ls_R_factor_R_work':
                            r_work=lines[i].split()[1]
                            r_free=lines[i+1].split()[1]
                            return [method,resolution,r_work,r_free]
                    if method=="'ELECTRON MICROSCOPY'":
                        if lines[i][:33]=='_em_3d_reconstruction.resolution ':
                            resolution=lines[i].split()[1]
                            return [method,resolution]      
                else:
                    if lines[i][:13]=='_exptl.method':
                        method=' '.join(lines[i].split()[1:])
                        if method=="'SOLUTION NMR'":
                            return [method,str(i)]       

    def batr(self):
        path_batr=os.path.join(self.path,"CIF")
        for parent,dirnames,filenames in os.walk(path_batr): 
            for f in filenames:
                path_f = os.path.join(parent,f)
                f_read= open(path_f, 'r')
                lines = f_read.readlines()
                try:
                    method = self.get_attributes(lines)
                    #global path_method
                    #path_method=os.path.join(self.path,"method.txt")
                    fw = open(path_method, 'a') # 若是'wb'就表示写二进制文件                
                    fw.write(f+": "+' '.join(method))
                    fw.write('\n')
                    fw.close()
                except :
                    path_ex = os.path.join(self.path,"ex.txt")
                    fw_ex = open(path_ex, 'a')
                    fw_ex.write(lines[0])
                    fw_ex.close()
                f_read.close()
                
    #清洗point3数据 以及提取多链信息
    def point3(self):
        bc_30=[]
        #打开method.txt
        file=open(path_method,'r')
        lines=file.readlines()
        #global path_n
        #path_n = os.path.join(self.path,"bc_30.txt")
        fn = open(path_n, 'a')
        
        #打开bc_30 拆成单个列表
        path_bc30 = os.path.join(self.path,"duqu.txt")
        file_bc30=open(path_bc30,'r')
        f_lines=file_bc30.readlines()
        for i in range(len(f_lines)):
            bc_30.extend(f_lines[i].split())

        #遍历筛选大于0.3的值 并保留拆链信息
        for line in lines:
            a=line.split()
            if len(a)==4:
                continue
            elif line.split()[-2]=='?' or line.split()[-3]=='?':
                continue
            elif line.split()[-2]=='.' or line.split()[-3]=='.':
                continue
            elif float(line.split()[-3])>0 and float(line.split()[-2])>0:      
                if 1/float(line.split()[-3])-float(line.split()[-2])>0.3:
                    for line_bc30 in bc_30:
                        if a[0][0:4].upper()==line_bc30[0:4]:  
                            fn.write(line_bc30+'\n')
                            print(line_bc30)
        fn.close()
        file.close()
        file_bc30.close()
    
    def get_chains(self):#txt文件位置 cif读取位置 cif保存位置
        #for parent,dirnames,filenames in os.walk('D:\CIF_'):
            #for f in filenames:
        global path_cif
        path_cif=os.path.join(self.path,"CIF\\")
        path_name=os.path.join(self.path,"bc_30.txt")
        with open(path_name,'r')as file:
            chains=file.readlines()
        for i in range(len(chains)):
            chains[i]=chains[i][:-1]
        pros=[]
        for chain in chains:
            pros.append(chain[:4].lower())
        pros=list(set(pros))
        pros.sort()
        dic={}
        chain_list={}
        for pro in pros:
            dic.update({pro:[]})
        for chain in chains:
            dic[chain[:4].lower()].append(chain[-1])
            chain_list.update({chain:[]})
        for pro in pros:
            diry=pro[1:3]+'/'
            with open(path_cif+diry+pro+'.cif','r') as file:
                message=file.readlines()
            for line in message:
                if line[:4]=='ATOM':
                    chain_num=line.split()[-3]
                    if chain_num in dic[pro]:
                        if line.split()[-2]=='CA':
                            chain_list[pro.upper()+'_'+chain_num].append(line)
        print('next:')
        #global save_path
        #save_path=os.path.join(self.path,"cif_list\\")
        os.mkdir(save_path)
        for chain in chains:
            if chain.upper()==chain:
                with open(save_path+chain+'.cif','w') as writer:
                    writer.write(''.join(chain_list[chain]))
            else:
                with open(save_path+chain+chain[-1]+'.cif','w') as writer:
                    writer.write(''.join(chain_list[chain]))
                        #因为windows系统下命名不区分大小写，修改小写链命名（6CUE_r.chain --> 6CUE_rr.chain)
                             
    #删除概率较小的一条链
    def remove_lines(self):
        a=0
        global path_remove
        path_remove=os.path.join(self.path,"remove.txt")
        fd=open(path_remove,'a')
        #global path_filtered
        #path_filtered=save_path.replace("cif_list","cif_filtered")
        os.mkdir(path_filtered)
        for parent,dirnames,filenames in os.walk(save_path):
            for f in filenames:
                path=os.path.join(parent,f)
                f_read=open(path,'r')
                f_lines=f_read.readlines()
                #first=f_lines[0].split()
                last_number=1122
                idx=[]
                for i,f_lines_l in enumerate(f_lines):
                    f_lines_l=f_lines_l.split()
                    if i==0:
                        b=int(f_lines_l[8])
                        if b != 1:
                            a=a+1
                    if  f_lines_l[8] == last_number:
                        if f_lines_l[13]<=prob:
                            idx.append(i)
                            fd.write(f+'--'+last_number+'--'+'remove'+'\n')
                        else:
                            idx.append(i-1)
                    else:
                        last_number=f_lines_l[8]
                    prob=f_lines_l[13]
                for j in idx[::-1]:
                    f_lines.pop(j)
                    #print(path+'--'+str(j))
                path_w = path.replace("cif_list","cif_filtered")#考虑全局变量
                fw=open(path_w,'w')
                for i in f_lines:
                    fw.write(i)
                fw.close()
                f_read.close()
        fd.close()
        print(a)

    #移除整个断链pdb文件
    def remove_pdb(self):
        idx=[]
        fd=open(path_remove,'a')
        for parent,dirnames,filenames in os.walk(path_filtered): 
            for f in filenames:
                path=os.path.join(parent,f)
                f_read=open(path,'r')
                f_lines=f_read.readlines()
                for i,f_lines_l in enumerate(f_lines):
                    f_lines_l=f_lines_l.split()
                    if i==0:    
                        last_number=int(f_lines_l[8])-1
                    a=int(f_lines_l[8])
                    if  a == last_number+1:
                        last_number=int(f_lines_l[8])
                    else:
                        idx.append(f)
                        fd.write(f+str(last_number)+'\n')
                        f_read.close()
                        break
                f_read.close()
        fd.close()
        for j in idx[::-1]:
            path_d=os.path.join(parent,j)
            if os.path.exists(path_d):
                os.unlink(path_d)       
        print(len(idx))
        
    #删除文件大小为空的cif文件                               
    def delete_none(self):
        path_none=path_filtered.replace("cif_filtered\\","delete_none.txt")
        fw=open(path_none,'a')
        for parent,dirnames,filenames in os.walk(path_filtered):
            for f in filenames:
                path=os.path.join(parent,f)
                size = os.path.getsize(path)
                if size == 0:
                    os.unlink(path)
                    fw.write(f+'\n')
                    print(f)
        
    #删除不常规的氨基酸  
    def delete_acid(self):
        acid=[]
        path_d_acid=path_filtered.replace("cif_filtered\\","delete_acid.txt")
        fw=open(path_d_acid,'a')
        for parent,dirnames,filenames in os.walk(path_filtered): 
            for f in filenames:
                path=os.path.join(parent,f)
                f_read=open(path,'r')
                f_lines=f_read.readlines()
                for i in range(len(f_lines)):
                    if f_lines[i].split()[-4]=='UNK':
                        acid.append(f)
                        print(f)
                        break
                f_read.close()
        for j in acid[::-1]:
            path_d=os.path.join(parent,j)
            if os.path.exists(path_d):
                os.unlink(path_d)
                fw.write(path_d+'\n')
        fw.close()
                    
        
    def anti_acid(self):
        for parent,dirnames,filenames in os.walk('D:\WGAN-GP-tensorflow-master\src\pdb128/'):
            for f in filenames:
                path=os.path.join(parent,f)
                npy_file=np.load(path)
                array=np.array(npy_file)
                for i in range (64):
                    for j in range (64):
                        for k in range (3):
                            if isnan(array[i,j,k]):
                                path=os.path.join(parent,j)
                                if os.path.exists(path):
                                    os.unlink(path)
                                    print(path)
                                    

                    
