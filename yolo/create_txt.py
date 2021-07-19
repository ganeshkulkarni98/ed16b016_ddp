import pandas as pd
import os

df_1 = pd.read_csv(r"/lfs/usrhome/btech/ed16b016/scratch/project/yolo/labels_7_8_9_10.csv")

def save_labels(name,class_id,row,DATADIR=r"/lfs/usrhome/btech/ed16b016/scratch/project/yolo/VinBig_7_8_9_10"):
    filepath=os.path.join(DATADIR,"{}.txt".format(name))
    x,y,w,h=row
    #x=x/1024
    #y=y/1024
    #w=w/1024
    #h=h/1024
    with open(filepath,'a') as f:
        if class_id == 4:
            f.close()
        else:
            line="{} {} {} {} {}".format(class_id,x,y,w,h)
            f.write('%s\n' % line)
            f.close()
        
for i in range(len(df_1)):
    save_labels(df_1['image_id'][i],int(df_1['class_id'][i]),[df_1['x'][i],df_1['y'][i],df_1['w'][i],df_1['h'][i]])