import h5py
import numpy as np
import shutil
from matplotlib import pyplot as plt
from os.path import join, isdir, isfile
from os import mkdir, listdir
import cv2
import os
from tqdm import tqdm

input_dir = 'data'
output_dir = 'data_viz'
fps = 30

if not isdir(input_dir): mkdir(input_dir)
if not isdir(output_dir): mkdir(output_dir)

"""
in each .hdf5 file, there are 
    action (28,)
    depth (128, 128, 1)
    highlight (128, 128, 1)
    ins_seg (128, 128, 1)
    proprioception (20,)
    rgb (128, 128, 3)
    seg (128, 128, 1)
    task_obs (456,)
"""


def download_data(chosen_activities=None):
    """ download one .hdf5 data file for each of the chosen activity (all 100 if not specified) """

    ## 'behavior.txt' is the list of all human demo file names from Behavior website
    lines = [l.replace('\n','') for l in open('behavior.txt', 'r').readlines()]
    tasks = {}  ## maps activity name to demo .hdf5 name, should be 100 activities, and I chose to download the first one
    for line in lines:
        task = line[:line.index('_0_')]
        if task not in tasks: tasks[task] = []
        tasks[task].append(line)

    ## generate script to download the first demo file of the specified tasks 
    downloader = 'downloader1.sh'
    f = open(downloader, 'w')
    for k, v in tasks.items():
        if chosen_activities != None and k not in chosen_activities: continue
        cmd = f'gsutil cp gs://gibsonchallenge/behavior_human_demos_v2/{v[0]} ./data/{k}_0.hdf5\n'
        f.write(cmd)
    f.close()

    return downloader


def make_video_from_folder(activity):
    
    act_dir = join(output_dir, activity)
    video_name = join(output_dir, f'{activity}.mp4')
    indices = [int(img.replace('.jpg','')) for img in listdir(act_dir) if img.endswith(".jpg")]
    if len(indices) == 0: 
        print(activity, 'has 0 frames')
        return
    images = [join(act_dir, f'{i}.jpg') for i in range(min(indices),max(indices)+1)]
    frame = cv2.imread(images[0])
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = cv2.VideoWriter(video_name, fourcc, fps, (width,height))

    for image in images:
        video.write(cv2.imread(image))

    cv2.destroyAllWindows()
    video.release()


def generate_images_videos(len_frames=600):

    files = [f for f in listdir(input_dir) if '.hdf5' in f]
    files.sort()

    for file in tqdm(files):
        activity = file[:file.index('.')]
        act_dir = join(output_dir, activity)

        ## use the change in action and proprioception value to determine when the human has started moving, and only reconstruct the frames after that 
        last_action = None
        started = False
        
        ## skip the activity if the subfolder already exist
        if isdir(act_dir): continue
        mkdir(act_dir)

        hf = h5py.File(join(input_dir, file))
        input_frames = hf['action'].shape[0]
        
        for j in range(len(range(input_frames))):
            
            this_action = np.asarray(hf["action"][j])
            this_proprioception = np.asarray(hf["proprioception"][j])
            
            if not started and j != 0:
                diff_action = np.abs(np.max(this_action - last_action))
                diff_proprioception = np.abs(np.max(this_proprioception - last_proprioception))
                if diff_action > 0.5 and diff_proprioception > 0.5:
                    started = len_frames
                    print(activity, j, started)
                    
            if not started:  
                last_action = np.asarray(hf["action"][j])
                last_proprioception = np.asarray(hf["proprioception"][j])
                continue
            
            img_name = join(act_dir, f'{j}.jpg')
            if not isfile(img_name) and j != 0:
                f = plt.figure(figsize=(9, 6.5)) 
                f.suptitle(f"{activity} ... {j}/{input_frames}", fontsize=16)
                i = 0
                for k, v in hf.items():
                    if len(v[j].shape) >= 2:
                        i += 1
                        f.add_subplot(2, 3, i)
                        plt.imshow(v[j])
                        plt.title(k)
                        plt.axis('off')
                f.add_subplot(2, 3, 6)
                text = f'len(action) = {len(this_action)}\n'
                text += f'abs(diff(action)) = {round( np.abs(np.max(this_action - last_action)), 2)}\n'
                text += f'\nlen(proprioception) = {len(this_proprioception)}\n'
                num = round(float(abs(max(this_proprioception - last_proprioception))), 2)
                text += f'abs(diff(prop)) = {num}\n'
                text += f'\nlen(task_obs) = {len(hf["task_obs"][j])}\n'
                plt.text(0, 0.1, text, fontsize = 11)
                plt.axis('off')
                plt.title('other data')
        #         plt.show()
                plt.savefig(img_name, dpi=100, bbox_inches='tight')  
                plt.close()
                
            last_action = np.asarray(hf["action"][j])
            last_proprioception = np.asarray(hf["proprioception"][j])
            started -= 1
            if started == 0: break
            
        make_video_from_folder(activity)

if __name__ == "__main__":

    chosen_activities = ['waxing_cars_or_other_vehicles', 'cleaning_windows', 'locking_every_window', 'thawing_frozen_food', 'cleaning_the_pool']
    len_frames = 600 ## 20 sec each video for fps=30
    len_frames = 1000000 ## some large number (larger than data length) so you reconstruct the whole video

    ## create a script to download one data file for each activity and rename it /data/{activity}_0.hdf5
    download_script = download_data(chosen_activities=chosen_activities)

    ## download the .hdf5 files to the /data folder
    os.system(f'chmod +x {download_script}')
    os.system(f'./{download_script}')

    ## reconstruct image files to the /data_viz/{activity}_0/{frame_index}.jpg
    ## generate video files to the /data_viz/{activity}_0.mp4
    generate_images_videos(len_frames=len_frames)

