import os 
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)



def make_imageclip_dataset(dir, nframes, class_to_idx, vid_diverse_sampling, split='all'):
    """
    TODO: add xflip
    """
    def _sort(path):
        return sorted(os.listdir(path))

    images = []
    n_video = 0
    n_clip = 0
    
    dir_list = sorted(os.listdir(dir))

    for target in dir_list:
        if split == 'train':
            if 'val' in target: dir_list.remove(target)
            if 'test' in target: dir_list.remove(target)
        elif split == 'val' or split == 'test':
            if 'train' in target: dir_list.remove(target)
            if 'full_images' in target: dir_list.remove(target)
    
    for target in dir_list:
        if os.path.isdir(os.path.join(dir,target))==True:
            n_video +=1
            subfolder_path = os.path.join(dir, target)
            for subsubfold in sorted(os.listdir(subfolder_path) ):
                if os.path.isdir(os.path.join(subfolder_path, subsubfold) ):
                    subsubfolder_path = os.path.join(subfolder_path, subsubfold)
                    i = 1

                    if nframes > 0 and vid_diverse_sampling:
                        n_clip += 1

                        item_frames_0 = []
                        item_frames_1 = []
                        item_frames_2 = []
                        item_frames_3 = []

                        for fi in _sort(subsubfolder_path):
                            if is_image_file(fi):
                                file_name = fi
                                file_path = os.path.join(subsubfolder_path, file_name)
                                item = (file_path, class_to_idx[target])

                                if i % 4 == 0:
                                    item_frames_0.append(item)
                                elif i % 4 == 1:
                                    item_frames_1.append(item)
                                elif i % 4 == 2:
                                    item_frames_2.append(item)
                                else:
                                    item_frames_3.append(item)

                                if i %nframes == 0 and i > 0:
                                    images.append(item_frames_0) # item_frames is a list containing n frames.
                                    images.append(item_frames_1) # item_frames is a list containing n frames.
                                    images.append(item_frames_2) # item_frames is a list containing n frames.
                                    images.append(item_frames_3) # item_frames is a list containing n frames.
                                    item_frames_0 = []
                                    item_frames_1 = []
                                    item_frames_2 = []
                                    item_frames_3 = []

                                i = i+1
                    else:
                        item_frames = []
                        for fi in _sort(subsubfolder_path):
                            if is_image_file(fi):
                                # fi is an image in the subsubfolder
                                file_name = fi
                                file_path = os.path.join(subsubfolder_path, file_name)
                                item = (file_path, class_to_idx[target])
                                item_frames.append(item)
                                if i % nframes == 0 and i > 0:
                                    images.append(item_frames)  # item_frames is a list containing 32 frames.
                                    item_frames = []
                                i = i + 1
            
    return images

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx



def make_imageclip_dataset1(dir, nframes, class_to_idx, vid_diverse_sampling, split='all'):
    """
    TODO: add xflip
    """
    def _sort(path):
        return sorted(os.listdir(path))

    images = []
    n_video = 0
    n_clip = 0
    
    dir_list = sorted(os.listdir(dir))
    for target in dir_list:
        if split == 'train':
            if 'val' in target: dir_list.remove(target)
            if 'test' in target: dir_list.remove(target)
        elif split == 'val' or split == 'test':
            if 'train' in target: dir_list.remove(target)
            if 'full_images' in target: dir_list.remove(target)
    
    for target in dir_list:
        if os.path.isdir(os.path.join(dir,target))==True:
            n_video +=1
            subfolder_path = os.path.join(dir, target)
            for subsubfold in sorted(os.listdir(subfolder_path) ):
                if os.path.isdir(os.path.join(subfolder_path, subsubfold) ):
                    subsubfolder_path = os.path.join(subfolder_path, subsubfold)
                    i = 1

                    if nframes > 0 and vid_diverse_sampling:
                        n_clip += 1

                        item_frames_0 = []
                        item_frames_1 = []
                        item_frames_2 = []
                        item_frames_3 = []

                        for fi in _sort(subsubfolder_path):
                            if is_image_file(fi):
                                file_name = fi
                                file_path = os.path.join(subsubfolder_path, file_name)
                                item = (file_path, class_to_idx[target])

                                if i % 4 == 0:
                                    item_frames_0.append(item)
                                elif i % 4 == 1:
                                    item_frames_1.append(item)
                                elif i % 4 == 2:
                                    item_frames_2.append(item)
                                else:
                                    item_frames_3.append(item)

                                if i %nframes == 0 and i > 0:
                                    images.append(item_frames_0) # item_frames is a list containing n frames.
                                    images.append(item_frames_1) # item_frames is a list containing n frames.
                                    images.append(item_frames_2) # item_frames is a list containing n frames.
                                    images.append(item_frames_3) # item_frames is a list containing n frames.
                                    item_frames_0 = []
                                    item_frames_1 = []
                                    item_frames_2 = []
                                    item_frames_3 = []

                                i = i+1
                    else:
                        item_frames = []
                        for fi in _sort(subsubfolder_path):
                            if is_image_file(fi):
                                # fi is an image in the subsubfolder
                                file_name = fi
                                file_path = os.path.join(subsubfolder_path, file_name)
                                item = (file_path, class_to_idx[target])
                                item_frames.append(item)
                                if i % nframes == 0 and i > 0:
                                    images.append(item_frames)  # item_frames is a list containing 32 frames.
                                    item_frames = []
                                i = i + 1
            
    return images