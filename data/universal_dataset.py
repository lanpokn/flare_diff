import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset, make_dataset_all, make_dataset_all_text, make_dataset_3, make_dataset_5, make_dataset_6, make_dataset_4, make_dataset_2
from PIL import Image
from pathlib import Path
import numpy as np
import random
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import Augmentor
import cv2
import torch
from torch.distributions import Normal
import io
class RandomGammaCorrection(object):
	def __init__(self, gamma = None):
		self.gamma = gamma
	def __call__(self,image):
		if self.gamma == None:
			# more chances of selecting 0 (original image)
			gammas = [0.5,1,2]
			self.gamma = random.choice(gammas)
			return TF.adjust_gamma(image, self.gamma, gain=1)
		elif isinstance(self.gamma,tuple):
			gamma=random.uniform(*self.gamma)
			return TF.adjust_gamma(image, gamma, gain=1)
		elif self.gamma == 0:
			return image
		else:
			return TF.adjust_gamma(image,self.gamma,gain=1)
def remove_background(image):
	#the input of the image is PIL.Image form with [H,W,C]
	image=np.float32(np.array(image))
	_EPS=1e-7
	rgb_max=np.max(image,(0,1))
	rgb_min=np.min(image,(0,1))
	image=(image-rgb_min)*rgb_max/(rgb_max-rgb_min+_EPS)
	image=torch.from_numpy(image)
	return image
def resize_to_256(image):
    # 如果图像没有 batch 维度，则添加一个（interpolate 需要 4D 输入）
    if image.dim() == 3:
        image = image.unsqueeze(0)  # 变为 [1, C, H, W]
    
    # 使用双线性插值调整大小
    resized_image = F.interpolate(image, size=(256, 256), mode='bilinear', align_corners=False)
    
    # 去掉 batch 维度
    resized_image = resized_image.squeeze(0)  # 变回 [C, H, W]
    return resized_image
def tensor_to_pil(image_tensor):
    # 将张量转换为 [H, W, C] 格式，并转换为 numpy 数组
    image_tensor = image_tensor.cpu().detach()  # 确保张量在 CPU 上
    if image_tensor.dim() == 3 and image_tensor.shape[0] == 3:  # [C, H, W]
        image_tensor = image_tensor.permute(1, 2, 0)  # 变为 [H, W, C]
    image_np = image_tensor.numpy()  # 转换为 numpy 数组
    image_np = (image_np * 255).astype('uint8')  # 转换为 0-255 的 uint8 格式
    return Image.fromarray(image_np)  # 转换为 PIL 图像

# 保存图像到文件夹
def save_image_to_folder(image, folder, filename):
    if not os.path.exists(folder):
        os.makedirs(folder)  # 如果文件夹不存在，则创建
    filepath = os.path.join(folder, filename)
    image.save(filepath)  # 保存图像
    print(f"Saved: {filepath}")
    return filepath
def tensor_to_pil(image_tensor):
    # 将张量转换为 [H, W, C] 格式，并转换为 numpy 数组
    image_tensor = image_tensor.cpu().detach()  # 确保张量在 CPU 上
    if image_tensor.dim() == 3 and image_tensor.shape[0] == 3:  # [C, H, W]
        image_tensor = image_tensor.permute(1, 2, 0)  # 变为 [H, W, C]
    image_np = image_tensor.numpy()  # 转换为 numpy 数组
    image_np = (image_np * 255).astype('uint8')  # 转换为 0-255 的 uint8 格式
    return Image.fromarray(image_np)  # 转换为 PIL 图像

# 将 PIL 图像保存为 JPEG 格式到内存中
def pil_to_jpeg_bytes(pil_image):
    img_byte_arr = io.BytesIO()  # 创建一个内存中的字节流
    pil_image.save(img_byte_arr, format='JPEG')  # 将 PIL 图像保存为 JPEG 格式到字节流
    img_byte_arr.seek(0)  # 将指针移动到字节流的开头
    return img_byte_arr

class AlignedDataset_all(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt, image_size, augment_flip=True, equalizeHist=True, crop_patch=True, generation=False, task=None):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        #         transform_base:
        #   img_size: 512
        # transform_flare:
        #   scale_min: 0.8
        #   scale_max: 1.5
        #   translate: 300
        #   shear: 20
        self.img_size = 512
        self.transform_base=transforms.Compose([transforms.RandomCrop((self.img_size,self.img_size),pad_if_needed=True,padding_mode='reflect'),
							  transforms.RandomHorizontalFlip(),
							  transforms.RandomVerticalFlip()
                              ])

        self.transform_flare=transforms.Compose([transforms.RandomAffine(degrees=(0,360),scale=(0.8,1.5),translate=(300/1440,300/1440),shear=(-20,20)),
									transforms.CenterCrop((self.img_size,self.img_size)),
									transforms.RandomHorizontalFlip(),
									transforms.RandomVerticalFlip()
									])
        
        
        BaseDataset.__init__(self, opt)
        self.equalizeHist = equalizeHist
        self.augment_flip = augment_flip
        self.crop_patch = crop_patch
        self.generation = generation
        self.image_size = image_size
        self.opt = opt
        #origin----------------------------------------------------------------------------------------------------------
        self.dir_Arain = os.path.join(opt.dataroot, 'rain1400/' + opt.phase + '/rainy_image')
        self.dir_Brain = os.path.join(opt.dataroot, 'rain1400/' + opt.phase + '/ground_truth')
        self.dir_Alsrw = os.path.join(opt.dataroot, 'LSRW/' + opt.phase + '/low')
        self.dir_Blsrw = os.path.join(opt.dataroot, 'LSRW/' + opt.phase + '/high')
        self.dir_Alol = os.path.join(opt.dataroot, 'LOL/' + opt.phase + '/low')
        self.dir_Blol = os.path.join(opt.dataroot, 'LOL/' + opt.phase + '/high')
        self.dir_Aflare = os.path.join(opt.dataroot, 'flare/' + opt.phase + '/scene')
        self.dir_Bflare = os.path.join(opt.dataroot, 'flare/' + opt.phase + '/Scatter')
        self.C_paths = ''
        self.D_paths = ''
        if opt.phase == 'train':
            self.dir_Asnow = os.path.join(opt.dataroot, 'Snow100K/' + opt.phase + '/synthetic')
            self.dir_Bsnow = os.path.join(opt.dataroot, 'Snow100K/' + opt.phase + '/gt')
            self.dir_Arain_syn = os.path.join(opt.dataroot, 'syn_rain/' + opt.phase + '/input')
            self.dir_Brain_syn = os.path.join(opt.dataroot, 'syn_rain/' + opt.phase + '/target')
            self.dir_Ablur = os.path.join(opt.dataroot, 'Deblur/' + opt.phase + '/input')
            self.dir_Bblur = os.path.join(opt.dataroot, 'Deblur/' + opt.phase + '/target')

            flog_prefix = os.path.join(opt.dataroot, 'RESIDE/OTS_ALPHA/')
            self.dir_Afog = flog_prefix + 'haze/OTS'
            self.dir_Bfog = flog_prefix + 'clear/clear_images'
            
            self.dir_Cflare = os.path.join(opt.dataroot, 'flare/' + opt.phase + '/Reflective')
            self.dir_Dflare = os.path.join(opt.dataroot, 'flare/' + opt.phase + '/light')

        else:
            self.dir_Asnow = os.path.join(opt.dataroot, 'Snow100K/' + opt.phase + '/Snow100K-S/synthetic') #Snow100K-S Snow100K-L
            self.dir_Bsnow = os.path.join(opt.dataroot, 'Snow100K/' + opt.phase + '/Snow100K-S/gt')
            # self.dir_Asnow = os.path.join(opt.dataroot, 'Snow100K/' + 'realistic') #Snow100K-S Snow100K-L
            # self.dir_Bsnow = os.path.join(opt.dataroot, 'Snow100K/' + 'realistic')
            self.dir_Arain_syn = os.path.join(opt.dataroot, 'syn_rain/' + opt.phase + '/Test2800/input') #Rain100H, Rain100L, Test100, Test1200,
            self.dir_Brain_syn = os.path.join(opt.dataroot, 'syn_rain/' + opt.phase + '/Test2800/target')   #Test2800
            self.dir_Ablur = os.path.join(opt.dataroot, 'Deblur/' + opt.phase + '/GoPro/input')  #GoPro, HIDE,  Reblur_J, Reblur_R
            self.dir_Bblur = os.path.join(opt.dataroot, 'Deblur/' + opt.phase + '/GoPro/target')
            self.dir_Afog = os.path.join(opt.dataroot, 'RESIDE/SOTS/outdoor/hazy')
            self.dir_Bfog = os.path.join(opt.dataroot, 'RESIDE/SOTS/outdoor/gt')
            self.dir_Aasd = os.path.join(opt.dataroot, 'temp')
            self.dir_Basd = os.path.join(opt.dataroot, 'temp')
        
        #test
        if task == 'light':
            if opt.phase == 'train':
                self.A_paths = sorted(make_dataset_2(self.dir_Alol, self.dir_Alsrw, opt.max_dataset_size))
                self.B_paths = sorted(make_dataset_2(self.dir_Blol, self.dir_Blsrw, opt.max_dataset_size))
            else:
                self.A_paths = sorted(make_dataset(self.dir_Alol, opt.max_dataset_size))
                self.B_paths = sorted(make_dataset(self.dir_Blol, opt.max_dataset_size))
        elif task == 'light_only':
            self.A_paths = sorted(make_dataset(self.dir_Alol, opt.max_dataset_size))
            self.B_paths = sorted(make_dataset(self.dir_Blol, opt.max_dataset_size))
        elif task == 'rain':
            self.A_paths = sorted(make_dataset(self.dir_Arain_syn, opt.max_dataset_size))
            self.B_paths = sorted(make_dataset(self.dir_Brain_syn, opt.max_dataset_size))
        elif task == 'snow':
            self.A_paths = sorted(make_dataset(self.dir_Asnow, opt.max_dataset_size))
            self.B_paths = sorted(make_dataset(self.dir_Bsnow, opt.max_dataset_size))
        elif task == 'blur':
            self.A_paths = sorted(make_dataset(self.dir_Ablur, opt.max_dataset_size))
            self.B_paths = sorted(make_dataset(self.dir_Bblur, opt.max_dataset_size))
        elif task == 'fog':
            self.A_paths = sorted(make_dataset(self.dir_Afog, opt.max_dataset_size))
            self.B_paths = sorted(make_dataset(self.dir_Bfog, opt.max_dataset_size))
        elif task == '4':
            self.A_paths = sorted(make_dataset_4(self.dir_Arain_syn, self.dir_Alsrw, self.dir_Alol, self.dir_Asnow, opt.max_dataset_size))
            self.B_paths = sorted(make_dataset_4(self.dir_Brain_syn, self.dir_Blsrw, self.dir_Blol, self.dir_Bsnow, opt.max_dataset_size))
        elif task == '5':
            self.A_paths = sorted(make_dataset_5(self.dir_Arain_syn, self.dir_Alsrw, self.dir_Alol, self.dir_Asnow, self.dir_Ablur, opt.max_dataset_size))
            self.B_paths = sorted(make_dataset_5(self.dir_Brain_syn, self.dir_Blsrw, self.dir_Blol, self.dir_Bsnow, self.dir_Bblur, opt.max_dataset_size))
        elif task == '6':
            self.A_paths = sorted(make_dataset_6(self.dir_Arain_syn, self.dir_Alol, self.dir_Asnow, self.dir_Ablur, self.dir_Afog, opt.max_dataset_size))
            self.B_paths = sorted(make_dataset_6(self.dir_Brain_syn, self.dir_Blol, self.dir_Bsnow, self.dir_Bblur, self.dir_Bfog, opt.max_dataset_size))
        elif task == 'flare':
            self.A_paths = sorted(make_dataset(self.dir_Aflare, opt.max_dataset_size))
            self.B_paths = sorted(make_dataset(self.dir_Bflare, opt.max_dataset_size))
            if opt.phase == 'train':
                self.C_paths = sorted(make_dataset(self.dir_Cflare, opt.max_dataset_size))
                self.D_paths = sorted(make_dataset(self.dir_Dflare, opt.max_dataset_size))
        else:
            self.A_paths = sorted(make_dataset(self.dir_Aasd, opt.max_dataset_size))
            self.B_paths = sorted(make_dataset(self.dir_Basd, opt.max_dataset_size))
    

        self.A_size = len(self.A_paths)  # get the size of dataset A
        print(self.A_size)
        self.B_size = len(self.B_paths)  # get the size of dataset B
        print(self.B_size)
        self.C_size = len(self.C_paths)
        self.D_size = len(self.D_paths)
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        if self.opt.phase == 'train':
            # TODO random here
            # flare_index = random.randint(0, self.B_size - 1)
            # A_path = self.A_paths[random.randint(0, self.A_size - 1)]  # make sure index is within then range
            # B_path = self.B_paths[flare_index]
            # C_path = self.C_paths[random.randint(0, self.C_size - 1)]
            # D_path = self.D_paths[flare_index]
            A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
            B_path = self.B_paths[index % self.B_size]
            C_path = self.C_paths[index % self.C_size]
            D_path = self.D_paths[index % self.D_size]
            #TODO,read scene reflective scatter, and use them to generate
            base_img = Image.open(A_path).convert('RGB') #condition
            flare_img = Image.open(B_path).convert('RGB') #gt
            reflective_img = Image.open(C_path).convert('RGB')
            light_img = Image.open(D_path).convert('RGB')

        else:
            # read a image given a random integer index
            A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
            B_path = self.B_paths[index % self.B_size]

        # if 'LOL' in A_path or 'LSRW' in A_path:
        #     condition = cv2.cvtColor(np.asarray(condition), cv2.COLOR_RGB2BGR)
        #     gt = cv2.cvtColor(np.asarray(gt), cv2.COLOR_RGB2BGR)
        
        #     if self.crop_patch:
        #         gt, condition = self.get_patch([gt, condition], self.image_size)
        #     if 'LOL' in A_path:
        #         condition = self.cv2equalizeHist(condition) if self.equalizeHist else condition
        #     else:
        #         condition = condition

        #     images = [[gt, condition]]
        #     p = Augmentor.DataPipeline(images)
        #     if self.augment_flip:
        #         p.flip_left_right(1)
        #     g = p.generator(batch_size=1)
        #     augmented_images = next(g)
        #     gt = cv2.cvtColor(augmented_images[0][0], cv2.COLOR_BGR2RGB)
        #     condition = cv2.cvtColor(augmented_images[0][1], cv2.COLOR_BGR2RGB)
        
        #     gt = self.to_tensor(gt)
        #     condition = self.to_tensor(condition)
        # else:

        if self.opt.phase == 'train':
            #TODO 此处写死transform方法，仿照flare7kpp中写，只保证最后是256即可
            #最终还要在这里实现加法生成flare和gt组合图。
            #但这样还有一个大问题，数量对不上，因为不是随机取的。。
            #只处理flare的话，干脆不要用get_transform了，用给定的transform，读取完就用flare7K中的方法，就结束了
            # w, h = condition.size

            # condition = A_transform(condition)
            # gt = B_transform(gt)
            # reflective = C_transform(reflective)
            # if h < 512 or w < 512:
            #     osize = [512, 512]
            #     resi = transforms.Resize(osize, transforms.InterpolationMode.BICUBIC)
            #     condition = resi(condition)
            #     gt = resi(gt)
            #     reflective = resi(reflective)
            
            gamma=np.random.uniform(1.8,2.2)
            to_tensor=transforms.ToTensor()
            adjust_gamma=RandomGammaCorrection(gamma)
            adjust_gamma_reverse=RandomGammaCorrection(1/gamma)
            color_jitter=transforms.ColorJitter(brightness=(0.8,3),hue=0.0)
            if self.transform_base is not None:
                base_img=to_tensor(base_img)
                base_img=adjust_gamma(base_img)
                base_img=self.transform_base(base_img)
            else:
                base_img=to_tensor(base_img)
                base_img=adjust_gamma(base_img)
            sigma_chi=0.01*np.random.chisquare(df=1)
            base_img=Normal(base_img,sigma_chi).sample()
            gain=np.random.uniform(0.5,1.2)
            base_img=gain*base_img
            base_img=torch.clamp(base_img,min=0,max=1)


            light_img=to_tensor(light_img)
            light_img=adjust_gamma(light_img)
            flare_img=to_tensor(flare_img)
            flare_img=adjust_gamma(flare_img)
            


            if self.transform_flare is not None:
                flare_merge=torch.cat((flare_img, light_img), dim=0)
                flare_merge=self.transform_flare(flare_merge)
                

            flare_img, light_img = torch.split(flare_merge, 3, dim=0)
            flare_img=color_jitter(flare_img)
    
            reflective_img=to_tensor(reflective_img)
            reflective_img=adjust_gamma(reflective_img)
            reflective_img=self.transform_flare(reflective_img)
            flare_img = torch.clamp(flare_img+reflective_img,min=0,max=1)

            flare_img=remove_background(flare_img)
    

            #flare blur
            #mannuly blur here, interteresting
            blur_transform=transforms.GaussianBlur(21,sigma=(0.1,3.0))
            flare_img=blur_transform(flare_img)
            #flare_img=flare_img+flare_DC_offset
            # #TODO, may need test
            # distortion = np.random.uniform(-1.5,0)
            # if np.random.uniform(0,1.0) > 0.5:
            #     flare_img = undistort_image(flare_img,k1 = distortion)
            #     light_img = undistort_image(light_img,k1 = distortion)

            flare_img=torch.clamp(flare_img,min=0,max=1)           
            

            merge_img = flare_img+base_img
            merge_img=torch.clamp(merge_img,min=0,max=1)
            base_img=base_img+light_img
            base_img=torch.clamp(base_img,min=0,max=1)
            flare_img=flare_img-light_img
            flare_img=torch.clamp(flare_img,min=0,max=1)
            # AE_gain=np.random.uniform(0.95,1.0)
            # AE_gain=1
            # flare_img=flare_img+AE_gain*base_img
            # # the artifact on the lens will also cause the scene to become unclear
            blur_transform=transforms.GaussianBlur(3,sigma=(0.01,0.5))

            merge_img=torch.clamp(merge_img,min=0,max=1)
            base_img=base_img+light_img
            base_img=torch.clamp(base_img,min=0,max=1)
            flare_img=flare_img-light_img
            flare_img=torch.clamp(flare_img,min=0,max=1)
            

            condition = adjust_gamma_reverse(merge_img)
            gt = adjust_gamma_reverse(base_img)
            # resized_condition = resize_to_256(condition)
            # resized_gt = resize_to_256(gt)
            # transform_params = get_params(self.opt, condition.size)
            # A_transform = get_transform(self.opt, transform_params, grayscale=False)
            # B_transform = get_transform(self.opt, transform_params, grayscale=False)
            # condition = A_transform(condition)
            # gt = B_transform(gt)
            # 将调整大小后的张量转换为 PIL 图像
            # pil_condition = tensor_to_pil(condition)
            # pil_gt = tensor_to_pil(gt)

            # # 保存图像到文件夹
            # output_folder = "debug_images"  # 保存图像的文件夹
            # A_path = save_image_to_folder(pil_condition, output_folder, "condition"+str(index)+".jpg")
            # B_path = save_image_to_folder(pil_gt, output_folder, "gt"+str(index)+".jpg")
            

        else:
            # no change here
            condition = Image.open(B_path).convert('RGB')
            gt = Image.open(A_path).convert('RGB')
            w, h = condition.size
            transform_params = get_params(self.opt, condition.size)
            A_transform = get_transform(self.opt, transform_params, grayscale=False)
            B_transform = get_transform(self.opt, transform_params, grayscale=False)
            condition = A_transform(condition)
            gt = B_transform(gt)
            
                
        return {'adap': condition, 'gt': gt, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return max(self.A_size, self.B_size)
    
    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist

        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                return [p for ext in self.exts for p in Path(f'{flist}').glob(f'**/*.{ext}')]

            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                except:
                    return [flist]
        return []

    def cv2equalizeHist(self, img):
        (b, g, r) = cv2.split(img)
        b = cv2.equalizeHist(b)
        g = cv2.equalizeHist(g)
        r = cv2.equalizeHist(r)
        img = cv2.merge((b, g, r))
        return img

    def to_tensor(self, img):
        img = Image.fromarray(img)  # returns an image object.
        img_t = TF.to_tensor(img).float()
        return img_t

    def load_name(self, index, sub_dir=False):
        if self.condition:
            # condition
            name = self.input[index]
            if sub_dir == 0:
                return os.path.basename(name)
            elif sub_dir == 1:
                path = os.path.dirname(name)
                sub_dir = (path.split("/"))[-1]
                return sub_dir+"_"+os.path.basename(name)

    def get_patch(self, image_list, patch_size):
        i = 0
        h, w = image_list[0].shape[:2]
        rr = random.randint(0, h-patch_size)
        cc = random.randint(0, w-patch_size)
        for img in image_list:
            image_list[i] = img[rr:rr+patch_size, cc:cc+patch_size, :]
            i += 1
        return image_list

    def pad_img(self, img_list, patch_size, block_size=8):
        i = 0
        for img in img_list:
            img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            h, w = img.shape[:2]
            bottom = 0
            right = 0
            if h < patch_size:
                bottom = patch_size-h
                h = patch_size
            if w < patch_size:
                right = patch_size-w
                w = patch_size
            bottom = bottom + (h // block_size) * block_size + \
                (block_size if h % block_size != 0 else 0) - h
            right = right + (w // block_size) * block_size + \
                (block_size if w % block_size != 0 else 0) - w
            img_list[i] = cv2.copyMakeBorder(
                img, 0, bottom, 0, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            i += 1
        return img_list

    def get_pad_size(self, index, block_size=8):
        img = Image.open(self.input[index])
        patch_size = self.image_size
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]
        bottom = 0
        right = 0
        if h < patch_size:
            bottom = patch_size-h
            h = patch_size
        if w < patch_size:
            right = patch_size-w
            w = patch_size
        bottom = bottom + (h // block_size) * block_size + \
            (block_size if h % block_size != 0 else 0) - h
        right = right + (w // block_size) * block_size + \
            (block_size if w % block_size != 0 else 0) - w
        return [bottom, right]
