import argparse
import cv2
import glob
import numpy as np
import os
import torch
from basicsr.utils import imwrite

from gfpgan import GFPGANer


def main():
    """Inference demo for GFPGAN (for users).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--input',
        type=str,
        default='inputs/whole_imgs',
        help='Input image or folder. Default: inputs/whole_imgs')
    parser.add_argument('-o', '--output', type=str, default='results', help='Output folder. Default: results')
    # we use version to select models, which is more user-friendly
    parser.add_argument(
        '-v', '--version', type=str, default='1.4', help='GFPGAN model version. Option: 1 | 1.2 | 1.3 | 1.4. Default: 1.4')
    parser.add_argument(
        '-s', '--upscale', type=float, default=2, help='The final upsampling scale of the image. Default: 2')
    parser.add_argument(
        '-tw', '--target_width', type=int, help='The target image width after upscaling. Only one of target_width or target_height can be provided, or neither can be provided.')
    parser.add_argument(
        '-th', '--target_height', type=int, help='The target image height after upscaling. Only one of target_width or target_height can be provided, or neither can be provided.')

    parser.add_argument(
        '--bg_upsampler', type=str, default='realesrgan', help='background upsampler. Default: realesrgan')
    parser.add_argument(
        '--bg_tile',
        type=int,
        default=400,
        help='Tile size for background sampler, 0 for no tile during testing. Default: 400')
    parser.add_argument('--suffix', type=str, default=None, help='Suffix of the restored faces')
    parser.add_argument('--only_center_face', action='store_true', help='Only restore the center face')
    parser.add_argument('--aligned', action='store_true', help='Input are aligned faces')
    parser.add_argument('-sub', '--search_subfolders', action='store_true', help='Decide whether to search subfolders.')
    parser.add_argument(
        '--ext',
        type=str,
        default='auto',
        help='Image extension. Options: auto | jpg | png, auto means using the same extension as inputs. Default: auto')
    parser.add_argument('-w', '--weight', type=float, default=0.5, help='Adjustable weights.')
    args = parser.parse_args()

    args = parser.parse_args()

    # ------------------------ input & output ------------------------
    if args.input.endswith('/'):
        args.input = args.input[:-1]
    if os.path.isfile(args.input):
        img_list = [args.input]
    else:
        img_list = sorted(glob.glob(os.path.join(args.input, '*')))

    os.makedirs(args.output, exist_ok=True)

    # ------------------------ set up background upsampler ------------------------
    if args.bg_upsampler == 'realesrgan':
        if not torch.cuda.is_available():  # CPU
            import warnings
            warnings.warn('The unoptimized RealESRGAN is slow on CPU. We do not use it. '
                          'If you really want to use it, please modify the corresponding codes.')
            bg_upsampler = None
        else:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            bg_upsampler = RealESRGANer(
                scale=2,
                model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
                model=model,
                tile=args.bg_tile,
                tile_pad=10,
                pre_pad=0,
                half=True)  # need to set False in CPU mode
    else:
        bg_upsampler = None

    # ------------------------ set up GFPGAN restorer ------------------------
    if args.version == '1':
        arch = 'original'
        channel_multiplier = 1
        model_name = 'GFPGANv1'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/GFPGANv1.pth'
    elif args.version == '1.2':
        arch = 'clean'
        channel_multiplier = 2
        model_name = 'GFPGANCleanv1-NoCE-C2'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth'
    elif args.version == '1.3':
        arch = 'clean'
        channel_multiplier = 2
        model_name = 'GFPGANv1.3'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth'
    elif args.version == '1.4':
        arch = 'clean'
        channel_multiplier = 2
        model_name = 'GFPGANv1.4'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'
    elif args.version == 'RestoreFormer':
        arch = 'RestoreFormer'
        channel_multiplier = 2
        model_name = 'RestoreFormer'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth'
    else:
        raise ValueError(f'Wrong model version {args.version}.')

    # determine model paths
    model_path = os.path.join('experiments/pretrained_models', model_name + '.pth')
    if not os.path.isfile(model_path):
        model_path = os.path.join('gfpgan/weights', model_name + '.pth')
    if not os.path.isfile(model_path):
        # download pre-trained models from url
        model_path = url

    restorer = GFPGANer(
        model_path=model_path,
        upscale=args.upscale,
        target_width=args.target_width,
        target_height=args.target_height,
        arch=arch,
        channel_multiplier=channel_multiplier,
        bg_upsampler=bg_upsampler)

    # ------------------------ restore ------------------------
    processing(args, restorer, img_list)

    print(f'Results are in the [{args.output}] folder.')


def processing(args, restorer, img_list: list[str], folder_name: str =""):
    if folder_name:
        print(f"\n________________________________________________\nCurrently processing the subfolder: {folder_name}\n________________________________________________\n")

    imgsLen = len(img_list)
    for idx, img_path in enumerate(img_list):
        # read image
        try:
            img_name = os.path.basename(img_path)
            if os.path.isdir(img_path) and args.search_subfolders:
                print(f"path {img_path}")
                processing(args, restorer, sorted(glob.glob(os.path.join(img_path, '*'))), img_name)
                continue

            print(f'[{idx}/{imgsLen}] Processing {img_name} ...')
            basename, ext = os.path.splitext(img_name)
            input_img = cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_COLOR) #input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            
            if input_img is None:
                print("  cv2.imread result None, continue")
                continue
        except Exception as err:
            print("  cv2.imread result Exception: " + str(err))
            continue
        restorer_enhance(args, restorer, idx, input_img, basename, ext, folder_name)


def restorer_enhance(args, restorer: GFPGANer, idx: int, input_img: cv2.typing.MatLike, basename: str, ext: str, folder_name: str =""):
    # restore faces and background if necessary
    cropped_faces, restored_faces, restored_img = restorer.enhance(
        input_img,
        has_aligned=args.aligned,
        only_center_face=args.only_center_face,
        paste_back=True,
        weight=args.weight)
    
    # save faces
    for idx, (cropped_face, restored_face) in enumerate(zip(cropped_faces, restored_faces)):
        # save cropped face
        save_crop_path = os.path.join(args.output, folder_name, 'cropped_faces', f'{basename}_{idx:02d}.png')
        imwriteUTF8(cropped_face, save_crop_path, '.png') #imwrite(cropped_face, save_crop_path)
        # save restored face
        if args.suffix is not None:
            save_face_name = f'{basename}_{idx:02d}_{args.suffix}.png'
        else:
            save_face_name = f'{basename}_{idx:02d}.png'
        save_restore_path = os.path.join(args.output, folder_name, 'restored_faces', save_face_name)
        imwriteUTF8(restored_face, save_restore_path, '.png') #imwrite(restored_face, save_restore_path)
        
        # save comparison image
        cmp_img = np.concatenate((cropped_face, restored_face), axis=1)
        imwriteUTF8(cmp_img, os.path.join(args.output, folder_name, 'cmp', f'{basename}_{idx:02d}.png'), '.png') #imwrite(cmp_img, os.path.join(args.output, 'cmp', f'{basename}_{idx:02d}.png'))
    
    # save restored img
    if restored_img is not None:
        if args.ext == 'auto':
            extension = ext[1:]
        else:
            extension = args.ext
    
        if args.suffix is not None:
            save_restore_path = os.path.join(args.output, folder_name, 'restored_imgs', f'{basename}_{args.suffix}.{extension}')
        else:
            save_restore_path = os.path.join(args.output, folder_name, 'restored_imgs', f'{basename}.{extension}')
        imwriteUTF8(restored_img, save_restore_path, f'.{extension}') #imwrite(restored_img, save_restore_path)

def imwriteUTF8(image, save_path, extension):
    is_success, im_buf_arr = cv2.imencode(extension, image)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    im_buf_arr.tofile(save_path) #imwrite(cropped_face, save_path)

if __name__ == '__main__':
    main()
