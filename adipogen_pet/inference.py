import os
import argparse
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from network import CGAN

fat_HU = (-140, -30)
def normalization_function(ct):
    ct = ct.astype("float32")
    fat_mean = (fat_HU[1] - fat_HU[0])/2.0
    ct -= (fat_HU[1] - fat_mean)
    ct /= fat_mean
    return ct

def main(input_path, output_path, model_weights, gpu):
    os.environ["CUDA_VISIBLE_DEVICES"]=gpu

    context_levels = 5
    cGAN = CGAN(kernel_size=4, strides=(2, 2), n_slices=context_levels*2+1, fat_masked=True)
    cGAN.load_checkpoint(model_weights)

    ct_image = sitk.ReadImage(input_path)
    ct_arr = normalization_function(np.swapaxes(sitk.GetArrayFromImage(ct_image).astype("int16"), 0, -1))

    pet_arr = np.zeros((128, 128, ct_arr.shape[-1]))
    nslices = ct_arr.shape[-1]
    for i in tqdm(range(context_levels, nslices-context_levels, 1)):
        ct_slice = ct_arr[:, :, i-context_levels:i+context_levels+1]
        prediction = cGAN.predict(ct_slice[np.newaxis, ..., np.newaxis]).numpy()
        pet_arr[:, :, i] = prediction[0, ..., 0]

    pet_arr = np.swapaxes(pet_arr, 0, -1)

    pet_image = sitk.GetImageFromArray(pet_arr)
    pet_image.SetDirection(ct_image.GetDirection())
    pet_image.SetOrigin(ct_image.GetOrigin())
    pet_image.SetSpacing(ct_image.GetSpacing()*np.array([4.0, 4.0, 1.0]))

    sitk.WriteImage(pet_image, output_path, useCompression=True)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='airway inference script.')
    parser.add_argument('--input', type=str, help='Input path with CT scans.', required=True)
    parser.add_argument('--output', type=str, help='Output path for the airway segmentation.', required=True)
    parser.add_argument('--model', type=str, help='Model file containing the weights.', required=True)
    parser.add_argument('--gpu', type=str, help='GPU to use.', required=False)
    
    parser.set_defaults(gpu="")
    args = parser.parse_args()
    
    print("input: %s" % args.input)
    print("output: %s" % args.output)
    print("model: %s" % args.model)
    print("gpu: %s" % args.gpu)
    
    main(args.input, args.output, args.model, args.gpu)