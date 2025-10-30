import json
from monai.data import NibabelWriter
# Changed: Replace 'AddChannel' with 'EnsureChannelFirst' (or 'EnsureChannelFirstD' if working with dicts)
# Since 'save_img' operates on a single image array, 'EnsureChannelFirst' is the appropriate replacement.
from monai.transforms import EnsureChannelFirst


def save_json(data, file_path, sort_keys=True):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4, sort_keys=sort_keys)
    print(f'save json to {file_path}')


def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        print(f'load json from {file_path}')
        return data


def save_img(img, img_meta_dict, pth):
    writer = NibabelWriter()
    # Updated: Use EnsureChannelFirst() instead of AddChannel()
    writer.set_data_array(EnsureChannelFirst()(img))
    writer.set_metadata(img_meta_dict)
    writer.write(pth, verbose=True)
