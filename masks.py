
from PIL import Image
import os


class CityscapesDataset(Dataset):
    def __init__(self, split, root_dir, target_type='semantic', mode='fine', transform=None, eval=False):
        self.transform = transform

        self.split = split
        self.yLabel_list = []
        self.XImg_list = []
        self.eval = eval

        # Preparing a list of all labelTrainIds rgb and
        # ground truth images. Setting relabbelled=True is recommended.

        self.label_path = os.path.join(os.getcwd(), root_dir + '/' + self.split)
        self.rgb_path = os.path.join(os.getcwd(), root_dir + '/leftImg8bit/' + self.split)
        city_list = os.listdir(self.label_path)
        for city in city_list:
            temp = os.listdir(self.label_path + '/' + city)
            list_items = temp.copy()

            # 19-class label items being filtered
            for item in temp:
                if not item.endswith('labelTrainIds.png', 0, len(item)):
                    list_items.remove(item)

            # defining paths
            list_items = ['/' + city + '/' + path for path in list_items]

            self.yLabel_list.extend(list_items)
            self.XImg_list.extend(
                ['/' + city + '/' + path for path in os.listdir(self.rgb_path + '/' + city)]
            )

    def __len__(self):
        length = len(self.XImg_list)
        return length

    def __getitem__(self, index):
        image = Image.open(self.rgb_path + self.XImg_list[index])
        y = Image.open(self.label_path + self.yLabel_list[index])

        if self.transform is not None:
            image = self.transform(image)
            y = self.transform(y)

        image = transforms.ToTensor()(image)
        y = np.array(y)
        y = torch.from_numpy(y)

        y = y.type(torch.LongTensor)
        if self.eval:
            return image, y, self.XImg_list[index]
        else:
            return image, y


class RuledDataset:
    def __getitem__(self, index):
        print("==================")
        image_name = 'j01-045' #.".join(self.images_paths[index].split('.')[:-1])
        #print(self.images_paths.split('.')[:-1])
        print(image_name)
        image = Image.open(os.path.join(self.image_dir, f"{image_name}.png")).convert("RGB")
        seg = Image.open(os.path.join(self.segmentation_dir, f"{image_name}.png")).convert("L")

        image = self.transform_image(image)
        seg = self.transform_mask(seg)

        return image, seg

    def __init__(self, image_paths, image_dir, segmentation_dir, transform_image, transform_mask):
        super(RuledDataset).__init__()
        self.image_dir = image_dir
        self.segmentation_dir = segmentation_dir
        self.transform_image = transform_image
        self.transform_mask = transform_mask
        self.images_paths = image_paths
        print("hi----------")
        print(self.images_paths)

    def __len__(self):
        return len(self.images_paths)



