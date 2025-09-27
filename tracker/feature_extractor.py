import torch
import numpy as np
from PIL import Image
from lavis.models import load_model_and_preprocess

def read_coords(file_path):
    coords = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                coords.append([float(parts[1]), float(parts[2])])
    return coords
class ReIDFeatureExtractor:
    
    def __init__(self, device=None, batch_size = 16, crop_size=10):

        self.device = device if device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.crop_size = crop_size
        
        # loda CLIP
        self.model, self.vis_processors, self.txt_processors = load_model_and_preprocess(
            "clip_feature_extractor",  model_type="ViT-B-16",  is_eval=True,device=self.device
        )
        
        print(f"✅ ReID model is ready,start to use": {self.device}")
    
    def extract_features_from_coordinates(self, image, coordinate_path):
        #  dtype
        dtype = np.dtype([
            ("id", np.int32),
            ("coordinate", np.float32, 2),
            ("feature", np.float32, 512)
        ])

        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        else:
            img = image.convert("RGB") if image.mode != "RGB" else image
        
        if isinstance(coordinate_path, str):
            coordinates = read_coords(coordinate_path)
        if not coordinates:
            return np.array([], dtype=dtype)
        
        processed_coords = []
        for coord in coordinates:
            if len(coord) == 3:  # (id, x, y)
                _, x, y = coord
                processed_coords.append((x, y))
            elif len(coord) == 2:  # (x, y)
                x, y = coord
                processed_coords.append((x, y))
            else:
                raise ValueError(f"coordinate type error: {coord},should be(x,y) or (id,x,y)")
        
        crops = self._crop_images(img, processed_coords)
    
        features = self._batch_extract_features(crops)

        frame_data = []
        for idx, ((x, y), feature) in enumerate(zip(processed_coords, features)):
            frame_data.append((idx, (float(x), float(y)), feature))
        
        dets = np.array(frame_data, dtype=dtype)
        
        return dets
    
    def _crop_images(self, img, coordinates):

        crops = []
        W, H = img.size 
        
        for x, y in coordinates:
            left = max(0, x - self.crop_size)
            top = max(0, y - self.crop_size) 
            right = min(W, x + self.crop_size)
            bottom = min(H, y + self.crop_size)
            
            cropped_img = img.crop((left, top, right, bottom))
            crops.append(cropped_img)
        
        return crops
    
    def _batch_extract_features(self, crops):

        all_features = []
        
        for i in range(0, len(crops), self.batch_size):
            batch_crops = crops[i:i + self.batch_size]
            
            # trans -> tensor
            image_tensors = torch.stack([
                self.vis_processors["eval"](crop) for crop in batch_crops
            ]).to(self.device)
            
            # txt _input
            text_input = self.txt_processors["eval"]("A pedestrian")
            sample = {
                "image": image_tensors, 
                "text_input": [text_input] * len(batch_crops)
            }
            
            # feature_extract
            with torch.no_grad():
                features = self.model.extract_features(sample)
                image_embeds = features.image_embeds.cpu().detach().numpy()
                all_features.extend(image_embeds)
        
        return all_features
    
    def process_single_frame(self, image_path, coordinates):

        return self.extract_features_from_coordinates(image_path, coordinates)
    
    def clear_cache(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def example_usage():

    reid_extractor = ReIDFeatureExtractor(batch_size=16, crop_size=10)
    

    
    # coordinates_with_id = [(1, 100, 200), (2, 150, 300), (3, 200, 400)]  # (id, x, y)
    # dets = reid_extractor.extract_features_from_coordinates("image.jpg", coordinates_with_id)
    
 
    # dets = [
    #     {"coordinate": (100.0, 200.0), "features": numpy_array_512d},
    #     {"coordinate": (150.0, 300.0), "features": numpy_array_512d},
    #     ...
    # ]
    
    print("ReID is ready,start to use")
    return reid_extractor

if __name__ == "__main__":
    extractor = example_usage()










