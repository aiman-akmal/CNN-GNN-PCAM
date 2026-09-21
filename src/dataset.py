import torch
import torch.nn.functional as F
import scipy.ndimage as ndi
import numpy as np

def extract_cell_crops(image_tensor):
    img_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    
    gray = np.dot(img_np[..., :3], [0.2989, 0.5870, 0.1140])
    
    thresh = np.percentile(gray, 20)
    binary_mask = gray < thresh 
    
    labeled_array, num_features = ndi.label(binary_mask)
    
    centroids = []
    for i in range(1, num_features + 1):
        mask = (labeled_array == i)
        if np.sum(mask) >= 4: 
            com = ndi.center_of_mass(mask)
            centroids.append((int(com[0]), int(com[1])))
            
    if len(centroids) == 0:
        return torch.zeros((1, 3, 96, 96))
        
    padded_img = np.pad(img_np, ((16, 16), (16, 16), (0, 0)), mode='reflect')
    
    crops = []
    for r, c in centroids:
        r_p, c_p = r + 16, c + 16
        crop = padded_img[r_p-16:r_p+16, c_p-16:c_p+16, :]
        
        crop_tensor = torch.tensor(crop, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        crop_resized = F.interpolate(crop_tensor, size=(96, 96), mode='bilinear', align_corners=False)
        crops.append(crop_resized)
        
    return torch.cat(crops, dim=0)

def create_knn_edges(node_features):
    num_nodes = node_features.shape[0]
    if num_nodes <= 1:
        return torch.empty((2, 0), dtype=torch.long)
        
    k = min(6, num_nodes - 1)
    dist = torch.cdist(node_features, node_features, p=2.0)
    
    _, indices = torch.topk(-dist, k=k+1, dim=1) 
    
    edge_src, edge_dst = [], []
    for i in range(num_nodes):
        for j in range(1, k+1): 
            neighbor = indices[i, j].item()
            edge_src.append(i)
            edge_dst.append(neighbor)
            
    return torch.tensor([edge_src, edge_dst], dtype=torch.long)