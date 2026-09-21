import torch
import cv2
import numpy as np

def compute_local_gradcams(cnn_model, crop_data_list, crop_size=32, device="cpu"):
    cnn_model.to(device)
    cnn_model.eval()
    target_layer = cnn_model.features[-1]
    
    for crop_dict in crop_data_list:
        crop_tensor = crop_dict['tensor'].to(device) 
        
        activations, gradients = [], []
        def fw_hook(module, input, output): activations.append(output)
        def bw_hook(module, grad_input, grad_output): gradients.append(grad_output[0])
        
        h1 = target_layer.register_forward_hook(fw_hook)
        h2 = target_layer.register_full_backward_hook(bw_hook)
        
        cnn_model.zero_grad()
        features = cnn_model(crop_tensor) 
        
        l2_norm_target = torch.norm(features, p=2)
        l2_norm_target.backward()
        
        h1.remove()
        h2.remove()
        
        act = activations[0].detach()[0]
        grad = gradients[0].detach()[0]
        weights = torch.mean(grad, dim=[1, 2], keepdim=True)
        local_cam = torch.sum(weights * act, dim=0).cpu().numpy()
        local_cam = np.maximum(local_cam, 0)
        
        local_cam_resized = cv2.resize(local_cam, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
        if np.max(local_cam_resized) > 0:
            local_cam_resized /= np.max(local_cam_resized)
            
        crop_dict['heatmap'] = local_cam_resized
        
        del crop_tensor, features, l2_norm_target
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    return crop_data_list

def stitch_global_canvas(crop_data_list, canvas_size=(96, 96), crop_size=32):
    global_heatmap = np.zeros(canvas_size, dtype=np.float32)
    half_crop = crop_size // 2
    
    for crop_dict in crop_data_list:
        local_heat = crop_dict['heatmap']
        cx, cy = crop_dict['centroid']
        
        x_start = max(0, cx - half_crop)
        x_end = min(canvas_size[0], cx + half_crop)
        y_start = max(0, cy - half_crop)
        y_end = min(canvas_size[1], cy + half_crop)
        
        crop_x_start = max(0, half_crop - cx)
        crop_x_end = crop_size - max(0, (cx + half_crop) - canvas_size[0])
        crop_y_start = max(0, half_crop - cy)
        crop_y_end = crop_size - max(0, (cy + half_crop) - canvas_size[1])
        
        local_region = local_heat[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
        current_canvas_region = global_heatmap[y_start:y_end, x_start:x_end]
        
        global_heatmap[y_start:y_end, x_start:x_end] = np.maximum(current_canvas_region, local_region)

    if np.max(global_heatmap) > 0:
        global_heatmap /= np.max(global_heatmap)
        
    return global_heatmap

def compute_final_fidelity_scores(p_orig, p_delete_array, p_retain_array):
    sparsity_levels = np.linspace(0.05, 0.95, 19)
    
    y_plus = p_orig - np.array(p_delete_array)  
    y_minus = np.array(p_retain_array)         
    
    auc_plus_raw = np.trapz(y=y_plus, x=sparsity_levels)
    auc_minus_raw = np.trapz(y=y_minus, x=sparsity_levels)
    
    fid_plus = (2 * auc_plus_raw) - 1
    fid_minus = (2 * auc_minus_raw) - 1
    
    return fid_plus, fid_minus