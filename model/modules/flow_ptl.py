import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import os 
import cv2 
import numpy as np
import ptlflow
from cvbase import flow2rgb
import time
import openvino as ov

class raft_ptlflow(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.fix_raft = ptlflow.get_model(
            "rapidflow_it6",pretrained_ckpt='things'
        )  
        self.fix_raft.to(device)

        for p in self.fix_raft.parameters():
            p.requires_grad = False
        self.fix_raft.eval()

    def forward(self, gt_local_frames, iters=20):
        b, l_t, c, h, w = gt_local_frames.size()

        with torch.no_grad():
            gtlf_1 = gt_local_frames[0, :-1, :, :, :].contiguous() 
            gtlf_2 = gt_local_frames[0, 1:, :, :, :].contiguous()

            gt_flows_forward = self.fix_raft(
                {"images": torch.stack((gtlf_1, gtlf_2), dim=1)}
            )["flows"]
            gt_flows_backward = self.fix_raft(
                {"images": torch.stack((gtlf_2, gtlf_1), dim=1)}
            )["flows"]

        gt_flows_forward = gt_flows_forward.view(b, l_t - 1, 2, h, w)
        gt_flows_backward = gt_flows_backward.view(b, l_t - 1, 2, h, w)
        

        return gt_flows_forward, gt_flows_backward
    

class raft_ptlflow_openvino:
    def __init__(self):  
        core = ov.Core()
        # ov_model = ov.convert_model('/root/ProPainter/model/modules/rapidflow_it6_dynamicaxes.onnx')
        # ov.save_model(ov_model, '/root/ProPainter/model/modules/rapidflow_it6_dynamicaxes.xml')
        
        ov_model = core.read_model('/root/ProPainter/model/modules/rapidflow_it6_dynamicaxes.xml')
        
        hint = 'THROUGHPUT' 
        stream_num = 2
        config = {"ENABLE_HYPER_THREADING": True} 
        config['NUM_STREAMS'] = str(stream_num)
        config['PERF_COUNT'] = 'NO'
        config['INFERENCE_PRECISION_HINT'] = 'bf16'
        config['PERFORMANCE_HINT'] = hint
    
        self.compiled_model = core.compile_model(ov_model, "CPU", config=config)
        
    def forward(self, gt_local_frames, iters=20):
        b, l_t, c, h, w = gt_local_frames.size()

        gtlf_1 = gt_local_frames[0, :-1, :, :, :].contiguous() 
        gtlf_2 = gt_local_frames[0, 1:, :, :, :].contiguous()

        inputs_f = torch.stack((gtlf_1, gtlf_2), dim=1).numpy()
        inputs_b = torch.stack((gtlf_2, gtlf_1), dim=1).numpy()
                
        # forward
        ov_out_f = self.compiled_model(inputs_f)[self.compiled_model.output(0)]
        gt_flows_forward = torch.Tensor(ov_out_f).view(b, l_t - 1, 2, h, w)
        # backward
        ov_out_b = self.compiled_model(inputs_b)[self.compiled_model.output(0)]
        gt_flows_backward = torch.Tensor(ov_out_b).view(b, l_t - 1, 2, h, w)    

        return gt_flows_forward, gt_flows_backward
    

def load_images_from_floder(folder):
    images = []
    for filename in sorted(os.listdir(folder)):
        frame_path = os.path.join(folder, filename)
        frame = cv2.imread(frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        images.append(frame)
    stacked_images = np.stack(images, axis=0)
    if len(stacked_images.shape) == 3:
        stacked_images = stacked_images[..., np.newaxis]
    tensor_images = torch.from_numpy(stacked_images).permute(0, 3, 1, 2)
    tensor_images = tensor_images / 255.0
    return tensor_images
    
    
def save_flow(flows,save_dir='flow_results'):
    os.makedirs(save_dir, exist_ok=True)
    np_flows = flows.to('cpu').detach().permute(0, 1, 3, 4, 2).numpy()

    for i, flow in enumerate(np_flows[0]):
        flow = flow2rgb(flow) 
        flow = (flow * 255).astype(np.uint8)
        flow = cv2.cvtColor(flow, cv2.COLOR_BGR2RGB)

        cv2.imwrite(os.path.join(save_dir, f'{i:06d}_flow.png'), flow)
