"""Real recognition adapters, deliberately independent of VEIL/InsightFace.

Official references and model hashes are emitted by each adapter. A recognition
backend alone does not establish calibrated thresholds or human ground truth.
"""
from __future__ import annotations
import hashlib
import importlib.util
from pathlib import Path
import numpy as np
from PIL import Image


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


class FaceRecognitionVerifier:
    paper_valid=True

    def __init__(self, backend, model_path=None, *, device=None, adaface_net=None):
        import torch
        from facenet_pytorch import MTCNN, InceptionResnetV1
        self.torch=torch
        self.name=backend
        self.device=device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.detector=MTCNN(image_size=160, margin=0, keep_all=True,
                            post_process=False, device=self.device)
        self.metadata={'backend':backend,'device':self.device,'embedding_dimension':512,
            'detector':'facenet-pytorch MTCNN; no VEIL detector or cache',
            'similarity':'cosine; unit normalized embeddings',
            'facenet_pytorch_source':'https://github.com/timesler/facenet-pytorch',
            'code_license':'MIT; training dataset and checkpoint terms are separate'}
        if backend=='facenet':
            if model_path:
                self.model=InceptionResnetV1(classify=False).eval().to(self.device)
                state=torch.load(model_path,map_location='cpu',weights_only=True)
                # Official checkpoints contain the VGGFace2 classifier as well.
                self.model=InceptionResnetV1(num_classes=8631, classify=False).eval().to(self.device)
                self.model.load_state_dict(state,strict=True)
            else:
                self.model=InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
                from facenet_pytorch.models.inception_resnet_v1 import get_torch_home
                model_path=Path(get_torch_home())/'checkpoints/20180402-114759-vggface2.pt'
            self.metadata.update(model='InceptionResnetV1 VGGFace2', input_size=[160,160],
                channel_order='RGB', normalization='(pixel-127.5)/128',
                alignment='official MTCNN bbox crop, margin 0; no landmark warp',
                checkpoint_source='https://github.com/timesler/facenet-pytorch/releases/tag/v2.2.9')
        elif backend=='adaface':
            if model_path is None: raise ValueError('AdaFace requires an official local checkpoint')
            net_path=Path(adaface_net or Path(__file__).resolve().parents[1]/'local_research/vendor/adaface_net.py')
            spec=importlib.util.spec_from_file_location('official_adaface_net',net_path)
            net=importlib.util.module_from_spec(spec)
            spec.loader.exec_module(net)
            self.model=net.build_model('ir_50').eval().to(self.device)
            # The official Lightning checkpoint stores a callback CLASS as
            # metadata. Substitute an inert type while retaining weights_only;
            # do not import or execute checkpoint-supplied Python code.
            inert_callback=type('ModelCheckpoint',(),{})
            with torch.serialization.safe_globals([(inert_callback,
                    'pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint')]):
                ckpt=torch.load(model_path,map_location='cpu',weights_only=True)
            state={k[len('model.'):]:v for k,v in ckpt['state_dict'].items() if k.startswith('model.')}
            self.model.load_state_dict(state,strict=True)
            self.metadata.update(model='AdaFace IR50 MS1MV2', input_size=[112,112],channel_order='BGR',
                normalization='(pixel/255-0.5)/0.5',
                alignment='MTCNN five landmarks; least-squares similarity to official 112x112 template',
                source='https://github.com/mk-minchul/AdaFace',
                checkpoint_source='https://drive.google.com/file/d/1eUaSHG4pGlIZK7hBkqjyp2fc2epKoBvI/view',
                architecture_sha256=file_hash(net_path))
        else: raise ValueError(f'Unsupported independent backend: {backend}')
        self.metadata['checkpoint_sha256']=file_hash(model_path)
        self.last_failure=None

    def extract_embedding(self,image:Image.Image):
        import cv2
        from skimage.transform import SimilarityTransform
        self.last_failure=None
        image=image.convert('RGB')
        try:
            boxes,probs,landmarks=self.detector.detect(image,landmarks=True)
        except (ValueError,RuntimeError,IndexError) as exc:
            self.last_failure='detection_error:'+type(exc).__name__
            return None
        if boxes is None or len(boxes)==0:
            self.last_failure='face_not_detected'
            return None
        # Crop callers supply one annotation ROI. Prefer its central face; this
        # avoids silently selecting a neighboring large face from a padded ROI.
        center=np.array(image.size)/2
        idx=min(range(len(boxes)),key=lambda i:np.linalg.norm((boxes[i,:2]+boxes[i,2:])/2-center))
        if self.name=='facenet':
            face=self.detector.extract(image,boxes[idx:idx+1],save_path=None)
            if face is None:
                self.last_failure='crop_failed'; return None
            tensor=(face.to(self.device)-127.5)/128.0
        else:
            reference=np.array([[38.2946,51.6963],[73.5318,51.5014],[56.0252,71.7366],
                [41.5493,92.3655],[70.7299,92.2041]],dtype=np.float32)
            transform=SimilarityTransform()
            if not transform.estimate(np.asarray(landmarks[idx],dtype=np.float64),reference):
                self.last_failure='alignment_failed'; return None
            aligned=cv2.warpAffine(np.asarray(image),transform.params[:2],(112,112),borderValue=0)
            arr=np.ascontiguousarray(aligned[:,:,::-1].transpose(2,0,1),dtype=np.float32)
            tensor=self.torch.from_numpy((arr/255.0-.5)/.5).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            features=self.model(tensor)
            if isinstance(features,tuple): features=features[0]
        result=features[0].detach().cpu().numpy().astype(np.float32)
        norm=float(np.linalg.norm(result))
        if not np.isfinite(result).all() or norm<=0:
            self.last_failure='invalid_embedding'; return None
        return result/norm

    def similarity(self,a,b):
        return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))
