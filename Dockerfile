FROM dataloopai/dtlpy-agent:gpu.cuda.11.8.py3.10.pytorch2

RUN pip install --user \
    ftfy \
    regex \
    'pillow>=11.0.0' \
    git+https://github.com/openai/CLIP.git

# Pre-download CLIP ViT-B/32 model weights
RUN python -c "import clip; import torch; model, preprocess = clip.load('ViT-B/32', device='cpu', jit=False, download_root='/tmp/weights'); print('ViT-B/32 model weights downloaded successfully')"

# docker build --no-cache -t gcr.io/viewo-g/piper/agent/runner/apps/clip-model-adapter:0.1.0 -f Dockerfile .
# docker push gcr.io/viewo-g/piper/agent/runner/apps/clip-model-adapter:0.1.0
