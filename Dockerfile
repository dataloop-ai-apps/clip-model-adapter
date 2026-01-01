FROM hub.dataloop.ai/dtlpy-runner-images/gpu:python3.10_cuda11.8_pytorch2

# Install dependencies with no cache
RUN pip install --no-cache-dir --user \
    ftfy \
    regex \
    'pillow>=11.0.0' \
    git+https://github.com/openai/CLIP.git

# Pre-download CLIP ViT-B/32 model weights
RUN python3 -c "import clip; import torch; model, preprocess = clip.load('ViT-B/32', device='cpu', jit=False, download_root='/tmp/weights'); print('ViT-B/32 model weights downloaded successfully')" && \
    rm -rf /tmp/weights /tmp/* /var/tmp/* && \
    pip cache purge || true

# docker build --no-cache -t gcr.io/viewo-g/piper/agent/runner/apps/clip-model-adapter:0.1.0 -f Dockerfile .
# docker push gcr.io/viewo-g/piper/agent/runner/apps/clip-model-adapter:0.1.0
