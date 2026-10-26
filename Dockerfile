FROM hub.dataloop.ai/dtlpy-runner-images/cpu:python3.13_full_bci

# Limit threading in numerical libraries to prevent CPU oversubscription on CPU-limited containers
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1

RUN echo "OMP_NUM_THREADS=$OMP_NUM_THREADS" && \
    echo "MKL_NUM_THREADS=$MKL_NUM_THREADS" && \
    echo "OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS" && \
    echo "NUMEXPR_NUM_THREADS=$NUMEXPR_NUM_THREADS"

# Install dependencies with no cache
RUN pip install --no-cache-dir --user \
    ftfy \
    regex \
    'pillow>=11.0.0' \
    git+https://github.com/openai/CLIP.git

# Pre-download CLIP ViT-B/32 model weights
RUN python3 -c "import clip; import torch; model, preprocess = clip.load('ViT-B/32', device='cpu', jit=False, download_root='/tmp/weights'); print('ViT-B/32 model weights downloaded successfully')"

# docker build --no-cache -t hub.dataloop.ai/customerhub/piper/agent/runner/apps/clip-model-adapter:0.1.4 -f Dockerfile .
# docker push hub.dataloop.ai/customerhub/piper/agent/runner/apps/clip-model-adapter:0.1.4
