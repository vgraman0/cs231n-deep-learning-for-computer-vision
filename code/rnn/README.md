# Image Captioning Network

![alt text](image_captioning_arch.png)

## Data preparation
- Load COCO images and captions.
- Build a vocabulary from all captions and convert each caption into a sequence of token indices.
- Preprocess images using standard ResNet transforms (resize to 224×224, normalize).

## Encoding
1. Take a batch of input images `(B, 3, 224, 224)`.
2. Pass them through a pretrained ResNet-50 (with the final classification layer removed) to extract feature vectors `(B, 2048)`.
3. Use a linear layer to map these CNN features into the decoder’s embedding dimension `(B, embed_dim)`.

## Decoding
- Feed the image embedding and the caption tokens into an LSTM decoder.
- At each timestep, the LSTM predicts the next word using a linear layer + softmax.
- Training uses the ground-truth caption (“teacher forcing”); inference generates tokens one by one until `<EOS>`.
