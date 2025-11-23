import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    """
    CNN encoder using a pretrained ResNet-50 instead of Inception.
    Maps each image to an `embed_size` feature vector.
    """

    def __init__(self, embed_size, train_cnn=False):
        super().__init__()
        self.train_cnn = train_cnn

        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        modules = list(resnet.children())[:-1]
        self.cnn = nn.Sequential(*modules)

        self.fc = nn.Linear(resnet.fc.in_features, embed_size)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        for param in self.cnn.parameters():
            param.requires_grad = self.train_cnn

    def forward(self, images):
        # CNN forward
        with torch.set_grad_enabled(self.train_cnn):
            features = self.cnn(images)
            features = features.reshape(features.size(0), -1)

        features = self.fc(features)
        features = self.dropout(self.relu(features))
        return features


class DecoderRNN(nn.Module):
    """
    DecoderRNN

    Takes the encoded image feature vector + caption tokens
    and predicts the next word at each time step.

    Architecture:
      - Embedding layer converts token IDs → dense embeddings
      - LSTM processes the "image + caption sequence"
      - Final Linear layer converts LSTM output → vocabulary logits

    The decoder treats the image feature vector as the first input token:
        input sequence = [image_embedding; word1; word2; ...]
    """

    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        """
        features: (batch, embed_size)
        captions: (seq_len, batch)
        """

        embeddings = self.dropout(self.embed(captions))
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)

        hiddens, _ = self.lstm(embeddings)  # (seq_len+1, batch, hidden_size)
        logits = self.linear(hiddens)  # (seq_len+1, batch, vocab_size)

        return logits[1:, :, :]  # (seq_len, batch, vocab)


class ImageCaptionNet(nn.Module):
    """
    Combines EncoderCNN and DecoderRNN into a single end-to-end model for image captioning.

    forward():
        Used during training.
        Input: images + ground truth captions
        Output: vocabulary logits for each time step.

    caption_image():
        Used during inference.
        Greedily generates a caption word-by-word by feeding back the
        last predicted word into the decoder LSTM.
    """

    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super().__init__()

        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        features = self.encoder(images)
        return self.decoder(features, captions)

    def caption_image(self, image, vocabulary, max_length=50):
        """
        image: a single image tensor (1,3,299,299)
        vocabulary: maps indices → word strings
        Generates a caption greedily:
            hidden_{t+1} = LSTM(embed(predicted_t))
        """

        result_caption = []

        with torch.no_grad():
            x = self.encoder(image).unsqueeze(0)  # (1,1,embed_size)
            states = None

            for _ in range(max_length):
                hiddens, states = self.decoder.lstm(x, states)  # (1,1,H)
                output = self.decoder.linear(hiddens.squeeze(0))
                predicted = output.argmax(1)
                result_caption.append(predicted.item())

                x = self.decoder.embed(predicted).unsqueeze(0)

                if vocabulary.itos[predicted.item()] == "<EOS>":
                    break

        return [vocabulary.itos[idx] for idx in result_caption]
