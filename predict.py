from PIL import Image
import torch
import argparse
import numpy as np
import pandas as pd
import json

def main():
    user_input = get_input_args()
    path = user_input.path
    checkpoint = user_input.checkpoint
    top_k = user_input.top_k
    name = user_input.name
    gpu = user_input.gpu

    model = load_checkpoint(checkpoint).
    probs, classes = predict(path, model, int(top_k))
    view_classify(probs, classes, name)

def get_input_args():
    parser = argparse.ArgumentParser(description = 'image prediction')
    parser.add_argument('path',action = 'store')
    parser.add_argument('checkpoint',action = 'store')
    parser.add_argument('--top_k',type = int,dest = 'top_k',default = 5)
    parser.add_argument('--category_names ',type = str,dest = 'name')
    parser.add_argument('--gpu',dest = 'gpu',action = 'store_true')
    return parser.parse_args()

def load_checkpoint(path):
    checkpoint = torch.load(path)
    model = checkpoint['model']
    model.idx = checkpoint['idx']
    model.classifier = checkpoint['classifier']
    model.load_dict(checkpoint['dict'])
    for param in model.parameters():
        param.requires_grad = False
    return model

def predict(path, model, topk=5):
    image = torch.from_numpy(process_image(path))
    image = image.unsqueeze(0).float()
    model.eval()
    model.requires_grad = False
    outputs = torch.exp(model.forward(image)).topk(topk)
    probs, classes = outputs[0].data.cpu().numpy()[0], outputs[1].data.cpu().numpy()[0]
    idx_to_class = {key: value for value, key in model.idx.items()}
    classes = [idx_to_class[classes[i]] for i in range(classes.size)]
    return probs, classes

def view_classify(probs, classes, name):
    if name is None:
        name_classes = classes
    else:
        with open(name, 'r') as f:
            name_data = json.load(f)
        name_classes = [name_data[i] for i in classes]
    df = pd.DataFrame({
        'classes': pd.Series(data = name_classes),
        'values': pd.Series(data = probs, dtype='float64')
    })
    print(df)

def process_image(image):
    im = Image.open(image)
    im = im.resize((256, 256))
    im = im.crop((16, 16, 240, 240))
    np_image = np.array(im)
    np_image_norm = ((np_image / 255) - ([0.485, 0.456, 0.406])) / ([0.229, 0.224, 0.225])
    np_image_norm = np_image_norm.transpose((2, 0, 1))
    return np_image_norm

if __name__ == '__main__':
    main()
