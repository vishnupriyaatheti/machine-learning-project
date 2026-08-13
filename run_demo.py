import sys
sys.path.append('c:/Users/athet/OneDrive/Documents/ULL Admission/machine learning/FInal Project Models')

from adaptive_unet_transformer import build_adaptive_attention_unet
import numpy as np

def main():
    m = build_adaptive_attention_unet(input_shape=(64,64,1), base_filters=8, num_heads=2, embed_dim=32, mlp_dim=64)
    print('Model built, layer count:', len(m.layers))
    x = np.random.rand(1,64,64,1).astype('float32')
    y = m.predict(x)
    print('Forward pass output shape:', y.shape)

if __name__ == '__main__':
    main()
