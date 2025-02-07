import os
import numpy as np
import imageio.v3 as iio
from skimage.transform import resize

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def carregar_imagens(pasta, label, tamanho=(64, 64)):
    imagens = []
    labels = []
    
    for arquivo in os.listdir(pasta):
        caminho = os.path.join(pasta, arquivo)
        
        try:
            imagem = iio.imread(caminho)
            
            if len(imagem.shape) == 3:
                imagem = np.dot(imagem[..., :3], [0.2989, 0.5870, 0.1140])  

            imagem = resize(imagem, tamanho, anti_aliasing=True)

            imagens.append(imagem.flatten())  
            labels.append(label)
        
        except Exception as e:
            print(f"Erro no {arquivo}: {e} ⛔")

    return np.array(imagens), np.array(labels)

X_gatos, y_gatos = carregar_imagens(r"C:\Users\user\Pictures\Dataset\C", label=0)  # Gatos = 0
X_cachorros, y_cachorros = carregar_imagens(r"C:\Users\user\Pictures\Dataset\D", label=1)  # Cachorros = 1

X = np.vstack((X_gatos, X_cachorros))
y = np.hstack((y_gatos, y_cachorros))

X = X / 255.0

indices = np.random.permutation(len(y))
X, y = X[indices], y[indices]

np.random.seed(42)
pesos_entrada = np.random.rand(X.shape[1], 4)
pesos_saida = np.random.rand(4, 1)

taxa_aprendizagem = 0.5
for epoca in range(10000):
    camada_oculta = sigmoid(np.dot(X, pesos_entrada))
    saida = sigmoid(np.dot(camada_oculta, pesos_saida))

    erro = y.reshape(-1, 1) - saida
    ajuste_saida = erro * (saida * (1 - saida))
    ajuste_oculta = np.dot(ajuste_saida, pesos_saida.T) * (camada_oculta * (1 - camada_oculta))

    pesos_saida += np.dot(camada_oculta.T, ajuste_saida) * taxa_aprendizagem
    pesos_entrada += np.dot(X.T, ajuste_oculta) * taxa_aprendizagem

print("Rede neural treinada ✅")

def prever_imagem(caminho):
    try:
        imagem = iio.imread(caminho)
        
        if len(imagem.shape) == 3:
            imagem = np.dot(imagem[..., :3], [0.2989, 0.5870, 0.1140])

        imagem = resize(imagem, (64, 64), anti_aliasing=True)
        imagem = imagem.flatten() / 255.0

        camada_oculta = sigmoid(np.dot(imagem, pesos_entrada))
        saida = sigmoid(np.dot(camada_oculta, pesos_saida))

        resultado = "🐶" if saida > 0.5 else "🐱"
        print(f"Parece um: {resultado} (Compatibilidade: {saida[0]:.2f})")
    
    except Exception as e:
        print(f"Erro an imagem: {e} ⛔")

caminho_imagem = input("Digite o caminho da imagem para testar: ")
prever_imagem(caminho_imagem)
