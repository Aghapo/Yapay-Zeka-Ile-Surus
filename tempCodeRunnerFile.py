import numpy as np

class NeuralNetwork:
    # Başlangıçtaki değerler
    def __init__(self, input_size=5, hidden_size=8, output_size=2):
        # Ağırlıklar (Weights) - Nöronlar arasındaki bağlantı gücü
        self.w1 = np.random.uniform(-1, 1, (input_size, hidden_size))
        self.w2 = np.random.uniform(-1, 1, (hidden_size, output_size))
        
        # Sapmalar (Biases) - YENİ EKLENDİ
        # Nöronların ne kadar kolay tetikleneceğini belirler (Esneklik sağlar)
        self.b1 = np.random.uniform(-1, 1, (1, hidden_size))
        self.b2 = np.random.uniform(-1, 1, (1, output_size))

    def sigmoid(self, x):
        # YENİ: np.clip eklenerek matematiksel taşma (overflow) hataları önlendi
        return 1 / (1 + np.exp(-np.clip(x, -10, 10)))
    
    def forward(self, inputs): 
        # Girdileri tek satırlık bir matrise (1x5) dönüştür
        inputs = np.array(inputs, ndmin=2)
        
        # Gizli katmana yollananlar (Bias eklendi)
        self.hidden = self.sigmoid(np.dot(inputs, self.w1) + self.b1)
        
        # Çıkışa yollanan (Bias eklendi)
        self.output = self.sigmoid(np.dot(self.hidden, self.w2) + self.b2)
        
        # [[0.7, 0.4]] gibi gelen sonucu [0.7, 0.4] yapmak için flatten() kullandık
        return self.output.flatten()

    def mutate(self, rate=0.1):
        # YENİ EKLENDİ: Ağı geliştirirken (evrim) ağırlıkları ufak ufak değiştirmek için
        if np.random.rand() < rate:
            self.w1 += np.random.normal(0, 0.2, self.w1.shape)
        if np.random.rand() < rate:
            self.w2 += np.random.normal(0, 0.2, self.w2.shape)
        if np.random.rand() < rate:
            self.b1 += np.random.normal(0, 0.2, self.b1.shape)
        if np.random.rand() < rate:
            self.b2 += np.random.normal(0, 0.2, self.b2.shape)

# --- Test Kısmı ---
if __name__ == "__main__":
    nn = NeuralNetwork(input_size=5, hidden_size=8, output_size=2)
    
    # Örnek sensör değerleri (-90, -45, 0, 45, 90 derecelik açılardaki duvar mesafeleri)
    sensor_data = [0.8, 0.6, 1.0, 0.6, 0.8]  
    
    # Ağa karar verdiriyoruz
    decision = nn.forward(sensor_data)

    print(f"Gaz Kararı:       {decision[0]:.3f}")
    print(f"Direksiyon Kararı:{decision[1]:.3f}")
    print("-" * 30)
    print("MANTIK:")
    print("Gaz        -> (0.5 altı = Fren/Yavaşlama, 0.5 üstü = Gaz)")
    print("Direksiyon -> (0.5 altı = Sola Dön,       0.5 üstü = Sağa Dön)")