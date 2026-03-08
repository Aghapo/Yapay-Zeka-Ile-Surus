import numpy as np 
class NeuralNetwork:

    #Başlangıçtaki değerler
    def __init__(self, input_size = 5, hidden_size = 8, output_size = 2):
        self.w1 = np.random.uniform(-1, 1, (input_size,hidden_size))
        self.w2 = np.random.uniform(-1, 1, (hidden_size, output_size))

    def sigmoid (self, x):
        return 1/ (1+np.exp(-x))
    
    
    def forward (self, inputs): 
        #Gizli katmana yollananlar
        self.hidden = self.sigmoid(np.dot(inputs,self.w1))
        #Çıkışa yollanan
        self.output = self.sigmoid(np.dot(self.hidden, self.w2))
        return self.output
    

nn = NeuralNetwork()
sensor_data = np.array([0.8, 0.6, 1.0, 0.6, 0.8])  # örnek sensör değerleri
decision = nn.forward(sensor_data)

print(f"Gaz:   {decision[0]:.3f}")
print(f"Dönüş: {decision[1]:.3f}")
print(f"(0.5 altı = sol/fren, 0.5 üstü = sağ/gaz)")