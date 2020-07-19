# UAV 3D Virtual Environment 

Ambiente virtual 3D para simulação da estimação de posição, velocidade e atitude de um Veículo Aéreo Não Tripulado.

Status do projeto:
- É possível obter a posição e atitude, em relação ao referencial da câmera,  do veículo a partir de um tabuleiro de xadrez que fica na parte superior do veículo. A posição e atitude são encontradas através do algoritmo solvePNP disponibilizado na biblioteca de visão computacional OpenCV.
- Aplica-se uma matriz de transformação entre o referencial do objeto e o referencial inercial (solo). Esta matriz é encontrada através do algoritmo proposto por J.Cashbaugh e C.Kitts (2018). A partir desta transformação, encontra-se a posição e atitude do veículo em relação ao solo, sendo possível assim comparar com os valores encontrados pelos sensores simulados no ambiente virtual.

Próximos passos:
 - Utilizar marcações definidas no quadrirrotor e substituir o tabuleiro de xadrez.
 - Implementar o filtro de Kalman para melhorar a precisão das medidas.
