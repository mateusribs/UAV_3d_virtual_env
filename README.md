# UAV 3D Virtual Environment 

Ambiente virtual 3D, desenvolvido por @rafaelcostafrf, para simulação da estimação de posição, velocidade e atitude de um Veículo Aéreo Não Tripulado.

Nesta simulação foi implementado o sistema de detecção por marcadores fiduciais ArUco desenvolvido por (https://www.uco.es/investiga/grupos/ava/node/26) [1]. A partir da detecção é possível realizar o rastreamento do marcador e encontrar sua postura (orientação e posição) em relação a câmera. 

Juntamente com as informações obtidas pela câmera, utilizou-se a biblioteca https://github.com/mateusribs/quadrotor_environment para simular os sensores presentes no quadrirrotor. As informações obtidas pelo acelerômetro juntamente com o giroscópio foram fundidas através do filtro complementar de Madgwick [2]. O filtro de Madgwick tem como saída o quatérnio que descreve a atitude do corpo. 


[1] "Generation of fiducial marker dictionaries using mixed integer linear programming",S. Garrido-Jurado, R. Muñoz Salinas, F.J. Madrid-Cuevas, R. Medina-Carnicer, Pattern Recognition:51, 481-491,2016

[2] Sebastian OH Madgwick, Andrew JL Harrison, and Ravi Vaidyanathan. Estimation of IMU and MARG orientation using a gradient descent algorithm, 2011 IEEE international conference on rehabilitation robotics. IEEE, 2011.
