from Lab4_keras import NIN


NIN('leaky',True,True,'leaky_HE_BN')
NIN('leaky',False,False,'leaky_nonHE_nonBN')
NIN('leaky',False,True,'leaky_nonHE_BN')
NIN('leaky',True,False,'leaky_HE_nonBN')
NIN('relu' or 'elu'  , w/o HE's , w/o BN)
NIN('relu', False , False,'relu_nonHE_nonBN')
NIN('relu', False , True,'relu_nonHE_BN')
NIN('elu', False , True,'elu_nonHE_BN')
NIN('relu', True , True,'elu_HE_BN')
NIN('elu', True , True,'relu_HE_BN')


NIN('elu', False , False,'elu_nonHE_nonBN')
NIN('elu', True , False,'elu_HE_nonBN')
NIN('relu', True , False,'relu_HE_nonBN')
