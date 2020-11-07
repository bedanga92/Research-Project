from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.vgg19 import VGG19

def loadRestNetModel(input_shape,out_shape):

    pre_trained_model = ResNet50(input_shape = input_shape, # Shape of our images
                                include_top = False, # Leave out the last fully connected layer
                                weights = 'imagenet')
    for layer in pre_trained_model.layers:
        layer.trainable = False
    
    x = layers.Flatten()(pre_trained_model.output)
    # Add a fully connected layer with 1,024 hidden units and ReLU activation
    x = layers.Dense(1024, activation='relu')(x)
    # Add a dropout rate of 0.2
    x = layers.Dropout(0.2)(x)                  
    # Add a final softmax layer for classification
    x = layers.Dense  (out_shape, activation='softmax')(x)           

    model = Model( pre_trained_model.input, x) 

    return model

#restnet = loadRestModel((224,224,3),3)
#restnet.summary()

def loadVGG19Model(input_shape,out_shape):
    #new_input = layers.Input(shape=input_shape)
    modelvgg19 = VGG19(input_shape = input_shape,
        include_top=False,weights="imagenet")
    for layer in modelvgg19.layers:
        layer.trainable = False
    x = layers.Flatten()(modelvgg19.output)
    x = layers.Dense(1024,activation='relu')(x)
    x = layers.Dense(512,activation='relu')(x)
    x = layers.Dense(out_shape, activation='softmax')(x)

    model = Model(modelvgg19.input,x)

    return model

#vgg19 = loadVGG19Model((224,224,3),3)
#vgg19.summary()

def loadDenseNetModel(input_shape,out_shape):
    pre_trained_model = DenseNet121(input_shape = input_shape, # Shape of our images
                                include_top = False, # Leave out the last fully connected layer
                                weights = 'imagenet')
    for layer in pre_trained_model.layers:
        layer.trainable = False
    
    x = layers.Flatten()(pre_trained_model.output)
    # Add a fully connected layer with 1,024 hidden units and ReLU activation
    x = layers.Dense(1024, activation='relu')(x)
    # Add a dropout rate of 0.2
    x = layers.Dropout(0.2)(x)                  
    # Add a final sigmoid layer for classification
    x = layers.Dense  (out_shape, activation='softmax')(x)           

    model = Model( pre_trained_model.input, x) 

    return model

#densenet = loadDenseNetModel((224,224,3),3)
#densenet.summary()
