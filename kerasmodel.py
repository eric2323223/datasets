from keras.layers import Conv2D, Dense, concatenate, Flatten, Cropping2D, ZeroPadding2D, merge, MaxPooling2D, Activation, AveragePooling2D, Concatenate, BatchNormalization, Input
from keras.models import Sequential, Model
import h5py
from keras.preprocessing import image
import numpy as np

def setWeights(model):
    f = h5py.File('./models/model.h5', 'r')
    for key in f.keys():
        layer = list(f[key].items())[0][1]
        layerName = (list(f[key].items())[0][0])
        weights = list(layer.items())
        # print(weights)
        kernel = weights[1][1]
        bias = weights[0][1]
        # print(layerName)
        model.get_layer(layerName).set_weights([kernel.value, bias.value])
    print("weights are set")

def tain_model(model):
    model.compile()

def build_model():
    input = Input(shape=(150, 150, 3))
    conv1 = (Conv2D(64, kernel_size=7, strides=2, name='conv1', activation='relu'))(input)
    pool1 = (MaxPooling2D((3,3), strides=2, name='pool1'))(conv1)
    norm1 = (BatchNormalization(name='norm1'))(pool1)
    reduction2 = (Conv2D(64, kernel_size=1, strides=1, name='reduction2', activation='relu'))(norm1)
    conv2 = (Conv2D(192, kernel_size=3, strides=1, name='conv2', activation='relu'))(reduction2)
    norm2 = (BatchNormalization(name='norm2'))(conv2)
    pool2 = (MaxPooling2D((3, 3), strides=2, name='pool2'))(norm2)
    # print(pool2.output_shape)

    # # Inception block1
    icp1_reduction1 = (Conv2D(96, (1, 1), strides=(1,1), name='icp1_reduction1', activation='relu'))(pool2)
    icp1_out1 = (Conv2D(128, (3, 3), strides=(1,1), name='icp1_out1', activation='relu'))(icp1_reduction1)

    icp1_reduction2 = Conv2D(16, (1, 1), strides=(1,1), name='icp1_reduction2', activation='relu')(pool2)
    icp1_padding2 = ZeroPadding2D(padding=(1,1))(icp1_reduction2)
    icp1_out2 = Conv2D(32, (5, 5), strides=(1,1), padding='valid', name='icp1_out2', activation='relu')(icp1_padding2)

    icp1_pool = MaxPooling2D((3, 3), strides=(1,1), name='icp1_pool')(pool2)
    icp1_out3 = Conv2D(32, (1, 1), strides=(1,1), name='icp1_out3', activation="relu")(icp1_pool)

    icp1_out0 = Conv2D(64, (1, 1), padding="valid", name='icp1_out0', activation='relu')(Cropping2D(cropping=1)(pool2))
    #
    #
    # # inception block2
    icp2_in = concatenate([icp1_out0, icp1_out1, icp1_out2, icp1_out3], axis=3, name='icp2_in')
    icp2_reduction1 = Conv2D(128, (1,1), strides=1, name='icp2_reduction1', activation='relu')(icp2_in)
    icp2_out1 = Conv2D(192, (3, 3), strides=1, name='icp2_out1', activation='relu')(icp2_reduction1)

    icp2_reduction2 = Conv2D(32, (1,1), strides=1, name='icp2_reduction2', activation='relu')(icp2_in)
    icp2_padding = ZeroPadding2D((1,1))(icp2_reduction2)
    icp2_out2 = Conv2D(96, (5, 5), strides=1, name='icp2_out2', activation='relu')(icp2_padding)

    icp2_pool = MaxPooling2D((3, 3), strides=1, name='icp2_pool')(icp2_in)
    icp2_out3 = Conv2D(64, (1, 1), strides=1, name='icp2_out3', activation='relu')(icp2_pool)

    icp2_out0 = Conv2D(128, (1, 1), strides=1, name='icp2_out0', activation='relu')(Cropping2D(cropping=1)(icp2_in))


    # inception block3
    icp2_out = concatenate([icp2_out0, icp2_out1, icp2_out2, icp2_out3], name='icp2_out')
    # icp3_in = MaxPooling2D((3,3), strides=2, name='icp3_in')(icp2_out)
    icp3_reduction1 = Conv2D(112, (1,1), strides=1, name='icp3_reduction1',activation='relu')(icp2_out)
    icp3_out1 = Conv2D(224, (3,3), strides=1, name='icp3_out1', activation='relu')(icp3_reduction1)

    icp3_reduction2 = Conv2D(24, (1,1), strides=1, name='icp3_reduction2', activation='relu')(icp2_out)
    icp3_padding = ZeroPadding2D((1,1))(icp3_reduction2)
    icp3_out2 = Conv2D(64, (5,5), strides=1, name='icp3_out2', activation='relu')(icp3_padding)

    icp3_pool = MaxPooling2D((3,3), strides=1, name='icp3_pool')(icp2_out)
    icp3_out3 = Conv2D(64, (1,1), strides=1, name='icp3_out3', activation='relu')(icp3_pool)

    icp3_out0 = Conv2D(160, (1,1), strides=1, name='icp3_out0', activation='relu')(Cropping2D(cropping=1)(icp2_out))


    # inception block4
    icp3_out = concatenate([icp3_out0, icp3_out1, icp3_out2, icp3_out3], name='icp3_out')
    icp4_reduction1 = Conv2D(160, (1,1), strides=1, name='icp4_reduction1', activation='relu')(icp3_out)
    icp4_out1 = Conv2D(320, (3,3), strides=1, name='icp4_out1', activation='relu')(icp4_reduction1)

    icp4_reduction2 = Conv2D(32, (1,1), strides=1, name='icp4_reduction2', activation='relu')(icp3_out)
    icp4_padding = ZeroPadding2D((1,1))(icp4_reduction2)
    icp4_out2 = Conv2D(128, (5,5), strides=1, name='icp4_out2', activation='relu')(icp4_padding)

    icp4_pool = MaxPooling2D((3,3), strides=1, name='icp4_pool')(icp3_out)
    icp4_out3 = Conv2D(128, (1,1), strides=1, name='icp4_out3', activation='relu')(icp4_pool)

    icp4_out0 = Conv2D(256, (1,1), strides=1, name='icp4_out0', activation='relu')(Cropping2D(cropping=1)(icp3_out))
    #
    #
    icp4_out = concatenate([icp4_out0, icp4_out1, icp4_out2, icp4_out3], name='icp4_out')
    cls3_pool = AveragePooling2D((5,5), strides=3, name='cls3_pool')(icp4_out)
    cls3_reduction = Conv2D(128, (1,1), strides=1, name='cls3_reduction', activation='relu')(cls3_pool)
    cls3_flatten = Flatten()(cls3_reduction)
    cls3_fc1 = Dense(1024, name='cls3_fc1')(cls3_flatten)
    cls3_fc2 = Dense(7354, name='cls3_fc2')(cls3_fc1)
    loss = Activation('softmax')(cls3_fc2)

    model = Model(inputs=input, outputs=loss)
    model.summary()
    setWeights(model)
    return model


def classify(filenames, model):
    for name in filenames:
        filePath = "./images/"+name
        img = image.load_img(filePath, target_size=(150,150))
        x = image.img_to_array(img)
        # print(x.shape)
        x = np.expand_dims(x, axis=0)
        # print(x.shape)
        y = model.predict(x)
        print(np.argmax(y[0]))
        print(np.sort(y[0]))


# def test():
#     input_a = np.reshape([1, 2, 3], (1, 1, 3))
#     input_b = np.reshape([4, 5, 6], (1, 1, 3))
#
#     a = Input(shape=(1, 3))
#     b = Input(shape=(1, 3))
#
#     concat = merge([a, b], mode='concat', concat_axis=-1)
#     dot = merge([a, b], mode='dot', dot_axes=2)
#     cos = merge([a, b], mode='cos', dot_axes=2)
#
#     model_concat = Model(input=[a, b], output=concat)
#     model_dot = Model(input=[a, b], output=dot)
#     model_cos = Model(input=[a, b], output=cos)
#
#     print(model_concat.predict([input_a, input_b]))
#     print(model_dot.predict([input_a, input_b]))
#     print(model_cos.predict([input_a, input_b]))


if __name__ == '__main__':
    model = build_model()
    # setWeights(model)
    # classify(model)

    classify(["zhong1.png", "zhong2.png", "zhong3.png", "zhong4.png"], model)
