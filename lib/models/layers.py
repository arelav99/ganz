from tensorflow.keras import layers as l
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input


def encoder_block(layer, ngf, kernel_size, strides, padding="same"):
    layer = l.Conv2D(ngf, kernel_size, strides=strides, padding=padding)(layer)
    layer = l.LeakyReLU(alpha=0.05)(layer)
    return l.BatchNormalization()(layer)


def decoder_block(layer, ngf, kernel_size, strides, padding="same"):
    layer = l.Conv2DTranspose(
        ngf, kernel_size, strides=strides, padding=padding)(layer)
    layer = l.ReLU()(layer)
    return l.BatchNormalization()(layer)


def create_encoder(opts, prev_model=None):
    ngf = opts.ngf
    size = opts.size

    if prev_model is None:
        input_ = Input((opts.size, opts.size, opts.nc))
    else:
        layer = prev_model

    inx = 0
    while size != 4:
        if inx == 0 and prev_model is None:
            layer = encoder_block(input_, ngf, (3, 3), (2, 2))
            inx += 1
        else:
            layer = encoder_block(layer,  ngf, (3, 3), (2, 2))

        ngf *= 2
        size //= 2

    last_conv = encoder_block(layer, opts.nz, (4, 4), (1, 1), padding="valid")
    if prev_model is None:
        return last_conv, input_
    else:
        return last_conv


def create_decoder(opts, prev_model):
    curr_size = 1
    itr_size = 4
    ngf = opts.ngf
    inx = 0

    while itr_size != opts.size // 2:
        ngf *= 2
        itr_size *= 2

    while curr_size != opts.size / 4:
        if inx == 0:
            layer = decoder_block(prev_model, ngf, kernel_size=(
                4, 4), strides=(1, 1), padding="valid")
            inx += 1
        else:
            layer = decoder_block(
                layer, ngf, kernel_size=(3, 3), strides=(2, 2))

        curr_size *= 2
        ngf //= 2

    return l.Conv2DTranspose(opts.nc, (3, 3), strides=(2, 2), padding="same",               activation="tanh")(layer)


def build_generator_skipgan(opts):
    size, ngf = opts.size, opts.ngf
    input_ = Input((opts.size, opts.size, opts.nc))
    layers = []
    inx = 0

    while True:
        if inx == 0:
            layer = encoder_block(input_, ngf, (4, 4), (2, 2))
            inx += 1
        else:
            layer = encoder_block(layer, ngf, (4, 4), (2, 2))
        size /= 2
        if size == 1:
            break
        ngf *= 2
        layers.append(layer)

    ngf /= 2
    inx = 1
    while size != opts.size // 2:
        layer = decoder_block(layer, ngf, (4, 4), (2, 2))
        layer = l.concatenate([layers[-inx], layer])
        inx += 1
        size *= 2
        ngf /= 2

    final_gen = l.Conv2DTranspose(opts.nc, (3, 3), strides=(
        2, 2), padding="same",  activation="tanh")(layer)
    return Model(inputs=input_, outputs=[final_gen])


def build_discriminator_skipgan(opts):
    inx, size, ngf = 0, opts.size, opts.ngf
    input_ = Input((opts.size, opts.size, opts.nc))

    while True:
        if inx == 0:
            layer = encoder_block(input_, ngf, (4, 4), (2, 2))
            inx += 1
        else:
            layer = encoder_block(layer, ngf, (4, 4), (2, 2))

        ngf *= 2
        size /= 2
        if size == 4:
            break

    conv_disc = encoder_block(layer, opts.nz, (4, 4), (1, 1), padding="valid")
    final_disc = l.Dense(1, activation="sigmoid")(conv_disc)
    return Model(inputs=input_, outputs=[conv_disc, final_disc])


def build_generator_gan(opts):
    enc1, input_ = create_encoder(opts)
    dec = create_decoder(opts, prev_model=enc1)
    enc2 = create_encoder(opts, prev_model=dec)

    return Model(inputs=input_, outputs=[enc1, dec, enc2])


def build_discriminator_gan(opts):
    labels, input_ = create_encoder(opts)
    enc = l.Dense(1, activation="sigmoid")(labels)

    return Model(inputs=input_, outputs=[labels, enc])
