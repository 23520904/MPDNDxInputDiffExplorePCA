"""CNN architecture for the differential-neural distinguisher
(Gohr-style wide-narrow dilated conv blocks)."""

from tensorflow.keras.layers import Add, Activation, BatchNormalization, Conv1D, Dense, Flatten, Input, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

def get_dilation_rates(input_size):
    """Dilation rates for successive wide-narrow conv blocks."""
    rates = []
    while input_size >= 8:
        rates.append(int(input_size / 2 - 1))
        input_size = input_size // 2
    return rates

def make_model(input_size=224, n_filters=32, n_add_filters=16):
    """Build the wide-narrow dilated-conv distinguisher model.

    :param input_size: Total bits (e.g., 224 for 'full' mode in SPECK32).
    """
    dilation_rates = get_dilation_rates(input_size)

    d1 = 256
    d2 = 64
    reg_param = 1e-5

    inputs = Input(shape=(input_size,))
    # Reshape to (input_size, 1) for Conv1D
    x = Reshape((input_size, 1))(inputs)

    # NOTE: Normalization is handled by the Generator (0,1 -> -1,1),
    # so we do not repeat it here to avoid scaling errors.

    for dilation_rate in dilation_rates:
        # wide-narrow block
        x = Conv1D(filters=n_filters, kernel_size=2, padding='valid',
                   dilation_rate=dilation_rate, strides=1, activation='relu')(x)
        x = BatchNormalization()(x)

        x_skip = x
        x = Conv1D(filters=n_filters, kernel_size=2, padding='causal',
                   dilation_rate=1, activation='relu')(x)
        x = Add()([x, x_skip])
        x = BatchNormalization()(x)

        n_filters += n_add_filters

    out = Flatten()(x)
    dense0 = Dense(d1, kernel_regularizer=l2(reg_param))(out)
    dense0 = BatchNormalization()(dense0)
    dense0 = Activation('relu')(dense0)
    
    dense1 = Dense(d1, kernel_regularizer=l2(reg_param))(dense0)
    dense1 = BatchNormalization()(dense1)
    dense1 = Activation('relu')(dense1)
    
    dense2 = Dense(d2, kernel_regularizer=l2(reg_param))(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Activation('relu')(dense2)
    
    out = Dense(1, activation='sigmoid', kernel_regularizer=l2(reg_param))(dense2)

    return Model(inputs, out)