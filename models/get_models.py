# get_models.py

from . import encoders


def get_encoder(enc_name, bottleneck_size, latent_size, input_dim, pe_dim):
    encoder = getattr(encoders, enc_name)(bottleneck_size=bottleneck_size,
                                            latent_size=latent_size,
                                            input_dim=input_dim,
                                            pe_dim=pe_dim)
    return encoder


def get_decoder(dec_name, bottleneck_size, coord_dim, latent_dim):
    decoder = getattr(encoders, dec_name)(bottleneck_size=bottleneck_size,
                                          coord_dim=coord_dim,
                                          latent_dim=latent_dim)
    return decoder
