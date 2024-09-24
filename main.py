import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import librosa

# 1. Data Preprocessing
def preprocess_audio(file_path, sr=22050, duration=5):
    audio, _ = librosa.load(file_path, sr=sr, duration=duration)
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
    return librosa.power_to_db(mel_spec, ref=np.max)

# 2. VAE-GAN Model
class VAEGAN(keras.Model):
    def __init__(self, latent_dim):
        super(VAEGAN, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.discriminator = self.build_discriminator()

    def build_encoder(self):
        encoder_inputs = keras.Input(shape=(128, 87, 1))
        x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
        x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Flatten()(x)
        x = layers.Dense(16 * self.latent_dim)(x)

        z_mean = layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(x)
        z = layers.Lambda(self.sampling, output_shape=(self.latent_dim,), name="z")([z_mean, z_log_var])
        
        return keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    def build_decoder(self):
        latent_inputs = keras.Input(shape=(self.latent_dim,))
        x = layers.Dense(32 * 22 * 64, activation="relu")(latent_inputs)
        x = layers.Reshape((32, 22, 64))(x)
        x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
        decoder_outputs = layers.Conv2DTranspose(1, 3, activation="linear", padding="same")(x)
        
        return keras.Model(latent_inputs, decoder_outputs, name="decoder")

    def build_discriminator(self):
        inputs = keras.Input(shape=(128, 87, 1))
        x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(inputs)
        x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Flatten()(x)
        x = layers.Dense(1)(x)
        
        return keras.Model(inputs, x, name="discriminator")

    def sampling(self, args):
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        batch_size = tf.shape(data)[0]

        # VAE training
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                keras.losses.mse(data, reconstruction)
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.encoder.trainable_weights + self.decoder.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.encoder.trainable_weights + self.decoder.trainable_weights))

        # Discriminator training
        generated_data = self.decoder(tf.random.normal(shape=(batch_size, self.latent_dim)))
        with tf.GradientTape() as tape:
            real_output = self.discriminator(data)
            fake_output = self.discriminator(generated_data)
            d_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)

        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        # Generator training
        with tf.GradientTape() as tape:
            generated_data = self.decoder(tf.random.normal(shape=(batch_size, self.latent_dim)))
            fake_output = self.discriminator(generated_data)
            g_loss = -tf.reduce_mean(fake_output)

        grads = tape.gradient(g_loss, self.decoder.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.decoder.trainable_weights))

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
            "d_loss": d_loss,
            "g_loss": g_loss,
        }

# 3. Training function
def train_model(model, data, epochs=50, batch_size=32):
    model.compile(optimizer=keras.optimizers.Adam())
    model.fit(data, epochs=epochs, batch_size=batch_size)

# 4. Voice conversion function
def convert_voice(model, input_audio):
    z_mean, _, _ = model.encoder(input_audio[np.newaxis, ..., np.newaxis])
    return model.decoder(z_mean).numpy()[0, ..., 0]

# 5. Main application
def main():
    # Load and preprocess data
    source_audio = preprocess_audio('path_to_source_audio.wav')
    target_audio = preprocess_audio('path_to_target_audio.wav')
    
    # Combine source and target data
    combined_data = np.concatenate([source_audio, target_audio], axis=1)
    combined_data = combined_data[..., np.newaxis]
    
    # Build and train the model
    model = VAEGAN(latent_dim=128)
    train_model(model, combined_data)
    
    # Convert a new audio file
    new_audio = preprocess_audio('path_to_new_audio.wav')
    converted_audio = convert_voice(model, new_audio)
    
    # Save the converted audio
    librosa.output.write_wav('converted_audio.wav', 
                             librosa.feature.inverse.mel_to_audio(librosa.db_to_power(converted_audio)), 
                             sr=22050)

if __name__ == "__main__":
    main()
