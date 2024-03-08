from keras.preprocessing.image import ImageDataGenerator

def create_data_augmentation_generators(train_dir, validation_dir, test_dir, batch_size=32):
    """
    Create data augmentation generators for training, validation, and testing datasets.

    Parameters:
    - train_dir: Directory with training images.
    - validation_dir: Directory with validation images.
    - test_dir: Directory with test images.
    - batch_size: Size of the batches of data.

    Returns:
    - train_generator: Generator for training data.
    - validation_generator: Generator for validation data.
    - test_generator: Generator for test data.
    """
    # Training data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Validation data should not be augmented, just rescaled
    validation_datagen = ImageDataGenerator(rescale=1./255)

    # Test data should not be augmented, just rescaled
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Set up the generators
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary'
    )

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(
