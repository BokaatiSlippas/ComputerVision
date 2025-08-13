from helper_functions import load_and_rebuild_generator, cnn_model



loaded = load_and_rebuild_generator(filepath="fer2013_preprocessed.pkl")
train_generator = loaded["train_generator"]
validation_data = loaded["validation_data"]
class_weights = loaded["class_weights"]
metadata = loaded["metadata"]


model, history = cnn_model(train_generator=train_generator, validation_data=validation_data, class_weights=class_weights)

model.save('fer_model.keras')