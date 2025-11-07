def train_model(model, train_gen, val_gen, epochs=20):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_gen, validation_data=val_gen, epochs=epochs, verbose=1)
    return model, history
