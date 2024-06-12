epochs=12
history=model.fit(train_gen,epochs=epochs ,validation_data=valid_gen ,verbose=0)
