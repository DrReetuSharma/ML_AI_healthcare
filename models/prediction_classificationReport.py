prediction = model.predict(test_gen)

y_pred = np.argmax(prediction , axis = 1)

print(classification_report(test_gen.classes, y_pred , target_names= classes ))
