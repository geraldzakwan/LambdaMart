from lambdamart import LambdaMART
from test import get_data

import numpy as np
import warnings
warnings.filterwarnings('ignore')

def main():
	training_data = get_data("example_data/train.txt")
	test_data = get_data("example_data/test.txt")

	print("-" * 50)
	print("Train data length: {}".format(len(training_data)))
	print("Test data length: {}".format(len(test_data)))

	print("-" * 50)
	print("Model training...")
	model = LambdaMART(training_data=training_data, number_of_trees=2, learning_rate=0.1)
	model.fit()

	print("-" * 50)
	print("Sample test data query IDs: ")
	sample_size = 10
	print(test_data[:sample_size, 1])

	print("-" * 50)
	print("Sample scores when predicting at once: ")
	predicted_scores_1 = model.predict(test_data[:, 1:])
	predicted_scores_1 = predicted_scores_1[:sample_size]
	print(predicted_scores_1)

	print("-" * 50)
	print("Sample scores when predicting per item: ")
	predicted_scores_2 = []
	for item in test_data[:sample_size]:
		predicted_scores_2.append(model.predict(np.array([item[1:]]))[0])
	predicted_scores_2 = np.array(predicted_scores_2)
	print(predicted_scores_2)

	np.testing.assert_array_equal(predicted_scores_1, predicted_scores_2)
	print("-" * 50)

if __name__ == '__main__':
    main()
