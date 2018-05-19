from utils.dataset import Dataset
from utils.improved import Improved
from utils.scorer import report_score


def execute_demo(language, size=0):
	
	data = Dataset(language)
	if size:
		data.trainset = data.trainset[0:size]

	print("{}: {} training - {} dev - {} test".format(language, len(data.trainset), len(data.devset), len(data.testset)))

	improved = Improved(language)

	improved.train(data.trainset)

	predictions_dev = improved.test(data.devset)
	predictions_test = improved.test(data.testset)

	gold_labels_dev = [sent['gold_label'] for sent in data.devset]
	gold_labels_test = [sent['gold_label'] for sent in data.testset]

	if size:
		print("dev score size = " + str(size))
		report_score(gold_labels_dev, predictions_dev)
		print("test score size = " + str(size))
		report_score(gold_labels_test, predictions_test)
		print('-' * 50)
	else:
		print("dev score")
		report_score(gold_labels_dev, predictions_dev)
		print("test score")
		report_score(gold_labels_test, predictions_test)
		print('-' * 50)


if __name__ == '__main__':
	i = 100
	while i <= 1000:
		execute_demo('english', i)
		execute_demo('spanish', i)
		i += 100
	print('\n' + '*' * 50 + '\n')
	execute_demo('english')
	execute_demo('spanish')

	


