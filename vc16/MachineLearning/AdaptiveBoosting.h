#ifndef ADAPTIVE_BOOSTING_H
#define ADAPTIVE_BOOSTING_H

namespace mll
{
	// The Stump Data Structure
	typedef struct _stump
	{
		double eps;
		nml::numat pred;
		int dim;
		double thres;
		int type;
		double alpha;
	} stump;

	// The Inequality Type for Adaboost
	typedef enum _ietype
	{
		INEQUAL_UNKNOWN = -1,
		INEQUAL_LT,
		INEQUAL_GT,
	} ietype;

	// The Adaptive Boosting Classifier
	class adaboost : public mllclassifier
	{
		// Variables
	public:

		
		// Functions
	public:
		// Set a train condition
		// nwc : target number of weak classifiers
		void condition(const int nwc);
		// Train the dataset
		// dataset : training dataset
		void train(const mlldata& dataset);
		// dataset : training dataset
		// nwc : target number of weak classifiers
		void train(const mlldata& dataset, const int nwc);
		// Predict a response
		const double predict(const nml::numat& x);
		// Open the trained parameters
		const int open(const std::string path, const std::string prefix = "");
		// Save the trained parameters
		const int save(const std::string path, const std::string prefix = "");

		// Operators
	public:
		adaboost& operator=(const adaboost& obj);

		// Constructors & Destructor
	public:
		adaboost();
		// nwc : target number of weak classifiers
		adaboost(const int nwc);
		// dataset : training dataset
		// nwc : target number of weak classifiers
		adaboost(const mlldata& dataset, const int nwc);
		adaboost(const adaboost& obj);
		~adaboost();

		// Variables
	private:
		// Number of iterations
		int nwc;
		// Number of feature dimensions
		int fdim;
		// Feature vector
		nml::numat X;
		// Target vector
		nml::numat T;
		// Class vector
		nml::numat C;
		// Weight vector
		nml::numat D;
		// Error vector
		nml::numat E;
		// Weak classifiers
		std::vector<stump> WC;

		// Functions
	private:
		// Set an object
		void setObject();
		// Copy the object
		void copyObject(const nml::object& obj);
		// Clear the object
		void clearObject();
		// Build a stump
		const stump buildStump();
		// Classify a stump
		const nml::numat classifyStump(const nml::numat& feature, const double thres, const int inequal) const;
		// Compare classify results
		const nml::numat compareResults(const nml::numat& pred, const nml::numat& real) const;
		// Calculate an error rate
		const double calculateError(const stump& bestStump) const;
		// Get a signed vector
		const nml::numat sign(const nml::numat& vector) const;
		// Open train condition information
		const int openTrainCondInfo(const std::string path, const std::string prefix);
		// Open weak classifier information
		const int openWeakClassifierInfo(const std::string path, const std::string prefix);

	};
}

#endif