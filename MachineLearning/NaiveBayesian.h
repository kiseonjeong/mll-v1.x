#ifndef NAIVE_BAYESIAN_H
#define NAIVE_BAYESIAN_H

namespace mll
{
	// The Naive Bayesian Classifier
	class naivebayes : public mllclassifier
	{
		// Variables
	public:


		// Functions
	public:
		// Train the dataset
		void train(const mlldata& dataset);
		// Predict a response
		const double predict(const nml::numat& x);
		// Open the trained parameters
		const int open(const std::string path, const std::string prefix = "");
		// Save the trained parameters
		const int save(const std::string path, const std::string prefix = "");

		// Operators
	public:
		naivebayes& operator=(const naivebayes& obj);

		// Constructors & Destructor
	public:
		naivebayes();
		naivebayes(const mlldata& dataset);
		naivebayes(const naivebayes& obj);
		~naivebayes();

		// Variables
	private:
		// Prior probability
		nml::numat prior;
		// Conditional probability
		nml::numat cond;
		// Normalization matrix
		nml::numat denom;
		// Class vector
		nml::numat C;
		// Rows length
		int vrows;
		// Column length
		int vcols;

		// Functions
	private:
		// Set an object
		void setObject();
		// Copy the object
		void copyObject(const nml::object& obj);
		// Clear the object
		void clearObject();
		// Open label information
		const int openLabelInfo(const std::string path, const std::string prefix);
		// Open probability information
		const int openProbInfo(const std::string path, const std::string prefix);

	};
}

#endif