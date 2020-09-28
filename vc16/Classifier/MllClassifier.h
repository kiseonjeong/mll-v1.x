#ifndef MLL_CLASSIFIER_H
#define MLL_CLASSIFIER_H

namespace mll
{
	// The Classifier for Machine Learning
	class mllclassifier : public mllutil
	{
		// Variables
	public:


		// Functions
	public:
		// Train the dataset
		virtual void train(const mlldata& dataset) = 0;
		// Predict a response
		virtual const double predict(const nml::numat& x) = 0;
		// Open the trained parameters
		virtual const int open(const std::string path, const std::string prefix = "") = 0;
		// Save the trained parameters
		virtual const int save(const std::string path, const std::string prefix = "") = 0;

		// Variables
	protected:


		// Functions
	protected:
		// Find section name
		const bool findSectionName(std::ifstream& reader, const std::string sectionName);
		// Get section name for parameter writing
		const std::string getSectionName(const std::string name, const std::string prefix);

	};
}

#endif