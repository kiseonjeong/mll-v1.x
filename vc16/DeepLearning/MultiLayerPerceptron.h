#ifndef MULTI_LAYER_PERCEPTRON_H
#define MULTI_LAYER_PERCEPTRON_H

namespace mll
{
	// The Training Type for Multi Layer Perceptron
	typedef enum _trtype
	{
		MLP_TRAIN_UNKNOWN = -1,
		MLP_TRAIN_CLASSIFICATION,
		MLP_TRAIN_REGRESSION,
		MLP_TRAIN_GENERATION,
	} trtype;

	// The Multi Layer Perceptron Classifier
	class MLP : public mllclassifier
	{
		// Variables
	public:
		// Progress interval
		int progInterval;
		// Network weight
		nml::numem<nml::numat> W;
		// Train dataset and cost
		mlldata trD;
		std::vector<double> trC;
		// Test dataset and cost
		mlldata teD;
		std::vector<double> teC;
		// Validation dataset and cost
		mlldata vdD;
		std::vector<double> vdC;

		// Functions
	public:
		// Set a train condition
		// type : training type (0 = classification, 1 = regression)
		// N : mini batch size
		// E : target error rate
		// maxIter : maximum iterations
		// hl : hidden layer architecture
		// init : initializer
		// opt : optimizer
		// lamda : panelty term for regularization (0.01 ~ 0.00001)
		// idprob : dropout probability on the input layer
		// annealer : annealing function
		void condition(const int type, const int N, const double E, const int maxIter, const std::vector<netlayer>& hl, const initializer& init, const optimizer& opt, const regularizer& reg);
		// Train the dataset
		// dataset : training dataset
		// type : training type (0 = classification, 1 = regression)
		// N : mini batch size
		// E : target error rate
		// maxIter : maximum iterations
		// hl : hidden layer architecture
		// init : initializer
		// opt : optimizer
		// lamda : panelty term for regularization (0.01 ~ 0.00001)
		// idprob : dropout probability on the input layer
		// annealer : annealing function
		virtual void train(const mlldata& dataset);
		void train(const mlldata& dataset, const int type, const int N, const double E, const int maxIter, const std::vector<netlayer>& hl, const initializer& init, const optimizer& opt, const regularizer& reg);
		// Predict a response
		const double predict(const nml::numat& x);
		// Open the trained parameters
		const int open(const std::string path, const std::string prefix = "");
		// Save the trained parameters
		virtual const int save(const std::string path, const std::string prefix = "");

		// Operators
	public:
		MLP& operator=(const MLP& obj);

		// Constructors & Destructor
	public:
		MLP();
		// type : training type (0 = classification, 1 = regression)
		// N : mini batch size
		// E : target error rate
		// maxIter : maximum iterations
		// hl : hidden layer architecture
		// init : initializer
		// opt : optimizer
		// lamda : panelty term for regularization (0.01 ~ 0.00001)
		// idprob : dropout probability on the input layer
		// annealer : annealing function
		MLP(const int type, const int N, const double E, const int maxIter, const std::vector<netlayer>& hl, const initializer& init, const optimizer& opt, const regularizer& reg);
		// dataset : training dataset
		// type : training type (0 = classification, 1 = regression)
		// N : mini batch size
		// E : target error rate
		// maxIter : maximum iterations
		// hl : hidden layer architecture
		// init : initializer
		// opt : optimizer
		// lamda : panelty term for regularization (0.01 ~ 0.00001)
		// idprob : dropout probability on the input layer
		// annealer : annealing function
		MLP(const mlldata& dataset, const int type, const int N, const double E, const int maxIter, const std::vector<netlayer>& hls, const initializer& init, const optimizer& opt, const regularizer& reg);
		MLP(const MLP& obj);
		~MLP();

		// Variables
	protected:
		// Mini batch size
		int N;
		// Error rate
		double E;
		// Max iteration
		int maxIter;
		// Hidden layer information
		std::vector<netlayer> hl;
		// Network architecture
		nml::numem<netlayer> net;
		// Weight initializer
		initializer* init;
		// Gradient optimizer
		optimizer* opt;
		// Weight Regularizer
		regularizer* reg;
		// Train type
		int type;
		// Feature vector
		nml::numat X;
		// Target vector
		nml::numat T;
		// Class vector
		nml::numat C;
		// Node output
		nml::numem<nml::numat> nout;
		// Layer output
		nml::numem<nml::numat> lout;
		// Gradient matrix
		nml::numem<nml::numat> grad;
		// Delta matrix
		nml::numem<nml::numat> delta;
		// Number of input nodes
		int inode;
		// Number of output nodes
		int onode;
		// Number of hidden nodes
		std::vector<int> hnode;

		// Functions
	protected:
		// Set an object
		virtual void setObject();
		// Copy the object
		virtual void copyObject(const nml::object& obj);
		// Clear the object
		virtual void clearObject();
		// Copy the initializer
		void copyInitializer(const initializer& init);
		// Copy the optimizer
		void copyOptimizer(const optimizer& opt);
		// Copy the regularizer
		void copyRegularizer(const regularizer& reg);
		// Backup the dataset
		virtual void backupDataset(const mlldata& dataset);
		// Create a network architecture
		virtual void createNetwork();
		// Create the cache memories
		void createCaches();
		// Set the dropout layers
		void setDropoutLayers();
		// Initialize a gradient matrix
		void initGradient();
		// Compute forward propagations
		virtual const double computeForwardProps(const mlldata& dataset, nml::numat& target, const bool tr = true);
		// Set an input and an output vector
		virtual void setInoutVectors(const mlldata& dataset, nml::numat& target);
		// Calculate a cost value on the Mean Squared Error
		virtual const double calculateMeanSquaredError(const nml::numat& y, const nml::numat& t, const bool tr = true) const;
		// Calculate a cost value on the Cross Entropy Error
		virtual const double calculateCrossEntropyError(const nml::numat& y, const nml::numat& t, const bool tr = true) const;
		// Calculate a cost value on the Cross Entropy Error
		virtual const double calculateNegLogLikelihoodError(const nml::numat& y, const nml::numat& t, const bool tr = true) const;
		// Compute backward propagations
		virtual void computeBackwardProps(const nml::numat& target);
		// Get the weight matrix
		const nml::numat getWeightMatrix(const nml::numat& Wi);
		// Get the bias matrix
		const nml::numat getBiasMatrix(const nml::numat& Wi);
		// Update the network weight
		void updateNetwork(const double iter, const int N);
		// Get a sign matrix
		const nml::numat getSignMatrix(const nml::numat& Wi);
		// Get an Argmax
		const double getArgmax(const nml::numat& vec);
		// Open train condition information
		virtual const int openTrainCondInfo(const std::string path, const std::string prefix);
		// Open label information
		virtual const int openLabelInfo(const std::string path, const std::string prefix);
		// Open layer information
		virtual const int openLayerInfo(const std::string path, const std::string prefix);
		// Open the hidden architecture
		virtual const int openHiddenArchitecture(const std::string path, const std::string prefix);
		// Open weight information
		virtual const int openWeightInfo(const std::string path, const std::string prefix);

		// Variables
	private:


		// Functions
	private:

	};

	// Nick name
	typedef MLP FFNN;			// feed-forward neural network
}

#endif