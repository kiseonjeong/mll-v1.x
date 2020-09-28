#include "stdafx.h"
#include "MultiLayerPerceptron.h"

namespace mll
{
	MLP::MLP()
	{
		// Set an object
		setObject();
	}

	MLP::MLP(const int type, const int N, const double E, const int maxIter, const vector<netlayer>& hl, const initializer& init, const optimizer& opt, const regularizer& reg)
	{
		// Set an object
		setObject();

		// Set a train condition
		condition(type, N, E, maxIter, hl, init, opt, reg);
	}

	MLP::MLP(const mlldata& dataset, const int type, const int N, const double E, const int maxIter, const vector<netlayer>& hl, const initializer& init, const optimizer& opt, const regularizer& reg)
	{
		// Set an object
		setObject();

		// Train the dataset
		train(dataset, type, N, E, maxIter, hl, init, opt, reg);
	}

	MLP::MLP(const MLP& obj)
	{
		// Set an object
		setObject();

		// Clone the object
		*this = obj;
	}

	MLP::~MLP()
	{
		// Clear the object
		clearObject();
	}

	MLP& MLP::operator=(const MLP& obj)
	{
		// Clear the object
		clearObject();

		// Set an object
		setObject();

		// Copy the object
		copyObject(obj);

		return *this;
	}

	void MLP::setObject()
	{
		// Set the parameters
		setType(*this);
		progInterval = 1;
		N = 0;
		E = 0.0;
		maxIter = 0;
		type = MLP_TRAIN_UNKNOWN;

		// Set the memories
		hl.clear();
		net.release();
		init = nullptr;
		opt = nullptr;
		reg = nullptr;
		X.release();
		T.release();
		C.release();
		nout.release();
		lout.release();
		W.release();
		grad.release();
		delta.release();
	}

	void MLP::copyObject(const object& obj)
	{
		// Do down casting
		const MLP* _obj = static_cast<const MLP*>(&obj);

		// Copy the parameters
		progInterval = _obj->progInterval;
		N = _obj->N;
		E = _obj->E;
		maxIter = _obj->maxIter;
		type = _obj->type;

		// Copy the memories
		if (_obj->init != nullptr)
		{
			copyInitializer(*_obj->init);
		}
		if (_obj->opt != nullptr)
		{
			copyOptimizer(*_obj->opt);
		}
		if (_obj->reg != nullptr)
		{
			copyRegularizer(*_obj->reg);
		}
	}

	void MLP::clearObject()
	{
		// Clear the memories
		hl.clear();
		net.release();
		if (init != nullptr)
		{
			delete init;
		}
		if (opt != nullptr)
		{
			delete opt;
		}
		if (reg != nullptr)
		{
			delete reg;
		}
		X.release();
		T.release();
		C.release();
		nout.release();
		lout.release();
		W.release();
		grad.release();
		delta.release();
	}

	void MLP::copyInitializer(const initializer& init)
	{
		// Check the old memory
		if (this->init != nullptr)
		{
			delete this->init;
		}

		// Create an initializer
		this->init = new initializer(init);
	}

	void MLP::copyOptimizer(const optimizer& opt)
	{
		// Check the old memory
		if (this->opt != nullptr)
		{
			delete this->opt;
		}

		// Create an optimizer
		switch (opt.type)
		{
		case OPTIMIZER_SGD: this->opt = new sgd((sgd&)opt); break;
		case OPTIMIZER_MOMENTUN: this->opt = new momentum((momentum&)opt); break;
		case OPTIMIZER_NESTEROV: this->opt = new nesterov((nesterov&)opt); break;
		case OPTIMIZER_ADAGRAD: this->opt = new adagrad((adagrad&)opt); break;
		case OPTIMIZER_RMSPROP: this->opt = new rmsprop((rmsprop&)opt); break;
		case OPTIMIZER_ADADELTA: this->opt = new adadelta((adadelta&)opt); break;
		case OPTIMIZER_ADAM: this->opt = new adam((adam&)opt); break;
		default: this->opt = nullptr; break;
		}
	}

	void MLP::copyRegularizer(const regularizer& reg)
	{
		// Check the old memory
		if (this->reg != nullptr)
		{
			delete this->reg;
		}

		// Create an initializer
		this->reg = new regularizer(reg);
	}

	void MLP::condition(const int type, const int N, const double E, const int maxIter, const std::vector<netlayer>& hl, const initializer& init, const optimizer& opt, const regularizer& reg)
	{
		// Set the conditions
		this->type = type;
		this->N = N;
		this->E = E;
		this->maxIter = maxIter;
		this->hl = hl;
		copyInitializer(init);
		copyOptimizer(opt);
		copyRegularizer(reg);
	}

	void MLP::train(const mlldata& dataset)
	{
		// Backup the dataset for training
		backupDataset(dataset);

		// Create a network architecture
		createNetwork();

		// Create the cache memories
		createCaches();

		// Clear the cost memories
		trC.clear();
		teC.clear();
		vdC.clear();

		// Optimize the network weight matrix using the Back Propagation Algorithm
		for (int i = 0; i < maxIter; i++)
		{
			// Generate the mini batch dataset
			vector<mlldata> mini = dataset.subdata(N);
			numat target;

			// Calculate a cost value on the train dataset
			if (i % progInterval == 0 && trD.empty() == false)
			{
				trC.push_back(computeForwardProps(trD, target, false));
			}

			// Calculate a cost value on the test dataset
			if (i % progInterval == 0 && teD.empty() == false)
			{
				teC.push_back(computeForwardProps(teD, target, false));
			}

			// Calculate a cost value on the validation dataset
			if (i % progInterval == 0 && vdD.empty() == false)
			{
				vdC.push_back(computeForwardProps(vdD, target, false));
			}

			// Do the Gradient Descent Optimization
			double batchCost = 0.0;
			const int nmini = (int)mini.size();
			for (int j = 0; j < nmini; j++)
			{
				// Set the dropout layers
				setDropoutLayers();

				// Initialize the gradients
				initGradient();

				// Compuate forward computations
				batchCost += computeForwardProps(mini[j], target, true);

				// Compute backward computations
				computeBackwardProps(target);

				// Update the network weight matrix
				updateNetwork(j + 1, mini[j][0].rows);
			}
			batchCost /= nmini;

			// Do annealing the learning rate
			opt->update(i);

			// Calculate the mean of the cost on the mini batch dataset
			const double trainCost = computeForwardProps(dataset, target, false);
			if (i % progInterval == 0)
			{
				cout << "Epoch : " << i + 1 << ", Cost Value : " << trainCost << endl;
			}

			// Check the stop condition
			if (trainCost < E || i >= maxIter - 1)
			{
				// Stop the training
				break;
			}
		}
	}

	void MLP::train(const mlldata& dataset, const int type, const int N, const double E, const int maxIter, const vector<netlayer>& hl, const initializer& init, const optimizer& opt, const regularizer& reg)
	{
		// Set a train condition
		condition(type, N, E, maxIter, hl, init, opt, reg);

		// Train the dataset
		train(dataset);
	}

	void MLP::backupDataset(const mlldata& dataset)
	{
		// Check the train type
		if (type == MLP_TRAIN_REGRESSION)
		{
			// Backup the dataset
			X = dataset[0];
			T = dataset[1];
			C = numat::zeros(msize(1));
		}
		else
		{
			// Backup the dataset
			X = dataset[0];
			T = dataset[1];
			C = dataset[2];
		}

		// Set an input node
		inode = 1 + X.cols;

		// Set hidden nodes
		for (int i = 0; i < (int)hl.size(); i++)
		{
			hnode.push_back(1 + hl[i].node);
		}

		// Set an output node
		onode = C.length();
	}

	void MLP::createNetwork()
	{
		// Initialize the parameters
		const int numlayers = 1 + (int)hl.size() + 1;
		nout = numem<numat>(msize(numlayers));
		lout = numem<numat>(msize(numlayers));
		net = numem<netlayer>(msize(numlayers));
		W = numem<numat>(msize(numlayers - 1));

		// Create a network architecture
		for (int i = 0; i < numlayers; i++)
		{
			// Check the layer index
			if (i == 0)
			{
				// Set the input layer
				net(i) = netlayer(X.cols, nn::identity());
				net(i).set(NET_LAYER_INPUT);
			}
			else if (i == numlayers - 1)
			{
				// Set the output layer
// 				net(i) = (type == MLP_TRAIN_REGRESSION) ? netlayer(C.length(), nn::identity()) : netlayer(C.length(), nn::softmax());
				net(i) = (type == MLP_TRAIN_REGRESSION) ? netlayer(C.length(), nn::identity()) : netlayer(C.length(), nn::logsoftmax());
				net(i).set(NET_LAYER_OUTPUT);
			}
			else
			{
				// Set the hidden layer
				net(i) = hl[i - 1];
				net(i).set(NET_LAYER_HIDDEN);
			}
		}

		// Create a network weight
		for (int i = 0; i < W.length(); i++)
		{
			// Create a matrix memory
			W(i) = numat::zeros(msize(net(i + 1).node, 1 + net(i).node));

			// Set the network weight
			init->generate(net(i + 1).getActFunc().type, W(i));
		}
	}

	void MLP::createCaches()
	{
		// Create a delta cache
		delta = numem<numat>(msize(W.length()));
		for (int i = 0; i < delta.length(); i++)
		{
			delta(i) = numat::zeros(msize(1, net(i + 1).node));
		}

		// Create a gradient cache
		grad = numem<numat>(msize(W.length()));
		for (int i = 0; i < grad.length(); i++)
		{
			grad(i) = numat::zeros(msize(W(i).rows, W(i).cols));
		}

		// Create a cache for the optimizer
		opt->create(W);
	}

	void MLP::setDropoutLayers()
	{
		// Check the number of the hidden layers
		const int n = (int)hnode.size();
		for (int i = 0; i < n + 1; i++)
		{
			if (i == 0)
			{
				reg->dolayers[i].generate(inode);
			}
			else
			{
				reg->dolayers[i].generate(hnode[i - 1]);
			}
		}
	}

	void MLP::initGradient()
	{
		// Initialize the gradient matrix
		for (int i = 0; i < grad.length(); i++)
		{
			for (int j = 0; j < grad(i).length(); j++)
			{
				grad(i)(j) = 0.0;
			}
		}
	}

	const double MLP::computeForwardProps(const mlldata& dataset, numat& target, const bool tr)
	{
		// Set the input and output vectors
		setInoutVectors(dataset, target);

		// Do forward propagations
		for (int i = 0; i < W.length(); i++)
		{
			// Check the training flag
			if (tr == true)
			{
				// Activate the dropout layer
				lout(i) = reg->dolayers[i].forward(lout(i));
			}

			// Activate the affine layer
			// from the current layer output to the next node output
			nout(i + 1) = lout(i) * W(i).T();

			// Check the current layer is output or not
			if (i != W.length() - 1)
			{
				// Check the training flag
				if (tr == true)
				{
					// Activate the batch normalization layer
					nout(i + 1) = reg->bnlayers[i].forward(nout(i + 1));
				}
				else
				{
					// Activate the batch normalization layer
					nout(i + 1) = reg->bnlayers[i].inference(nout(i + 1));
				}

				// Activatve the current nodes
				lout(i + 1) = numat::ones(msize(dataset[0].rows, 1)).happend(net(i + 1).activative(nout(i + 1)));
			}
			else
			{
				// Activatve the current nodes
				lout(i + 1) = net(i + 1).activative(nout(i + 1));
			}
		}

		// Check the train type
		double cost = 0.0;
		if (type == MLP_TRAIN_REGRESSION)
		{
			// Calculate the mean squared error
			cost = calculateMeanSquaredError(lout(lout.length() - 1), target, tr);
		}
		else
		{
			// Calculate the cross entropy error
// 			cost = calculateCrossEntropyError(lout[lout.length() - 1], target, tr);
			cost = calculateNegLogLikelihoodError(lout(lout.length() - 1), target, tr);
			lout(lout.length() - 1) = numat::exp(lout(lout.length() - 1));
		}

		return cost;
	}

	void MLP::setInoutVectors(const mlldata& dataset, numat& target)
	{
		// Set the input vector
		nout(0) = dataset[0];
		lout(0) = net(0).activative(nout(0));
		lout(0) = numat::ones(msize(dataset[0].rows, 1)).happend(lout(0));

		// Check the train type
		if (type == MLP_TRAIN_REGRESSION)
		{
			// Set a target value
			target = dataset[1];
		}
		else
		{
			// Set a target vector
			target = numat::zeros(msize(dataset[1].rows, C.length()));

			// Encode the target vector
			for (int i = 0; i < dataset[0].rows; i++)
			{
				// Find a label index
				int label = 0;
				for (int j = 0; j < C.length(); j++)
				{
					if (C(j) == dataset[1](i))
					{
						label = j;
						break;
					}
				}

				// Do one-hot encoding
				target(i, label) = 1.0;
			}
		}
	}

	const double MLP::calculateMeanSquaredError(const numat& y, const nml::numat& t, const bool tr) const
	{
		// Check the training flag
		if (tr == true)
		{
			// Calculate the mean squared error
			double panelty = 0.0;
			if (reg->type == REGULARIZE_L1)
			{
				for (int i = 0; i < W.length(); i++)
				{
					panelty += numat::sum(numat::abs(W(i)));
				}

				return (numat::sum(0.5 * numat::sumr((y - t).mul(y - t))) + reg->lamda * panelty) / y.rows;
			}
			else if (reg->type == REGULARIZE_L2)
			{
				for (int i = 0; i < W.length(); i++)
				{
					panelty += numat::sum(W(i).mul(W(i)));
				}

				return (numat::sum(0.5 * numat::sumr((y - t).mul(y - t))) + 0.5 * reg->lamda * panelty) / y.rows;
			}
		}

		return numat::sum(0.5 * numat::sumr((y - t).mul(y - t))) / y.rows;
	}

	const double MLP::calculateCrossEntropyError(const numat& y, const nml::numat& t, const bool tr) const
	{
		// Check the training flag
		if (tr == true)
		{
			// Calculate the cross entropy error
			double panelty = 0.0;
			if (reg->type == REGULARIZE_L1)
			{
				for (int i = 0; i < W.length(); i++)
				{
					panelty += numat::sum(numat::abs(W(i)));
				}

				return (numat::sum(-numat::sumr(t.mul(numat::log(y)))) + reg->lamda * panelty) / y.rows;
			}
			else if (reg->type == REGULARIZE_L2)
			{
				for (int i = 0; i < W.length(); i++)
				{
					panelty += numat::sum(W(i).mul(W(i)));
				}

				return (numat::sum(-numat::sumr(t.mul(numat::log(y)))) + 0.5 * reg->lamda * panelty) / y.rows;
			}
		}

		return numat::sum(-numat::sumr(t.mul(numat::log(y)))) / y.rows;
	}

	const double MLP::calculateNegLogLikelihoodError(const numat& y, const nml::numat& t, const bool tr) const
	{
		// Check the training flag
		if (tr == true)
		{
			// Calculate the negative log likelihood error
			double panelty = 0.0;
			if (reg->type == REGULARIZE_L1)
			{
				for (int i = 0; i < W.length(); i++)
				{
					panelty += numat::sum(numat::abs(W(i)));
				}

				return (numat::sum(-numat::sumr(t.mul(y))) + reg->lamda * panelty) / y.rows;
			}
			else if (reg->type == REGULARIZE_L2)
			{
				for (int i = 0; i < W.length(); i++)
				{
					panelty += numat::sum(W(i).mul(W(i)));
				}

				return (numat::sum(-numat::sumr(t.mul(y))) + 0.5 * reg->lamda * panelty) / y.rows;
			}
		}

		return numat::sum(-numat::sumr(t.mul(y))) / y.rows;
	}

	void MLP::computeBackwardProps(const numat& target)
	{
		// Do backward propagations
		for (int i = W.length() - 1; i >= 0; i--)
		{
			// Calculate the delta values
			if (i == W.length() - 1)
			{
				// Derivate the output layer
				delta(i) = lout(i + 1) - target;
			}
			else
			{
				// Derivate the hidden layer
				delta(i) = (delta(i + 1) * getWeightMatrix(W(i + 1))).mul(net(i + 1).derivative());

				// Derivate the batch normalization layer
				delta(i) = reg->bnlayers[i].backward(delta(i));
			}

			// Calculate the gradients
			grad(i) = delta(i).T() * lout(i);
		}
	}

	const numat MLP::getWeightMatrix(const numat& Wi)
	{
		// Get the weight matrix
		numat mat(msize(Wi.rows, Wi.cols - 1));
		for (int i = 0; i < Wi.rows; i++)
		{
			for (int j = 1; j < Wi.cols; j++)
			{
				mat(i, j - 1) = Wi(i, j);
			}
		}

		return mat;
	}

	const numat MLP::getBiasMatrix(const numat& Wi)
	{
		// Get the bias matrix
		numat mat(msize(Wi.rows, 1));
		for (int i = 0; i < Wi.rows; i++)
		{
			mat(i) = Wi(i, 0);
		}

		return mat;
	}

	void MLP::updateNetwork(const double iter, const int N)
	{
		// Update the network weight
		for (int i = W.length() - 1; i >= 0; i--)
		{
			// Set an weight matrix for the update
			numat L = numat::zeros(msize(W(i).rows, 1)).happend(getWeightMatrix(W(i)));

			// Update the parameter space
			if (reg->type == REGULARIZE_L1)
			{
				W(i) = W(i) - (opt->epsilon * reg->lamda / N * getSignMatrix(L)) + opt->calculate(i, iter, grad(i) / N);
			}
			else if (reg->type == REGULARIZE_L2)
			{
				W(i) = (1.0 - opt->epsilon * reg->lamda / N) * L + opt->calculate(i, iter, grad(i) / N);
			}
			else
			{
				W(i) = L + opt->calculate(i, iter, grad(i) / N);
			}
		}

		// Update the batch normalization layer
		for (int i = W.length() - 2; i >= 0; i--)
		{
			// Update the scale parameters
			reg->bnlayers[i].update(opt->epsilon);
		}
	}

	const numat MLP::getSignMatrix(const numat& Wi)
	{
		// Generate a sign matrix
		numat sign(msize(Wi.rows, Wi.cols));
		for (int i = 0; i < Wi.length(); i++)
		{
			if (Wi(i) > 0.0)
			{
				sign(i) = 1.0;
			}
			else if (Wi(i) < 0.0)
			{
				sign(i) = -1.0;
			}
			else
			{
				sign(i) = 0.0;
			}
		}

		return sign;
	}

	const double MLP::predict(const numat& x)
	{
		// Initialize the parameters
		const int numlayers = net.length();
		if (nout.empty() == true)
		{
			nout = numem<numat>(msize(numlayers));
		}
		if (lout.empty() == true)
		{
			lout = numem<numat>(msize(numlayers));
		}

		// Set the input vector
		nout(0) = x;
		lout(0) = net(0).activative(nout(0));
		lout(0) = numat::ones(msize(1)).happend(lout(0));

		// Do forward propagations
		for (int i = 0; i < W.length(); i++)
		{
			// Activate the affine layer
			// from the current layer output to the next node output
			nout(i + 1) = lout(i) * W(i).T();

			// Check the current layer is output or not
			if (i != W.length() - 1)
			{
				// Check the batch normalization layer
				if (reg->bnlayers.empty() == false)
				{
					// Activate the batch normalization layer
					nout(i + 1) = reg->bnlayers[i].inference(nout(i + 1));
				}

				// Activatve the current nodes
				lout(i + 1) = numat::ones(msize(1)).happend(net(i + 1).activative(nout(i + 1)));
			}
			else
			{
				// Activatve the current nodes
				lout(i + 1) = net(i + 1).activative(nout(i + 1));
			}
		}

		// Check the train type
		if (type == MLP_TRAIN_REGRESSION)
		{
			return lout(numlayers - 1)(0);
		}
		else
		{
// 			return getArgmax(lout[numlayers - 1]);
			return getArgmax(numat::exp(lout(numlayers - 1)));
		}
	}

	const double MLP::getArgmax(const nml::numat& vec)
	{
		// Find an argmax value
		int maxIndex = 0;
		double maxValue = vec(0);
		for (int i = 1; i < vec.length(); i++)
		{
			if (vec(i) > maxValue)
			{
				maxValue = vec(i);
				maxIndex = i;
			}
		}

		return C(maxIndex);
	}

	const int MLP::open(const string path, const string prefix)
	{
		// Open train condition information
		if (openTrainCondInfo(path, prefix) != 0)
		{
			return -1;
		}

		// Open label information
		if (openLabelInfo(path, prefix) != 0)
		{
			return -1;
		}

		// Open layer information
		if (openLayerInfo(path, prefix) != 0)
		{
			return -1;
		}

		// Open the hidden architecture
		if (openHiddenArchitecture(path, prefix) != 0)
		{
			return -1;
		}

		// Open weight information
		if (openWeightInfo(path, prefix) != 0)
		{
			return -1;
		}

		return 0;
	}

	const int MLP::openTrainCondInfo(const string path, const string prefix)
	{
		// Create a result reader
		ifstream reader(path);
		if (reader.is_open() == false)
		{
			return -1;
		}

		// Find section name
		if (findSectionName(reader, getSectionName("MLP_LEARNING_INFO", prefix)) == false)
		{
			type = MLP_TRAIN_CLASSIFICATION;		// Default training mode
			reader.close();

			return -1;
		}

		// Find key name
		vector<string> splitStrs;
		string lineStr;
		string trimStr;
		double mu = -1.0;
		double sigma = -1.0;
		gwmode gw = GAUSSIAN_WEIGHT_UNKNOWN;
		int op = OPTIMIZER_UNKNOWN;
		double epsilon = -1.0;
		double delta = -1.0;
		double decay = -1.0;
		double beta1 = -1.0;
		double beta2 = -1.0;
		while (!reader.eof())
		{
			// Check the end of the section
			getline(reader, lineStr);
			trimStr = trim(lineStr);
			if (trimStr[0] == '[' && trimStr[trimStr.size() - 1] == ']')
			{
				break;
			}

			// Check the string format
			splitStrs = split(trimStr, "=");
			if (splitStrs.size() == 1)
			{
				continue;
			}

			// Check key name
			if (splitStrs[0] == "Target_Error")
			{
				E = atof(splitStrs[1].c_str());
				continue;
			}
			if (splitStrs[0] == "Max_Iter")
			{
				maxIter = atoi(splitStrs[1].c_str());
				continue;
			}
			if (splitStrs[0] == "Batch_Size")
			{
				N = atoi(splitStrs[1].c_str());
				continue;
			}
			if (splitStrs[0] == "Gaussian_Weight_Mean")
			{
				mu = atof(splitStrs[1].c_str());
				continue;
			}
			if (splitStrs[0] == "Gaussian_Weight_Sigma")
			{
				sigma = atof(splitStrs[1].c_str());
				continue;
			}
			if (splitStrs[0] == "Gaussian_Weight_Type")
			{
				gw = (gwmode)atoi(splitStrs[1].c_str());
				continue;
			}
			if (splitStrs[0] == "Train_Type")
			{
				type = (trtype)atoi(splitStrs[1].c_str());
				continue;
			}
			if (splitStrs[0] == "Optimizer_Type")
			{
				op = (optype)atoi(splitStrs[1].c_str());
				continue;
			}
			if (splitStrs[0] == "Epsilon")
			{
				epsilon = atof(splitStrs[1].c_str());
				continue;
			}
			if (splitStrs[0] == "Delta")
			{
				delta = atof(splitStrs[1].c_str());
				continue;
			}
			if (splitStrs[0] == "Decay")
			{
				decay = atof(splitStrs[1].c_str());
				continue;
			}
			if (splitStrs[0] == "Beta1")
			{
				beta1 = atof(splitStrs[1].c_str());
				continue;
			}
			if (splitStrs[0] == "Beta2")
			{
				beta2 = atof(splitStrs[1].c_str());
				continue;
			}
			if (splitStrs[0] == "Batch_Normalization_Num_Layers")
			{
				reg->bnlayers = vector<batchnorm>(atoi(splitStrs[1].c_str()));
				while (!reader.eof())
				{
					// Check the end of the section
					getline(reader, lineStr);
					trimStr = trim(lineStr);
					if (trimStr[0] == '[' && trimStr[trimStr.size() - 1] == ']')
					{
						break;
					}

					// Check the string format
					splitStrs = split(trimStr, "=");
					if (splitStrs.size() == 1)
					{
						continue;
					}

					// Check key format
					vector<string> indexStrs = split(splitStrs[0], "_");
					if (indexStrs.size() != 5)
					{
						continue;
					}

					// Check key name
					if (indexStrs[2] == "Activation" && indexStrs[3] == "Mode")
					{
						if (atoi(splitStrs[1].c_str()) == 1)
						{
							reg->bnlayers[atoi(indexStrs[4].c_str())].set(true);
						}
						else
						{
							reg->bnlayers[atoi(indexStrs[4].c_str())].set(false);
						}
						continue;
					}
					if (indexStrs[2] == "Num" && indexStrs[3] == "Nodes")
					{
						if (reg->bnlayers[atoi(indexStrs[4].c_str())].act == true)
						{
							reg->bnlayers[atoi(indexStrs[4].c_str())].mu_p = numat::ones(msize(1, atoi(splitStrs[1].c_str())));
							reg->bnlayers[atoi(indexStrs[4].c_str())].var_p = numat::ones(msize(1, atoi(splitStrs[1].c_str())));
							reg->bnlayers[atoi(indexStrs[4].c_str())].gamma = numat::ones(msize(1, atoi(splitStrs[1].c_str())));
							reg->bnlayers[atoi(indexStrs[4].c_str())].beta = numat::ones(msize(1, atoi(splitStrs[1].c_str())));
						}
						continue;
					}
					if (indexStrs[2] == "Mu")
					{
						if (reg->bnlayers[atoi(indexStrs[3].c_str())].act == true)
						{
							reg->bnlayers[atoi(indexStrs[3].c_str())].mu_p(atoi(indexStrs[4].c_str())) = atof(splitStrs[1].c_str());
						}
						continue;
					}
					if (indexStrs[2] == "Var")
					{
						if (reg->bnlayers[atoi(indexStrs[3].c_str())].act == true)
						{
							reg->bnlayers[atoi(indexStrs[3].c_str())].var_p(atoi(indexStrs[4].c_str())) = atof(splitStrs[1].c_str());
						}
						continue;
					}
					if (indexStrs[2] == "Gamma")
					{
						if (reg->bnlayers[atoi(indexStrs[3].c_str())].act == true)
						{
							reg->bnlayers[atoi(indexStrs[3].c_str())].gamma(atoi(indexStrs[4].c_str())) = atof(splitStrs[1].c_str());
						}
						continue;
					}
					if (indexStrs[2] == "Beta")
					{
						if (reg->bnlayers[atoi(indexStrs[3].c_str())].act == true)
						{
							reg->bnlayers[atoi(indexStrs[3].c_str())].beta(atoi(indexStrs[4].c_str())) = atof(splitStrs[1].c_str());
						}
						continue;
					}
				}
				break;
			}
		}
		reader.close();

		// Check the old memories
		if (init != nullptr)
		{
			delete init;
			init = nullptr;
		}
		if (opt != nullptr)
		{
			delete opt;
			opt = nullptr;
		}

		// Create an initializer
		if (mu != -1.0 && sigma != -1.0 && gw != GAUSSIAN_WEIGHT_UNKNOWN)
		{
			init = new initializer(mu, sigma, gw);
		}
		else
		{
			return -1;
		}

		// Create an optimizer
		switch (op)
		{
		case OPTIMIZER_SGD: 
			if (epsilon != -1.0)
			{
				opt = new sgd(epsilon);
			}
			else
			{
				return -1;
			} break;
		case OPTIMIZER_MOMENTUN: 
			if (epsilon != -1.0 && delta != -1.0)
			{
				opt = new momentum(epsilon, delta);
			}
			else
			{
				return -1;
			} break;
		case OPTIMIZER_NESTEROV: 
			if (epsilon != -1.0 && delta != -1.0)
			{
				opt = new nesterov(epsilon, delta);
			}
			else
			{
				return -1;
			} break;
		case OPTIMIZER_ADAGRAD: 
			if (epsilon != -1.0)
			{
				opt = new adagrad(epsilon);
			}
			else
			{
				return -1;
			} break;
		case OPTIMIZER_RMSPROP: 
			if (epsilon != -1.0 && delta != -1.0)
			{
				opt = new rmsprop(epsilon, delta);
			}
			else
			{
				return -1;
			} break;
		case OPTIMIZER_ADADELTA: 
			if (decay != -1.0)
			{
				opt = new adadelta(decay);
			}
			else
			{
				return -1;
			} break;
		case OPTIMIZER_ADAM: 
			if (epsilon != -1.0 && beta1 != -1.0 && beta2 != -1.0)
			{
				opt = new adam(epsilon, beta1, beta2);
			}
			else
			{
				return -1;
			} break;
		default: 
			return -1;
		}

		return 0;
	}

	const int MLP::openLabelInfo(const string path, const string prefix)
	{
		// Create a result reader
		ifstream reader(path);
		if (reader.is_open() == false)
		{
			return -1;
		}

		// Find section name
		if (findSectionName(reader, getSectionName("MLP_LABEL_INFO", prefix)) == false)
		{
			reader.close();

			return -1;
		}

		// Find key name
		vector<string> splitStrs;
		string lineStr;
		string trimStr;
		while (!reader.eof())
		{
			// Check the end of the section
			getline(reader, lineStr);
			trimStr = trim(lineStr);
			if (trimStr[0] == '[' && trimStr[trimStr.size() - 1] == ']')
			{
				break;
			}

			// Check the string format
			splitStrs = split(trimStr, "=");
			if (splitStrs.size() == 1)
			{
				continue;
			}

			// Check key name
			if (splitStrs[0] == "Num_C")
			{
				C = numat(msize(1, atoi(splitStrs[1].c_str())), 0.0);
				while (!reader.eof())
				{
					// Check the end of the section
					getline(reader, lineStr);
					trimStr = trim(lineStr);
					if (trimStr[0] == '[' && trimStr[trimStr.size() - 1] == ']')
					{
						break;
					}

					// Check the string format
					splitStrs = split(trimStr, "=");
					if (splitStrs.size() == 1)
					{
						continue;
					}

					// Check key format
					vector<string> indexStrs = split(splitStrs[0], "_");
					if (indexStrs.size() != 2)
					{
						continue;
					}

					// Set a value
					C(atoi(indexStrs[1].c_str())) = atof(splitStrs[1].c_str());
				}
				break;
			}
		}
		reader.close();

		return 0;
	}

	const int MLP::openLayerInfo(const string path, const string prefix)
	{
		// Create a result reader
		ifstream reader(path);
		if (reader.is_open() == false)
		{
			return -1;
		}

		// Find section name
		if (findSectionName(reader, getSectionName("MLP_LAYER_INFO", prefix)) == false)
		{
			reader.close();

			return -1;
		}

		// Find key name
		vector<string> splitStrs;
		string lineStr;
		string trimStr;
		while (!reader.eof())
		{
			// Check the end of the section
			getline(reader, lineStr);
			trimStr = trim(lineStr);
			if (trimStr[0] == '[' && trimStr[trimStr.size() - 1] == ']')
			{
				break;
			}

			// Check the string format
			splitStrs = split(trimStr, "=");
			if (splitStrs.size() == 1)
			{
				continue;
			}

			// Check key name
			if (splitStrs[0] == "Input_Neurons")
			{
				inode = atoi(splitStrs[1].c_str());
				continue;
			}
			if (splitStrs[0] == "Output_Neurons")
			{
				onode = atoi(splitStrs[1].c_str());
				continue;
			}
			if (splitStrs[0] == "Num_Hidden_Layers")
			{
				hnode = vector<int>(atoi(splitStrs[1].c_str()));
				while (!reader.eof())
				{
					// Check the end of the section
					getline(reader, lineStr);
					trimStr = trim(lineStr);
					if (trimStr[0] == '[' && trimStr[trimStr.size() - 1] == ']')
					{
						break;
					}

					// Check the string format
					splitStrs = split(trimStr, "=");
					if (splitStrs.size() == 1)
					{
						continue;
					}

					// Check key format
					vector<string> indexStrs = split(splitStrs[0], "_");
					if (indexStrs.size() != 3)
					{
						continue;
					}

					// Set a value
					hnode[atoi(indexStrs[2].c_str())] = atoi(splitStrs[1].c_str());
				}
				break;
			}
		}
		reader.close();

		// Check the network architecture
		if (inode != -1 && onode != -1 && hnode.empty() == false)
		{
			// Create a network architecture
			const int numlayers = 1 + (int)hnode.size() + 1;
			net = numem<netlayer>(msize(numlayers));
			hl = vector<netlayer>(numlayers - 2);
			for (int i = 0; i < numlayers - 2; i++)
			{
				hl[i] = netlayer(hnode[i]);
			}
		}
		else
		{
			return -1;
		}

		return 0;
	}

	const int MLP::openHiddenArchitecture(const string path, const string prefix)
	{
		// Create a result reader
		ifstream reader(path);
		if (reader.is_open() == false)
		{
			return -1;
		}

		// Find section name
		if (findSectionName(reader, getSectionName("MLP_WEIGHT_INFO", prefix)) == false)
		{
			reader.close();

			return -1;
		}

		// Find key name
		vector<string> splitStrs;
		string lineStr;
		string trimStr;
		nn::aftype aftype = nn::ACT_FUNC_UNKNOWN;
		double afparam = -1.0;
		int i = -1;
		int j = -1;
		int count = 0;
		while (!reader.eof())
		{
			// Check the end of the section
			getline(reader, lineStr);
			trimStr = trim(lineStr);
			if (trimStr[0] == '[' && trimStr[trimStr.size() - 1] == ']')
			{
				break;
			}

			// Check the string format
			splitStrs = split(trimStr, "=");
			if (splitStrs.size() == 1)
			{
				continue;
			}

			// Check key format
			vector<string> indexStrs = split(splitStrs[0], "_");
			if (indexStrs.size() != 4)
			{
				continue;
			}

			// Check key name
			if (indexStrs[0] == "Activation" && indexStrs[1] == "Func" && indexStrs[2] == "Type")
			{
				aftype = (nn::aftype)atoi(splitStrs[1].c_str());
				i = atoi(indexStrs[3].c_str());
			}
			if (indexStrs[0] == "Activation" && indexStrs[1] == "Func" && indexStrs[2] == "Param")
			{
				afparam = atof(splitStrs[1].c_str());
				j = atoi(indexStrs[3].c_str());
			}

			// Check the read index
			if (i != -1 && j != -1 && i == j)
			{
				// Create the hidden layer
				switch (aftype)
				{
				case nn::ACT_FUNC_LINEAR: hl[i].set(nn::linear(afparam)); break;
				case nn::ACT_FUNC_SIGMOID: hl[i].set(nn::sigmoid(afparam)); break;
				case nn::ACT_FUNC_HYPER_TANGENT: hl[i].set(nn::tanh(afparam)); break;
				case nn::ACT_FUNC_ReLU: hl[i].set(nn::relu()); break;
				case nn::ACT_FUNC_ReLU6: hl[i].set(nn::relu6()); break;
				case nn::ACT_FUNC_LEAKY_ReLU: hl[i].set(nn::leakyrelu()); break;
				case nn::ACT_FUNC_PReLU: hl[i].set(nn::prelu(afparam)); break;
				case nn::ACT_FUNC_ELU: hl[i].set(nn::elu(afparam)); break;
				case nn::ACT_FUNC_SOFTSIGN: hl[i].set(nn::softsign()); break;
				case nn::ACT_FUNC_SOFTPLUS: hl[i].set(nn::softplus()); break;
// 				case nn::ACT_FUNC_SOFTMAX: hl[i].set(nn::softmax()); break;
				case nn::ACT_FUNC_SOFTMAX: hl[i].set(nn::logsoftmax()); break;
				case nn::ACT_FUNC_IDENTITY: hl[i].set(nn::identity()); break;
				default: return -1;
				}

				// Check the architecture
				if (count != i)
				{
					reader.close();

					return -1;
				}
				else
				{
					count++;
				}

				// Initialize the index
				aftype = nn::ACT_FUNC_UNKNOWN;
				afparam = -1.0;
				i = -1;
				j = -1;
			}
		}
		reader.close();

		// Generate the network architecture
		for (int i = 0; i < net.length(); i++)
		{
			// Check the layer index
			if (i == 0)
			{
				// Set the input layer
				net(i) = netlayer(inode, nn::identity());
				net(i).set(NET_LAYER_INPUT);
			}
			else if (i == net.length() - 1)
			{
				// Set the output layer
// 				net(i) = (type == MLP_TRAIN_REGRESSION) ? netlayer(onode, nn::identity()) : netlayer(onode, nn::softmax());
				net(i) = (type == MLP_TRAIN_REGRESSION) ? netlayer(onode, nn::identity()) : netlayer(onode, nn::logsoftmax());
				net(i).set(NET_LAYER_OUTPUT);
			}
			else
			{
				// Set the hidden layer
				net(i) = hl[i - 1];
				net(i).set(NET_LAYER_HIDDEN);
			}
		}

		// Create an weight matrix
		W = numem<numat>(msize(net.length() - 1));
		for (int i = 0; i < W.length(); i++)
		{
			// Initialize the matrix memory
			W(i) = numat::zeros(msize(net(i + 1).node, 1 + net(i).node));
		}

		return 0;
	}

	const int MLP::openWeightInfo(const string path, const string prefix)
	{
		// Create a result reader
		ifstream reader(path);
		if (reader.is_open() == false)
		{
			return -1;
		}

		// Find section name
		if (findSectionName(reader, getSectionName("MLP_WEIGHT_INFO", prefix)) == false)
		{
			reader.close();

			return -1;
		}

		// Find key name
		vector<string> splitStrs;
		string lineStr;
		string trimStr;
		nn::aftype aftype = nn::ACT_FUNC_UNKNOWN;
		double afparam = -1.0;
		while (!reader.eof())
		{
			// Check the end of the section
			getline(reader, lineStr);
			trimStr = trim(lineStr);
			if (trimStr[0] == '[' && trimStr[trimStr.size() - 1] == ']')
			{
				break;
			}

			// Check the string format
			splitStrs = split(trimStr, "=");
			if (splitStrs.size() == 1)
			{
				continue;
			}

			// Check key format
			vector<string> indexStrs = split(splitStrs[0], "_");
			if (indexStrs.size() != 4)
			{
				continue;
			}

			// Check key name
			if (indexStrs[0] == "Weight")
			{
				// Set a value
				W(atoi(indexStrs[1].c_str()))(atoi(indexStrs[2].c_str()), atoi(indexStrs[3].c_str())) = atof(splitStrs[1].c_str());
			}
		}
		reader.close();

		return 0;
	}

	const int MLP::save(const string path, const string prefix)
	{
		// Create a result writer
		ofstream writer(path, ios::trunc);
		if (writer.is_open() == false)
		{
			return -1;
		}

		// Save train condition information
		writer << getSectionName("MLP_LEARNING_INFO", prefix) << endl;
		writer << "Target_Error=" << E << endl;
		writer << "Max_Iter=" << maxIter << endl;
		writer << "Batch_Size=" << N << endl;
		writer << "Gaussian_Weight_Mean=" << init->mu << endl;
		writer << "Gaussian_Weight_Sigma=" << init->sigma << endl;
		writer << "Gaussian_Weight_Type=" << init->mode << endl;
		writer << "Train_Type=" << type << endl;
		writer << "Optimizer_Type=" << opt->type << endl;
		switch (opt->type)
		{
		case OPTIMIZER_SGD:
		case OPTIMIZER_ADAGRAD:
			writer << "Epsilon=" << opt->epsilon << endl;
			break;
		case OPTIMIZER_MOMENTUN:
		case OPTIMIZER_NESTEROV:
			writer << "Epsilon=" << opt->epsilon << endl;
			writer << "Delta=" << opt->delta << endl;
			break;
		case OPTIMIZER_RMSPROP:
			writer << "Epsilon=" << opt->epsilon << endl;
			writer << "Decay=" << opt->decay << endl;
			break;
		case OPTIMIZER_ADADELTA:
			writer << "Decay=" << opt->decay << endl;
			break;
		case OPTIMIZER_ADAM:
			writer << "Epsilon=" << opt->epsilon << endl;
			writer << "Beta1=" << opt->beta1 << endl;
			writer << "Beta2=" << opt->beta2 << endl;
			break;
		}
		writer << "Regularizer_Lamda=" << reg->lamda << endl;
		for (int i = 0; i < (int)reg->dolayers.size(); i++)
		{
			writer << "Dropout_Probability_" << i << "=" << reg->dolayers[i].kprob << endl;
		}
		if (reg->bnlayers.empty() == false)
		{
			const int lcount = (int)reg->bnlayers.size();
			bool flag = false;
			for (int i = 0; i < lcount; i++)
			{
				if (reg->bnlayers[i].act == true)
				{
					flag = true;
					break;
				}
			}
			if (flag == true)
			{
				writer << "Batch_Normalization_Num_Layers=" << lcount << endl;
				for (int i = 0; i < lcount; i++)
				{
					if (reg->bnlayers[i].act == true)
					{
						writer << "Batch_Normalization_Activation_Mode_" << i << "=1" << endl;
					}
					else
					{
						writer << "Batch_Normalization_Activation_Mode_" << i << "=0" << endl;
					}
				}
				for (int i = 0; i < lcount; i++)
				{
					writer << "Batch_Normalization_Num_Nodes_" << i << "=" << reg->bnlayers[i].gamma.length() << endl;
				}
				for (int i = 0; i < lcount; i++)
				{
					writer << "Batch_Normalization_Momentum_Value_" << i << "=" << reg->bnlayers[i].momentum << endl;
				}
				for (int i = 0; i < lcount; i++)
				{
					for (int j = 0; j < reg->bnlayers[i].mu_p.length(); j++)
					{
						writer << "Batch_Normalization_Mu_" << i << "_" << j << "=" << reg->bnlayers[i].mu_p(j) << endl;
					}
				}
				for (int i = 0; i < lcount; i++)
				{
					for (int j = 0; j < reg->bnlayers[i].var_p.length(); j++)
					{
						writer << "Batch_Normalization_Var_" << i << "_" << j << "=" << reg->bnlayers[i].var_p(j) << endl;
					}
				}
				for (int i = 0; i < lcount; i++)
				{
					for (int j = 0; j < reg->bnlayers[i].gamma.length(); j++)
					{
						writer << "Batch_Normalization_Gamma_" << i << "_" << j << "=" << reg->bnlayers[i].gamma(j) << endl;
					}
				}
				for (int i = 0; i < lcount; i++)
				{
					for (int j = 0; j < reg->bnlayers[i].beta.length(); j++)
					{
						writer << "Batch_Normalization_Beta_" << i << "_" << j << "=" << reg->bnlayers[i].beta(j) << endl;
					}
				}
			}
		}
		writer << endl;

		// Save label information
		writer << getSectionName("MLP_LABEL_INFO", prefix) << endl;
		writer << "Num_C=" << C.length() << endl;
		for (int i = 0; i < C.length(); i++)
		{
			writer << "C_" << i << "=" << C(i) << endl;
		}
		writer << endl;

		// Save layer information
		writer << getSectionName("MLP_LAYER_INFO", prefix) << endl;
		writer << "Input_Neurons=" << net(0).node << endl;
		writer << "Output_Neurons=" << net(net.length() - 1).node << endl;
		writer << "Num_Hidden_Layers=" << net.length() - 2 << endl;
		for (int i = 1; i < net.length() - 1; i++)
		{
			writer << "Hidden_Neurons_" << i - 1 << "=" << net(i).node << endl;
		}
		writer << endl;

		// Save weight information
		writer << getSectionName("MLP_WEIGHT_INFO", prefix) << endl;
		for (int i = 0; i < (int)hl.size(); i++)
		{
			writer << "Activation_Func_Type_" << i << "=" << hl[i].getActFunc().type << endl;
			writer << "Activation_Func_Param_" << i << "=" << hl[i].getActFunc().alpha << endl;
		}
		for (int i = 0; i < W.length(); i++)
		{
			for (int y = 0; y < W(i).rows; y++)
			{
				for (int x = 0; x < W(i).cols; x++)
				{
					writer << "Weight_" << i << "_" << y << "_" << x << "=" << W(i)(y, x) << endl;
				}
			}
		}
		writer << endl;
		writer.close();

		return 0;
	}
}