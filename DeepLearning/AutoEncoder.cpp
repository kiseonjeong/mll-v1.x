#include "stdafx.h"
#include "AutoEncoder.h"

namespace mll
{
	autoEncoder::autoEncoder()
	{
		// Set an object
		setObject();
	}

	autoEncoder::autoEncoder(const int N, const double E, const int maxIter, const vector<netlayer>& hl, const initializer& init, const optimizer& opt, const regularizer& reg)
	{
		// Set an object
		setObject();

		// Set a train condition
		condition(N, E, maxIter, hl, init, opt, reg);
	}

	autoEncoder::autoEncoder(const mlldata& dataset, const int N, const double E, const int maxIter, const vector<netlayer>& hl, const initializer& init, const optimizer& opt, const regularizer& reg)
	{
		// Set an object
		setObject();

		// Train the dataset
		train(dataset, N, E, maxIter, hl, init, opt, reg);
	}

	autoEncoder::autoEncoder(const autoEncoder& obj)
	{
		// Set an object
		setObject();

		// Clone the object
		*this = obj;
	}

	autoEncoder::~autoEncoder()
	{
		// Clear the object
		clearObject();
	}

	autoEncoder& autoEncoder::operator=(const autoEncoder& obj)
	{
		// Clear the object
		clearObject();

		// Set an object
		setObject();

		// Copy the object
		copyObject(obj);

		return *this;
	}

	void autoEncoder::setObject()
	{
		// Set the parameters
		MLP::setObject();
		setType(*this);
		type = MLP_TRAIN_GENERATION;

		// Set the memories

	}

	void autoEncoder::copyObject(const object& obj)
	{
		// Do down casting
		MLP::copyObject(obj);
		const autoEncoder* _obj = static_cast<const autoEncoder*>(&obj);

		// Copy the parameters
		sparsity = _obj->sparsity;

		// Copy the memories
		rho0 = _obj->rho0;
		rho1 = _obj->rho1;
		KL = _obj->KL;
	}

	void autoEncoder::clearObject()
	{
		// Clear the memories
		MLP::clearObject();
		rho0.clear();
		rho1.clear();
		KL.clear();
	}

	void autoEncoder::condition(const int N, const double E, const int maxIter, const std::vector<netlayer>& hl, const initializer& init, const optimizer& opt, const regularizer& reg)
	{
		// Set the conditions
		this->type = MLP_TRAIN_GENERATION;
		this->N = N;
		this->E = E;
		this->maxIter = maxIter;
		this->hl = hl;
		copyInitializer(init);
		copyOptimizer(opt);
		copyRegularizer(reg);
	}

	void autoEncoder::train(const mlldata& dataset)
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

		// Check the regularization type
		if (reg->type == REGULARIZE_SPARSITY)
		{
			sparsity = true;
		}
		else
		{
			sparsity = false;
		}

		// Check the activation functions
		for (int i = 0; i < (int)hl.size(); i++)
		{
			if (hl[i].getActFunc().type != nn::ACT_FUNC_SIGMOID)
			{
				sparsity = false;
				break;
			}
		}

		// Optimize the network weight matrix using the Back Propagation Algorithm
		numat target;
		for (int i = 0; i < maxIter; i++)
		{
			// Generate the mini batch dataset
			vector<mlldata> mini = dataset.subdata(N);

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

				// Initialize the sparsity parameters
				initSparsity();

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

	void autoEncoder::train(const mlldata& dataset, const int N, const double E, const int maxIter, const vector<netlayer>& hl, const initializer& init, const optimizer& opt, const regularizer& reg)
	{
		// Set a train condition
		condition(N, E, maxIter, hl, init, opt, reg);

		// Train the dataset
		train(dataset);
	}

	void autoEncoder::backupDataset(const mlldata& dataset)
	{
		// Check the train type
		if (type == MLP_TRAIN_REGRESSION)
		{
			// Backup the dataset
			X = dataset[0];
			T = dataset[0];
		}
		else
		{
			// Backup the dataset
			X = dataset[0];
			T = dataset[0];
		}

		// Set an input node
		inode = 1 + X.cols;

		// Set hidden nodes
		for (int i = 0; i < (int)hl.size(); i++)
		{
			hnode.push_back(1 + hl[i].node);
		}

		// Set an output node
		onode = inode - 1;
	}

	void autoEncoder::createNetwork()
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
				net(i) = netlayer(T.cols, nn::identity());
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

	void autoEncoder::initSparsity()
	{
		// Check the number of the hidden layers
		const int n = (int)hl.size();

		// Check the memory status
		if (rho1.empty() == true)
		{
			// Create the memories
			for (int i = 0; i < n; i++)
			{
				rho1.push_back(numat::zeros(msize(1, hnode[i] - 1)));
			}
		}
		else
		{
			// Initialize the memories
			for (int i = 0; i < n; i++)
			{
				for (int j = 0; j < rho1[i].length(); j++)
				{
					rho1[i](j) = 0.0;
				}
			}
		}
		if (KL.empty() == true)
		{
			// Create the memories
			for (int i = 0; i < n; i++)
			{
				KL.push_back(numat::zeros(msize(1, hnode[i] - 1)));
			}
		}
	}

	const double autoEncoder::computeForwardProps(const mlldata& dataset, numat& target, const bool tr)
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

					// Check the sparsity flag
					if (sparsity == true)
					{
						// Calculate activations on the units
						rho1[i] = numat::sumc(net(i + 1).activative(nout(i + 1))) / X.rows;

						// Calculate moving average activations
						if (rho0.size() != rho1.size())
						{
							rho0.push_back(rho1[i]);
						}
						else
						{
							rho1[i] = reg->momentum * rho0[i] + (1.0 - reg->momentum) * rho1[i];
							rho0[i] = rho1[i];
						}

						// Calculate a KL divergence for the regularization
						KL[i] = reg->rho * numat::log(reg->rho / rho1[i]) + (1.0 - reg->rho) * numat::log((1.0 - reg->rho) / (1.0 - rho1[i]));
					}
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

		// Calculate the mean squared error
		return calculateMeanSquaredError(lout(lout.length() - 1), target, tr);
	}

	void autoEncoder::setInoutVectors(const mlldata& dataset, numat& target)
	{
		// Set the input vector
		nout(0) = dataset[0];
		lout(0) = net(0).activative(nout(0));
		lout(0) = numat::ones(msize(dataset[0].rows, 1)).happend(lout(0));

		// Set a target value
		target = dataset[1];
	}

	const double autoEncoder::calculateMeanSquaredError(const numat& y, const numat& t, const bool tr) const
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
			else if (reg->type == REGULARIZE_SPARSITY)
			{
				for (int i = 0; i < (int)KL.size(); i++)
				{
					for (int j = 0; j < KL[i].length(); j++)
					{
						panelty += KL[i](j);
					}
				}

				return numat::sum(0.5 * numat::sumr((y - t).mul(y - t))) / y.rows + reg->beta * panelty;
			}
		}

		return numat::sum(0.5 * numat::sumr((y - t).mul(y - t))) / y.rows;
	}

	void autoEncoder::computeBackwardProps(const numat& target)
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
				// Check the sparsity flag
				if (sparsity == true)
				{
					// Derivate the hidden layer
					delta(i) = (delta(i + 1) * getWeightMatrix(W(i + 1)) + numat::ones(msize(delta(i + 1).rows, 1)) * (reg->beta * (-reg->rho / rho1[i] + (1.0 + reg->rho) / (1.0 - rho1[i])))).mul(net(i + 1).derivative());
				}
				else
				{
					// Derivate the hidden layer
					delta(i) = (delta(i + 1) * getWeightMatrix(W(i + 1))).mul(net(i + 1).derivative());
				}

				// Derivate the batch normalization layer
				delta(i) = reg->bnlayers[i].backward(delta(i));
			}

			// Calculate the gradients
			grad(i) = delta(i).T() * lout(i);
		}
	}

	const numat autoEncoder::generate(const numat& x)
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

		return lout(numlayers - 1);
	}

	const int autoEncoder::openTrainCondInfo(const string path, const string prefix)
	{
		// Create a result reader
		ifstream reader(path);
		if (reader.is_open() == false)
		{
			return -1;
		}

		// Find section name
		if (findSectionName(reader, getSectionName("AE_LEARNING_INFO", prefix)) == false)
		{
			type = MLP_TRAIN_GENERATION;		// Default training mode
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
						reg->bnlayers[atoi(indexStrs[4].c_str())].mu_p = numat::ones(msize(1, atoi(splitStrs[1].c_str())));
						reg->bnlayers[atoi(indexStrs[4].c_str())].var_p = numat::ones(msize(1, atoi(splitStrs[1].c_str())));
						reg->bnlayers[atoi(indexStrs[4].c_str())].gamma = numat::ones(msize(1, atoi(splitStrs[1].c_str())));
						reg->bnlayers[atoi(indexStrs[4].c_str())].beta = numat::ones(msize(1, atoi(splitStrs[1].c_str())));
						continue;
					}
					if (indexStrs[2] == "Mu")
					{
						reg->bnlayers[atoi(indexStrs[3].c_str())].mu_p(atoi(indexStrs[4].c_str())) = atof(splitStrs[1].c_str());
						continue;
					}
					if (indexStrs[2] == "Var")
					{
						reg->bnlayers[atoi(indexStrs[3].c_str())].var_p(atoi(indexStrs[4].c_str())) = atof(splitStrs[1].c_str());
						continue;
					}
					if (indexStrs[2] == "Gamma")
					{
						reg->bnlayers[atoi(indexStrs[3].c_str())].gamma(atoi(indexStrs[4].c_str())) = atof(splitStrs[1].c_str());
						continue;
					}
					if (indexStrs[2] == "Beta")
					{
						reg->bnlayers[atoi(indexStrs[3].c_str())].beta(atoi(indexStrs[4].c_str())) = atof(splitStrs[1].c_str());
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

	const int autoEncoder::openLabelInfo(const string path, const string prefix)
	{
		// Do nothing
		return 0;
	}

	const int autoEncoder::openLayerInfo(const string path, const string prefix)
	{
		// Create a result reader
		ifstream reader(path);
		if (reader.is_open() == false)
		{
			return -1;
		}

		// Find section name
		if (findSectionName(reader, getSectionName("AE_LAYER_INFO", prefix)) == false)
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

	const int autoEncoder::openHiddenArchitecture(const string path, const string prefix)
	{
		// Create a result reader
		ifstream reader(path);
		if (reader.is_open() == false)
		{
			return -1;
		}

		// Find section name
		if (findSectionName(reader, getSectionName("AE_WEIGHT_INFO", prefix)) == false)
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
				net(i) = netlayer(onode, nn::identity());
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

	const int autoEncoder::openWeightInfo(const string path, const string prefix)
	{
		// Create a result reader
		ifstream reader(path);
		if (reader.is_open() == false)
		{
			return -1;
		}

		// Find section name
		if (findSectionName(reader, getSectionName("AE_WEIGHT_INFO", prefix)) == false)
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

	const int autoEncoder::save(const string path, const string prefix)
	{
		// Create a result writer
		ofstream writer(path, ios::trunc);
		if (writer.is_open() == false)
		{
			return -1;
		}

		// Save train condition information
		writer << getSectionName("AE_LEARNING_INFO", prefix) << endl;
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

		// Save layer information
		writer << getSectionName("AE_LAYER_INFO", prefix) << endl;
		writer << "Input_Neurons=" << net(0).node << endl;
		writer << "Output_Neurons=" << net(net.length() - 1).node << endl;
		writer << "Num_Hidden_Layers=" << net.length() - 2 << endl;
		for (int i = 1; i < net.length() - 1; i++)
		{
			writer << "Hidden_Neurons_" << i - 1 << "=" << net(i).node << endl;
		}
		writer << endl;

		// Save weight information
		writer << getSectionName("AE_WEIGHT_INFO", prefix) << endl;
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