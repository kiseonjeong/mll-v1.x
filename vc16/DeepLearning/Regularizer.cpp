#include "stdafx.h"
#include "Regularizer.h"

namespace mll
{
	regularizer::regularizer() : nhids(_nhids), lamda(_lamda), type(_type), beta(_beta), rho(_rho), momentum(_momentum)
	{
		// Set an object
		setObject();
	}

	regularizer::regularizer(const int nhids, const double lamda, const rgtype type, const vector<dropout>& dolayers, const vector<batchnorm>& bnlayers) : nhids(_nhids), lamda(_lamda), type(_type), beta(_beta), rho(_rho), momentum(_momentum)
	{
		// Set an object
		setObject();

		// Set the gaussian parameters
		set(nhids, lamda, type, dolayers, bnlayers);
	}

	regularizer::regularizer(const int nhids, const double lamda, const double beta, const double rho, const double momentum, const rgtype type, const vector<dropout>& dolayers, const vector<batchnorm>& bnlayers) : nhids(_nhids), lamda(_lamda), type(_type), beta(_beta), rho(_rho), momentum(_momentum)
	{
		// Set an object
		setObject();

		// Set the gaussian parameters
		set(nhids, lamda, beta, rho, momentum, type, dolayers, bnlayers);
	}

	regularizer::regularizer(const regularizer& obj) : nhids(_nhids), lamda(_lamda), type(_type), beta(_beta), rho(_rho), momentum(_momentum)
	{
		// Set an object
		setObject();

		// Clone the object
		*this = obj;
	}

	regularizer::~regularizer()
	{
		// Clear the object
		clearObject();
	}

	regularizer& regularizer::operator=(const regularizer& obj)
	{
		// Clear the object
		clearObject();

		// Set an object
		setObject();

		// Copy the object
		copyObject(obj);

		return *this;
	}

	void regularizer::setObject()
	{
		// Set the parameters
		setType(*this);
		_nhids = -1;
		_lamda = 0.0;
		_type = REGULARIZE_NONE;
		_beta = 1.0;
		_rho = 0.05;
		_momentum = 0.9;
	}

	void regularizer::copyObject(const object& obj)
	{
		// Do down casting
		const regularizer* _obj = static_cast<const regularizer*>(&obj);

		// Copy the parameters
		_nhids = _obj->_nhids;
		_lamda = _obj->_lamda;
		_type = _obj->_type;
		_beta = _obj->_beta;
		_rho = _obj->_rho;
		_momentum = _obj->_momentum;

		// Copy the memories
		dolayers = _obj->dolayers;
		bnlayers = _obj->bnlayers;
	}

	void regularizer::clearObject()
	{
		// Clear the memories
		dolayers.clear();
		bnlayers.clear();
	}

	void regularizer::set(const int nhids, const double lamda, const rgtype type, const vector<dropout>& dolayers, const vector<batchnorm>& bnlayers)
	{
		// Set the number of the hidden layers
		_nhids = nhids;

		// Set the wegith decay parameters
		_lamda = lamda;

		// Set the regularization type
		_type = type;

		// Check the dropout layers
		if (dolayers.empty() == true)
		{
			// Set the dropout layers
			this->dolayers.push_back(dropout(1.0, DROPOUT_INPUT_LAYER));
			for (int i = 0; i < nhids; i++)
			{
				this->dolayers.push_back(dropout(1.0, DROPOUT_HIDDEN_LAYER));
			}
		}
		else
		{
			// Set the dropout layers
			int icount = 0, hcount = 0;
			for (int i = 0; i < (int)dolayers.size(); i++)
			{
				if (dolayers[i].type == DROPOUT_INPUT_LAYER)
				{
					icount++;
				}
				else if (dolayers[i].type == DROPOUT_HIDDEN_LAYER)
				{
					hcount++;
				}
			}
			if (icount == 0)
			{
				this->dolayers.push_back(dropout(1.0, DROPOUT_INPUT_LAYER));
				icount++;
			}
			else
			{
				for (int i = 0; i < (int)dolayers.size(); i++)
				{
					this->dolayers.push_back(dolayers[i]);
				}
			}
			if (hcount == 0)
			{
				for (int i = 0; i < nhids; i++)
				{
					this->dolayers.push_back(dropout(1.0, DROPOUT_HIDDEN_LAYER));
					hcount++;
				}
			}
			else
			{
				for (int i = 0; i < (int)dolayers.size(); i++)
				{
					this->dolayers.push_back(dolayers[i]);
				}
			}
			assert(nhids + 1 == icount + hcount);
		}

		// Check the batch normalization layers
		if (bnlayers.empty() == true)
		{
			// Set the batch normalization layers
			for (int i = 0; i < nhids; i++)
			{
				this->bnlayers.push_back(batchnorm(false));
			}
		}
		else
		{
			// Set the batch normalization layers
			assert(nhids == (int)bnlayers.size());
			for (int i = 0; i < (int)bnlayers.size(); i++)
			{
				this->bnlayers.push_back(bnlayers[i]);
			}
		}
	}

	void regularizer::set(const int nhids, const double lamda, const double beta, const double rho, const double momentum, const rgtype type, const vector<dropout>& dolayers, const vector<batchnorm>& bnlayers)
	{
		// Set the number of the hidden layers
		_nhids = nhids;

		// Set the wegith decay parameters
		_lamda = lamda;

		// Set the regularization type
		_type = type;

		// Set the sparsity parameters
		_beta = beta;
		_rho = rho;
		_momentum = momentum;

		// Check the dropout layers
		if (dolayers.empty() == true)
		{
			// Set the dropout layers
			this->dolayers.push_back(dropout(1.0, DROPOUT_INPUT_LAYER));
			for (int i = 0; i < nhids; i++)
			{
				this->dolayers.push_back(dropout(1.0, DROPOUT_HIDDEN_LAYER));
			}
		}
		else
		{
			// Set the dropout layers
			int icount = 0, hcount = 0;
			for (int i = 0; i < (int)dolayers.size(); i++)
			{
				if (dolayers[i].type == DROPOUT_INPUT_LAYER)
				{
					icount++;
				}
				else if (dolayers[i].type == DROPOUT_HIDDEN_LAYER)
				{
					hcount++;
				}
			}
			if (icount == 0)
			{
				this->dolayers.push_back(dropout(1.0, DROPOUT_INPUT_LAYER));
				icount++;
			}
			else
			{
				for (int i = 0; i < (int)dolayers.size(); i++)
				{
					this->dolayers.push_back(dolayers[i]);
				}
			}
			if (hcount == 0)
			{
				for (int i = 0; i < nhids; i++)
				{
					this->dolayers.push_back(dropout(1.0, DROPOUT_HIDDEN_LAYER));
					hcount++;
				}
			}
			else
			{
				for (int i = 0; i < (int)dolayers.size(); i++)
				{
					this->dolayers.push_back(dolayers[i]);
				}
			}
			assert(nhids + 1 == icount + hcount);
		}

		// Check the batch normalization layers
		if (bnlayers.empty() == true)
		{
			// Set the batch normalization layers
			for (int i = 0; i < nhids; i++)
			{
				this->bnlayers.push_back(batchnorm(false));
			}
		}
		else
		{
			// Set the batch normalization layers
			assert(nhids == (int)bnlayers.size());
			for (int i = 0; i < (int)bnlayers.size(); i++)
			{
				this->bnlayers.push_back(bnlayers[i]);
			}
		}
	}
}