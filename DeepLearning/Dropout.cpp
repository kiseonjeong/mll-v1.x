#include "stdafx.h"
#include "Dropout.h"

namespace mll
{
	dropout::dropout() : type(_type), kprob(_kprob)
	{
		// Set an object
		setObject();
	}

	dropout::dropout(const double kprob, const dotype type) : type(_type), kprob(_kprob)
	{
		// Set an object
		setObject();

		// Set the parameters
		set(kprob, type);
	}

	dropout::dropout(const dropout& obj) : type(_type), kprob(_kprob)
	{
		// Set an object
		setObject();

		// Clone the object
		*this = obj;
	}

	dropout::~dropout()
	{
		// Clear the object
		clearObject();
	}

	dropout& dropout::operator=(const dropout& obj)
	{
		// Clear the object
		clearObject();

		// Set an object
		setObject();

		// Copy the object
		copyObject(obj);

		return *this;
	}

	void dropout::setObject()
	{
		// Set the parameters
		setType(*this);

		// Set the parameters
		_type = DROPOUT_HIDDEN_LAYER;
		_kprob = 1.0;

		// Set the memories
		M.release();
	}

	void dropout::copyObject(const object& obj)
	{
		// Do down casting
		dropout* _obj = (dropout*)&obj;

		// Copy the parameters
		_type = _obj->_type;
		_kprob = _obj->_kprob;

		// Copy the memories
		M = _obj->M;
	}

	void dropout::clearObject()
	{
		// Clear the memories
		M.release();
	}

	void dropout::set(const double kprob, const dotype type)
	{
		// Set a keep probability
		if (kprob < 0.0)
		{
			_kprob = 0.0;
		}
		else if (kprob > 1.0)
		{
			_kprob = 1.0;
		}
		else
		{
			_kprob = kprob;
		}
		
		// Set a layer type
		_type = type;
	}

	void dropout::generate(const int length)
	{
		// Create an uniform random number generator
		random_device dev;
		mt19937 gen(dev());
		uniform_real_distribution<double> rand(0.0, 1.0);

		// Generate a selection matrix
		M = numat::ones(msize(1, length));
		for (int i = 1; i < length; i++)
		{
			if (rand(gen) > _kprob)
			{
				M(i) = 0.0;
			}
			else
			{
				M(i) = 1.0;
			}
		}
	}

	const numat dropout::forward(const numat& net) const
	{
		// Activative the data
		numat out = net;
		for (int i = 0; i < net.rows; i++)
		{
			for (int j = 1; j < net.cols; j++)
			{
				if (_kprob == 0.0)
				{
					out(i, j) = 0.0;
				}
				else
				{
					out(i, j) = net(i, j) * M(j) / _kprob;
				}
			}
		}

		return out;
	}

	const numat dropout::backward(const numat& dout) const
	{
		// Derivative the data
		numat deriv(msize(dout.rows, dout.cols), 0.0);
		for (int i = 0; i < dout.rows; i++)
		{
			for (int j = 0; j < dout.cols; j++)
			{
				deriv(i, j) = dout(i, j) * M(j);
			}
		}

		return deriv;
	}
}