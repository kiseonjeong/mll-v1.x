#include "stdafx.h"
#include "Initializer.h"

namespace mll
{
	initializer::initializer() : mu(_mu), sigma(_sigma), mode(_mode)
	{
		// Set an object
		setObject();
	}

	initializer::initializer(const double mu, const double sigma, const gwmode mode) : mu(_mu), sigma(_sigma), mode(_mode)
	{
		// Set an object
		setObject();

		// Set the gaussian parameters
		set(mu, sigma, mode);
	}

	initializer::initializer(const initializer& obj) : mu(_mu), sigma(_sigma), mode(_mode)
	{
		// Set an object
		setObject();

		// Clone the object
		*this = obj;
	}

	initializer::~initializer()
	{
		// Clear the object
		clearObject();
	}

	initializer& initializer::operator=(const initializer& obj)
	{
		// Clear the object
		clearObject();

		// Set an object
		setObject();

		// Copy the object
		copyObject(obj);

		return *this;
	}

	void initializer::setObject()
	{
		// Set the parameters
		setType(*this);
		_mu = 0.0;
		_sigma = 1.0;
		_mode = GAUSSIAN_WEIGHT_AUTO;
	}

	void initializer::copyObject(const object& obj)
	{
		// Do down casting
		initializer* _obj = (initializer*)&obj;

		// Copy the parameters
		_mu = _obj->_mu;
		_sigma = _obj->_sigma;
		_mode = _obj->_mode;
	}

	void initializer::clearObject()
	{

	}

	void initializer::set(const double mu, const double sigma, const gwmode mode)
	{
		// Set the gaussian parameters
		this->_mu = mu;
		this->_sigma = sigma;
		this->_mode = mode;
	}

	void initializer::generate(const int aftype, numat& W) const
	{
		// Set the gaussian parameters
		double mean = _mu;
		double stddev = _sigma;
		double nnode = W.cols - 1;
		switch (mode)
		{
		case GAUSSIAN_WEIGHT_XAVIER:
			stddev = sqrt(1.0 / nnode);
			break;
		case GAUSSIAN_WEIGHT_HE:
			stddev = sqrt(2.0 / nnode);
			break;
		case GAUSSIAN_WEIGHT_AUTO:
			if (aftype == nn::ACT_FUNC_ReLU || aftype == nn::ACT_FUNC_ReLU6)
			{
				stddev = sqrt(2.0 / nnode);
			}
			else
			{
				stddev = sqrt(1.0 / nnode);
			}
			break;
		}

		// Generate a gaussian weight matrix
		for (int i = 0; i < W.rows; i++)
		{
			random_device rd;
			mt19937 gen(rd());
			normal_distribution<double> dist(mean, stddev);
			for (int j = 1; j < W.cols; j++)
			{
				W(i, j) = dist(gen);
			}
		}
	}
}